# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A unit source for GPTQ / AutoRound int4 MoE checkpoints in safetensors.

It lets the expert store be built outside the engine, on the CPU, straight
from a model's weight files: ``UnitSource`` for
``expert_store_build build --source checkpoint --model <dir>``.

What it does, and what it does not do itself:

1. It reads the checkpoint's safetensors headers (all shard files, or the ones
   the index names), finds the routed experts' tensors by a name pattern, and
   reads *only* the byte ranges of the units asked for (``os.preadv`` at the
   header offsets; nothing is mapped or loaded whole).
2. It stacks gate and up into one ``w13`` input and hands the tensors to the
   engine's own functions: :func:`...xpu_expert_tiers.kernel_layout_unit` (the
   transpose, the kernels' ``implement_zp`` and the scales' cast) for the
   bytes, :func:`...expert_tier_store.build_schema`, ``checkpoint_identity``
   and ``store_identity`` plus the kernels' transform fingerprint for the
   schema and the identity. The transform is not re-implemented here.
3. ``implement_zp`` is taken from the kernels package's own source file
   without importing the package (its import loads native extensions and
   needs the accelerator runtime): see :func:`load_kernels_transform`.

The unit is the full expert (no tensor-parallel shard), which is what the
engine's store holds when experts are placed by expert parallelism.
"""

import ast
import json
import os
import re
import struct
import sys
import threading
import types
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.machinery import PathFinder
from types import SimpleNamespace
from typing import Any

import torch

# Name patterns of the routed experts' tensors, tried in order unless one is
# given. ``{layer}``, ``{expert}``, ``{proj}`` and ``{kind}`` are the fields.
DEFAULT_NAME_FORMATS = (
    "model.language_model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}",
    "model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}",
    "language_model.model.layers.{layer}.mlp.experts.{expert}.{proj}.{kind}",
    "model.layers.{layer}.block_sparse_moe.experts.{expert}.{proj}.{kind}",
)
GATE, UP, DOWN = "gate_proj", "up_proj", "down_proj"
QWEIGHT, QZEROS, SCALES = "qweight", "qzeros", "scales"

_DTYPES: dict[str, torch.dtype] = {
    "I32": torch.int32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "F32": torch.float32,
    "U8": torch.uint8,
    "I8": torch.int8,
    "I64": torch.int64,
    "BOOL": torch.bool,
}
_ITEMSIZE = {"I32": 4, "F16": 2, "BF16": 2, "F32": 4, "U8": 1, "I8": 1, "I64": 8}
_SCALE_DTYPES = ("F16", "BF16", "F32")
# Symmetric int4: the zero point is 8, stored as 7 (GPTQ v1 packs zp-1) or 8.
_SYM_QZEROS = (0x77777777, -0x77777778)
_MAX_HEADER = 1 << 30
_TORCH_DTYPE_NAMES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


class CheckpointError(ValueError):
    """The checkpoint is not what this reader can turn into a store."""


@dataclass(frozen=True)
class TensorRef:
    shard: int
    dtype: str
    shape: tuple[int, ...]
    offset: int
    nbytes: int


# --------------------------------------------------------------------------
# The kernels' transform, without the kernels' native code
# --------------------------------------------------------------------------


def load_kernels_transform(src: str | None = None) -> Any:
    """Make ``vllm_xpu_kernels.fused_moe_interface.implement_zp`` importable
    as the engine imports it, running the kernels' own source for it.

    Importing the package for real loads its native extensions (and, on an
    accelerator host, the device runtime), and its modules evaluate
    ``torch.ops`` attributes at import time, so they cannot be imported
    without them. ``implement_zp`` is pure torch: this finds the package's
    ``fused_moe_interface.py`` without executing the package, takes that
    function's source out of the file, compiles it and registers a module that
    holds it under the real module's name. A function that needs anything but
    ``torch`` fails at the probe run below, loudly.

    ``src`` is a directory holding the package (a source tree); by default the
    installed package is used. An already imported (or faked) module is kept.
    """
    name = "vllm_xpu_kernels.fused_moe_interface"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    # the path finder, not find_spec: a faked or partly built module in
    # sys.modules must not stand in for the files
    spec = PathFinder.find_spec("vllm_xpu_kernels", [src] if src else None)
    if spec is None or not spec.submodule_search_locations:
        raise ImportError(
            "vllm_xpu_kernels is not installed here; give the directory that "
            "holds the package (--kernels-src)"
        )
    path = os.path.join(
        list(spec.submodule_search_locations)[0], "fused_moe_interface.py"
    )
    with open(path) as handle:
        source = handle.read()
    tree = ast.parse(source, filename=path)
    node = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "implement_zp"
        ),
        None,
    )
    if node is None:
        raise ImportError(f"{path}: no function implement_zp")
    module = types.ModuleType(name)
    module.__file__ = path
    module.__dict__["torch"] = torch
    code = compile(
        ast.Module(body=[node], type_ignores=[]), path, "exec"
    )  # keeps the kernels' own file and line numbers in tracebacks
    exec(code, module.__dict__)
    with torch.device("cpu"):
        probe = module.implement_zp(torch.arange(256, dtype=torch.uint8, device="cpu"))
    if probe.dtype != torch.uint8 or probe.shape != (256,):
        raise ImportError(f"{path}: implement_zp gave {probe.dtype} {probe.shape}")
    if "vllm_xpu_kernels" not in sys.modules:
        parent = types.ModuleType("vllm_xpu_kernels")
        parent.__path__ = list(spec.submodule_search_locations)
        sys.modules["vllm_xpu_kernels"] = parent
    sys.modules[name] = module
    return module


# --------------------------------------------------------------------------
# Safetensors headers
# --------------------------------------------------------------------------


class SafetensorsDir:
    """The tensors of a model directory's safetensors files, by header.

    With a ``*.safetensors.index.json`` only the files its ``weight_map`` names
    are read (the engine's loader ignores the others); without one, every
    ``*.safetensors`` file is. Headers are parsed once, at construction;
    tensor bytes are read on demand with ``preadv``.
    """

    def __init__(self, path: str):
        self.path = path
        names = sorted(f for f in os.listdir(path) if f.endswith(".safetensors"))
        self.index_file: str | None = None
        for f in sorted(os.listdir(path)):
            if f.endswith(".safetensors.index.json"):
                self.index_file = f
                break
        self.ignored_files: list[str] = []
        self.index_map: dict[str, str] | None = None
        if self.index_file:
            with open(os.path.join(path, self.index_file)) as handle:
                self.index_map = json.load(handle)["weight_map"]
            named = set(self.index_map.values())
            missing = sorted(named - set(names))
            if missing:
                raise CheckpointError(
                    f"{self.index_file} names files that are not here: {missing[:3]}"
                )
            self.ignored_files = [n for n in names if n not in named]
            names = [n for n in names if n in named]
        if not names:
            raise CheckpointError(f"{path}: no safetensors files")
        self.files = names
        self.refs: dict[str, TensorRef] = {}
        self._fds: dict[int, int] = {}
        self._lock = threading.Lock()
        for shard, name in enumerate(self.files):
            self._parse(shard, os.path.join(path, name))
        if self.index_map is not None:
            wrong = [
                n
                for n, f in self.index_map.items()
                if n in self.refs and self.files[self.refs[n].shard] != f
            ]
            if wrong:
                raise CheckpointError(
                    f"{self.index_file}: {len(wrong)} tensors sit in another file "
                    f"than the index says, first {wrong[0]!r}"
                )

    def _parse(self, shard: int, file: str) -> None:
        with open(file, "rb") as handle:
            size = os.fstat(handle.fileno()).st_size
            raw = handle.read(8)
            if len(raw) != 8:
                raise CheckpointError(f"{file}: shorter than a header")
            (length,) = struct.unpack("<Q", raw)
            if length > min(_MAX_HEADER, size - 8):
                raise CheckpointError(f"{file}: header length {length} is not valid")
            header = json.loads(handle.read(length))
        base = 8 + length
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            dtype, shape = entry["dtype"], tuple(entry["shape"])
            begin, end = entry["data_offsets"]
            if dtype in _ITEMSIZE:
                want = _ITEMSIZE[dtype]
                for dim in shape:
                    want *= dim
                if want != end - begin:
                    raise CheckpointError(
                        f"{file}: {name} is {end - begin} B, its shape says {want}"
                    )
            if base + end > size:
                raise CheckpointError(f"{file}: {name} runs past the end (truncated)")
            if name in self.refs:
                raise CheckpointError(f"{name} is in two files")
            self.refs[name] = TensorRef(shard, dtype, shape, base + begin, end - begin)

    def read(self, name: str) -> torch.Tensor:
        ref = self.refs[name]
        if ref.dtype not in _DTYPES:
            raise CheckpointError(f"{name}: dtype {ref.dtype} not handled")
        fd = self._fd(ref.shard)
        buf = bytearray(ref.nbytes)
        view = memoryview(buf)
        got = 0
        while got < ref.nbytes:
            n = os.preadv(fd, [view[got:]], ref.offset + got)
            if n <= 0:
                raise CheckpointError(f"{name}: file ended at {got} of {ref.nbytes} B")
            got += n
        return torch.frombuffer(buf, dtype=_DTYPES[ref.dtype]).reshape(ref.shape)

    def _fd(self, shard: int) -> int:
        fd = self._fds.get(shard)
        if fd is None:
            with self._lock:
                fd = self._fds.get(shard)
                if fd is None:
                    fd = os.open(
                        os.path.join(self.path, self.files[shard]), os.O_RDONLY
                    )
                    self._fds[shard] = fd
        return fd

    def close(self) -> None:
        with self._lock:
            for fd in self._fds.values():
                os.close(fd)
            self._fds.clear()


# --------------------------------------------------------------------------
# The source
# --------------------------------------------------------------------------


def _name_regex(fmt: str, projs: tuple[str, ...], kinds: tuple[str, ...]) -> re.Pattern:
    rx = re.escape(fmt)
    fields = {
        "layer": r"(?P<layer>\d+)",
        "expert": r"(?P<expert>\d+)",
        "proj": "(?P<proj>" + "|".join(re.escape(p) for p in projs) + ")",
        "kind": "(?P<kind>" + "|".join(re.escape(k) for k in kinds) + ")",
    }
    for field, pattern in fields.items():
        token = re.escape("{" + field + "}")
        if token not in rx:
            raise CheckpointError(f"name_format {fmt!r} has no {{{field}}}")
        rx = rx.replace(token, pattern)
    return re.compile(rx)


def _read_config(path: str) -> dict[str, Any]:
    """The numbers of ``config.json`` this reader cross-checks (all optional)."""
    file = os.path.join(path, "config.json")
    if not os.path.isfile(file):
        return {}
    with open(file) as handle:
        cfg = json.load(handle)
    text = cfg.get("text_config") or cfg
    quant = cfg.get("quantization_config") or text.get("quantization_config") or {}

    def first(*keys: str) -> Any:
        for src in (text, cfg):
            for key in keys:
                if src.get(key) is not None:
                    return src[key]
        return None

    return {
        "hidden_size": first("hidden_size"),
        "moe_intermediate_size": first("moe_intermediate_size"),
        "num_experts": first("num_experts", "num_local_experts", "n_routed_experts"),
        "num_hidden_layers": first("num_hidden_layers"),
        "bits": quant.get("bits"),
        "group_size": quant.get("group_size"),
        "sym": quant.get("sym"),
        "quant_method": quant.get("quant_method"),
    }


# (allowed safetensors dtypes, shape) of one tensor of a unit
_Expect = tuple[tuple[str, ...], tuple[int, ...]]


class CheckpointExperts:
    """``UnitSource`` over a GPTQ/AutoRound int4 MoE checkpoint.

    ``model``: the checkpoint directory (read here). The engine's identity of
    the store is a function of how the engine names the checkpoint, so the
    values the engine will have are given: ``engine_model`` (the string it is
    served with; default ``model``), ``engine_quantization`` (what the engine
    resolved and logs as ``quantization=``; required), ``engine_dtype``
    (default ``bfloat16``; also the scales' dtype in the units) and
    ``engine_revision``. ``name_format`` overrides the tensor name pattern
    (fields ``{layer}``, ``{expert}``, ``{proj}``, ``{kind}``), ``proj_names``
    the (gate, up, down) names, ``kernels_src`` where the kernels' source is.
    ``strict_qzeros`` makes a non-symmetric checkpoint an error instead of a
    count in :meth:`info` (the engine ignores zero points on XPU either way).
    """

    def __init__(
        self,
        model: str,
        engine_quantization: str | None = None,
        engine_model: str | None = None,
        engine_dtype: str = "bfloat16",
        engine_revision: str | None = None,
        name_format: str | None = None,
        proj_names: Sequence[str] = (GATE, UP, DOWN),
        kernels_src: str | None = None,
        strict_qzeros: bool = False,
        check_qzeros: bool = True,
    ):
        from vllm.model_executor.layers.fused_moe import xpu_expert_tiers as xt
        from vllm.model_executor.offloader import expert_tier_store as tier_store

        if engine_quantization is None:
            raise CheckpointError(
                "engine_quantization is required: the store's identity includes "
                "the quantization name the engine resolves (the engine logs it "
                "as quantization=...; --engine-quantization)"
            )
        if engine_dtype not in _TORCH_DTYPE_NAMES:
            raise CheckpointError(
                f"engine_dtype {engine_dtype!r} not one of {sorted(_TORCH_DTYPE_NAMES)}"
            )
        self.model = model
        self.scales_dtype = _TORCH_DTYPE_NAMES[engine_dtype]
        self.strict_qzeros = bool(strict_qzeros)
        self.check_qzeros = bool(check_qzeros)
        self.proj_names = tuple(proj_names)
        if len(self.proj_names) != 3:
            raise CheckpointError("proj_names is (gate, up, down)")
        self._gate, self._up, self._down = self.proj_names
        self.config = _read_config(model)
        self.files = SafetensorsDir(model)
        self._discover(name_format)
        self._derive_shapes()
        # The engine's own functions: schema, identity, bytes.
        load_kernels_transform(kernels_src)
        self._xt = xt
        quant_stub = SimpleNamespace(
            quant_config=SimpleNamespace(group_size=self.group_size),
            moe=SimpleNamespace(w13_num_shards=2),
        )
        self.unit_tensor_bytes = xt.slot_tensor_bytes(
            quant_stub, self.hidden, self.inter, self.scales_dtype
        )
        self.unit_nbytes = sum(self.unit_tensor_bytes.values())
        self.schema = tier_store.build_schema(
            self.layers, self.experts, self.unit_tensor_bytes
        )
        self.engine_inputs = {
            "model": engine_model if engine_model is not None else model,
            "revision": engine_revision,
            "quantization": engine_quantization,
            "dtype": self.scales_dtype,
        }
        self.checkpoint_identity = tier_store.checkpoint_identity(
            SimpleNamespace(**self.engine_inputs)
        )
        self.transform_fingerprint = xt._layout_fingerprint()
        self.identity = tier_store.store_identity(
            self.schema, self.checkpoint_identity, self.transform_fingerprint
        )
        self._lock = threading.Lock()
        self._qz_checked = 0
        self._qz_odd: list[tuple[int, int]] = []

    # -- discovery ---------------------------------------------------------

    def _discover(self, name_format: str | None) -> None:
        projs = self.proj_names
        kinds = (QWEIGHT, QZEROS, SCALES)
        formats = [name_format] if name_format else list(DEFAULT_NAME_FORMATS)
        names = list(self.files.refs)
        best: tuple[int, str, re.Pattern, dict[tuple[int, int], int]] | None = None
        for fmt in formats:
            rx = _name_regex(fmt, projs, kinds)
            present: dict[tuple[int, int], int] = {}
            hits = 0
            for name in names:
                m = rx.fullmatch(name)
                if m is None:
                    continue
                key = (int(m["layer"]), int(m["expert"]))
                bit = 1 << (projs.index(m["proj"]) * 3 + kinds.index(m["kind"]))
                present[key] = present.get(key, 0) | bit
                hits += 1
            if hits and (best is None or hits > best[0]):
                best = (hits, fmt, rx, present)
            if hits and name_format is None:
                break
        if best is None:
            raise CheckpointError(
                f"{self.model}: no routed-expert tensors match "
                f"{formats if len(formats) > 1 else formats[0]!r} (set name_format)"
            )
        self.tensors_used, self.name_format, rx, present = best
        self._rx = rx
        full = (1 << 9) - 1
        partial = [k for k, bits in present.items() if bits != full]
        if partial:
            raise CheckpointError(
                f"unit (layer, expert) {min(partial)} lacks some of its "
                f"{len(projs) * 3} tensors ({len(partial)} such units)"
            )
        layers = sorted({k[0] for k in present})
        self.layers = len(layers)
        if layers != list(range(self.layers)):
            raise CheckpointError(
                f"routed experts are in layers {layers[:6]}..., the store needs "
                "every decoder layer from 0 (a model with dense layers is not "
                "supported by the engine's store either)"
            )
        per_layer = {n: sum(1 for k in present if k[0] == n) for n in layers}
        counts = set(per_layer.values())
        self.experts = max(counts)
        if len(counts) != 1 or any(
            (n, e) not in present for n in layers for e in range(self.experts)
        ):
            raise CheckpointError(
                "experts are not 0..E-1 in every layer, or their number differs "
                f"between layers ({sorted(counts)[:4]})"
            )
        self._skipped = self._classify_skipped(names)

    def _classify_skipped(self, names: list[str]) -> list[dict[str, Any]]:
        groups: Counter[str] = Counter()
        other = 0
        for name in names:
            if self._rx.fullmatch(name):
                continue
            if "expert" not in name:
                other += 1
                continue
            groups[re.sub(r"\d+", "N", name)] += 1
        out = []
        for pattern, count in groups.most_common(24):
            if pattern.startswith("mtp.") or ".mtp." in pattern:
                reason = (
                    "MTP / drafter layer: not a decoder layer, its experts stay "
                    "resident in the engine (xpu_expert_tiers._LAYER_RE)"
                )
            elif "shared_expert" in pattern:
                reason = "shared expert: not a routed expert, it stays in the layer"
            elif pattern.endswith((".weight", ".bias")):
                reason = "unquantized tensor: not int4 GPTQ material"
            else:
                reason = (
                    "name does not match the pattern (set name_format if these "
                    "are routed experts)"
                )
            out.append({"pattern": pattern, "tensors": count, "reason": reason})
        if other:
            out.append(
                {
                    "pattern": "(tensors that are not experts)",
                    "tensors": other,
                    "reason": "attention, norms, embeddings, router, dense parts: "
                    "loaded by the engine itself, not store material",
                }
            )
        return out

    def _name(self, layer: int, expert: int, proj: str, kind: str) -> str:
        return self.name_format.format(layer=layer, expert=expert, proj=proj, kind=kind)

    def _derive_shapes(self) -> None:
        refs = self.files.refs
        n = self._name
        gate_q = refs[n(0, 0, self._gate, QWEIGHT)]
        gate_s = refs[n(0, 0, self._gate, SCALES)]
        cfg = self.config
        bits = cfg.get("bits") or 4
        if bits != 4:
            raise CheckpointError(
                f"{bits}-bit weights: the engine's store is int4 only"
            )
        pack = 32 // bits
        k8, inter = gate_q.shape
        hidden = k8 * pack
        groups = gate_s.shape[0]
        if cfg.get("group_size") is not None:
            group_size = int(cfg["group_size"])
        else:
            group_size = -1 if groups == 1 else hidden // groups
        want_groups = 1 if group_size == -1 else hidden // group_size
        if hidden % pack or groups != want_groups:
            raise CheckpointError(
                f"gate_proj scales have {groups} groups, hidden {hidden} with "
                f"group size {group_size} needs {want_groups}"
            )
        self.hidden, self.inter, self.group_size, self.pack = (
            hidden,
            inter,
            group_size,
            pack,
        )
        self.groups13 = want_groups
        self.groups2 = 1 if group_size == -1 else inter // group_size
        for field, have in (
            ("hidden_size", hidden),
            ("moe_intermediate_size", inter),
            ("num_experts", self.experts),
            ("num_hidden_layers", self.layers),
        ):
            want = cfg.get(field)
            if want is not None and int(want) != have:
                raise CheckpointError(
                    f"config.json says {field} {want}, the tensors say {have}"
                )
        if cfg.get("sym") is False:
            self._sym_note = "config.json says sym=false"
        else:
            self._sym_note = ""
        # Every unit's tensors must have the shapes of the first one.
        expect = self._expected_shapes()
        for layer in range(self.layers):
            for expert in range(self.experts):
                for (proj, kind), (dtypes, shape) in expect.items():
                    ref = refs[n(layer, expert, proj, kind)]
                    if ref.shape != shape or ref.dtype not in dtypes:
                        raise CheckpointError(
                            f"{n(layer, expert, proj, kind)}: {ref.dtype} "
                            f"{list(ref.shape)}, expected {list(dtypes)} "
                            f"{list(shape)}"
                        )

    def _expected_shapes(
        self,
    ) -> dict[tuple[str, str], tuple[tuple[str, ...], tuple[int, ...]]]:
        k, n, p = self.hidden, self.inter, self.pack
        g13, g2 = self.groups13, self.groups2
        i32 = ("I32",)
        out: dict[tuple[str, str], _Expect] = {}
        for proj in (self._gate, self._up):
            out[(proj, QWEIGHT)] = (i32, (k // p, n))
            out[(proj, QZEROS)] = (i32, (g13, n // p))
            out[(proj, SCALES)] = (_SCALE_DTYPES, (g13, n))
        out[(self._down, QWEIGHT)] = (i32, (n // p, k))
        out[(self._down, QZEROS)] = (i32, (g2, k // p))
        out[(self._down, SCALES)] = (_SCALE_DTYPES, (g2, k))
        return out

    # -- the UnitSource protocol ---------------------------------------------

    def unit_keys(self) -> list[tuple[int, int]]:
        return [(n, e) for n in range(self.layers) for e in range(self.experts)]

    def read(self, key: tuple[int, int]) -> dict[str, torch.Tensor]:
        layer, expert = key
        if not (0 <= layer < self.layers and 0 <= expert < self.experts):
            raise KeyError(key)

        def get(proj: str, kind: str) -> torch.Tensor:
            return self.files.read(self._name(layer, expert, proj, kind))

        if self.check_qzeros:
            self._check_qzeros(key, get)
        # gate then up, along the output dimension: the loader's w1 / w3 order
        w13_qweight = torch.cat([get(self._gate, QWEIGHT), get(self._up, QWEIGHT)], 1)
        w13_scales = torch.cat([get(self._gate, SCALES), get(self._up, SCALES)], 1)
        unit = self._xt.kernel_layout_unit(
            w13_qweight,
            w13_scales,
            get(self._down, QWEIGHT),
            get(self._down, SCALES),
            self.scales_dtype,
        )
        return {k: v.reshape(-1).view(torch.uint8) for k, v in unit.items()}

    def _check_qzeros(self, key: tuple[int, int], get: Any) -> None:
        odd = False
        for proj in (self._gate, self._up, self._down):
            flat = get(proj, QZEROS).flatten()
            first = int(flat[0])
            odd = odd or first not in _SYM_QZEROS or not bool((flat == first).all())
        with self._lock:
            self._qz_checked += 1
            if odd:
                self._qz_odd.append(key)
        if odd and self.strict_qzeros:
            raise CheckpointError(
                f"unit {key}: zero points are not the symmetric int4 value; the "
                "engine ignores them on XPU (strict_qzeros)"
            )

    def info(self) -> dict[str, Any]:
        """What was found, what is skipped and why, what the store will be."""
        from vllm.model_executor.offloader import expert_store as es

        layout = es._Layout(self.schema, 1, self.identity)
        return {
            "model": self.model,
            "name_format": self.name_format,
            "files_read": self.files.files,
            "files_ignored_not_in_index": self.files.ignored_files,
            "tensors_in_files": len(self.files.refs),
            "tensors_used": self.tensors_used,
            "layers": self.layers,
            "experts_per_layer": self.experts,
            "units": self.layers * self.experts,
            "hidden_size": self.hidden,
            "moe_intermediate_size": self.inter,
            "bits": 4,
            "group_size": self.group_size,
            "config_json": self.config,
            "unit_tensor_bytes": self.unit_tensor_bytes,
            "unit_payload_bytes": self.unit_nbytes,
            "unit_stored_bytes": layout.slot_bytes,
            "store_bytes": layout.slot_bytes * self.layers * self.experts,
            "skipped": self._skipped,
            "qzeros": {
                "checked_units": self._qz_checked,
                "not_symmetric_units": len(self._qz_odd),
                "first_not_symmetric": self._qz_odd[:3],
                "note": self._sym_note,
            },
            "identity": self.identity,
            "identity_inputs": {
                "engine": {k: str(v) for k, v in self.engine_inputs.items()},
                "transform_fingerprint": self.transform_fingerprint,
                "layout_version": self._layout_version(),
                "checkpoint_files": json.loads(self.checkpoint_identity)[4:],
            },
        }

    @staticmethod
    def _layout_version() -> str:
        from vllm.model_executor.offloader import expert_tier_store as tier_store

        return str(tier_store.LAYOUT_VERSION)

    def close(self) -> None:
        self.files.close()


def format_info(info: dict[str, Any]) -> str:
    gib = 1 << 30
    lines = [
        f"checkpoint  {info['model']}",
        f"names       {info['name_format']}",
        (
            f"shape       {info['layers']} layers x {info['experts_per_layer']} "
            f"experts, hidden {info['hidden_size']}, moe intermediate "
            f"{info['moe_intermediate_size']}, int4 group {info['group_size']}"
        ),
        (
            f"files       {len(info['files_read'])} read, "
            f"{len(info['files_ignored_not_in_index'])} not in the index "
            f"(ignored, as the engine's loader does)"
        ),
        (f"tensors     {info['tensors_used']} used of {info['tensors_in_files']}"),
        (
            f"unit        {info['unit_payload_bytes']} B payload, "
            f"{info['unit_stored_bytes']} B stored; store "
            f"{info['store_bytes'] / gib:.2f} GiB for {info['units']} units"
        ),
        f"identity    {info['identity']}",
    ]
    for item in info["skipped"]:
        lines.append(
            f"skipped     {item['tensors']} x {item['pattern']}: {item['reason']}"
        )
    return "\n".join(lines)
