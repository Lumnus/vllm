# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests of the store build tool (``expert_store_build``)."""

import contextlib
import json
import os
import pathlib
import subprocess
import sys
import time

import pytest
import torch

from vllm.model_executor.offloader import expert_store_build as ebuild
from vllm.model_executor.offloader.expert_store import (
    DATA_FILE,
    ExpertStore,
    StoreIncompleteError,
)
from vllm.model_executor.offloader.weight_units import ComponentStore

MODULE = "vllm.model_executor.offloader.expert_store_build"
SMALL = ["--arg", "layers=3", "--arg", "experts=8"]


def cli(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", VLLM_TARGET_DEVICE="cpu")
    proc = subprocess.run(
        [sys.executable, "-m", MODULE, *args],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if check:
        assert proc.returncode == 0, proc.stderr + proc.stdout
    return proc


def popen(*args: str) -> subprocess.Popen:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", VLLM_TARGET_DEVICE="cpu")
    return subprocess.Popen(
        [sys.executable, "-m", MODULE, *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )


def dirs_of(tmp_path, n: int) -> list[str]:
    out = []
    for i in range(n):
        d = tmp_path / f"drive{i}"
        d.mkdir()
        out.append(str(d))
    return out


def assert_store_matches_source(dirs: list[str], source) -> None:
    layout_slot = max(
        sum(s.nbytes for s in source.specs) + 8192, 1 << 16
    )  # a pool of a few slots
    with ExpertStore.open(
        dirs, source.identity, 8 * layout_slot, pin=False, check_every=1
    ) as store:
        for key in source.unit_keys():
            want = source.read(key)
            with store.get(key) as entry:
                blob = bytes(entry.raw)
                for place in entry.tensor_places():
                    assert (
                        blob[place.offset : place.offset + place.nbytes]
                        == want[place.name]
                    )


def test_build_info_verify_roundtrip(tmp_path):
    dirs = dirs_of(tmp_path, 2)
    proc = cli("build", "--dirs", *dirs, "--source", "synthetic", *SMALL, "--finalize")
    assert "24 written" in proc.stderr
    info = json.loads(cli("info", "--dirs", *dirs, "--json").stdout)
    assert info["kind"] == "experts" and info["complete"]
    assert info["units"] == 24 and info["num_stripes"] == 2
    assert info["identity"] == "synthetic-experts"
    assert info["groups"][0]["unit_bytes"] == 4096
    assert info["unit_checksums"]["count"] == 24
    text = cli("info", "--dirs", *dirs).stdout
    assert "complete" in text and "synthetic-experts" in text
    report = json.loads(
        cli(
            "verify", "--dirs", *dirs, "--source", "synthetic", *SMALL, "--sample", "10"
        ).stdout
    )
    assert report["ok"] and report["sha256"] == "ok"
    assert report["units_checked"] == 24 and report["source_sampled"] == 10
    assert_store_matches_source(dirs, ebuild.SyntheticExperts(layers=3, experts=8))


def test_two_processes_then_one_finalize(tmp_path):
    dirs = dirs_of(tmp_path, 2)
    common = ["--dirs", *dirs, "--source", "synthetic", *SMALL]
    first = popen("build", *common, "--part", "0/2")
    second = popen("build", *common, "--part", "1/2")
    for proc in (first, second):
        out, err = proc.communicate(timeout=300)
        assert proc.returncode == 0, err
    info = json.loads(cli("info", "--dirs", *dirs, "--json").stdout)
    assert not info["complete"] and info["units_recorded"] == 24
    assert {w["id"] for w in info["writers"]} == {"part0of2", "part1of2"}
    cli("finalize", "--dirs", *dirs)
    assert (
        cli("verify", "--dirs", *dirs, "--source", "synthetic", *SMALL).returncode == 0
    )
    assert_store_matches_source(dirs, ebuild.SyntheticExperts(layers=3, experts=8))


def test_finalize_refuses_while_a_part_is_missing(tmp_path):
    dirs = dirs_of(tmp_path, 1)
    cli("build", "--dirs", *dirs, "--source", "synthetic", *SMALL, "--part", "0/2")
    proc = cli("finalize", "--dirs", *dirs, check=False)
    assert proc.returncode == 2
    assert "12 of 24 units are not recorded" in proc.stderr
    with pytest.raises(StoreIncompleteError):
        ExpertStore.open(dirs, "synthetic-experts", 1 << 20, pin=False)


def test_resume_after_the_process_is_killed(tmp_path):
    dirs = dirs_of(tmp_path, 2)
    args = ["--dirs", *dirs, "--source", "synthetic", *SMALL]
    slow = popen(
        "build", *args, "--workers", "1", "--flush-every", "1", "--arg", "delay_ms=40"
    )
    record = pathlib.Path(dirs[0], "writers", "part0of1.json")
    recorded = 0
    deadline = time.time() + 120
    while time.time() < deadline and recorded < 6:
        time.sleep(0.05)
        if record.exists():
            with contextlib.suppress(ValueError):  # replaced while reading
                recorded = json.loads(record.read_text())["units"]
    slow.kill()  # SIGKILL, no cleanup
    slow.communicate()
    assert 6 <= recorded < 24, "the build finished or never started before the kill"
    manifest = pathlib.Path(dirs[0], "manifest.json").read_text()
    assert '"complete": true' not in manifest
    proc = cli("build", *args, "--finalize")
    skipped = int(proc.stderr.split(" written, ")[1].split(" already")[0])
    assert skipped >= recorded
    assert (
        cli("verify", "--dirs", *dirs, "--source", "synthetic", *SMALL).returncode == 0
    )
    assert_store_matches_source(dirs, ebuild.SyntheticExperts(layers=3, experts=8))


def test_rebuild_refusals_and_flag(tmp_path):
    dirs = dirs_of(tmp_path, 1)
    cli("build", "--dirs", *dirs, "--source", "synthetic", *SMALL, "--finalize")
    again = cli("build", "--dirs", *dirs, "--source", "synthetic", *SMALL, check=False)
    assert again.returncode == 2 and "rebuild" in again.stderr
    cli(
        "build",
        "--dirs",
        *dirs,
        "--source",
        "synthetic",
        *SMALL,
        "--rebuild",
        "--finalize",
    )


def flip_byte(path: str, pos: int) -> None:
    with open(path, "r+b") as handle:
        handle.seek(pos)
        byte = handle.read(1)[0]
        handle.seek(pos)
        handle.write(bytes([byte ^ 1]))


def test_verify_finds_a_corrupt_unit_by_key(tmp_path):
    dirs = dirs_of(tmp_path, 2)
    cli("build", "--dirs", *dirs, "--source", "synthetic", *SMALL, "--finalize")
    layout = ebuild._load_layout(ebuild._manifest(dirs))
    stripe, offset, _ = layout.locate(2, 5)
    flip_byte(os.path.join(dirs[stripe], DATA_FILE), offset + 17)
    proc = cli("verify", "--dirs", *dirs, "--no-sha256", check=False)
    assert proc.returncode == 1
    report = json.loads(proc.stdout)
    assert not report["ok"]
    assert "(2, 5)" in report["problems"][0] or "[2, 5]" in report["problems"][0]
    # the whole-file hash alone sees it too
    sha_only = cli("verify", "--dirs", *dirs, "--no-units", check=False)
    assert sha_only.returncode == 1


def test_verify_against_source_finds_a_wrong_identity_of_bytes(tmp_path):
    dirs = dirs_of(tmp_path, 1)
    cli("build", "--dirs", *dirs, "--source", "synthetic", *SMALL, "--finalize")
    other = cli(
        "verify", "--dirs", *dirs, "--source", "synthetic", *SMALL, "--arg", "seed=9",
        "--sample", "5", check=False,
    )  # fmt: skip
    assert other.returncode == 1
    assert "differ from the source" in other.stdout


def test_build_from_an_iterator_skips_what_is_recorded(tmp_path):
    dirs = dirs_of(tmp_path, 1)
    source = ebuild.SyntheticExperts(layers=2, experts=4)
    writer = ebuild.open_writer(dirs, source.schema, source.identity, writer_id="a")
    pairs = [(k, source.read(k)) for k in source.unit_keys()]
    first = ebuild.build_from_iter(writer, iter(pairs[:5]), flush_every=2)
    assert (first.written, first.skipped) == (5, 0)
    again = ebuild.build_from_iter(writer, iter(pairs), flush_every=2, workers=3)
    assert (again.written, again.skipped) == (3, 5)
    writer.close()
    ebuild.finalize_store(dirs)
    assert ebuild.verify_store(dirs, source=source)["ok"]


def test_components_build_two_parts_info_verify(tmp_path):
    dirs = dirs_of(tmp_path, 2)
    common = ["--dirs", *dirs, "--source", "synthetic-components"]
    for part in ("0/2", "1/2"):
        cli("build", *common, "--part", part)
    cli("finalize", "--dirs", *dirs)
    info = json.loads(cli("info", "--dirs", *dirs, "--json").stdout)
    assert info["kind"] == "components" and info["complete"]
    by_name = {c["name"]: c for c in info["components"]}
    assert set(by_name) == {"tower", "small"}
    assert by_name["tower"]["extents"] > 1 and by_name["small"]["tensors"] == 6
    assert by_name["tower"]["largest_extent"] <= info["max_extent_bytes"]
    report = json.loads(
        cli("verify", "--dirs", *dirs, "--source", "synthetic-components").stdout
    )
    assert report["ok"] and report["units_checked"] == info["units"]
    source = ebuild.SyntheticComponents()
    with ComponentStore.open(
        dirs, source.identity, 1 << 20, pin=False, check_every=1
    ) as cs:
        for name in source.unit_keys():
            want = source.read(name)
            loaded = cs[name].load()
            for spec in cs[name].tensor_specs:
                blob = loaded.tensors[spec.name].reshape(-1).view(torch.uint8)
                assert blob.numpy().tobytes() == want[spec.name]


def test_bad_source_spec_is_reported(tmp_path):
    proc = cli("build", "--dirs", str(tmp_path), "--source", "nonsense", check=False)
    assert proc.returncode == 2 and "module:callable" in proc.stderr
