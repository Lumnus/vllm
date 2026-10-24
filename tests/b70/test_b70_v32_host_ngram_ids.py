import types, torch, numpy as np
from vllm.models.qwen4_exp.nvidia.ngram_embedding import Qwen4ExpNGramEmbedding as C, b70_host_ngram_ids
rng = np.random.default_rng(0)
for trial in range(200):
    n = int(rng.integers(2, 5)); hpn = int(rng.integers(1, 5)); heads = (n - 1) * hpn
    eos = 7
    mult = [2 * int(rng.integers(1, 2**60)) + 1 for _ in range(n)]
    sizes = torch.tensor(rng.integers(1000, 2**31, heads), dtype=torch.long)
    offs = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long), sizes[:-1]]), 0)
    f = types.SimpleNamespace(layer_multipliers=torch.tensor(mult, dtype=torch.long), ngram_heads_vocab_sizes=sizes,
        ngram_heads_offsets=offs, eos_token_id=eos, heads_per_ngram=hpn, ngram_size=n,
        _shift_precompute=C._shift_precompute, _shift_apply=C._shift_apply)
    R = int(rng.integers(1, 9)); lens = rng.integers(1, 12, R)
    qsl = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64); T = int(qsl[-1])
    ids = rng.integers(0, 20, T).astype(np.int64)
    ctx = rng.integers(0, 20, (R, n - 1)).astype(np.int64)
    ref = C.compute_ngram_ids(f, torch.from_numpy(ids), torch.from_numpy(qsl), torch.from_numpy(ctx)).numpy()
    got = b70_host_ngram_ids(ids, qsl, ctx, mult, sizes.numpy(), offs.numpy(), eos, hpn, n)
    assert ref.shape == got.shape and np.array_equal(ref, got), (trial, ref, got)
print("host ngram ids == torch eager path: 200/200")
