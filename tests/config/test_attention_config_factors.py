# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from vllm.config import AttentionConfig
from vllm.config.utils import hash_factors


def test_qsa_int8_dot_off_is_not_a_compile_factor():
    factors = AttentionConfig().compile_factors()
    assert "qsa_int8_dot" not in factors
    assert "backend" in factors


def test_qsa_int8_dot_set_is_a_compile_factor():
    off = hash_factors(AttentionConfig().compile_factors())
    qk = AttentionConfig(qsa_int8_dot="qk").compile_factors()
    qk_pv = AttentionConfig(qsa_int8_dot="qk_pv").compile_factors()
    assert qk["qsa_int8_dot"] == "qk"
    assert len({off, hash_factors(qk), hash_factors(qk_pv)}) == 3
