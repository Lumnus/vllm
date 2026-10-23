# wu1ff B70 Flash-Next 1.0.5 chain (vendored as commits W032-01..16 on this branch)

Source: wu1ff/B70-LLM-Controller@5e5cf43f048d9b32146bb22925d4a2cf198400b0, model pack qwen3.8-flash-next v1.0.5, patches/production/patch-chain.
License of the Controller repository: MIT (LICENSE here). The patches modify vLLM files, which stay Apache-2.0 with
their per-file notices; NOTICE.md is the pack's own notice file, copied unchanged. Authors named there: Wulf
(disk/checkpoint/frontend/serving integration, residency shim) and Astra (DFlash2, native/graph/ownership/prefill).
The vllm_xpu_kernels/* hunks of the same patches live in the kernels-python series (a separate
vllm-xpu-kernels-py series for the 0.32 line), not in this tree.
The twelve native libraries they load are built from the pack's native/ sources.
