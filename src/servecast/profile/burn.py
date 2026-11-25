"""The load a clock lock is checked under (`gpu.under_load`), in a process of its own.

    python -m servecast.profile.burn

A dense GEMM, the paper's lock check's heavy load: fp16 products of 4096 x 4096 matrices, eight queued per
synchronisation so the SMs stay busy, which draws at least the power LLM serving draws, so a level that holds here
holds while vLLM serves. The operands are scaled by 1/sqrt(4096) so every product stays finite: products that overflow
to inf or NaN draw a power that depends on the library. It prints "ready" once the first product ran, then runs until
it is stopped.
"""

from __future__ import annotations

N = 4096
QUEUED = 8


def main() -> int:  # pragma: no cover - needs a GPU and torch
    import torch

    a = torch.randn(N, N, device="cuda", dtype=torch.float16) / N ** 0.5
    b = torch.randn(N, N, device="cuda", dtype=torch.float16) / N ** 0.5
    c = torch.empty_like(a)
    torch.matmul(a, b, out=c)
    torch.cuda.synchronize()
    print("ready", flush=True)
    while True:
        for _ in range(QUEUED):
            torch.matmul(a, b, out=c)
        torch.cuda.synchronize()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
