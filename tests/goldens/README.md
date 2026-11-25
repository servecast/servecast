# Test data

The model's tests compare against `newest-a100.json`: the newest cost law's predictions on the A100-SXM4-40GB, next
to the measurements they are compared with. It is test data and makes no claim about accuracy; the published
accuracy figures are the ones in the arXiv version. The paper's own campaigns and tables ship with the package
(`src/servecast/data/paper/`, described in its README).

Times are in seconds and throughput in requests per second. Every float is written as Python writes it, so it reads
back bit for bit. A null means that the value does not exist.

The predicted means and 95th percentiles are computed over the simulated requests kept in the steady-state window,
with `numpy.mean` and `numpy.quantile(..., 0.95)` (numpy's default linear interpolation). A port must use the same
estimators to match them exactly. The measured 95th percentiles use the same quantile, over the measured requests in
the window that the measured means were computed on.

Each campaign's `prediction` holds the settings its predictions were made with, as in the paper's data, with two
more stop targets, `ttft_p95` and `tpot_p95`, judged by steady-state estimator 2026-11. `max_completions` is null:
every prediction ran until its precision targets were met.

Each load point holds:

- `predicted`: the throughput; the means and 95th percentiles of TTFT and TPOT; the means the simulator
  reported itself; and the number of samples.
- `measured`: the measured references for the same quantities.
- `scored_means`, `scored_p95_ttft`, `scored_p95_tpot`: whether the newest law's comparison scores this load
  point. These flags are copied from that comparison's record and cannot be derived from this file, because
  they also depend on runs that are not included here.

