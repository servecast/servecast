"""The home page's Predict card shows what the page's own examples print, so its numbers cannot go stale: the tiles are
the first prediction's row at 64 clients, rounded as shown, and the curve plots the load curve's requests per second
from zero, one point per number of clients. tests/test_docs_examples.py checks the printed rows against servecast."""

from __future__ import annotations

import re
from pathlib import Path

from tests.test_docs_examples import blocks

ROOT = Path(__file__).resolve().parents[1]
PAGE = "docs/index.md"
SOURCE = (ROOT / PAGE).read_text(encoding="utf-8")
#: the columns of a printed row
CLIENTS, TTFT, P95_TTFT, TPOT, P95_TPOT, REQ_S = 0, 2, 3, 4, 5, 6


def printed() -> dict[int, list[str]]:
    """Each row the page's commands print, by its number of clients."""
    rows: dict[int, list[str]] = {}
    for _, shown in blocks(PAGE):
        for line in (shown or "").splitlines():
            cells = line.split()
            if cells and cells[CLIENTS].isdigit():
                rows[int(cells[CLIENTS])] = cells
    return rows


def tiles(source: str) -> list[tuple[str, str, str]]:
    return re.findall(r'<span class="sc-k">([^<]+)</span><span class="sc-v">([^<]+)</span>'
                      r'<span class="sc-s">([^<]+)</span>', source)


def curve(source: str) -> tuple[list[str], list[float], list[str]]:
    """The curve's value labels, its points' heights and its ticks."""
    match = re.search(r'<div class="sc-curve">(.*?)</div>', source)
    assert match is not None, "the home page has no curve"
    svg = match.group(1)
    return (re.findall(r'class="v">([^<]+)</text>', svg),
            [float(y) for y in re.findall(r'<circle cx="[^"]+" cy="([^"]+)"', svg)],
            re.findall(r'text-anchor="middle">(\d+)</text>', svg))


def expected_tiles() -> list[tuple[str, str, str]]:
    row = [float(c) for c in printed()[64][TTFT:REQ_S + 1]]
    ttft, p95_ttft, tpot, p95_tpot, req_s = row
    return [("TTFT", f"{ttft:.0f} ms", f"P95 {p95_ttft:.0f} ms"), ("TPOT", f"{tpot:.1f} ms", f"P95 {p95_tpot:.1f} ms"),
            ("Throughput", f"{req_s:.2f}", "requests / s")]


def expected_curve() -> tuple[list[str], list[float], list[str]]:
    rows = printed()
    points = [(c, float(rows[c][REQ_S])) for c in (16, 32, 64, 128)]
    # the plot's zero is at y = 100 and one request per second is 9.5 units up
    return ([f"{v:.1f}" for _, v in points], [round(100 - v * 9.5, 1) for _, v in points], [str(c) for c, _ in points])


def test_the_tiles_are_the_first_predictions_row_at_64_clients() -> None:
    assert tiles(SOURCE) == expected_tiles()


def test_the_curve_plots_the_load_curves_requests_per_second_from_zero() -> None:
    assert curve(SOURCE) == expected_curve()


def test_a_tile_or_a_point_that_differs_from_the_printed_rows_is_caught() -> None:
    assert tiles(SOURCE.replace(">213 ms<", ">214 ms<")) != expected_tiles()
    assert curve(SOURCE.replace('class="v">7.5<', 'class="v">7.6<')) != expected_curve()
