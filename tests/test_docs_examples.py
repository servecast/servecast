"""Every command in the README and the docs runs as a reader would type it, and the pages hold nothing a renderer
would drop.

A page is one session: its commands run in order, in one scratch directory, each presuming what the lines before it
made, after the page's SETUP (what the page tells the reader to have, such as a manifest). A fence with a title
(```yaml title="my-gpu.yaml"```) is a file the page tells the reader to write, and it is written before the page's
first line, a gpus.json in the data directory. Pages are read with a
CommonMark parser, so a fence is found at any depth (in a list item, in a quote) and with any delimiter. A command
block is a fence marked bash, console, sh or shell; every code block names its language, so no command can hide in
an unmarked one. A line
that is not a `servecast` command is in NOT_RUN with the reason. A command that cannot succeed here is in EXPECTED
(the help pages' list) or DOCS_EXPECTED with the status it ends with and a phrase its error must contain; one that
runs for hours is in SLOW. On the reference environment, a `text` fence right after a command block must be exactly
what that block's last command printed, so no number on a page can go stale.

The command-line reference is generated from the help pages, whose examples tests/test_cli_examples.py runs.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest
from markdown_it import MarkdownIt
from markdown_it.token import Token

from tests.test_cli_examples import EXPECTED, SLOW, SLOW_ON, run

ROOT = Path(__file__).resolve().parents[1]
GENERATED = {"docs/reference/cli.md", "docs/reference/formats.md"}
PAGES = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "README.md", *ROOT.glob("docs/**/*.md"))
               if p.relative_to(ROOT).as_posix() not in GENERATED)
SHELL = {"bash", "console", "sh", "shell"}
PARSER = MarkdownIt("commonmark")

#: lines of command blocks that are not run, with the reason
NOT_RUN = {
    "pip install servecast": "installs the release from PyPI; the tests run the checkout",
    "pip install 'servecast[profile]'": "installs the release and vLLM from PyPI; the tests run the checkout",
    "git clone https://github.com/servecast/servecast.git": "the tests run in the checkout already",
    "cd servecast": "the tests run in the checkout already",
    "pip install '.[profile]'": "installs the checkout and vLLM; the tests run the checkout",
    "pip install .": "installs the checkout; the tests run the checkout",
    "sbatch servecast.sbatch": "submits a Slurm job; tests/test_examples_slurm.py runs its script",
}


def _after_get_started(d: Path) -> None:
    """What the On your GPU guide presumes: the calibration my-a100, which Get started saves."""
    assert run(d, "servecast calibrate --data a100_sxm_40gb/c62ebe75 --name my-a100").returncode == 0


#: what each page presumes before its first line
SETUP: dict[str, Callable[[Path], None]] = {"docs/guides/calibrate.md": _after_get_started}
#: the GPU flow of the README and of the On your GPU guide needs an NVIDIA GPU: its profile commands are refused here,
#: so there are no profiling data to calibrate and no measurement to compare (its predict runs, with my-a100)
GPU_FLOW = {
    "servecast profile calibration -c example-a100.yaml --output-dir profiling": (3, "this computer cannot profile"),
    "servecast calibrate --data profiling --name my-a100 --overwrite": (2, "no profiling data"),
    "servecast profile serving -c example-a100.yaml --output-dir measurement": (3, "this computer cannot profile"),
    "servecast compare prediction measurement": (2, "no measurement.json"),
}
#: the On your GPU guide's flows for a GPU, an LLM and a workload of your own, with the gpus.json and the manifests it
#: shows: refused here for the same reason. The GPU's predict finds prediction/, which step 3 wrote above; a reader
#: with another GPU starts in an empty directory. Without a GPU there is no calibration my-gpu, so the GPU's serving
#: is refused for it, before the GPU is looked at
OWN_FLOWS = {
    "servecast profile calibration -c my-gpu.yaml --output-dir profiling": (3, "this computer cannot profile"),
    "servecast calibrate --data profiling --name my-gpu": (2, "no profiling data"),
    "servecast predict -c my-gpu.yaml --output-dir prediction": (3, "is not empty"),
    "servecast profile serving -c my-gpu.yaml --output-dir measurement": (2, "no calibration 'my-gpu'"),
    "servecast profile serving -c my-llm.yaml --output-dir measurement-3b": (3, "this computer cannot profile"),
    "servecast compare prediction-3b measurement-3b": (2, "no measurement.json"),
    "servecast profile serving -c my-workload.yaml --output-dir measurement-workload": (3, "this computer cannot "
                                                                                          "profile"),
    "servecast compare prediction-workload measurement-workload": (2, "no measurement.json"),
}
#: commands of the docs that cannot succeed here: (page, line) -> (status, phrase in the error)
DOCS_EXPECTED: dict[tuple[str, str], tuple[int, str]] = {
    **{(page, line): expected for page in ("README.md", "docs/guides/calibrate.md") for line, expected in
       GPU_FLOW.items()},
    **{("docs/guides/calibrate.md", line): expected for line, expected in OWN_FLOWS.items()}}
#: commands of the docs that run for hours
DOCS_SLOW = {"servecast replicate run rq1 --jobs 8", "servecast replicate run rq2 --gpu h100_sxm --jobs 8"}


def _code(tokens: list[Token]) -> list[Token]:
    return [t for t in tokens if t.type in ("fence", "code_block")]


def fences_in(text: str) -> list[tuple[str, str]]:
    """(language, body) of every code block of a text, in order; an indented block names no language."""
    return [(t.info.strip() if t.type == "fence" else "", t.content) for t in _code(PARSER.parse(text))]


def fences(page: str) -> list[tuple[str, str]]:
    return fences_in((ROOT / page).read_text(encoding="utf-8"))


#: the files a page shows that belong in the data directory, not where the commands run
DATA_FILES = {"gpus.json"}


def files(page: str) -> dict[str, str]:
    """The files a page shows in full, by the title of their fence; a text fence is what a command printed."""
    out = {}
    for info, body in fences(page):
        title = re.search(r'title="([^"]+)"', info)
        if title is not None and info.split(" ")[0] != "text":
            out[title.group(1)] = body
    return out


def write_files(page: str, d: Path) -> None:
    for name, body in files(page).items():
        target = d / "data" / name if name in DATA_FILES else d / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body, encoding="utf-8")


def _lines(body: str) -> list[str]:
    joined = body.replace("\\\n", " ")
    return [re.sub(r"\s+", " ", line.removeprefix("$ ")).strip() for line in joined.splitlines()
            if line.strip() and not line.lstrip().startswith("#")]  # a full-line comment runs nothing


def blocks(page: str) -> list[tuple[list[str], str | None]]:
    """Each command block's lines, and the text block right after it (what it prints), if there is one."""
    tokens = PARSER.parse((ROOT / page).read_text(encoding="utf-8"))
    out: list[tuple[list[str], str | None]] = []
    for i, t in enumerate(tokens):
        if t.type == "fence" and t.info.strip().split(" ")[0] in SHELL:
            after = tokens[i + 1] if i + 1 < len(tokens) else None
            shown = after is not None and after.type == "fence" and after.info.strip().split(" ")[0] == "text"
            out.append((_lines(t.content), after.content if shown and after is not None else None))
    return out


def commands(page: str) -> list[str]:
    return [line for lines, _ in blocks(page) for line in lines if line.startswith("servecast ")]


RUN_PAGES = [page for page in PAGES if commands(page)]


def test_the_docs_examples_are_found() -> None:
    assert "README.md" in PAGES and "docs/guides/predict.md" in RUN_PAGES
    assert set(files("docs/guides/calibrate.md")) == {"gpus.json", "my-gpu.yaml", "my-llm.yaml", "my-workload.yaml"}
    assert set(files("docs/get-started.md")) == {"example-a100.yaml"}  # its Output fence is what a command printed
    assert commands("docs/guides/predict.md") == [
        "servecast manifest init --template predict > m.yaml",
        "servecast manifest explain engine.workers.aggregated.scheduler.max_sequences",
        "servecast predict -c m.yaml --with a100_sxm_40gb/c62ebe75 --to Qwen/Qwen2.5-3B",
        "servecast predict -c m.yaml --sweep traffic.load.concurrency=16,32,64,128 --jobs 4 --output-dir sweep",
    ]


@pytest.mark.parametrize("page", PAGES)
def test_every_fence_names_its_language_and_every_command_line_is_run_or_says_why_not(page: str) -> None:
    for language, body in fences(page):
        assert language, f"{page}: a fence without a language: {body[:60]!r}"
        if language.split(" ")[0] not in SHELL:
            assert not any(line.startswith("servecast ") for line in _lines(body)), \
                f"{page}: a servecast command in a {language} fence, which is never run"
    for lines, _ in blocks(page):
        for line in lines:
            assert line.startswith("servecast ") or line in NOT_RUN, f"{page}: {line!r} is neither run nor in NOT_RUN"


def prose(page: str) -> str:
    """A page without its code blocks and inline code: what a reader sees as text."""
    lines = (ROOT / page).read_text(encoding="utf-8").splitlines()
    for t in _code(PARSER.parse("\n".join(lines))):
        if t.map is not None:
            lines[t.map[0]:t.map[1]] = [""] * (t.map[1] - t.map[0])
    return re.sub(r"`[^`\n]*`", "", "\n".join(lines))


@pytest.mark.parametrize("page", PAGES)
def test_no_page_holds_a_placeholder_markdown_drops(page: str) -> None:
    """Rendered Markdown reads <word> outside code as an HTML tag and drops it."""
    assert not re.findall(r"<[A-Za-z_][\w-]*>", prose(page))


def test_the_generated_reference_pages_hold_no_placeholder_markdown_drops() -> None:
    for page in sorted(GENERATED):
        test_no_page_holds_a_placeholder_markdown_drops(page)


def _session(page: str, d: Path) -> list[tuple[str, str]]:
    """Run the page's commands in order; (line, stdout) of each."""
    setup = SETUP.get(page)
    if setup is not None:
        setup(d)
    write_files(page, d)
    printed = []
    for line in commands(page):
        if (line in SLOW or line in DOCS_SLOW) and not SLOW_ON:
            continue
        result = run(d, line)
        status, phrase = DOCS_EXPECTED.get((page, line)) or EXPECTED.get(line, (0, ""))
        assert result.returncode == status, f"{page}: {line}: status {result.returncode}, expected {status}\n" \
                                            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        assert "Traceback" not in result.stderr, result.stderr[-2000:]
        assert phrase in result.stderr, f"{page}: {line}: the error does not say {phrase!r}: {result.stderr[-500:]}"
        printed.append((line, result.stdout))
    return printed


@pytest.mark.parametrize("page", RUN_PAGES)
def test_every_command_of_a_page_runs_in_order(tmp_path: Path, page: str) -> None:
    _session(page, tmp_path)


SHOWN = [page for page in RUN_PAGES if any(printed is not None for _, printed in blocks(page))]


@pytest.mark.exact
@pytest.mark.parametrize("page", SHOWN)
def test_what_a_page_shows_a_command_print_is_what_it_prints(tmp_path: Path, page: str) -> None:
    out = dict(_session(page, tmp_path))
    for lines, shown in blocks(page):
        if shown is not None:
            last = [line for line in lines if line.startswith("servecast ")][-1]
            assert out[last].rstrip() == shown.rstrip(), f"{page}: {last} printed\n{out[last]}"


def _merged(base: dict, part: dict) -> dict:
    """`part` written over `base`; a mapping of another `type` (a trace source for a synthetic one) replaces it."""
    out = dict(base)
    for key, value in part.items():
        old = out.get(key)
        if isinstance(value, dict) and isinstance(old, dict) and value.get("type", old.get("type")) == old.get("type"):
            out[key] = _merged(old, value)
        else:
            out[key] = value
    return out


@pytest.mark.parametrize("page", PAGES)
def test_every_yaml_fence_is_a_valid_part_of_a_manifest(page: str, tmp_path: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    """A manifest snippet on a page, put into the predict template, must still be a manifest ServeCast accepts, with
    the GPUs the page adds to the registry; a gpus.json a page shows must be one ServeCast reads."""
    import yaml

    from servecast.core import gpus, manifest, versions

    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path))
    for name, body in files(page).items():
        if name in DATA_FILES:
            (tmp_path / name).write_text(body, encoding="utf-8")
    if (tmp_path / "gpus.json").is_file():
        gpus.read(tmp_path / "gpus.json")  # raises if ServeCast cannot read the entry a reader copies
    template = yaml.safe_load(versions.template("predict"))
    for language, body in fences(page):
        if language.split(" ")[0] == "yaml":
            part = yaml.safe_load(body)
            assert isinstance(part, dict) and set(part) <= set(manifest.Manifest.model_fields), body
            _, problems = manifest.validate(_merged(template, part))
            assert problems == [], f"{page}: {body}\n{problems}"


@pytest.mark.parametrize("text", [
    "1. Run it:\n\n    ```bash\n    servecast nope\n    ```\n",
    "~~~bash\nservecast nope\n~~~\n",
    "````bash\nservecast nope\n````\n",
    "> ```bash\n> servecast nope\n> ```\n",
])
def test_a_command_in_any_kind_of_fence_is_found(text: str) -> None:
    assert [(language, body.strip()) for language, body in fences_in(text)] == [("bash", "servecast nope")]


def test_an_indented_code_block_is_found_and_names_no_language() -> None:
    assert fences_in("Text.\n\n    servecast nope\n") == [("", "servecast nope\n")]


def test_a_failing_command_piped_into_another_still_fails(tmp_path: Path) -> None:
    assert run(tmp_path, "servecast nope | cat").returncode != 0


@pytest.mark.parametrize("page", PAGES)
def test_every_relative_link_leads_to_a_file(page: str) -> None:
    for target in re.findall(r"\]\(([^)\s]+)\)", prose(page)):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        assert ((ROOT / page).parent / target.split("#")[0]).exists(), f"{page}: a link to {target}, which is not there"
