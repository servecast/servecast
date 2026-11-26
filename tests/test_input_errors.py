"""A file, a path or a value ServeCast cannot use is the user's to fix: an input error (exit 2) or a refused resource
(exit 3), named with its file and line, never "internal error ... please report it" (findings F06, F07, Q8, site-9,
F12, F15, F16, F01, MF-10, Q7 and MF-11 of the outside review)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from servecast.cli.app import app

RUN = CliRunner()


def run(*args: str) -> tuple[int, str]:
    result = RUN.invoke(app, list(args))
    return result.exit_code, result.output


def template(tmp_path: Path) -> Path:
    path = tmp_path / "m.yaml"
    assert run("manifest", "init", "--out", str(path))[0] == 0
    return path


@pytest.mark.parametrize(("name", "body", "expected"), [
    ("bin.jsonl", b"\xca\xfe\xba\xbe" * 40, "bin.jsonl: not UTF-8 text"),         # F07, Q8: a UnicodeDecodeError
    ("bad.jsonl", b"not json\n", "bad.jsonl:1: not a JSON line"),                   # F07: no file and no line
    ("big.jsonl", b'{"input_length": 1e400, "output_length": 3}\n', "big.jsonl:1"),  # Q8: an OverflowError
    ("float.jsonl", b'{"input_length": 2.5, "output_length": 3}\n', "float.jsonl:1"),  # MF-10: 2.5 was read as 2
    ("bool.jsonl", b'{"input_length": true, "output_length": 3}\n', "bool.jsonl:1"),   # MF-10: true was read as 1
])
def test_a_trace_that_cannot_be_read_names_its_file_and_line(tmp_path: Path, name: str, body: bytes,
                                                             expected: str) -> None:
    path = tmp_path / name
    path.write_bytes(body)
    code, output = run("workload", "describe", str(path))
    assert code == 2 and expected in output and "internal error" not in output, output


def test_an_integral_float_length_is_still_a_length(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text("".join(json.dumps({"input_length": 512.0, "output_length": 8}) + "\n" for _ in range(3)),
                    encoding="utf-8")
    code, output = run("workload", "describe", str(path), "--format", "json")
    assert code == 0 and json.loads(output)["input_tokens"]["mean"] == 512, output


def test_a_csv_trace_read_as_mooncake_says_which_format_it_needs(tmp_path: Path) -> None:
    """The review's MF-11: a JSON error without the file's name."""
    trace = tmp_path / "t.csv"
    trace.write_text("input_tokens,output_tokens\n" + "500,100\n" * 200, encoding="utf-8")
    code, output = run("manifest", "validate", "-c", str(template(tmp_path)), "--set",
                       f"traffic.source={{type: trace, paths: [{trace}], format: mooncake}}")
    assert code == 2 and "t.csv:1" in output and "format: lengths" in output, output


@pytest.mark.parametrize("command", [("workload", "sample", "-c", "{m}", "--out"), ("manifest", "init", "--out")])
def test_an_output_file_that_cannot_be_written_is_a_refused_resource(tmp_path: Path, command: tuple[str, ...]) -> None:
    """The review's F06: an internal error (FileNotFoundError)."""
    m = template(tmp_path)
    args = [a.format(m=m) for a in command] + [str(tmp_path / "no-such-dir" / "out")]
    code, output = run(*args)
    assert code == 3 and "cannot write" in output and "internal error" not in output, output


BAD_REGISTRY = {"schema": "servecast-gpus/1", "gpus": {"x": {"device_name": "X"}}}


def test_an_unreadable_gpu_registry_is_reported_not_crashed_on(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The review's site-9 and F12: list gpus crashed with an internal error, and doctor said nothing."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "gpus.json").write_text(json.dumps(BAD_REGISTRY), encoding="utf-8")
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(data))
    code, output = run("list", "gpus")
    assert code == 2 and "gpus.json" in output and "internal error" not in output, output
    code, output = run("doctor", "--format", "json")
    doc = json.loads(output)
    assert code == 0 and any("gpus.json" in p for p in doc["problems"]), doc


def test_doctor_names_the_data_and_cache_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The review's F11: error messages said that doctor names the data directory, and it did not."""
    monkeypatch.setenv("SERVECAST_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("SERVECAST_CACHE_DIR", str(tmp_path / "cache"))
    code, output = run("doctor", "--format", "json")
    doc = json.loads(output)
    assert code == 0 and doc["data_dir"] == str(tmp_path / "data") and doc["cache_dir"] == str(tmp_path / "cache")
    code, output = run("doctor")
    assert str(tmp_path / "data") in output and str(tmp_path / "cache") in output, output


@pytest.mark.parametrize(("config", "expected"), [
    ("{dir}", "a directory"),                                           # F16: "no such file" for a directory
    ("-", "standard input"),                                            # F16: "no such file" for -
])
def test_a_config_that_is_not_a_manifest_file_says_what_it_is(tmp_path: Path, config: str, expected: str) -> None:
    for command in (("predict", "--output-dir", str(tmp_path / "o")), ("manifest", "validate")):
        code, output = run(*command[:1 if command[0] == "predict" else 2], "-c", config.format(dir=tmp_path),
                           *command[1 if command[0] == "predict" else 2:])
        assert code == 2 and expected in output and "no such file" not in output, (command, output)


def test_an_empty_file_given_as_a_manifest_says_it_is_not_one(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    code, output = run("predict", "-c", str(empty), "--output-dir", str(tmp_path / "o"))
    assert code == 2 and "a manifest is a YAML mapping" in output, output


def test_a_concurrency_no_gpu_could_serve_is_refused(tmp_path: Path) -> None:
    """The review's F01: 10 million clients hung past --timeout and grew past 3 GB."""
    code, output = run("manifest", "validate", "-c", str(template(tmp_path)), "--set",
                       "traffic.load.concurrency=10000000")
    assert code == 2 and "traffic.load.concurrency" in output, output


def test_the_internal_error_hint_puts_debug_where_it_works(monkeypatch: pytest.MonkeyPatch,
                                                           capsys: pytest.CaptureFixture[str]) -> None:
    """The review's F15: the hint said "run again with --debug", which works only right after `servecast`."""
    from servecast.cli import app as cli

    def fails() -> None:
        raise RuntimeError("an unforeseen failure")

    monkeypatch.setattr(cli, "app", fails)
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 1 and "servecast --debug" in capsys.readouterr().err
