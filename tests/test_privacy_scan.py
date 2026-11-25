"""Tests for tools/privacy_scan.py.

Every sample that the scanner must flag is assembled at run time, so this file never contains one:
the scanner runs on the whole tracked tree, this file included.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCANNER = ROOT / "tools" / "privacy_scan.py"


def _load():
    spec = importlib.util.spec_from_file_location("privacy_scan", SCANNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["privacy_scan"] = module
    spec.loader.exec_module(module)
    return module


ps = _load()


def classes(text: str, path: str = "sample.txt") -> set[str]:
    return {finding.cls for finding in ps.scan_text(text, path)}


# ---- samples assembled at run time -------------------------------------------------------------

SLASH = "/"
HOME_MAC = SLASH.join(["", "Users", "alice", "work", "notes.txt"])
HOME_LINUX = SLASH.join(["", "home", "bob", ".cache"])
TMP_MAC = SLASH.join(["", "private", "var", "folders", "xy", "T"])
EMAIL = "alice" + "@" + "uni-x" + ".edu"
GITHUB_TOKEN = "gh" + "p_" + "A1b2C3d4" * 5
HF_TOKEN = "h" + "f_" + "Zz9" * 10
AWS_KEY = "AK" + "IA" + "ABCDEFGHIJKLMNOP"
PRIVATE_KEY = "-----BEGIN " + "RSA PRIVATE" + " KEY-----"
HEX16 = "58f3" + "a2c4" + "9d0e" + "1b7f"
HEX32 = HEX16 * 2
HEX40 = "3d3c42e5" + "aac5ba80" + "5825da76" + "410c1812" + "73ba90b1"
HEX64 = HEX32 * 2
IPV4 = ".".join(["10", "0", "3", "17"])
IPV6 = ":".join(["fe80", "", "1ff", "fe23", "4567", "890a"])
GPU_UUID = "GPU-" + "-".join(["1a2b3c4d", "5e6f", "7a8b", "9c0d", "1e2f3a" + "4b5c6d"])
RUN_STAMP = "lock-" + "20261113" + "T" + "191353" + "Z"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"see {HOME_MAC}", "home-path"),
        (f"cache at {HOME_LINUX}", "home-path"),
        (f"temp in {TMP_MAC}", "home-path"),
        (f"contact {EMAIL} for access", "email"),
        (f"token={GITHUB_TOKEN}", "credential"),
        (f"HF={HF_TOKEN}", "credential"),
        (f"key {AWS_KEY}", "credential"),
        (PRIVATE_KEY, "credential"),
        (f"law {HEX16}", "hex-id"),
        (f"instance {HEX32}", "hex-id"),
        (f"host {IPV4}", "ip-address"),
        (f"link {IPV6}", "ip-address"),
        (f"device {GPU_UUID}", "uuid"),
        (f"run {RUN_STAMP}", "run-id"),
    ],
)
def test_each_class_is_flagged(text: str, expected: str) -> None:
    assert expected in classes(text)


def test_clean_text_passes() -> None:
    clean = "\n".join(
        [
            "ServeCast predicts TTFT and TPOT on a GPU running vLLM 0.25.1 with Python 3.11.",
            "Released on 2026-11-17; see https://github.com/servecast/servecast for the source.",
            "Clone with git@github.com:servecast/servecast.git and run `servecast predict -c m.yaml`.",
            "@dataclass(frozen=True) decorates a class; commit 8a9921a is short enough.",
            "Localhost is 127.0.0.1 and the wildcard address is 0.0.0.0.",
            "Author: ServeCast <servecast@users.noreply.github.com>",
        ]
    )
    assert ps.scan_text(clean, "README.md") == []


def test_a_pinned_action_may_carry_its_commit_sha() -> None:
    line = f"      - uses: actions/checkout@{HEX40} # v7.0.1"
    assert ps.scan_text(line, ".github/workflows/ci.yml") == []


def test_a_bare_40_hex_outside_a_uses_line_is_flagged() -> None:
    assert "hex-id" in classes(f"the run was {HEX40}")


def test_a_sha256_may_appear_under_its_key_or_in_sha256sums() -> None:
    assert ps.scan_text(f'"sha256": "{HEX64}"', "calibration.json") == []
    assert ps.scan_text(f"{HEX64}  servecast-0.0.1.tar.gz", "SHA256SUMS") == []


def test_a_bare_64_hex_is_flagged() -> None:
    assert "hex-id" in classes(f"digest {HEX64}")


def test_findings_never_print_the_whole_secret() -> None:
    [finding] = [f for f in ps.scan_text(f"token={GITHUB_TOKEN}") if f.cls == "credential"]
    assert GITHUB_TOKEN not in finding.excerpt
    assert GITHUB_TOKEN not in str(finding)


def test_versions_dates_and_dotted_numbers_are_not_addresses() -> None:
    assert "ip-address" not in classes("vLLM 0.25.1, numpy 2.5.0, release 1.14.3 on 2026-11-17")


def test_slices_and_scopes_are_not_ipv6() -> None:
    assert "ip-address" not in classes("x[a::b], std::vector, Class::method, 12:30:45")


@pytest.mark.parametrize(
    "data",
    [b"\x00\x01" + GITHUB_TOKEN.encode(), ("x " + GITHUB_TOKEN).encode("utf-16"), b"caf\xe9 " + GITHUB_TOKEN.encode()],
    ids=["nul-byte", "utf-16", "latin-1"],
)
def test_a_file_that_is_not_utf8_text_is_flagged_since_it_cannot_be_scanned(tmp_path: Path, data: bytes) -> None:
    blob = tmp_path / "data.bin"
    blob.write_bytes(data)
    assert [f.cls for f in ps.scan_paths([blob])] == ["unreadable"]


def test_the_command_exits_1_on_a_finding_and_0_when_clean(tmp_path: Path) -> None:
    dirty = tmp_path / "dirty.txt"
    dirty.write_text(f"mail {EMAIL}\n", encoding="utf-8")
    clean = tmp_path / "clean.txt"
    clean.write_text("nothing to see\n", encoding="utf-8")
    run = [sys.executable, str(SCANNER)]
    assert subprocess.run([*run, str(dirty)], capture_output=True).returncode == 1
    assert subprocess.run([*run, str(clean)], capture_output=True).returncode == 0


def test_an_exact_string_can_be_allowed_for_one_call_only() -> None:
    assert "email" in classes(f"Co-Authored-By: a bot <{EMAIL}>")
    assert ps.scan_text(f"Co-Authored-By: a bot <{EMAIL}>", "message", allow={EMAIL}) == []
    assert "email" in classes(f"Co-Authored-By: a bot <{EMAIL}>")  # nothing leaks into later calls


def test_the_command_reads_stdin_for_commit_metadata() -> None:
    proc = subprocess.run(
        [sys.executable, str(SCANNER), "--stdin", "commit message"],
        input=f"Signed-off-by: someone <{EMAIL}>\n",
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1
    assert "commit message" in proc.stdout


# ---- the 2026-11-17 evaluation: classes that were missed, and public text that was refused ----------

PYTEST_TMP = SLASH.join(["", "tmp", "pytest-of-" + "someone", "run1", "out.json"])
MAC_TMP = SLASH.join(["", "private", "tmp", "claude-" + "501", "x"])
HPC_SCRATCH = SLASH.join(["", "scratch", "netid" + "42", "runs", "out.csv"])
ROOT_HOME = SLASH.join(["", "root", ".cache", "huggingface"])
WIN_JSON = '"C:' + "\\\\" + "Users" + "\\\\" + "someone" + "\\\\" + 'x"'
PGP_KEY = "-----BEGIN " + "PGP PRIVATE" + " KEY BLOCK-----"
AWS_TEMP = "AS" + "IA" + "ABCDEFGHIJKLMNOP"
HEX_UPPER = "1F2E3D4C" + "5B6A7988"
SSH_TARGET = "ssh " + "someone" + "@" + "gpubox"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"see {PYTEST_TMP}", "home-path"),
        (f"see {MAC_TMP}", "home-path"),
        (f"data in {HPC_SCRATCH}", "home-path"),
        (f"cache {ROOT_HOME}", "home-path"),
        (f"path {WIN_JSON}", "home-path"),
        (f"file trace_{RUN_STAMP[5:]}.csv", "run-id"),
        ("run " + "20261113" + "-" + "191353", "run-id"),
        (PGP_KEY, "credential"),
        (f"key {AWS_TEMP}", "credential"),
        (f"law {HEX_UPPER}", "hex-id"),
        (SSH_TARGET, "ssh-target"),
    ],
)
def test_each_missed_class_is_now_flagged(text: str, expected: str) -> None:
    assert expected in classes(text)


@pytest.mark.parametrize(
    "text",
    [
        "nvidia-cudnn-cu12==" + ".".join(["9", "1", "0", "70"]),
        'version = "' + ".".join(["12", "4", "5", "8"]) + '"',
        f"https://github.com/vllm-project/vllm/blob/{HEX40}/vllm/config.py#L10",
        f"vllm @ git+https://github.com/vllm-project/vllm@{HEX40}",
        f"https://github.com/vllm-project/vllm/pull/1/commits/{HEX40}",
        "contact: maintainer" + "@" + "example.com",
        f"{HEX64}  servecast-0.1.0.tar.gz",
        '"preempted_fraction": 8.503' + "0627177789" + "e-05",
        "a rate of 1.04350095689" + "58335e+12 per second",
        "released on 2026-11-17",
    ],
)
def test_public_text_is_not_flagged(text: str) -> None:
    assert ps.scan_text(text, "README.md") == []


def test_the_co_author_trailer_of_a_commit_message_is_allowed() -> None:
    message = "Fix a thing\n\nCo-Authored-By: Claude Opus 5.5 <noreply" + "@" + "anthropic.com>\n"
    assert ps.scan_text(message, "COMMIT_EDITMSG") == []
    assert ps.scan_text("some" + "one" + "@" + "anthropic.com", "x")


def test_the_co_author_address_is_allowed_on_its_trailer_line_only() -> None:
    address = "noreply" + "@" + "anthropic.com"
    assert classes(f"For help, write to {address}.", "README.md") == {"email"}
    assert classes(f"Co-Authored-By: Claude <{address}> in the middle of a sentence", "README.md") == {"email"}
