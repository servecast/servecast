"""Scan text for things that must never appear in this public repository.

Only generic classes live here: absolute home, temporary and scratch paths, e-mail addresses, ssh
login targets, credentials, bare hexadecimal identifiers, IP addresses, UUIDs and run timestamps; and
a file that is not UTF-8 text is reported, since it cannot be scanned. Words specific to the people
behind the project are checked before every push by a separate guard that is not part of this
repository.

    python tools/privacy_scan.py                    # every tracked file
    python tools/privacy_scan.py PATH [PATH ...]    # the given files or directories
    git log -1 --format=%B | python tools/privacy_scan.py --stdin "commit message"

Exit status: 0 nothing found, 1 something found, 2 usage error.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

MAX_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    column: int
    cls: str
    excerpt: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.cls}: {self.excerpt}"


@dataclass(frozen=True)
class Rule:
    cls: str
    pattern: re.Pattern[str]
    allowed: Callable[[re.Match[str], str, str], bool] | None = None


#: the project's commit identity and GitHub's ssh user
ALLOWED_EMAILS = frozenset({"servecast@users.noreply.github.com", "git@github.com"})
#: the co-author trailer's address, allowed only on a commit message's trailer line (assembled, so this line is not one)
TRAILER_EMAIL = "noreply" + "@" + "anthropic.com"
TRAILER = re.compile(r"Co-Authored-By: [^<>\n]+ <" + re.escape(TRAILER_EMAIL) + r">", re.IGNORECASE)
#: RFC 2606 reserves these for documentation: no one can own an address there
RESERVED_EMAIL = re.compile(r"@(?:[\w-]+\.)*example\.(?:com|net|org)$", re.IGNORECASE)
ALLOWED_IPV4 = frozenset({"127.0.0.1", "0.0.0.0", "255.255.255.255"})

UUID = re.compile(
    r"\b(?:GPU-|MIG-)?[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
PINNED_ACTION = re.compile(r"^\s*(?:-\s*)?uses:\s*[\w.-]+/[\w./-]+@$")
SHA256_KEY = re.compile(r"sha256\W{0,4}$", re.IGNORECASE)
COMMIT_KEY = re.compile(r"(?:commit|sha|revision|rev)\W{0,4}$", re.IGNORECASE)
#: a public commit link or pin: GitHub's /blob/, /tree/, /commit/, /commits/, /raw/ paths, or a VCS `@<sha>`
COMMIT_LINK = re.compile(r"(?:/(?:blob|tree|commit|commits|raw)/|@)$")
#: what precedes a version, so a four-part version is not read as an address
VERSION_BEFORE = re.compile(r"(?:[=<>~!]=\s*|\bv|version\W{0,4})$", re.IGNORECASE)
OCTET = r"(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"


def _email_allowed(match: re.Match[str], line: str, _path: str) -> bool:
    address = match.group(0).lower()
    if address == TRAILER_EMAIL:
        return TRAILER.fullmatch(line.strip()) is not None
    return address in ALLOWED_EMAILS or bool(RESERVED_EMAIL.search(match.group(0)))


def _hex_allowed(match: re.Match[str], line: str, path: str) -> bool:
    """A bare hex string is an identifier unless its context says what it is."""
    value, before, after = match.group(0), line[: match.start()], line[match.end():]
    if not (re.search(r"[a-f]", value, re.IGNORECASE) and re.search(r"\d", value)):
        return True  # digits only is a number; letters only is a word
    if re.fullmatch(r"\d+[eE]\d*", value) and (re.match(r"[-+]\d", after) or value[-1].isdigit()):
        return True  # a float's mantissa and exponent: 8.5e-05, 1e10
    if any(u.start() <= match.start() and match.end() <= u.end() for u in UUID.finditer(line)):
        return True  # part of a UUID, which the uuid rule reports
    if len(value) == 40 and (PINNED_ACTION.search(before) or COMMIT_KEY.search(before) or COMMIT_LINK.search(before)):
        return True  # a pinned action, a commit under its key, or a public commit link
    if len(value) == 64 and not before.strip() and re.match(r"\s+\S", after):
        return True  # a `sha256sum` output line: the digest, then the file it is of
    return len(value) == 64 and (Path(path).name == "SHA256SUMS" or bool(SHA256_KEY.search(before)))


def _ipv4_allowed(match: re.Match[str], line: str, _path: str) -> bool:
    """A four-part version after `==`, `v` or `version` is not an address."""
    return match.group(0) in ALLOWED_IPV4 or bool(VERSION_BEFORE.search(line[: match.start()]))


def _ipv6_allowed(match: re.Match[str], _line: str, _path: str) -> bool:
    """Only the full eight-group form, or a `::` with at least three groups, is an address."""
    value = match.group(0)
    groups = [g for g in value.split(":") if g]
    full = value.count(":") == 7 and "::" not in value
    compressed = "::" in value and len(groups) >= 3
    return not (full or compressed) or not re.search(r"\d", value)


RULES: tuple[Rule, ...] = (
    Rule(
        "home-path",
        re.compile(
            # each literal is broken by a one-character class, so this file does not match its own rule
            r"(?:/Users|/home|/scratch)/[A-Za-z0-9._-]+|/ro[o]t/|(?:/private)?/var/folders/\w{2}/|/private/tm[p]/"
            r"|/tm[p]/pytest-of-|/tm[p]/claude-\d|[A-Za-z]:\\+Users\\+\w+"
        ),
    ),
    Rule("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"), _email_allowed),
    Rule("ssh-target", re.compile(r"\b(?:ssh|scp|sftp|rsync)\s+(?:-\S+\s+)*[A-Za-z0-9._-]+@[A-Za-z0-9.-]+")),
    Rule(
        "credential",
        re.compile(
            r"\b(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w{20,}|hf_[A-Za-z0-9]{20,}|sk-[\w-]{20,}"
            r"|(?:AKIA|ASIA)[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9-]{10,})"
            r"|-----BEGIN [A-Z ]*PRIVATE KEY(?: BLOCK)?-----"
        ),
    ),
    Rule("uuid", UUID),
    Rule("hex-id", re.compile(r"(?<![0-9A-Za-z])[0-9a-fA-F]{12,}(?![0-9A-Za-z])"), _hex_allowed),
    Rule("ip-address", re.compile(rf"(?<![\d.]){OCTET}(?:\.{OCTET}){{3}}(?![\d.])"), _ipv4_allowed),
    Rule(
        "ip-address",
        re.compile(r"(?<![0-9A-Fa-f:])(?:[0-9A-Fa-f]{1,4}:){1,7}:?(?:[0-9A-Fa-f]{1,4}:){0,6}[0-9A-Fa-f]{1,4}(?![0-9A-Fa-f:])"),
        _ipv6_allowed,
    ),
    Rule("run-id", re.compile(r"(?<!\d)\d{8}(?:T\d{6}Z|-\d{6})(?![0-9A-Za-z])")),
)


def mask(secret: str) -> str:
    """Enough of a match to find it, never enough to reuse it."""
    return secret[:4] + "..." if len(secret) > 4 else "..."


def scan_text(text: str, path: str = "<text>", allow: Iterable[str] = ()) -> list[Finding]:
    """Findings in `text`; `allow` lists exact strings accepted for this call only."""
    accepted = frozenset(allow)
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for rule in RULES:
            for match in rule.pattern.finditer(line):
                if match.group(0) in accepted:
                    continue
                if rule.allowed is not None and rule.allowed(match, line, path):
                    continue
                findings.append(Finding(path, number, match.start() + 1, rule.cls, mask(match.group(0))))
    return findings


def _read_text(path: Path) -> str | None:
    """The file as text, or None when it is not UTF-8 text (a NUL byte, UTF-16, Latin-1): not scannable."""
    data = path.read_bytes()[:MAX_BYTES]
    if b"\x00" in data:
        return None
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _expand(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(sorted(p for p in path.rglob("*") if p.is_file() and ".git" not in p.parts))
        else:
            files.append(path)
    return files


def scan_paths(paths: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for path in _expand(paths):
        text = _read_text(path)
        if text is None:
            findings.append(Finding(str(path), 0, 0, "unreadable", "not UTF-8 text: it cannot be scanned"))
        else:
            findings.extend(scan_text(text, str(path)))
    return findings


def tracked_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout
    return [Path(name) for name in out.decode("utf-8").split("\0") if name]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Scan for text that must not appear in a public repository.")
    parser.add_argument("paths", nargs="*", type=Path, help="files or directories (default: every tracked file)")
    parser.add_argument("--stdin", metavar="LABEL", help="scan standard input, reported under LABEL")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)
    if args.stdin is not None and args.paths:
        parser.error("give either --stdin or paths, not both")
    if args.stdin is not None:
        findings = scan_text(sys.stdin.read(), args.stdin)
    else:
        findings = scan_paths(args.paths or tracked_files())
    if args.format == "json":
        print(json.dumps([asdict(f) for f in findings], indent=2))
    else:
        for finding in findings:
            print(finding)
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
