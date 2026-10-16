#!/usr/bin/env python3
"""
Secret scan: fails if a credential is committed anywhere in the repository.

Author Name:  Imaad Ullah Khan
Author Email: yameenimaad@gmail.com
AI Helper:    Claude

Run:  python Tests/secret_scan_test.py        (from App/)
      python Tests/secret_scan_test.py -v     (show each check as it runs)

------------------------------------------------------------------------------------------------
WHY THIS EXISTS, WHEN privacy_leak_test.py ALREADY CHECKED THE KEY
------------------------------------------------------------------------------------------------
privacy_leak_test.py asserts that `App/roboflow_key.txt` is untracked and listed in .gitignore.
That check passed continuously while the live Roboflow key was, in fact, publicly readable in
github.com/yameenimaad/Pink_Edge_AI -- because the file got there through GitHub's web "Add files
via upload" button, which writes a commit server-side and never consults .gitignore at all. A
guard that only knows one filename and only understands `git add` cannot see that.

So this suite checks the thing that actually matters -- *is a secret present in the committed
content* -- rather than whether one known path was staged:

  1. the live key's own value, if a local key file exists, searched for across every tracked file.
     This is the strongest check available: it needs no pattern to be right, and it catches the
     key pasted into a .py, a notebook, a README, a JSON manifest or a file named anything at all.
  2. provider token formats (GitHub, AWS, Google, Slack, OpenAI, private-key PEM blocks), which
     catch a credential this project has never held but someone might paste in later.
  3. hardcoded assignments -- `API_KEY = "..."`, `token: '...'` -- with a long literal on the right.
  4. that no file whose name suggests a credential is tracked, under any directory.

A NOTE ON WHY A KEY IS NEVER PUT "INSIDE A PYTHON FILE". A .py file in a public repository is
exactly as readable as a .txt one: GitHub renders it in the browser, the API serves it, and the
automated scanners that trawl public pushes for credentials index it identically. Moving a secret
from a text file into source code hides it from nobody and additionally makes it impossible to
rotate without editing and redeploying code. The supported ways to supply this project's key --
all of which keep it out of the repository -- are, in the order inference.py._roboflow_api_key()
tries them:

    ROBOFLOW_API_KEY in the environment          <- preferred for servers and CI
    st.secrets["ROBOFLOW_API_KEY"]               <- Streamlit Community Cloud's Secrets manager
    App/roboflow_key.txt                         <- local development only; gitignored

------------------------------------------------------------------------------------------------
WHAT THIS SUITE CANNOT DO
------------------------------------------------------------------------------------------------
It reads the CURRENT checkout, so it cannot tell you whether a secret is still reachable in an
earlier commit of a repository that has already been pushed. Git keeps every version: deleting a
key in a later commit leaves it readable at the old revision, and on a public repo it must be
assumed harvested the moment it was pushed. Removal is housekeeping; ROTATION at the provider is
the only thing that actually revokes an exposed key.
"""

import os
import re
import subprocess
import sys

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(APP_DIR)
VERBOSE = "-v" in sys.argv

# This file necessarily contains the patterns it searches for, so it would match itself. Listing
# it by name is deliberate and narrow -- anything else, including other test files, is scanned.
SELF = os.path.relpath(os.path.abspath(__file__), REPO_DIR).replace("\\", "/")

# Binary and dataset content: scanning 22k JPEGs for text patterns is pure cost, and a credential
# cannot hide in a JPEG in a way that matters for a source repository.
SKIP_SUFFIXES = (
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".webp", ".pdf", ".mp4", ".zip",
    ".pt", ".pth", ".onnx", ".tflite", ".db", ".xlsx", ".pyc", ".woff", ".woff2", ".ttf",
)
MAX_BYTES = 2 * 1024 * 1024   # a file larger than this is data, not source

# Documentation in this repo legitimately discusses the incident and names the file; those
# sentences must not be read as a leak. Only exact prose markers are allowed, never a key shape.
ALLOWED_MENTION = re.compile(r"roboflow_key\.txt|ROBOFLOW_API_KEY|YOUR_TOKEN|your-key-here|<TOKEN>")

TOKEN_PATTERNS = [
    # GitHub: classic PATs and the newer fine-grained form.
    ("GitHub personal access token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36}\b")),
    ("GitHub fine-grained token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("AWS access key id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[0-9A-Za-z-]{10,}\b")),
    ("OpenAI API key", re.compile(r"\bsk-[A-Za-z0-9]{32,}\b")),
    ("private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH |PGP )?PRIVATE KEY-----")),
]

# `NAME = "longvalue"` where NAME looks credential-ish. Requires >=16 chars with no spaces, so
# ordinary prose and short placeholders do not trip it.
ASSIGNMENT = re.compile(
    r"""(?i)\b(api[_-]?key|apikey|secret|token|passwd|password|private[_-]?key)\b\s*[:=]\s*
        ['"]([^'"\s]{16,})['"]""",
    re.VERBOSE,
)
# Values that are obviously not secrets even though they sit in a credential-shaped assignment.
PLACEHOLDER = re.compile(
    r"(?i)^(your|my|the|some|example|placeholder|changeme|xxx+|\.\.\.|<|\$\{|%s|f?['\"]?\{)"
)

RESULTS = []


def check(label):
    def deco(fn):
        try:
            detail = fn()
            RESULTS.append((True, label, detail or ""))
            if VERBOSE:
                print(f"[PASS] {label:62s} {detail or ''}")
        except AssertionError as e:
            RESULTS.append((False, label, str(e)))
            if VERBOSE:
                print(f"[FAIL] {label:62s} {e}")
        return fn
    return deco


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)


def tracked_files():
    """Every file git actually tracks. Reading the index rather than walking the filesystem is the
    point: untracked scratch files are irrelevant, and tracked ones are what a clone exposes."""
    out = subprocess.run(["git", "-C", REPO_DIR, "ls-files", "-z"],
                         capture_output=True, text=True, encoding="utf-8")
    if out.returncode != 0:
        return []
    return [p for p in out.stdout.split("\0") if p]


def readable_text_files():
    for rel in tracked_files():
        if rel == SELF or rel.lower().endswith(SKIP_SUFFIXES):
            continue
        full = os.path.join(REPO_DIR, rel)
        try:
            if not os.path.isfile(full) or os.path.getsize(full) > MAX_BYTES:
                continue
            with open(full, "r", encoding="utf-8", errors="strict") as fh:
                yield rel, fh.read()
        except (OSError, UnicodeDecodeError):
            continue    # binary or unreadable: nothing text-scannable here


def local_key_values():
    """The live secret values present on THIS machine, so they can be searched for in the
    committed content. Values are never printed by this suite -- only the file and line that
    contains one -- because the report itself would otherwise become the leak."""
    values = []
    key_file = os.path.join(APP_DIR, "roboflow_key.txt")
    if os.path.isfile(key_file):
        try:
            with open(key_file, "r", encoding="utf-8") as fh:
                v = fh.read().strip()
            if len(v) >= 8:
                values.append(("Roboflow key from App/roboflow_key.txt", v))
        except OSError:
            pass
    env = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if len(env) >= 8:
        values.append(("Roboflow key from $ROBOFLOW_API_KEY", env))
    return values


# ============================================================
# CHECKS
# ============================================================
@check("the live Roboflow key value appears in no tracked file")
def _():
    secrets = local_key_values()
    if not secrets:
        return "(skipped: no local key configured to search for)"
    hits = []
    for rel, text in readable_text_files():
        for name, value in secrets:
            if value in text:
                line = text[:text.index(value)].count("\n") + 1
                hits.append(f"{rel}:{line} contains the {name}")
    require(not hits, "A LIVE KEY IS IN THE COMMITTED CONTENT -> " + "; ".join(hits[:5]) +
                      ". Remove it, then ROTATE the key: anything already pushed must be "
                      "treated as compromised regardless of later deletion.")
    return f"searched {len(secrets)} local secret value(s) across the tracked tree"


@check("no provider token format is committed")
def _():
    hits = []
    for rel, text in readable_text_files():
        for name, pattern in TOKEN_PATTERNS:
            for m in pattern.finditer(text):
                line = text[:m.start()].count("\n") + 1
                hits.append(f"{rel}:{line} looks like a {name}")
    require(not hits, "credential-shaped value(s) committed -> " + "; ".join(hits[:5]))
    return f"{len(TOKEN_PATTERNS)} provider formats checked"


@check("no hardcoded credential assignment in source")
def _():
    hits = []
    for rel, text in readable_text_files():
        for m in ASSIGNMENT.finditer(text):
            value = m.group(2)
            if PLACEHOLDER.match(value) or ALLOWED_MENTION.search(value):
                continue
            line = text[:m.start()].count("\n") + 1
            hits.append(f"{rel}:{line} assigns {m.group(1)} a literal value")
    require(not hits, "hardcoded credential(s) -> " + "; ".join(hits[:5]) +
                      ". Read it from the environment instead; a .py file in a public repo is as "
                      "readable as a .txt one.")
    return "no literal assigned to an api_key/secret/token/password name"


@check("no credential-looking file is tracked, under any path")
def _():
    suspicious = re.compile(
        r"(?i)(^|/)(.*_key|.*secret.*|.*credential.*|.*token.*|\.env|id_rsa|id_ed25519)"
        r"(\.txt|\.json|\.ya?ml|\.pem|\.key|\.cfg|\.ini)?$")
    hits = [rel for rel in tracked_files()
            if suspicious.search(rel) and not rel.lower().endswith((".py", ".md"))]
    require(not hits, "credential-looking file(s) tracked -> " + ", ".join(hits[:5]))
    return f"{len(tracked_files())} tracked paths checked"


@check("the key file stays gitignored and untracked")
def _():
    out = subprocess.run(["git", "-C", REPO_DIR, "ls-files", "App/roboflow_key.txt"],
                         capture_output=True, text=True, encoding="utf-8")
    require(not out.stdout.strip(), "App/roboflow_key.txt is TRACKED — the key is in the repo")
    gitignore_path = os.path.join(REPO_DIR, ".gitignore")
    require(os.path.isfile(gitignore_path), ".gitignore is missing")
    with open(gitignore_path, "r", encoding="utf-8") as fh:
        require("roboflow_key.txt" in fh.read(), "roboflow_key.txt is not in .gitignore")
    # .gitignore governs `git add`, NOT GitHub's web upload button, which commits server-side.
    # That is how the key reached the public repo while this very assertion was passing.
    return "untracked and ignored (note: web-UI uploads bypass .gitignore entirely)"


@check("the key is read from the environment first, not from source")
def _():
    with open(os.path.join(APP_DIR, "inference.py"), "r", encoding="utf-8") as fh:
        src = fh.read()
    require("os.environ.get(\"ROBOFLOW_API_KEY\")" in src,
            "inference.py no longer reads ROBOFLOW_API_KEY from the environment")
    env_at = src.index("os.environ.get(\"ROBOFLOW_API_KEY\")")
    file_at = src.index("roboflow_key.txt")
    require(env_at < file_at,
            "the local key file is consulted before the environment variable; env must win so a "
            "deployment can override a stale developer file")
    return "env -> st.secrets -> gitignored file, in that order"


def main():
    print("=" * 86)
    print("PINK EDGE AI - SECRET SCAN")
    print("=" * 86)
    if not VERBOSE:
        print("(run with -v to see checks as they execute)\n")
    for ok, label, detail in RESULTS:
        print(f"[{'PASS' if ok else 'FAIL'}] {label:62s} {detail}")
    passed = sum(1 for ok, _, _ in RESULTS if ok)
    print(f"\n{passed}/{len(RESULTS)} checks passed.")
    if passed != len(RESULTS):
        print("\nA failing check means a secret is in the committed content. Removing it is not "
              "sufficient:\nif the commit was ever pushed, rotate the credential at the provider.")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
