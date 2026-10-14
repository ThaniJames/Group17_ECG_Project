#!/usr/bin/env bash
set -e

# always run relative to the project root, regardless of caller's cwd
cd "$(dirname "${BASH_SOURCE[0]}")"

MIN="3.12"  # numpy and scipy require >=3.12; nothing in requirements.txt needs more

# true if the given interpreter command is exactly $REQUIRED (unquoted so "py -3.14" splits)
is_required() {
    $1 -c "import sys; sys.exit(sys.version_info[:2] < tuple(map(int, '$MIN'.split('.'))))" >/dev/null 2>&1
}

# venv python lives in bin/ on Linux/Mac and Scripts/ on Windows
venv_python() {
    for p in .venv/bin/python .venv/Scripts/python.exe; do
        [ -f "$p" ] && echo "$p" && return 0
    done
    return 1
}

CREATED=0
# reuse the venv only if it works and has the right Python; otherwise rebuild it
if ! PY=$(venv_python) || ! is_required "$PY"; then
    SYS_PY=""
    for cmd in python3.14 python3.13 python3.12 python3 python "py -3"; do
        if is_required "$cmd"; then SYS_PY="$cmd"; break; fi
    done
    if [ -z "$SYS_PY" ]; then
        echo "ERROR: Python $MIN or newer not found. Install it from python.org and re-run." >&2
        exit 1
    fi
    rm -rf .venv
    $SYS_PY -m venv .venv
    PY=$(venv_python)
    CREATED=1
fi

# no-op when already satisfied; drop a half-built venv so the next run starts clean
if ! "$PY" -m pip install -r requirements.txt; then
    if [ "$CREATED" = 1 ]; then rm -rf .venv; fi
    exit 1
fi

"$PY" scripts/run_prepare.py
