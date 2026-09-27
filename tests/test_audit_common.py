"""Access to the research code of the study for the comparison tests. Set VIDEO_INDEX_RESEARCH_CODE to the directory
that holds `pipeline/` and `analysis/` of that code; the comparison tests are skipped when it is not set."""
import importlib
import os
import sys


def research_module(name, *subdirs):
    root = os.environ.get("VIDEO_INDEX_RESEARCH_CODE")
    if not root or not os.path.isdir(root):
        return None
    for d in ("pipeline",) + subdirs:
        p = os.path.join(root, d)
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        return importlib.import_module(name)
    except Exception as e:  # noqa: BLE001
        print(f"research module {name} not importable: {type(e).__name__}: {e}", file=sys.stderr)
        return None


def run_all(namespace):
    """Run every test_* function of a module without pytest."""
    n = 0
    for k, f in sorted(namespace.items()):
        if k.startswith("test_") and callable(f):
            f()
            n += 1
            print(f"ok  {k}")
    print(f"{n} tests passed")
