"""Every ```python block of README.md must run.

Blocks are executed in a fresh namespace, in order of appearance.  Blocks that
use ``copul.optim`` are skipped when ``cvxpy`` is not installed; a block
preceded by an HTML comment ``<!-- slow -->`` is marked ``slow``.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

README = Path(__file__).resolve().parents[1] / "README.md"
_BLOCK = re.compile(r"(<!--\s*slow\s*-->\s*\n)?```python\n(.*?)```", re.S)


def _has_cvxpy() -> bool:
    try:
        import cvxpy  # noqa: F401
    except ImportError:
        return False
    return True


def _blocks():
    text = README.read_text(encoding="utf-8")
    out = []
    for i, m in enumerate(_BLOCK.finditer(text)):
        code = m.group(2)
        line = text.count("\n", 0, m.start(2)) + 1
        marks = []
        if m.group(1):
            marks.append(pytest.mark.slow)
        if "copul.optim" in code and not _has_cvxpy():
            marks.append(pytest.mark.skip(reason="copul.optim needs cvxpy"))
        out.append(pytest.param(code, id=f"block{i}-line{line}", marks=marks))
    return out


BLOCKS = _blocks()


def test_readme_has_examples():
    assert len(BLOCKS) >= 8


@pytest.mark.parametrize("code", BLOCKS)
def test_readme_block_runs(code):
    import matplotlib.pyplot as plt

    namespace: dict = {"__name__": "__readme__"}
    try:
        exec(compile(code, str(README), "exec"), namespace)
    finally:
        plt.close("all")
