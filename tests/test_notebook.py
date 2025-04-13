import json
from pathlib import Path

NOTEBOOK = Path(__file__).parent.parent / "Technical_Assignment_Email_Matter_Grouping_System.ipynb"


def test_demo_notebook_is_committed_without_outputs():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code_cells
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in code_cells)
    assert "widgets" not in nb["metadata"]


def test_demo_notebook_has_no_shell_installs():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    sources = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    assert not any(line.lstrip().startswith("!") for s in sources for line in s.splitlines())
