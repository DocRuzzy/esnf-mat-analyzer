import re
from pathlib import Path


def test_task_ledger_exists():
    p = Path("task-ledger.md")
    assert p.exists(), "task-ledger.md must exist at repository root"


def test_copilot_instructions_contains_ledger_requirement():
    p = Path(".github/copilot-instructions.md")
    assert p.exists(), ".github/copilot-instructions.md must exist"
    text = p.read_text(encoding="utf-8")
    assert (
        "R-LEDGER-UPDATE" in text or "Task Ledger Update Requirement" in text
    ), "copilot instructions must include R-LEDGER-UPDATE requirement"


def test_table_integrity_and_statuses():
    p = Path("task-ledger.md")
    text = p.read_text(encoding="utf-8")

    # Ensure table header exists
    assert (
        "| ID | Title | Status | Prereq | Block Reason | Notes / Next Action |" in text
    ), "Expected ledger table header not found"

    # Find table rows starting with | T followed by digits
    rows = [ln for ln in text.splitlines() if re.match(r"^\|\s*T\d+\s*\|", ln)]
    assert rows, "No task rows found in ledger table"

    allowed_statuses = {"TODO", "IN-PROGRESS", "BLOCKED", "DONE", "DEFERRED"}
    done_ids = set()
    for row in rows:
        parts = [c.strip() for c in row.split("|")]
        # parts: ['', 'T001', 'Title', 'STATUS', 'Prereq', 'Block Reason', 'Notes', '']
        if len(parts) < 5:
            raise AssertionError(f"Malformed ledger row: {row}")
        tid = parts[1]
        status = parts[3]
        assert status in allowed_statuses, f"Invalid status '{status}' for task {tid}"
        if status == "DONE":
            done_ids.add(tid)

    # If any DONE in table, ensure Completed Tasks section lists them with a completion date
    completed_section = re.search(r"## Completed Tasks\n(.*?)(\n## |\Z)", text, re.S)
    if done_ids:
        assert completed_section, "Completed Tasks section missing"
        comp_text = completed_section.group(1)
        for tid in done_ids:
            # expect a line like '- T001 ... (YYYY-MM-DD)'
            m = re.search(
                rf"-\s*{re.escape(tid)}[^\(]*\((\d{{4}}-\d{{2}}-\d{{2}})\)", comp_text
            )
            assert (
                m
            ), f"DONE task {tid} not found with completion date in Completed Tasks section"
