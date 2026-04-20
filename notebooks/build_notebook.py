"""
build_notebook.py  — clean rebuild
-----------------------------------
Reads motor_fault_COMPLETE.py, strips BOM + non-printable characters,
splits on CELL markers, and writes a valid UTF-8 (no BOM) .ipynb file
that Google Colab can open and run without any SyntaxError.

Usage:
    python build_notebook.py
"""

import json, re, sys
from pathlib import Path

SOURCE = Path(__file__).parent / "motor_fault_COMPLETE.py"
OUTPUT = Path(__file__).parent / "motor_fault_COMPLETE.ipynb"

# ── 1. Read with explicit utf-8-sig so Python strips BOM automatically ──────
raw = SOURCE.read_text(encoding="utf-8-sig")   # utf-8-sig strips \ufeff

# Strip any remaining non-printable / zero-width characters (U+FEFF, U+200B …)
raw = raw.replace("\ufeff", "").replace("\u200b", "").replace("\u00a0", " ")

# Normalise Windows CRLF → LF
raw = raw.replace("\r\n", "\n").replace("\r", "\n")

# ── 2. Split on "# ====…\n# CELL N" markers ─────────────────────────────────
# Each block starts with the decorator line(s) + the CELL N heading
parts = re.split(r"(?m)(?=^# =+\n# CELL \d)", raw)
cells_src = [p for p in parts if p.strip()]

# ── 3. Helper builders ───────────────────────────────────────────────────────
def make_code_cell(lines):
    """lines: list[str]  — each string should end with '\n' (except last)."""
    # Ensure clean strings: no BOM, consistent newlines
    cleaned = []
    for i, ln in enumerate(lines):
        ln = ln.replace("\ufeff", "")
        if i < len(lines) - 1 and not ln.endswith("\n"):
            ln += "\n"
        cleaned.append(ln)
    return {
        "cell_type"      : "code",
        "execution_count": None,
        "metadata"       : {},
        "outputs"        : [],
        "source"         : cleaned,
    }

def make_md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata" : {},
        "source"   : [text],
    }

# ── 4. Build cells ───────────────────────────────────────────────────────────
nb_cells = []

# --- Title ---
nb_cells.append(make_md_cell(
    "# Motor Fault Prediction — Algerian Manufacturing Plant (M'Sila)\n"
    "> Production-grade predictive maintenance pipeline  \n"
    "> UCI AI4I 2020 | XGBoost + LightGBM + Optuna | SHAP | MotorFailurePredictor\n\n"
    "**Runtime**: GPU (T4) recommended — *Runtime > Change runtime type > T4 GPU*"
))

# --- Install cell (explicit, always first) ---
nb_cells.append(make_code_cell([
    "# ── Cell 0: Install dependencies ────────────────────────────────────────\n",
    "!pip install -q ucimlrepo 'xgboost>=2.0' 'lightgbm>=4.0' "
    "imbalanced-learn shap optuna tqdm\n",
    "print('All packages installed.')\n",
]))

# --- One cell per CELL block ---
for block in cells_src:
    lines = block.split("\n")

    # Collect heading comment lines vs. actual code
    heading_lines, code_lines = [], []
    in_header = True
    for ln in lines:
        if in_header and re.match(r"^# [=\-]+$|^# CELL", ln):
            heading_lines.append(ln)
        else:
            in_header = False
            code_lines.append(ln)

    # Markdown divider from "# CELL N — Description"
    md_text = ""
    for h in heading_lines:
        m = re.search(r"CELL (\d+)\s*[—\-]+\s*(.+)", h)
        if m:
            md_text = f"### Cell {m.group(1)} — {m.group(2).strip()}"
            break
    if md_text:
        nb_cells.append(make_md_cell(md_text))

    # Code cell — rejoin and re-split to get proper line endings
    code_body = "\n".join(code_lines)
    if not code_body.strip():
        continue

    # Re-split into lines with \n terminators (except last)
    final_lines = code_body.split("\n")
    with_newlines = [ln + "\n" for ln in final_lines[:-1]] + [final_lines[-1]]

    nb_cells.append(make_code_cell(with_newlines))

# ── 5. Assemble notebook ─────────────────────────────────────────────────────
notebook = {
    "nbformat"      : 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language"    : "python",
            "name"        : "python3",
        },
        "language_info": {
            "name"   : "python",
            "version": "3.10.0",
        },
        "accelerator": "GPU",
        "colab": {
            "name"      : "Motor_Fault_Prediction_M_Sila.ipynb",
            "provenance": [],
        },
    },
    "cells": nb_cells,
}

# ── 6. Write as UTF-8 without BOM ────────────────────────────────────────────
notebook_json = json.dumps(notebook, indent=1, ensure_ascii=False)

# Final safety check: no BOM in output
assert "\ufeff" not in notebook_json, "BOM found in output JSON — aborting!"

with open(OUTPUT, "w", encoding="utf-8", newline="\n") as fh:
    fh.write(notebook_json)

# ── 7. Report ─────────────────────────────────────────────────────────────────
print("[OK] Notebook written to:", str(OUTPUT))
print("     Cells      :", len(nb_cells))
print("     File size  :", OUTPUT.stat().st_size // 1024, "KB")
print("     BOM check  : CLEAN (no U+FEFF)")
print()
print("Upload: colab.research.google.com -> File -> Upload notebook")
print("        -> select motor_fault_COMPLETE.ipynb")
