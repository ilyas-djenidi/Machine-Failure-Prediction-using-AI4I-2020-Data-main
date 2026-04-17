# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
"""
build_notebook.py
-----------------
Reads motor_fault_COMPLETE.py and converts it into a valid Jupyter
notebook (.ipynb) that Google Colab can open without errors.

Run:  python build_notebook.py
Output: motor_fault_COMPLETE.ipynb  (same folder)
"""

import json, re
from pathlib import Path

SOURCE = Path(__file__).parent / "motor_fault_COMPLETE.py"
OUTPUT = Path(__file__).parent / "motor_fault_COMPLETE.ipynb"

# ---------------------------------------------------------------------------
# Read source and split on CELL markers
# ---------------------------------------------------------------------------
raw = SOURCE.read_text(encoding="utf-8")

# Split wherever a line starts with  "# CELL  <digit>"
# Keep the delimiter as part of the following chunk
parts = re.split(r'(?m)(?=^# =+\n# CELL \d)', raw)

# Clean up empty leading part
cells_src = [p for p in parts if p.strip()]

# ---------------------------------------------------------------------------
# Build notebook cells
# ---------------------------------------------------------------------------
def make_code_cell(source_lines: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines,
    }

def make_markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text],
    }

nb_cells = []

# Title markdown cell
nb_cells.append(make_markdown_cell(
    "# 🏭 Motor Fault Prediction — Algerian Manufacturing Plant (M'Sila)\n"
    "> Production-grade predictive maintenance pipeline  \n"
    "> UCI AI4I 2020 | XGBoost + LightGBM + Optuna | SHAP | MotorFailurePredictor\n\n"
    "**Runtime**: GPU (T4) recommended — *Runtime → Change runtime type → T4 GPU*"
))

# Install cell (always first code cell, clearly marked)
install_src = [
    "# Install all required packages\n",
    "!pip install -q ucimlrepo 'xgboost>=2.0' 'lightgbm>=4.0' "
    "imbalanced-learn shap optuna tqdm\n",
]
nb_cells.append(make_code_cell(install_src))

# Convert each code section into its own cell
for block in cells_src:
    lines = block.splitlines(keepends=True)
    if not lines:
        continue

    # Extract the CELL heading as a markdown divider
    heading_lines = []
    code_lines = []
    in_heading = True
    for line in lines:
        if in_heading and (line.startswith("# =") or line.startswith("# CELL")):
            heading_lines.append(line)
        else:
            in_heading = False
            code_lines.append(line)

    # Build a small markdown header from the CELL comment
    heading_text = ""
    for h in heading_lines:
        m = re.search(r"CELL (\d+)\s*[—–-]\s*(.+)", h)
        if m:
            heading_text = f"### Cell {m.group(1)} — {m.group(2).strip()}"
            break
    if heading_text:
        nb_cells.append(make_markdown_cell(heading_text))

    # Code cell (skip if only whitespace)
    code_body = "".join(code_lines)
    if code_body.strip():
        nb_cells.append(make_code_cell(code_lines))

# ---------------------------------------------------------------------------
# Assemble notebook JSON
# ---------------------------------------------------------------------------
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
        "accelerator": "GPU",
        "colab": {
            "name": "Motor_Fault_Prediction_M_Sila.ipynb",
            "provenance": [],
        },
    },
    "cells": nb_cells,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False), encoding="utf-8")
print("[OK] Notebook written to:", str(OUTPUT))
print("   Total cells:", len(nb_cells))
print("   File size  :", OUTPUT.stat().st_size // 1024, "KB")
print()
print("Upload to Colab: File > Upload notebook > select motor_fault_COMPLETE.ipynb")
