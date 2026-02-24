# scripts/06_export_tables_to_latex.py
import sys
from pathlib import Path
from typing import List

import pandas as pd


# Ensure project root is on PYTHONPATH when running scripts directly.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

# We split wide tables into visual chunks of:
# 1 index column + 4 data columns = 5 columns total per part.
# Why: keeps tables readable in PDF without dropping information.
MAX_DATA_COLS_PER_PART = 4


def _sanitize_latex_text(s: str) -> str:
    # We keep this minimal because pandas.to_latex already escapes cell content.
    # This function is only for labels shown in wrapper text (e.g., "Part 1/3").
    return str(s).replace("_", r"\_")


def _format_dataframe_for_latex(df: pd.DataFrame) -> pd.DataFrame:
    df_fmt = df.copy()

    # Format floats for report readability.
    # Why: default pandas float formatting is too noisy for PDF tables.
    for col in df_fmt.columns:
        if pd.api.types.is_float_dtype(df_fmt[col]):
            # Use 6 decimals for metrics-like tables; still compact enough.
            df_fmt[col] = df_fmt[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "")

    return df_fmt


def _split_dataframe_by_columns(df: pd.DataFrame, max_data_cols: int) -> List[pd.DataFrame]:
    # Keep row index in every part by splitting only data columns.
    cols = list(df.columns)
    if len(cols) <= max_data_cols:
        return [df]

    parts = []
    for i in range(0, len(cols), max_data_cols):
        part_cols = cols[i : i + max_data_cols]
        parts.append(df[part_cols].copy())
    return parts


def _latex_column_format(n_data_cols: int) -> str:
    # First column is the row index (left aligned); data columns centered.
    # Why: row labels are text-like and easier to scan left-aligned.
    return "l" + ("c" * n_data_cols)


def _df_to_tabular_latex(df_part: pd.DataFrame) -> str:
    n_data_cols = len(df_part.columns)
    colfmt = _latex_column_format(n_data_cols)

    latex = df_part.to_latex(
        index=True,
        escape=True,
        na_rep="",
        column_format=colfmt,
        longtable=False,  # We control page layout in wrapper, not here.
        multicolumn=True,
        multicolumn_format="c",
        bold_rows=False,
    )

    # pandas may emit "\toprule" etc. (good), but we do NOT want a full table env here.
    # to_latex returns only tabular by default, which is exactly what we need.
    return latex


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def export_csv_to_latex_parts(csv_path: Path) -> List[Path]:
    df = pd.read_csv(csv_path, index_col=0)

    # Report tables are display artifacts, so we format values after reading.
    df_fmt = _format_dataframe_for_latex(df)
    parts = _split_dataframe_by_columns(df_fmt, MAX_DATA_COLS_PER_PART)

    part_paths: List[Path] = []
    stem = csv_path.stem

    for idx, part_df in enumerate(parts, start=1):
        part_tex = _df_to_tabular_latex(part_df)
        part_path = TABLES_DIR / f"{stem}_part{idx}.tex"
        _write_text(part_path, part_tex)
        part_paths.append(part_path)

    return part_paths


def build_wrapper_table_tex(csv_path: Path, part_paths: List[Path]) -> Path:
    """
    Creates a wrapper file named *_table.tex with:
    - one logical table block (no caption here; caption lives in main report .tex)
    - parts stacked vertically
    - consistent width container for every part to align the index column visually
    """
    stem = csv_path.stem
    wrapper_path = TABLES_DIR / f"{stem}_table.tex"

    lines = []
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{4pt}")      # Slightly tighter columns for fit.
    lines.append(r"\renewcommand{\arraystretch}{1.08}")  # Mild row height for readability.

    n_parts = len(part_paths)

    for i, p in enumerate(part_paths, start=1):
        if n_parts > 1:
            # Part label kept small and consistent across parts.
            # Why: one logical table, but readers need orientation.
            lines.append(rf"{{\footnotesize\textit{{Part {i}/{n_parts}}}}}\par")
            lines.append(r"\vspace{0.15em}")

        # Fixed-width box + left-aligned content inside.
        # Why: every part starts at the same x-position, so the row-index column aligns.
        lines.append(r"\noindent\makebox[\textwidth][c]{%")
        lines.append(r"  \begin{minipage}{0.98\textwidth}")
        lines.append(r"  \raggedright")
        lines.append(rf"  \input{{tables/{p.name}}}")
        lines.append(r"  \end{minipage}%")
        lines.append(r"}")

        if i < n_parts:
            lines.append(r"\vspace{0.45em}")

    _write_text(wrapper_path, "\n".join(lines) + "\n")
    return wrapper_path


def main() -> None:
    if not TABLES_DIR.exists():
        raise FileNotFoundError(f"Tables directory not found: {TABLES_DIR}")

    csv_files = sorted(TABLES_DIR.glob("*.csv"))

    if not csv_files:
        print(f"No CSV tables found in: {TABLES_DIR}")
        return

    generated = []

    for csv_path in csv_files:
        # Skip previously generated helper CSVs if any naming convention changes later.
        # Why not over-filter now: we want the script to be generic for future report tables.
        part_paths = export_csv_to_latex_parts(csv_path)
        wrapper_path = build_wrapper_table_tex(csv_path, part_paths)
        generated.append((csv_path.name, len(part_paths), wrapper_path.name))

    print("Exported LaTeX table artifacts:")
    for csv_name, n_parts, wrapper_name in generated:
        print(f" - {csv_name} -> {n_parts} part(s), wrapper: {wrapper_name}")

    print("\nDone.")
    print("Use the wrappers in the report, e.g.: \\input{tables/<name>_table.tex}")


if __name__ == "__main__":
    main()
