import os
import shutil
from pathlib import Path

import kagglehub

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Download latest version to kagglehub cache
dataset_path = Path(kagglehub.dataset_download("adarshsng/lending-club-loan-data-csv"))
print("Kagglehub cache path:", dataset_path)

# Copy ALL files from cache to data/raw (so your pipeline finds them)
for file in dataset_path.rglob("*"):
    if file.is_file():
        dest = RAW_DIR / file.name
        shutil.copy2(file, dest)
        print("Copied:", file.name, "->", dest)

print("\nDone. Files in data/raw:")
for f in RAW_DIR.iterdir():
    print("-", f.name)