import pandas as pd
from pathlib import Path
from src.config import DDR_SEG_DIR, APTOS_DIR

print("=== APTOS DATA ===")
aptos = Path(APTOS_DIR)
print("Files:", [p.name for p in aptos.iterdir()])

for csv_file in sorted(aptos.glob("*.csv")):
    df = pd.read_csv(csv_file)
    print(f"\n--- {csv_file.name} ---")
    print("Columns:", list(df.columns))
    for label_col in ["diagnosis", "level"]:
        if label_col in df.columns:
            print(f"Label distribution ({label_col}):")
            print(df[label_col].value_counts().sort_index())
            break

print("\n=== DDR SEGMENTATION DATA ===")
ddr = Path(DDR_SEG_DIR)
for split in ["train", "val", "test"]:
    img_dir = ddr / "images" / split
    if img_dir.exists():
        print(f"{split}: {len(list(img_dir.glob('*')))} images")
    # check tet folder
    tet_dir = ddr / "images" / "tet"
    if split == "test" and not img_dir.exists() and tet_dir.exists():
        print(f"test (tet): {len(list(tet_dir.glob('*')))} images")

# Check annotation folders
print("\nAnnotation folders:")
for p in sorted((ddr / "annotations").iterdir()):
    if p.is_dir():
        files = list(p.glob("*"))
        print(f"  {p.name}/: {len(files)} files")
