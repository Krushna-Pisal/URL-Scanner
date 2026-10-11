"""Convert the supplied source CSV to the scanner's raw URL/label format."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "data" / "download" / "Dataset.csv"
OUTPUT_PATH = ROOT / "data" / "raw" / "urls.csv"


def prepare_dataset() -> pd.DataFrame:
    """Keep only cleaned URL text and binary labels, ignoring source features."""
    if not SOURCE_PATH.is_file():
        raise FileNotFoundError(f"Dataset CSV not found: {SOURCE_PATH}")

    data = pd.read_csv(SOURCE_PATH)
    original_columns = list(data.columns)
    data.columns = [str(column).strip().lower() for column in data.columns]
    print("Columns found:", data.columns.tolist())

    required = {"url", "label"}
    if not required.issubset(data.columns):
        raise ValueError(
            "Dataset must contain 'url' and 'label' columns after lower-casing and trimming. "
            f"Columns found: {data.columns.tolist()} (source columns: {original_columns})"
        )

    # Selecting these two columns discards all pre-computed dataset features.
    cleaned = data.loc[:, ["url", "label"]].dropna(subset=["url", "label"]).copy()
    cleaned["url"] = cleaned["url"].astype(str).str.strip()
    cleaned = cleaned.loc[cleaned["url"] != ""].drop_duplicates(subset="url").copy()
    numeric_labels = pd.to_numeric(cleaned["label"], errors="raise")
    if not numeric_labels.isin([0, 1]).all():
        values = sorted(numeric_labels.unique().tolist())
        raise ValueError(f"Labels must contain only 0 and 1; found: {values}")
    cleaned["label"] = numeric_labels.astype(int)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(OUTPUT_PATH, index=False)
    counts = cleaned["label"].value_counts().sort_index()
    print(f"Total rows written: {len(cleaned)}")
    print("Rows per label:")
    for label in (0, 1):
        print(f"  {label}: {int(counts.get(label, 0))}")
    print("Random sample URLs:")
    sample = cleaned.sample(n=min(8, len(cleaned)), random_state=42)
    for row in sample.itertuples(index=False):
        print(f"  label={row.label} url={row.url}")
    print(f"Output: {OUTPUT_PATH}")
    return cleaned


if __name__ == "__main__":
    prepare_dataset()
