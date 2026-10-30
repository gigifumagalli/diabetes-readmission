"""Download the UCI Diabetes 130-US Hospitals dataset into data/raw/.

Source: https://archive.ics.uci.edu/dataset/296 (CC BY 4.0).
The zip holds the encounters (diabetic_data.csv) and the lookup table for the
admission / discharge / source ID codes (IDS_mapping.csv), which ucimlrepo's
fetch_ucirepo(id=296) does not include.

Usage: uv run python scripts/download_data.py
"""

import io
import urllib.request
import zipfile
from pathlib import Path

URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
FILES = ("diabetic_data.csv", "IDS_mapping.csv")
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))

    # Extract by base name only: never trust paths stored inside an archive
    members = {Path(name).name: name for name in archive.namelist()}
    for filename in FILES:
        target = RAW_DIR / filename
        target.write_bytes(archive.read(members[filename]))
        print(f"{target.relative_to(RAW_DIR.parent.parent)}: {target.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
