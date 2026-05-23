"""Helper script to download and extract research artifacts from Zenodo."""

import urllib.request
import zipfile
from pathlib import Path

ZENODO_RECORD_ID = "20355140"
DATA_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/data.zip"
RESULTS_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}/files/results.zip"


def download_and_extract(url: str, output_zip: str, extract_to: Path) -> None:
    print(f"Downloading {output_zip} from Zenodo...")
    urllib.request.urlretrieve(url, output_zip)

    print(f"Extracting {output_zip}...")
    with zipfile.ZipFile(output_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    Path(output_zip).unlink()
    print(f"Finished processing {output_zip}\n")


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent

    (base_dir / "data").mkdir(exist_ok=True)
    (base_dir / "results").mkdir(exist_ok=True)

    print("=== Starting artifact download from Zenodo ===\n")

    download_and_extract(DATA_URL, "data.zip", base_dir)
    download_and_extract(RESULTS_URL, "results.zip", base_dir)

    print("=== Artifacts successfully downloaded and extracted! ===")
