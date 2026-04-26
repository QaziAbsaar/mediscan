"""
MediScan AI — Dataset Downloader
=================================
Downloads the disease-symptom dataset required by train.py.

Source: itachi9604/disease-symptom-description-dataset (Kaggle)
Mirror: GitHub raw content (no Kaggle API key required)

Usage:
    python download_data.py
"""

import os
import sys
import urllib.request

# ─────────────────────────────────────────────
# Dataset files and their public GitHub mirrors
# ─────────────────────────────────────────────
BASE_URL = (
    "https://raw.githubusercontent.com/"
    "itachi9604/healthcare-chatbot/master/dataset/"
)

FILES = {
    "dataset.csv": BASE_URL + "dataset.csv",
    "symptom_Description.csv": BASE_URL + "symptom_Description.csv",
    "symptom_precaution.csv": BASE_URL + "symptom_precaution.csv",
    "Symptom-severity.csv": BASE_URL + "Symptom-severity.csv",
}

DATA_DIR = "data"


def reporthook(block_num, block_size, total_size):
    """Simple download progress indicator."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        bar = "█" * (percent // 5) + "░" * (20 - percent // 5)
        sys.stdout.write(f"\r    [{bar}] {percent:3d}%")
        sys.stdout.flush()
        if downloaded >= total_size:
            print()  # newline when done
    else:
        sys.stdout.write(f"\r    Downloaded {downloaded:,} bytes")
        sys.stdout.flush()


def download_file(filename: str, url: str) -> bool:
    """Download a single file. Returns True on success."""
    dest = os.path.join(DATA_DIR, filename)

    if os.path.exists(dest):
        print(f"  [OK] Already exists - skipping: {filename}")
        return True

    print(f"  >> Downloading: {filename}")
    try:
        urllib.request.urlretrieve(url, dest, reporthook)
        size_kb = os.path.getsize(dest) / 1024
        print(f"    Saved to {dest}  ({size_kb:.1f} KB)")
        return True
    except Exception as exc:
        print(f"\n  [FAIL] Failed to download {filename}: {exc}")
        # Clean up partial file
        if os.path.exists(dest):
            os.remove(dest)
        return False


def main():
    print("=" * 60)
    print("  MediScan AI — Dataset Downloader")
    print("=" * 60)

    # Create data/ directory
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"\n  Target directory: {os.path.abspath(DATA_DIR)}\n")

    success_count = 0
    for filename, url in FILES.items():
        if download_file(filename, url):
            success_count += 1
        print()

    print("=" * 60)
    if success_count == len(FILES):
        print("  [SUCCESS] All files downloaded successfully!")
        print("\n  You can now run the training pipeline:")
        print("      python train.py")
    else:
        failed = len(FILES) - success_count
        print(f"  [WARNING] {failed} file(s) failed to download.")
        print("  Check your internet connection and try again.")
    print("=" * 60)


if __name__ == "__main__":
    main()
