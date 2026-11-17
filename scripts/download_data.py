# scripts/download_data.py
# Downloads the Kaggle dataset and copies it into data/raw/
import shutil
from pathlib import Path
import kagglehub

path = Path(kagglehub.dataset_download(
    "luthfimuthathohirin/billboard-hot-100-spotify-audio-features"))

try:
    root = Path(__file__).resolve().parent.parent
except NameError:
    root = Path.cwd()
raw_dir = root / "data" / "raw"
raw_dir.mkdir(parents=True, exist_ok=True)

for f in path.iterdir():
    shutil.copy2(f, raw_dir / f.name)

print("Dataset copied to:", raw_dir)