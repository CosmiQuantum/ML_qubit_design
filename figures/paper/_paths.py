from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SOURCE_MATERIALS_DIR = BASE_DIR / "source_materials"
OUTPUTS_DIR = BASE_DIR / "outputs"

OUTPUTS_DIR.mkdir(exist_ok=True)
