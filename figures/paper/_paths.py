from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
SOURCE_MATERIALS_DIR = BASE_DIR / "source_materials"
MANUSCRIPT_EXPORTS_DIR = BASE_DIR / "manuscript_exports"

MANUSCRIPT_EXPORTS_DIR.mkdir(exist_ok=True)
