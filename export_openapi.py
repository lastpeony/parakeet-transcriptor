"""
Write openapi.yaml for parakeet_service.main:app
Usage:  python export_openapi.py
"""

from importlib import import_module
from pathlib import Path
import sys
import yaml

APP_PATH = "parakeet_service.main"
APP_ATTR = "app"

def main() -> None:
    try:
        mod = import_module(APP_PATH)
        app = getattr(mod, APP_ATTR)
    except (ModuleNotFoundError, AttributeError) as exc:
        sys.exit(f"[export_openapi] Cannot import {APP_PATH}.{APP_ATTR}: {exc}")

    spec = app.openapi()
    out_file = Path("openapi.yaml")
    out_file.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    print(f"[export_openapi] wrote {out_file.resolve()}")

if __name__ == "__main__":
    main()
