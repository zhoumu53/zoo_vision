from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    app_path = root / "streamlit_labeller.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("Streamlit run interrupted, exiting cleanly.")


if __name__ == "__main__":
    main()
