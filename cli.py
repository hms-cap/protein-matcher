"""CLI launcher for the installable LockKey-Explorer web application."""

from __future__ import annotations

from pathlib import Path
import sys
from streamlit.web import cli as stcli


def main() -> None:
    app_path = Path(__file__).resolve().parent / "app.py"
    sys.argv = ["streamlit", "run", str(app_path)] + sys.argv[1:]
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
