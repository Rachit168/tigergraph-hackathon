"""Launch the judge investigation console.

Usage:
    python -m scripts.serve_ui
    python -m scripts.serve_ui --port 8765
"""

from __future__ import annotations

from ui.server import main

if __name__ == "__main__":
    raise SystemExit(main())
