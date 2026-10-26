"""Superseded: the old version summed the O* step into the CO2 -> CO profile and read a
fixed path. It now delegates to scripts/visualization/plot_co2rr_pathway.py."""
from pathlib import Path
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.visualization.plot_co2rr_pathway import main

    main()
