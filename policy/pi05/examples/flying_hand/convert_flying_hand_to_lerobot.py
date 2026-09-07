"""Pi05 uses the same LeRobot Flying-Hand converter as Pi0.

Run this file from the Pi05 repository with the same arguments documented in
``policy/pi0/examples/flying_hand/convert_flying_hand_to_lerobot.py``.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).parents[3] / "pi0" / "examples" / "flying_hand" / "convert_flying_hand_to_lerobot.py"),
        run_name="__main__",
    )
