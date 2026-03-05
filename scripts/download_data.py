from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_URL = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MatchIt/lalonde.csv"


def main() -> None:
    target = Path("data/raw/lalonde.csv")
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print(f"Found existing dataset: {target}")
        return

    frame = pd.read_csv(SOURCE_URL)
    frame.to_csv(target, index=False)
    print(f"Downloaded LaLonde dataset to {target}")


if __name__ == "__main__":
    main()
