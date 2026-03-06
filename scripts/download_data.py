from __future__ import annotations

from pathlib import Path

import pandas as pd

SOURCE_URLS = [
    "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/MatchIt/lalonde.csv",
    "https://vincentarelbundock.github.io/Rdatasets/csv/MatchIt/lalonde.csv",
    "https://github.com/vincentarelbundock/Rdatasets/raw/master/csv/MatchIt/lalonde.csv",
]


def main() -> None:
    target = Path("data/raw/lalonde.csv")
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        print(f"Found existing dataset: {target}")
        return

    errors: list[str] = []
    for url in SOURCE_URLS:
        try:
            frame = pd.read_csv(url)
            frame.to_csv(target, index=False)
            print(f"Downloaded LaLonde dataset to {target} from {url}")
            return
        except Exception as exc:  # broad by design: network conditions differ across environments
            errors.append(f"- {url}: {exc}")

    joined = "\n".join(errors)
    raise RuntimeError(f"Unable to download LaLonde dataset from all configured sources:\n{joined}")


if __name__ == "__main__":
    main()
