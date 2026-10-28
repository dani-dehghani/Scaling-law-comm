from __future__ import annotations

from src.reproducibility import environment_metadata


def main() -> None:
    info = environment_metadata()
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
