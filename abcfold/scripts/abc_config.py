from pathlib import Path


def main():
    config_file = Path(__file__).parent.parent.joinpath("data", "config.ini")
    print(f"Config file path: {config_file}")


if __name__ == "__main__":
    main()
