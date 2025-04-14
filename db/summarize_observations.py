import numpy as np
import pandas as pd
import psycopg2 as pg
from pathlib import Path
import argparse
from project_root import PROJECT_ROOT
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", "-d", type=str)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with (PROJECT_ROOT / "data/config.json").open() as f:
        config = json.load(f)

    conn = pg.connect("dbname=zoo_vision")
    df = pd.read_sql_query("select * from tracks", conn)
    print(df)


if __name__ == "__main__":
    main()
