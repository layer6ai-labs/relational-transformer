#!/usr/bin/env python
"""Build leakage-auditable rel-f1 dataset variants for temporal ablations.

The generated directories remain valid RelBench v3 datasets and deliberately
retain ``name: rel-f1`` so RT's standard evaluator can score them.  Put each
variant under a separate preprocessing root.

Variants:
  frozen   database tables are capped at the manifest test timestamp;
  rolling  full database, unchanged task tables;
  schedule full database plus task-row covariates computed only from races and
           circuits in ``(anchor, anchor + task timedelta]``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd
import yaml


TASKS = ("driver-dnf", "driver-position")


def _copy_common(source: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(f"destination already exists: {destination}")
    destination.mkdir(parents=True)
    manifest = yaml.safe_load((source / "manifest.yaml").read_text())
    shutil.copy2(source / "manifest.yaml", destination / "manifest.yaml")
    for optional in ("README.md", "schema.svg"):
        if (source / optional).exists():
            shutil.copy2(source / optional, destination / optional)
    shutil.copytree(source / "tasks", destination / "tasks")
    (destination / "db").mkdir()
    return manifest


def _copy_database(source: Path, destination: Path, manifest: dict, *, frozen: bool) -> None:
    cutoff = pd.Timestamp(manifest["test_timestamp"])
    for table_name, table_spec in manifest["tables"].items():
        src = source / "db" / f"{table_name}.parquet"
        dst = destination / "db" / src.name
        time_col = table_spec.get("time_col")
        if not frozen or not time_col:
            shutil.copy2(src, dst)
            continue
        frame = pd.read_parquet(src)
        frame = frame.loc[pd.to_datetime(frame[time_col]) <= cutoff].copy()
        frame.to_parquet(dst, index=False)


def _schedule_by_anchor(source: Path, anchors: pd.Series, horizon: pd.Timedelta) -> pd.DataFrame:
    races = pd.read_parquet(source / "db" / "races.parquet")
    circuits = pd.read_parquet(source / "db" / "circuits.parquet")
    races = races.merge(circuits, on="circuitId", how="left", suffixes=("", "_circuit"))
    rows: list[dict] = []
    for anchor in pd.to_datetime(anchors.drop_duplicates()).sort_values():
        upcoming = races.loc[
            (races["date"] > anchor) & (races["date"] <= anchor + horizon)
        ].sort_values(["date", "raceId"])
        row = {
            "date": anchor,
            f"schedule_n_races_{horizon.days}d": len(upcoming),
            f"schedule_n_circuits_{horizon.days}d": upcoming["circuitId"].nunique(),
            "schedule_days_to_next_race": (
                (upcoming.iloc[0]["date"] - anchor).days if len(upcoming) else pd.NA
            ),
        }
        for number, (_, race) in enumerate(upcoming.head(3).iterrows(), start=1):
            row[f"schedule_circuit_{number}"] = race["circuitId"]
        if len(upcoming):
            first = upcoming.iloc[0]
            row.update(
                schedule_next_round=first["round"],
                schedule_next_country=first["country"],
                schedule_next_lat=first["lat"],
                schedule_next_lng=first["lng"],
                schedule_next_alt=first["alt"],
            )
        rows.append(row)
    columns = [
        "date",
        f"schedule_n_races_{horizon.days}d",
        f"schedule_n_circuits_{horizon.days}d",
        "schedule_days_to_next_race",
        "schedule_circuit_1",
        "schedule_circuit_2",
        "schedule_circuit_3",
        "schedule_next_round",
        "schedule_next_country",
        "schedule_next_lat",
        "schedule_next_lng",
        "schedule_next_alt",
    ]
    # A stable column order is required because rustler copies train statistics
    # positionally onto val/test task tables. Dict insertion order alone is not
    # stable here: the first anchor of one split may have two future races while
    # another has three.
    return pd.DataFrame(rows).reindex(columns=columns)


def _add_schedule_columns(source: Path, destination: Path) -> None:
    horizons = {"driver-dnf": pd.Timedelta(days=30), "driver-position": pd.Timedelta(days=60)}
    for task_name, horizon in horizons.items():
        task_dir = destination / "tasks" / task_name
        for split in ("train", "val", "test"):
            path = task_dir / f"{split}.parquet"
            frame = pd.read_parquet(path)
            schedule = _schedule_by_anchor(source, frame["date"], horizon)
            frame = frame.merge(schedule, on="date", how="left", sort=False)
            frame.to_parquet(path, index=False)


def build_variant(source: Path, out_root: Path, variant: str) -> Path:
    destination = out_root / variant / "rel-f1"
    manifest = _copy_common(source, destination)
    _copy_database(source, destination, manifest, frozen=variant == "frozen")
    if variant == "schedule":
        _add_schedule_columns(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument(
        "--variants", nargs="+", choices=("frozen", "rolling", "schedule"),
        default=("frozen", "rolling", "schedule"),
    )
    args = parser.parse_args()
    for variant in args.variants:
        path = build_variant(args.source.resolve(), args.out_root.resolve(), variant)
        print(f"{variant}: {path}")


if __name__ == "__main__":
    main()
