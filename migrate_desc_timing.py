"""One-time migration of OIR timing metadata in PI_PP desc.yaml files.

Run a preview first:
    python migrate_desc_timing.py

Then perform the migration:
    python migrate_desc_timing.py --apply

For every desc.yaml below ROOT this script:
  * recalculates seconds_per_frame as movie_duration / (n_frames - 1);
  * adds a movie-local sync_coef only when the movie currently inherits the
    workflow-level sync_coef; and
  * renames the original file to desc_old.yaml before writing the replacement.

The new local coefficient preserves the old effective sampling interval:

    (duration / (n - 1)) * (1 + new_coef)
        == (duration / n) * (1 + old_effective_coef)

Existing movie-local coefficients are deliberately left untouched because they
are explicit per-movie calibration values.
"""

from __future__ import annotations

import argparse
import io
import os
from dataclasses import dataclass
from pathlib import Path

from ruamel.yaml import YAML


ROOT = Path(r"G:\LabWork\2-photon\PI_PP")
# Matches the new RunConfig default. It is used only if a desc.yaml has no
# workflow-level sync_coef at all.
DEFAULT_SYNC_COEF = -0.00274


@dataclass
class PlannedMigration:
    path: Path
    backup_path: Path
    output: str
    movies_updated: int
    inherited_syncs_localized: int
    existing_local_syncs_preserved: int


def corrected_sync_coef(old_sync_coef: float, n_frames: int) -> float:
    """Return a coefficient preserving the old effective sampling interval."""
    return (1.0 + old_sync_coef) * (n_frames - 1) / n_frames - 1.0


def make_yaml() -> YAML:
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.width = 4096  # Avoid wrapping existing compact trigger mappings.
    return yaml


def plan_file(path: Path) -> PlannedMigration:
    yaml = make_yaml()
    document = yaml.load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"Top-level YAML is not a mapping: {path}")

    global_sync_coef = float(document.get("sync_coef", DEFAULT_SYNC_COEF))
    movies = document.get("movies")
    if not isinstance(movies, list):
        raise ValueError(f"Missing or invalid movies list: {path}")

    localized = 0
    preserved = 0
    for index, movie in enumerate(movies):
        if not isinstance(movie, dict):
            raise ValueError(f"movies[{index}] is not a mapping: {path}")

        try:
            duration = float(movie["movie_duration"])
            n_frames = int(movie["n_frames"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"movies[{index}] needs numeric movie_duration and n_frames: {path}"
            ) from error

        if duration <= 0 or n_frames < 2:
            raise ValueError(
                f"movies[{index}] has invalid duration/frame count "
                f"({duration=}, {n_frames=}): {path}"
            )

        movie["seconds_per_frame"] = duration / (n_frames - 1)

        if "sync_coef" not in movie or movie["sync_coef"] is None:
            movie["sync_coef"] = corrected_sync_coef(global_sync_coef, n_frames)
            localized += 1
        else:
            preserved += 1

    rendered = io.StringIO()
    yaml.dump(document, rendered)
    return PlannedMigration(
        path=path,
        backup_path=path.with_name("desc_old.yaml"),
        output=rendered.getvalue(),
        movies_updated=len(movies),
        inherited_syncs_localized=localized,
        existing_local_syncs_preserved=preserved,
    )


def write_migration(plan: PlannedMigration) -> None:
    """Atomically stage the new YAML, then preserve the exact original as backup."""
    temporary_path = plan.path.with_name(plan.path.name + ".timing-migration.tmp")
    if temporary_path.exists():
        raise FileExistsError(f"Temporary migration file already exists: {temporary_path}")

    with temporary_path.open("w", encoding="utf-8", newline="") as stream:
        stream.write(plan.output)
        stream.flush()
        os.fsync(stream.fileno())

    try:
        # The original is retained byte-for-byte as desc_old.yaml.
        plan.path.replace(plan.backup_path)
        temporary_path.replace(plan.path)
    except Exception:
        # If replacing the new file fails, desc_old.yaml is the authoritative
        # original and can be renamed back manually. Do not delete either file.
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the migration; without this flag the script only previews it",
    )
    args = parser.parse_args()

    if not ROOT.is_dir():
        raise SystemExit(f"Root directory not found: {ROOT}")

    paths = sorted(ROOT.rglob("desc.yaml"))
    if not paths:
        raise SystemExit(f"No desc.yaml files found below: {ROOT}")

    # Fully validate and serialize every YAML before moving any original file.
    plans = [plan_file(path) for path in paths]
    occupied_backups = [plan.backup_path for plan in plans if plan.backup_path.exists()]
    if occupied_backups:
        names = "\n".join(f"  {path}" for path in occupied_backups)
        raise SystemExit(
            "Refusing to overwrite existing backup(s); no files were changed:\n" + names
        )

    movie_count = sum(plan.movies_updated for plan in plans)
    localized_count = sum(plan.inherited_syncs_localized for plan in plans)
    preserved_count = sum(plan.existing_local_syncs_preserved for plan in plans)
    verb = "Will migrate" if not args.apply else "Migrating"
    print(
        f"{verb} {len(plans)} desc.yaml file(s), {movie_count} movie(s): "
        f"{localized_count} inherited sync_coef value(s) will become movie-local; "
        f"{preserved_count} existing movie-local value(s) will be preserved."
    )

    for plan in plans:
        print(f"  {plan.path}")

    if not args.apply:
        print("Preview only. Re-run with --apply to rename originals to desc_old.yaml and write updates.")
        return 0

    for plan in plans:
        write_migration(plan)
    print("Migration complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
