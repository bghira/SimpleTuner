"""What was decided for each sample of one unit of a dataset: its size and where it is cropped."""

import json
import os
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

_PAIRS = {
    "target_size": ("target_width", "target_height"),
    "intermediary_size": ("intermediary_width", "intermediary_height"),
    "crop_coordinates": ("crop_top", "crop_left"),
}
_COLUMNS = ("sample_idx", "aspect_ratio", *(column for pair in _PAIRS.values() for column in pair))


class UnitDecisions:
    """
    One SQLite file per unit, holding a row for each sample the unit kept and the epoch they were
    decided in. A file is written whole under another name and then renamed, so a reader sees all
    of a unit's rows or none.
    """

    def __init__(self, directory):
        self.directory = Path(directory)

    def path(self, unit: int) -> Path:
        return self.directory / str(unit // 1000) / f"{unit}.sqlite"

    def read(self, unit: int) -> tuple[int, dict[int, dict]] | None:
        """The epoch a unit was decided in and its decisions by sample index, or None if it has not been decided."""
        path = self.path(unit)
        if not path.exists():
            return None
        with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro&immutable=1", uri=True)) as connection:
            epoch = connection.execute("PRAGMA user_version").fetchone()[0]
            rows = connection.execute(f"SELECT {', '.join(_COLUMNS)} FROM decisions ORDER BY sample_idx").fetchall()
        decisions = {}
        for sample_idx, aspect_ratio, *values in rows:
            pairs = zip(values[0::2], values[1::2])
            decisions[sample_idx] = {
                "aspect_ratio": aspect_ratio,
                **{key: None if pair == (None, None) else pair for key, pair in zip(_PAIRS, pairs)},
            }
        return epoch, decisions

    def read_samples_per_unit(self) -> float | None:
        """How many samples a unit kept on average, as measured when the first units were decided."""
        path = self.directory / "samples_per_unit.json"
        return json.loads(path.read_text())["samples_per_unit"] if path.exists() else None

    def write_samples_per_unit(self, value: float) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / "samples_per_unit.json"
        handle, temporary = tempfile.mkstemp(dir=self.directory, prefix=f".{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w") as file:
                json.dump({"samples_per_unit": value}, file)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def write(self, unit: int, epoch: int, decisions: dict[int, dict]) -> None:
        """Replace what is recorded for a unit."""
        path = self.path(unit)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            (sample_idx, decision["aspect_ratio"], *(value for key in _PAIRS for value in decision[key] or (None, None)))
            for sample_idx, decision in decisions.items()
        ]
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
        os.close(handle)
        try:
            with closing(sqlite3.connect(temporary)) as connection:
                connection.execute(
                    f"CREATE TABLE decisions (sample_idx INTEGER PRIMARY KEY, aspect_ratio REAL, {', '.join(_COLUMNS[2:])})"
                )
                connection.executemany(f"INSERT INTO decisions VALUES ({', '.join('?' * len(_COLUMNS))})", rows)
                connection.execute(f"PRAGMA user_version = {int(epoch)}")
                connection.commit()
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
