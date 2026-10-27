"""
FILENAME:       src/edgebench/core/results.py
DESCRIPTION:    CSV-backed ledger of completed work. Owns the "already done" decision for every
                stage of the pipeline, so a sweep can be interrupted and resumed without
                repeating finished work or, more importantly, without treating a partial result
                as finished. Writes are atomic and keyed on (entity, recipe digest), so a
                recipe change invalidates prior rows rather than silently mixing two
                experimental conditions in one table.
AUTHOR:         Kananelo Chabeli
DATE-CREATED:   2026-10-27
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Outcome recorded against every entity. A row that is not "completed" is retried on the next
# run, which is why a failure must still be written: an absent row and a failed row are
# indistinguishable otherwise, and the error message would be lost.
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"


class ResultsError(RuntimeError):
    """Raised when a results file cannot be read, written, or is missing required columns."""


def _require_pandas():
    """Import pandas, with an actionable message when it is absent.

    Returns:
        The pandas module.

    Raises:
        ResultsError: If pandas is not installed.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ResultsError("pandas is required for results storage: pip install pandas") from exc
    return pd


def atomic_write_text(path: Path, text: str) -> None:
    """Write a file by renaming a completed temporary file into place.

    A direct write leaves a truncated file if the process dies mid-write, and a truncated
    results table is worse than no table: it reads successfully and under-reports completed
    work.

    Args:
        path: Destination path.
        text: Content to write.

    Returns:
        None.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


@dataclass(frozen=True)
class ResultsLayout:
    """Filesystem locations for one task's results and artefacts.

    Attributes:
        results_dir: Directory holding the summary and per-epoch tables.
        artifact_dir: Directory holding the artefacts the summary refers to.
        summary_name: Filename of the one-row-per-entity table.
        epochs_name: Filename of the one-row-per-epoch table.
    """

    results_dir: Path
    artifact_dir: Path
    summary_name: str = "summary.csv"
    epochs_name: str = "epochs.csv"

    @property
    def summary_path(self) -> Path:
        """Path of the summary table.

        Returns:
            The summary CSV path.
        """
        return self.results_dir / self.summary_name

    @property
    def epochs_path(self) -> Path:
        """Path of the per-epoch table.

        Returns:
            The epochs CSV path.
        """
        return self.results_dir / self.epochs_name


class ResultsStore:
    """Reads and writes the ledger for one task, and answers the skip question.

    Entities are identified by name, which is the model identifier throughout the pipeline.
    """

    def __init__(self, layout: ResultsLayout, key: str = "model") -> None:
        """Initialise the store.

        Args:
            layout: Locations to read and write.
            key: Column naming the entity each summary row describes.
        """
        self.layout = layout
        self.key = key

    # ---- reads --------------------------------------------------------------------------

    def _read(self, path: Path):
        """Read a CSV into a DataFrame, returning an empty frame when absent.

        Args:
            path: CSV path.

        Returns:
            The loaded DataFrame, empty if the file does not exist.

        Raises:
            ResultsError: If the file exists but cannot be parsed.
        """
        pd = _require_pandas()
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except Exception as exc:
            raise ResultsError(f"could not read {path}: {exc}") from exc

    def summary(self):
        """Load the one-row-per-entity table.

        Returns:
            The summary DataFrame, empty if no runs have been recorded.
        """
        return self._read(self.layout.summary_path)

    def epochs(self):
        """Load the one-row-per-epoch table.

        Returns:
            The epochs DataFrame, empty if nothing has been logged.
        """
        return self._read(self.layout.epochs_path)

    # ---- the skip decision --------------------------------------------------------------

    def artifact_path(self, name: str, suffix: str = ".pt") -> Path:
        """Resolve the artefact path for an entity.

        Args:
            name: Entity identifier.
            suffix: Artefact extension.

        Returns:
            Path to the artefact.
        """
        return self.layout.artifact_dir / f"{name}{suffix}"

    def is_done(self, name: str, digest: str, suffix: str = ".pt") -> bool:
        """Report whether an entity is finished under the current recipe.

        Three conditions must hold together. The artefact must exist, so deleting it forces a
        rerun instead of leaving a row pointing at nothing. A completed row must exist, so a
        half-written artefact from an interrupted run is not mistaken for a result. And the
        row's digest must match, so changing the recipe invalidates earlier work rather than
        mixing two conditions in one table.

        Args:
            name: Entity identifier.
            digest: Recipe digest the current run is using.
            suffix: Artefact extension.

        Returns:
            True when the entity needs no further work.
        """
        if not self.artifact_path(name, suffix).exists():
            return False
        df = self.summary()
        if df.empty or not {self.key, "status", "recipe_digest"}.issubset(df.columns):
            return False
        rows = df[(df[self.key] == name) & (df["status"] == STATUS_COMPLETED)]
        if rows.empty:
            return False
        # Compared as text: a digest of all decimal digits is read back as a number by pandas,
        # and the comparison would then silently fail.
        return str(rows.iloc[-1]["recipe_digest"]) == str(digest)

    def pending(self, names: Sequence[str], digest: str, *,
                force: Iterable[str] = (), suffix: str = ".pt") -> tuple[list[str], list[str]]:
        """Partition entities into those needing work and those already finished.

        Args:
            names: Entity identifiers under consideration.
            digest: Recipe digest the current run is using.
            force: Identifiers to rerun regardless of recorded state.
            suffix: Artefact extension.

        Returns:
            The pending identifiers and the finished identifiers, preserving input order.
        """
        forced = set(force)
        todo, done = [], []
        for n in names:
            (todo if n in forced or not self.is_done(n, digest, suffix) else done).append(n)
        return todo, done

    # ---- writes -------------------------------------------------------------------------

    def record(self, row: Mapping[str, Any]) -> None:
        """Insert or replace the summary row for one entity.

        Replacing rather than appending keeps the table one row per entity, so a rerun
        supersedes its previous result instead of leaving two rows that disagree.

        Args:
            row: Row contents. Must contain the key column.

        Returns:
            None.

        Raises:
            ResultsError: If the key column is absent.
        """
        pd = _require_pandas()
        if self.key not in row:
            raise ResultsError(f"row is missing the {self.key!r} column")
        df = self.summary()
        if not df.empty and self.key in df.columns:
            df = df[df[self.key] != row[self.key]]
        out = pd.concat([df, pd.DataFrame([dict(row)])], ignore_index=True)
        if self.key in out.columns:
            out = out.sort_values(self.key)
        atomic_write_text(self.layout.summary_path, out.to_csv(index=False))

    def log_epoch(self, row: Mapping[str, Any]) -> None:
        """Append one row to the per-epoch table.

        Args:
            row: Row contents, expected to carry the key column and an "epoch" column.

        Returns:
            None.
        """
        pd = _require_pandas()
        df = self.epochs()
        out = pd.concat([df, pd.DataFrame([dict(row)])], ignore_index=True)
        atomic_write_text(self.layout.epochs_path, out.to_csv(index=False))

    def truncate_epochs(self, name: str, after_epoch: int = 0) -> None:
        """Discard logged epochs beyond a resume point.

        A run that crashes between logging an epoch and writing its checkpoint would otherwise
        produce a duplicate epoch number when it resumes, corrupting the learning curve.

        Args:
            name: Entity identifier.
            after_epoch: Highest epoch number to keep.

        Returns:
            None.
        """
        df = self.epochs()
        if df.empty or not {self.key, "epoch"}.issubset(df.columns):
            return
        keep = ~((df[self.key] == name) & (df["epoch"] > after_epoch))
        atomic_write_text(self.layout.epochs_path, df[keep].to_csv(index=False))

    def clear(self, name: str) -> None:
        """Remove every recorded row for one entity.

        Args:
            name: Entity identifier.

        Returns:
            None.
        """
        df = self.summary()
        if not df.empty and self.key in df.columns:
            atomic_write_text(self.layout.summary_path,
                              df[df[self.key] != name].to_csv(index=False))
        self.truncate_epochs(name, after_epoch=-1)

    # ---- reporting ----------------------------------------------------------------------

    def failures(self) -> list[dict[str, Any]]:
        """List entities whose last recorded run did not complete.

        Returns:
            One mapping per failed entity, carrying the identifier and error text.
        """
        df = self.summary()
        if df.empty or "status" not in df.columns:
            return []
        bad = df[df["status"] != STATUS_COMPLETED]
        return [{self.key: r[self.key], "status": r["status"], "error": r.get("error", "")}
                for _, r in bad.iterrows()]

    def digests(self) -> list[str]:
        """List the distinct recipe digests present in the summary.

        More than one value means the table mixes experimental conditions, which invalidates a
        direct comparison between its rows. Worth asserting before reporting results.

        Returns:
            Sorted distinct digests.
        """
        df = self.summary()
        if df.empty or "recipe_digest" not in df.columns:
            return []
        return sorted({str(v) for v in df["recipe_digest"].dropna().unique()})

    def write_sidecar(self, path: Path, data: Mapping[str, Any]) -> Path:
        """Write a JSON sidecar beside an artefact.

        Args:
            path: Artefact path; the sidecar takes the same stem with a .json suffix.
            data: Metadata to record.

        Returns:
            The sidecar path.
        """
        sc = path.with_suffix(".json")
        atomic_write_text(sc, json.dumps(dict(data), indent=2, default=str))
        return sc