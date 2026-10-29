"""
FILENAME:       src/edgexbench/core/paths.py

DESCRIPTION:    Canonical filesystem layout for the edgexbench framework. Every module resolves
                paths through this file so the directory structure is defined exactly once.
                Code root and data root are separable, which allows the repository to live on a
                home filesystem while datasets and artefacts sit on cluster scratch storage.

AUTHOR:         Kananelo Chabeli

DATE-CREATED:   2026-10-27

"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

# [AUDIT H2] A directory is the repository root only if its pyproject.toml declares this
# project. The generic markers (".git", "configs") matched unrelated directories, such as a
# dotfiles repository in $HOME above a site-packages install.
# [AUDIT H1] Also fixes the "repos11itory" typo.
# Before:
#   # Presence of any of these at a directory marks it as the repos11itory root. pyproject.toml is
#   # authoritative; the others cover a partially initialised checkout.
#   ROOT_MARKERS: Final[tuple[str, ...]] = ("pyproject.toml", ".git", "configs")
ROOT_MARKERS: Final[tuple[str, ...]] = ("pyproject.toml",)

# Project name a root's pyproject.toml must declare.
PROJECT_NAME: Final[str] = "edgexbench"
_PROJECT_NAME_LINE: Final[re.Pattern[str]] = re.compile(
    r'^\s*name\s*=\s*["\']' + re.escape(PROJECT_NAME) + r'["\']\s*$', re.MULTILINE)

# Environment overrides. ROOT relocates everything; 
# DATA_ROOT relocates only the generated
# trees, which is the common case on a cluster where code is on $HOME and data on scratch.
ENV_ROOT: Final[str] = "EDGEXBENCH_ROOT"
ENV_DATA_ROOT: Final[str] = "EDGEXBENCH_DATA_ROOT"

# Checkpoint representations. Each is a stage in the conversion chain, not an arbitrary label.
CHECKPOINT_STAGES: Final[tuple[str, ...]] = ("pytorch", "onnx")

# Result families written under the results tree.
RESULT_KINDS: Final[tuple[str, ...]] = ("training", "benchmark", "compile", "analysis")

# Path components are restricted to this character set. Anything else is rejected rather than
# silently creating a stray directory, and it blocks traversal via "..".
_SAFE_COMPONENT: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class PathError(RuntimeError):
    """Raised when the project root cannot be located or a path component is unsafe."""


def _validate_component(value: str, kind: str) -> str:
    """Reject path components that are empty, contain separators, or attempt traversal.

    Validation is on SHAPE only, not membership: this module is task-agnostic and must not
    carry a list of valid task names. Membership belongs to the task registry.

    Args:
        value: The candidate component, such as a task or backend name.
        kind: Human-readable role of the component, used in the error message.

    Returns:
        The validated component, unchanged.

    Raises:
        PathError: If the component is empty, malformed, or could escape its parent directory.
    """
    if not isinstance(value, str) or not _SAFE_COMPONENT.match(value):
        raise PathError(
            f"invalid {kind} name {value!r}: expected a single path component matching "
            f"[A-Za-z0-9][A-Za-z0-9._-]*"
        )
    return value


def _validate_choice(value: str, allowed: Iterable[str], kind: str) -> str:
    """Check a component against a closed set of permitted values.

    Used where the set genuinely is fixed, such as checkpoint stages, so that a typo fails
    immediately instead of producing an empty directory that later reads as "nothing to do".

    Args:
        value: The candidate value.
        allowed: The permitted values.
        kind: Human-readable role of the component, used in the error message.

    Returns:
        The validated value, unchanged.

    Raises:
        PathError: If the value is not in the permitted set.
    """
    allowed = tuple(allowed)
    if value not in allowed:
        raise PathError(f"unknown {kind} {value!r}: expected one of {allowed}")
    return value


def _is_project_root(candidate: Path) -> bool:
    """Report whether a directory is this project's repository root.

    Args:
        candidate: Directory to test.

    Returns:
        True if it holds a pyproject.toml declaring name = "edgexbench".
    """
    marker = candidate / ROOT_MARKERS[0]
    if not marker.is_file():
        return False
    try:
        return bool(_PROJECT_NAME_LINE.search(marker.read_text(errors="replace")))
    except OSError:
        return False


def find_project_root(start: Path | None = None) -> Path:
    """Locate the repository root by searching upward for this project's pyproject.toml.

    Resolution order is EDGEBENCH_ROOT, then an upward search from `start`, then an upward
    search from the current working directory. Marker search is used rather than arithmetic on
    __file__ because the package may be installed in editable or site-packages form, where the
    module's location says nothing about where the data trees are; the working-directory pass
    covers a site-packages install run from inside the checkout.

    Args:
        start: Directory to begin the upward search from. Defaults to this file's directory.

    Returns:
        Absolute path to the repository root.

    Raises:
        PathError: If EDGEBENCH_ROOT is set but does not exist, or no root is found.
    """
    env = os.environ.get(ENV_ROOT)
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise PathError(f"{ENV_ROOT} points at {root}, which is not a directory")
        return root

    # [AUDIT H2] Any of the generic markers used to qualify, and only the module's own location
    # was searched.
    # Before:
    #   here = (start or Path(__file__).parent).expanduser().resolve()
    #   for candidate in (here, *here.parents):
    #       if any((candidate / m).exists() for m in ROOT_MARKERS):
    #           return candidate
    searched = []
    for origin in (start or Path(__file__).parent, Path.cwd()):
        here = origin.expanduser().resolve()
        searched.append(str(here))
        for candidate in (here, *here.parents):
            if _is_project_root(candidate):
                return candidate

    raise PathError(
        f"could not locate the project root above {searched}: no pyproject.toml declaring "
        f"name = {PROJECT_NAME!r}. Set {ENV_ROOT} to the repository root."
    )


def ensure_dir(path: Path) -> Path:
    """Create a directory and its parents if absent.

    Directory creation is explicit rather than an import-time side effect, so importing this
    module never mutates the filesystem.

    Args:
        path: Directory to create.

    Returns:
        The same path, now guaranteed to exist.

    Raises:
        PathError: If the path exists as a non-directory.
    """
    if path.exists() and not path.is_dir():
        raise PathError(f"{path} exists but is not a directory")
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass(frozen=True)
class Paths:
    """
    Resolved filesystem layout for one edgexbench project.

    Frozen and explicitly constructed so tests can point an instance at a temporary directory.
    Accessors are pure path arithmetic and never touch the filesystem; call ensure_dir on the
    result when a directory is about to be written to.

    Attributes:
        root: Repository root. Holds code, configs and tests.
        data_root: Root of all generated trees. Equals `root` unless EDGEBENCH_DATA_ROOT is set.
    """

    root: Path
    data_root: Path

    # ---- construction -------------------------------------------------------------------

    @classmethod
    def from_env(cls, start: Path | None = None) -> "Paths":
        """Build a Paths instance from the environment and the detected repository root.

        Args:
            start: Directory to begin the root search from. Defaults to this file's directory.

        Returns:
            A resolved Paths instance.

        Raises:
            PathError: If the root cannot be located, or EDGEBENCH_DATA_ROOT is not a directory.
        """
        root = find_project_root(start)
        env_data = os.environ.get(ENV_DATA_ROOT)
        if env_data:
            data_root = Path(env_data).expanduser().resolve()
            if not data_root.is_dir():
                raise PathError(f"{ENV_DATA_ROOT} points at {data_root}, which is not a directory")
        else:
            data_root = root
        return cls(root=root, data_root=data_root)

    @classmethod
    def for_testing(cls, tmp: Path) -> "Paths":
        """Build a Paths instance rooted at a temporary directory.

        Args:
            tmp: Directory to use as both code and data root.

        Returns:
            A Paths instance confined to `tmp`.
        """
        tmp = tmp.expanduser().resolve()
        return cls(root=tmp, data_root=tmp)

    # ---- static trees -------------------------------------------------------------------

    @property
    def configs(self) -> Path:
        """Directory holding YAML recipe and hardware configurations.

        Returns:
            Path to the configs tree. Lives under the code root: configs are versioned
            alongside the code that interprets them.
        """
        return self.root / "configs"

    @property
    def datasets(self) -> Path:
        """Directory holding raw datasets and generated uint8 caches.

        Returns:
            Path to the datasets tree. Excluded from version control.
        """
        return self.data_root / "datasets"

    @property
    def calibration(self) -> Path:
        """Directory holding frozen INT8 calibration sets.

        Returns:
            Path to the calibration tree. Tracked in version control despite being generated:
            the calibration set must be byte-identical across every backend, and committing it
            makes that claim verifiable and removes the dataset dependency from machines that
            only compile.
        """
        return self.data_root / "calib_datasets"

    @property
    def checkpoints(self) -> Path:
        """Root of the model checkpoint tree.

        Returns:
            Path to the checkpoints tree, containing one sub-tree per representation stage.
        """
        return self.data_root / "checkpoint"

    @property
    def binaries(self) -> Path:
        """Root of the compiled hardware artefact tree.

        Returns:
            Path to the binaries tree, containing one sub-tree per hardware backend.
        """
        return self.data_root / "binaries"

    @property
    def results(self) -> Path:
        """Root of the results tree holding CSV and JSON outputs.

        Returns:
            Path to the results tree.
        """
        return self.data_root / "results"

    # ---- parameterised accessors --------------------------------------------------------

    def dataset_dir(self, name: str) -> Path:
        """Directory for one named dataset.

        Args:
            name: Dataset identifier, for example "imageNet-100".

        Returns:
            Path to that dataset's directory.

        Raises:
            PathError: If `name` is not a safe single path component.
        """
        return self.datasets / _validate_component(name, "dataset")

    def cache_dir(self, dataset: str) -> Path:
        """Directory for one dataset's generated uint8 caches.

        Kept as a sibling of the raw dataset rather than inside it, so the raw tree can be
        mounted read-only and the caches still written.

        Args:
            dataset: Dataset identifier the caches were built from.

        Returns:
            Path to that dataset's cache directory.

        Raises:
            PathError: If `dataset` is not a safe single path component.
        """
        return self.datasets / f"{_validate_component(dataset, 'dataset')}_cache"

    def calibration_dir(self, task: str) -> Path:
        """Directory holding the frozen calibration set for one task.

        Args:
            task: Task identifier, for example "classification".

        Returns:
            Path to that task's calibration directory.

        Raises:
            PathError: If `task` is not a safe single path component.
        """
        return self.calibration / _validate_component(task, "task")

    def checkpoint_dir(self, stage: str, task: str) -> Path:
        """Directory holding model checkpoints for one representation stage and task.

        Args:
            stage: Representation stage, one of CHECKPOINT_STAGES.
            task: Task identifier.

        Returns:
            Path to that stage and task's checkpoint directory.

        Raises:
            PathError: If `stage` is unknown or `task` is not a safe single path component.
        """
        return (self.checkpoints
                / _validate_choice(stage, CHECKPOINT_STAGES, "checkpoint stage")
                / _validate_component(task, "task"))

    def binary_dir(self, backend: str, task: str) -> Path:
        """Directory holding compiled artefacts for one hardware backend and task.

        The backend set is open rather than validated against a fixed list, so adding a target
        requires no change to this module.

        Args:
            backend: Hardware backend identifier, for example "hailo-8l".
            task: Task identifier.

        Returns:
            Path to that backend and task's binary directory.

        Raises:
            PathError: If either component is not a safe single path component.
        """
        return (self.binaries
                / _validate_component(backend, "backend")
                / _validate_component(task, "task"))

    def results_dir(self, kind: str, task: str) -> Path:
        """Directory holding one family of results for one task.

        Args:
            kind: Result family, one of RESULT_KINDS.
            task: Task identifier.

        Returns:
            Path to that result family and task's directory.

        Raises:
            PathError: If `kind` is unknown or `task` is not a safe single path component.
        """
        return (self.results
                / _validate_choice(kind, RESULT_KINDS, "result kind")
                / _validate_component(task, "task"))

    # ---- helpers ------------------------------------------------------------------------

    def relative(self, path: Path) -> Path:
        """Express a path relative to the data root where possible.

        Used for provenance fields written into sidecar metadata, so recorded paths stay
        portable between machines with different absolute layouts.

        Args:
            path: Path to relativise.

        Returns:
            The path relative to the data root, or unchanged if it lies outside it.
        """
        path = Path(path)
        try:
            return path.resolve().relative_to(self.data_root)
        except ValueError:
            return path

    def describe(self) -> dict[str, str]:
        """Summarise the resolved layout.

        Returns:
            Mapping of tree name to absolute path, suitable for logging at the start of a run
            or embedding in result metadata.
        """
        return {
            "root": str(self.root),
            "data_root": str(self.data_root),
            "configs": str(self.configs),
            "datasets": str(self.datasets),
            "calibration": str(self.calibration),
            "checkpoints": str(self.checkpoints),
            "binaries": str(self.binaries),
            "results": str(self.results),
        }


# [AUDIT H1] The comment named a module __getattr__ that does not exist.
# Before:
#   # Module-level instance for ordinary use. Construction is lazy via __getattr__ so that importing
#   # this module outside a project tree does not raise; ...
# Module-level instance for ordinary use. Construction is lazy, deferred to the first
# get_paths() call, so that importing this module outside a project tree does not raise; tests
# and tools that need an explicit layout should build their own Paths instance instead.
_PATHS: Paths | None = None


def get_paths() -> Paths:
    """Return the process-wide Paths instance, resolving it on first use.

    Returns:
        The shared Paths instance.

    Raises:
        PathError: If the project root cannot be located.
    """
    global _PATHS
    if _PATHS is None:
        _PATHS = Paths.from_env()
    return _PATHS

def set_paths(paths: Paths) -> None:
    """Override the process-wide Paths instance.

    Intended for tests and for tools that operate on a project other than the one containing
    this file.

    Args:
        paths: The instance to install.

    Returns:
        None.
    """
    global _PATHS
    _PATHS = paths