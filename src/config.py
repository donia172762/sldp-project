"""Configuration loading for the spoken-language-processing project.

Everything tunable lives in ``config.yaml`` at the project root. This module
turns that YAML file into a small object with attribute access, resolves every
path relative to the project root, and provides :func:`set_seed` so that a run
is reproducible from a single number.

Typical use::

    from src.config import load_config, set_seed

    cfg = load_config()
    set_seed(cfg.seed)
    print(cfg.audio.sample_rate)      # 16000
    print(cfg.path("manifests"))      # C:\\...\\sldp-project\\manifests
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

# The project root is the parent of the ``src`` package, i.e. the directory
# holding config.yaml. Resolving it from __file__ means scripts work no matter
# which directory they are launched from.
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH: Path = PROJECT_ROOT / "config.yaml"


class ConfigNode:
    """A read-friendly wrapper around a nested dict.

    Supports both ``cfg.audio.sample_rate`` and ``cfg["audio"]["sample_rate"]``.
    Nested dictionaries are wrapped lazily on construction; lists of dicts are
    wrapped element-wise. The node is intentionally *not* immutable - tests and
    ablation scripts legitimately want to poke a value before running - but the
    on-disk YAML remains the single source of truth for a real run.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        if not isinstance(data, dict):
            raise TypeError(f"ConfigNode expects a dict, got {type(data).__name__}")
        self._data: dict[str, Any] = {
            key: self._wrap(value) for key, value in data.items()
        }

    @staticmethod
    def _wrap(value: Any) -> Any:
        """Recursively wrap dicts so nested attribute access works."""
        if isinstance(value, dict):
            return ConfigNode(value)
        if isinstance(value, list):
            return [ConfigNode(v) if isinstance(v, dict) else v for v in value]
        return value

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup fails, so `self._data`
        # (set in __init__) never routes through here.
        try:
            return self._data[name]
        except KeyError as exc:
            raise AttributeError(
                f"No config key '{name}'. Available: {sorted(self._data)}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_data":
            super().__setattr__(name, value)
        else:
            self._data[name] = self._wrap(value)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-style ``get`` with a default, for optional keys."""
        return self._data.get(key, default)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain nested dict (useful for logging a run's settings)."""
        out: dict[str, Any] = {}
        for key, value in self._data.items():
            if isinstance(value, ConfigNode):
                out[key] = value.to_dict()
            elif isinstance(value, list):
                out[key] = [
                    v.to_dict() if isinstance(v, ConfigNode) else v for v in value
                ]
            else:
                out[key] = value
        return out

    def __repr__(self) -> str:
        return f"ConfigNode({sorted(self._data)})"


class Config(ConfigNode):
    """Top-level config: a :class:`ConfigNode` that also knows the project root."""

    def __init__(self, data: dict[str, Any], project_root: Path) -> None:
        super().__init__(data)
        # Bypass ConfigNode.__setattr__, which would push this into _data.
        object.__setattr__(self, "project_root", project_root)

    def path(self, key: str, *parts: str) -> Path:
        """Resolve ``paths.<key>`` against the project root.

        Extra ``parts`` are appended, so ``cfg.path("figures", "durations.png")``
        gives ``<root>/reports/figures/durations.png``. Absolute values in the
        YAML are honoured as-is, which lets a team member point ``data_root`` at
        an external drive holding the 60 GB corpora.
        """
        raw = getattr(self.paths, key)
        candidate = Path(raw).expanduser()
        resolved = candidate if candidate.is_absolute() else self.project_root / candidate
        return resolved.joinpath(*parts)

    def ensure_dir(self, key: str, *parts: str) -> Path:
        """Like :meth:`path`, but creates the directory if it does not exist."""
        target = self.path(key, *parts)
        target.mkdir(parents=True, exist_ok=True)
        return target


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load ``config.yaml`` (or another YAML file) into a :class:`Config`.

    Args:
        path: Config file to read. Defaults to ``config.yaml`` at the project
            root.

    Returns:
        The parsed configuration.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file does not contain a YAML mapping.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a YAML mapping.")

    # A config file next to the code defines its own root; this keeps test
    # fixtures (which write a temporary config.yaml) self-consistent.
    root = config_path.resolve().parent
    return Config(data, project_root=root)


def set_seed(seed: int) -> None:
    """Seed every RNG the project can touch.

    Seeds Python's :mod:`random`, NumPy's legacy global RNG, and PyTorch if it
    happens to be installed (Phase 3 uses it for the CNN; Phase 1 does not
    depend on it). ``PYTHONHASHSEED`` is set for completeness, although it only
    affects interpreters started *after* this call.

    Args:
        seed: The master seed, normally ``cfg.seed``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:  # torch is optional in Phase 1
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def new_rng(seed: int) -> np.random.Generator:
    """Return an explicit NumPy generator.

    Preferred over the global RNG wherever reproducibility actually matters
    (the speaker split, QC sampling, the synthetic dataset), because it cannot
    be disturbed by unrelated library calls that also draw random numbers.
    """
    return np.random.default_rng(seed)
