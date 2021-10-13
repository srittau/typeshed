from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from os import PathLike
from pathlib import Path
from typing import Tuple

import tomli

PY2_PATH = "@python2"

_STDLIB_VERSIONS_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_.]*): ([23]\.\d{1,2})-([23]\.\d{1,2})?")
_STUB_VERSION_RE = re.compile(r"\d+(\.\d+)+|\d+(\.\d+)*\.\*")

_PyVersion = Tuple[int, int]


class FormatError(Exception):
    pass


def stdlib_path(typeshed_path: str | PathLike[str], *, py2: bool = False) -> Path:
    path = Path(typeshed_path) / "stdlib"
    if py2:
        path /= PY2_PATH
    return path


def stubs_path(typeshed_path: str | PathLike[str]) -> Path:
    return Path(typeshed_path) / "stubs"


def distribution_path(typeshed_path: str | PathLike[str], distribution: str, *, py2: bool = False) -> Path:
    path = stubs_path(typeshed_path) / distribution
    if py2:
        path /= PY2_PATH
    return path


def distribution_source_path(typeshed_path: str | PathLike[str], distribution: str, *, py2: bool = False) -> Path:
    py2_path = distribution_path(typeshed_path, distribution, py2=True)
    if py2 and py2_path.is_dir():
        return py2_path
    else:
        return distribution_path(typeshed_path, distribution)


class VersionInfo:
    def __init__(self, versions: dict[str, tuple[_PyVersion, _PyVersion | None]]) -> None:
        self._versions = versions

    @property
    def py3_modules(self) -> set[str]:
        return set(mod for mod, v in self._versions.items() if v[1] is None or v[1][0] >= 3)

    def is_supported(self, mod: str, major: int, minor: int) -> bool:
        sub_mod = mod
        while True:
            if sub_mod in self._versions:
                min, max = self._versions[sub_mod]
                if max is None:
                    max = (99, 99)
                return min <= (major, minor) <= max
            if "." not in sub_mod:
                raise KeyError(f"unknown module '{mod}'")
            sub_mod = ".".join(sub_mod.split(".")[:-1])


def stdlib_versions(typeshed_path: str | PathLike[str]) -> VersionInfo:
    """Return Python versions supported by standard library modules.

    The returned dict has all standard library modules as keys. The values
    are 2-tuples with first/last Python version that contained the module.
    If Python 2.7 or earlier added the module, the first version will be
    "2.7". If the module is still supported by the latest Python version, the
    last version will be None.

    If a submodule has a lifetime that differs from the top module, it will
    be listed separately.
    """

    versions: dict[str, tuple[_PyVersion, _PyVersion | None]] = {}
    with open(stdlib_path(typeshed_path) / "VERSIONS") as f:
        data = f.read().splitlines()
    for line in data:
        line = line.split("#")[0].strip()
        if line == "":
            continue
        m = _STDLIB_VERSIONS_RE.fullmatch(line)
        if m is None:
            raise FormatError(f"Bad line in VERSIONS: {line}")
        module = m.group(1)
        assert module not in versions, f"Duplicate module {module} in VERSIONS"
        min: str = m.group(2)
        max: str | None = m.group(3)
        versions[module] = _parse_py_version(min), _parse_py_version(max) if max is not None else max
    return VersionInfo(versions)


def _parse_py_version(v: str) -> _PyVersion:
    major, minor = v.split(".")
    return int(major), int(minor)


def all_stdlib_modules(typeshed_path: str | PathLike[str]) -> list[str]:
    """Return a list of all standard library modules and sub-modules."""
    stdlib_p = stdlib_path(typeshed_path).absolute()
    prefix_len = len(str(stdlib_p))
    modules: set[str] = set()
    for p, _, files in os.walk(stdlib_p):
        path = p[prefix_len + 1 :]
        if path.startswith(PY2_PATH):
            continue
        for filename in files:
            base_module = ".".join(path.split(os.sep))
            if filename == "__init__.pyi":
                modules.add(base_module)
            elif filename.endswith(".pyi"):
                mod, _ = os.path.splitext(filename)
                modules.add(f"{base_module}.{mod}" if base_module else mod)
    return sorted(modules)


def third_party_distributions(typeshed_path: str | PathLike[str]) -> list[str]:
    return [e.name for e in stubs_path(typeshed_path).iterdir()]


class MetaData:
    def __init__(
        self,
        distribution: str,
        version: str,
        requires: Iterable[str] = [],
        *,
        extra_description: str | None = None,
        obsolete_since: str | None = None,
        python2: bool = False,
    ) -> None:  # noqa: B006
        if not _STUB_VERSION_RE.fullmatch(version):
            raise FormatError(f"Invalid version {version} for {distribution}")
        self.distribution = distribution
        self.version = version
        self.requires = list(requires)
        self.extra_description = extra_description
        self.obsolete_since = obsolete_since
        self.python2 = python2

    @property
    def base_version(self) -> str:
        if self.version.endswith(".*"):
            return self.version[:-2]
        else:
            return self.version


_metadata_keys = {"version": str, "python2": bool, "requires": list, "extra_description": str, "obsolete_since": str}


def read_metadata(typeshed_path: str | PathLike[str], distribution: str) -> MetaData:
    with open(distribution_path(typeshed_path, distribution) / "METADATA.toml") as f:
        data = tomli.loads(f.read())
    if "version" not in data:
        raise FormatError(f"Missing version for {distribution}")
    for key, expected_type in _metadata_keys.items():
        if key in data and not isinstance(data[key], expected_type):
            raise FormatError(f"Invalid {key} value for {distribution}")
    for key in data:
        if key not in _metadata_keys:
            raise FormatError(f"Unexpected key {key} for {distribution}")
    version: str = data["version"]
    requires: list[str] = data.get("requires", [])
    extra_description: str | None = data.get("extra_description")
    obsolete_since: str | None = data.get("obsolete_since")
    python2: bool = data.get("python2", False)
    for dep in requires:
        if not isinstance(dep, str):
            raise FormatError(f"Invalid dependency {dep} for {distribution}")
    return MetaData(
        distribution, version, requires, extra_description=extra_description, obsolete_since=obsolete_since, python2=python2
    )


def supported_python_versions(typeshed_path: str | PathLike[str], distribution: str) -> set[int]:
    supported: set[int] = set()
    data = read_metadata(typeshed_path, distribution)
    if data.python2 or distribution_path(typeshed_path, distribution, py2=True).is_dir():
        supported.add(2)
    if distribution_has_py3_stubs(typeshed_path, distribution):
        supported.add(3)
    assert 1 <= len(supported) <= 2
    return supported


def distribution_has_py3_stubs(typeshed_path: str | PathLike[str], distribution: str) -> bool:
    dist_path = distribution_path(typeshed_path, distribution)
    return len(list(dist_path.glob("*.pyi"))) > 0 or len(list(dist_path.glob("[!@]*/__init__.pyi"))) > 0


def stdlib_modules(typeshed_path: str | PathLike[str], *, py2: bool = False) -> Iterator[tuple[str, Path]]:
    root = stdlib_path(typeshed_path, py2=py2)
    for entry in root.iterdir():
        if entry.name in [PY2_PATH, "VERSIONS"]:
            continue
        yield entry.stem, entry


def distribution_modules(
    typeshed_path: str | PathLike[str], distribution: str, *, py2: bool = False
) -> Iterator[tuple[str, Path]]:
    root = distribution_source_path(typeshed_path, distribution, py2=py2)
    for entry in root.iterdir():
        if entry.name == PY2_PATH:
            continue
        yield entry.stem, entry
