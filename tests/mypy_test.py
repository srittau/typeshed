#!/usr/bin/env python3
"""Test runner for typeshed.

Depends on mypy being installed.

Approach:

1. Parse sys.argv
2. Compute appropriate arguments for mypy
3. Stuff those arguments into sys.argv
4. Run mypy.main('')
5. Repeat steps 2-4 for other mypy runs (e.g. --py2)
"""

import argparse
import os
import re
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any, NamedTuple

import tomli

typeshed_path = Path(__file__).parent.parent
sys.path.append(str(typeshed_path / "src"))

from typeshed_utils import (  # noqa: E402
    distribution_modules,
    stdlib_modules,
    stdlib_versions,
    supported_python_versions,
    third_party_distributions,
)

parser = argparse.ArgumentParser(description="Test runner for typeshed. Patterns are unanchored regexps on the full path.")
parser.add_argument("-v", "--verbose", action="count", default=0, help="More output")
parser.add_argument("-n", "--dry-run", action="store_true", help="Don't actually run mypy")
parser.add_argument("-x", "--exclude", type=str, nargs="*", help="Exclude pattern")
parser.add_argument("-p", "--python-version", type=str, nargs="*", help="These versions only (major[.minor])")
parser.add_argument("--platform", help="Run mypy for a certain OS platform (defaults to sys.platform)")
parser.add_argument(
    "--warn-unused-ignores",
    action="store_true",
    help="Run mypy with --warn-unused-ignores "
    "(hint: only get rid of warnings that are "
    "unused for all platforms and Python versions)",
)

parser.add_argument("filter", type=str, nargs="*", help="Include pattern (default all)")


def log(args, *varargs):
    if args.verbose >= 2:
        print(*varargs)


def match(fn_p: Path, args, exclude_list):
    fn = str(fn_p)
    if exclude_list.match(fn):
        log(args, fn, "exluded by exclude list")
        return False
    if not args.filter and not args.exclude:
        log(args, fn, "accept by default")
        return True
    if args.exclude:
        for f in args.exclude:
            if re.search(f, fn):
                log(args, fn, "excluded by pattern", f)
                return False
    if args.filter:
        for f in args.filter:
            if re.search(f, fn):
                log(args, fn, "accepted by pattern", f)
                return True
    if args.filter:
        log(args, fn, "rejected (no pattern matches)")
        return False
    log(args, fn, "accepted (no exclude pattern matches)")
    return True


def add_files(files: list[Path], seen: set[str], mod: str, path: Path, args, exclude_list) -> None:
    """Add all files in package or module represented by 'name' located in 'root'."""
    if path.suffix in [".pyi", ".py"]:
        if match(path, args, exclude_list):
            seen.add(mod)
            files.append(path)
    elif (path / "__init__.pyi").is_file() or (path / "__init__.py").is_file():
        for r, ds, fs in os.walk(path):
            ds.sort()
            fs.sort()
            for f in fs:
                _, x = os.path.splitext(f)
                if x in [".pyi", ".py"]:
                    fn = Path(r) / f
                    if match(fn, args, exclude_list):
                        seen.add(mod)
                        files.append(fn)


class MypyDistConf(NamedTuple):
    module_name: str
    values: dict[str, Any]


# The configuration section in the metadata file looks like the following, with multiple module sections possible
# [mypy-tests]
# [mypy-tests.yaml]
# module_name = "yaml"
# [mypy-tests.yaml.values]
# disallow_incomplete_defs = true
# disallow_untyped_defs = true


def add_configuration(configurations, seen_dist_configs, distribution):
    if distribution in seen_dist_configs:
        return

    with open(os.path.join("stubs", distribution, "METADATA.toml")) as f:
        data = dict(tomli.loads(f.read()))

    mypy_tests_conf = data.get("mypy-tests")
    if not mypy_tests_conf:
        return

    assert isinstance(mypy_tests_conf, dict), "mypy-tests should be a section"
    for section_name, mypy_section in mypy_tests_conf.items():
        assert isinstance(mypy_section, dict), "{} should be a section".format(section_name)
        module_name = mypy_section.get("module_name")

        assert module_name is not None, "{} should have a module_name key".format(section_name)
        assert isinstance(module_name, str), "{} should be a key-value pair".format(section_name)

        values = mypy_section.get("values")
        assert values is not None, "{} should have a values section".format(section_name)
        assert isinstance(values, dict), "values should be a section"

        configurations.append(MypyDistConf(module_name, values.copy()))
    seen_dist_configs.add(distribution)


def run_mypy(args, configurations: Iterable[Any], major, minor, files: list[Path], *, custom_typeshed=False):
    try:
        from mypy.main import main as mypy_main
    except ImportError:
        print("Cannot import mypy. Did you install it?")
        sys.exit(1)

    with tempfile.NamedTemporaryFile("w+") as temp:
        temp.write("[mypy]\n")
        for dist_conf in configurations:
            temp.write("[mypy-%s]\n" % dist_conf.module_name)
            for k, v in dist_conf.values.items():
                temp.write("{} = {}\n".format(k, v))
        temp.flush()

        flags = [
            "--python-version",
            "%d.%d" % (major, minor),
            "--config-file",
            temp.name,
            "--strict-optional",
            "--no-site-packages",
            "--show-traceback",
            "--no-implicit-optional",
            "--disallow-any-generics",
            "--disallow-subclassing-any",
            "--warn-incomplete-stub",
        ]
        if custom_typeshed:
            # Setting custom typeshed dir prevents mypy from falling back to its bundled
            # typeshed in case of stub deletions
            flags.extend(["--custom-typeshed-dir", os.path.dirname(os.path.dirname(__file__))])
        if args.warn_unused_ignores:
            flags.append("--warn-unused-ignores")
        if args.platform:
            flags.extend(["--platform", args.platform])
        sys.argv = ["mypy"] + flags + [str(f) for f in files]
        if args.verbose:
            print("running", " ".join(sys.argv))
        else:
            print("running mypy", " ".join(flags), "# with", len(files), "files")
        if not args.dry_run:
            try:
                mypy_main("", sys.stdout, sys.stderr)
            except SystemExit as err:
                return err.code
        return 0


def main():
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), "mypy_exclude_list.txt")) as f:
        exclude_list = re.compile("(%s)$" % "|".join(re.findall(r"^\s*([^\s#]+)\s*(?:#.*)?$", f.read(), flags=re.M)))

    versions = [(3, 10), (3, 9), (3, 8), (3, 7), (3, 6), (2, 7)]
    if args.python_version:
        versions = [v for v in versions if any(("%d.%d" % v).startswith(av) for av in args.python_version)]
        if not versions:
            print("--- no versions selected ---")
            sys.exit(1)

    code = 0
    runs = 0
    for major, minor in versions:
        seen = {"__builtin__", "builtins", "typing"}  # Always ignore these.

        # Test standard library files.
        files: list[Path] = []
        supported_versions = stdlib_versions(typeshed_path)
        for mod, path in stdlib_modules(typeshed_path, py2=major == 2):
            if mod in seen or mod.startswith("."):
                continue
            if supported_versions.is_supported(mod, major, minor):
                add_files(files, seen, mod, path, args, exclude_list)

        if files:
            this_code = run_mypy(args, [], major, minor, files, custom_typeshed=True)
            code = max(code, this_code)
            runs += 1

        # Test files of all third party distributions.
        configurations: list[Any] = []
        seen_dist_configs = set()
        files = []
        for distribution in third_party_distributions(typeshed_path):
            if major not in supported_python_versions(typeshed_path, distribution):
                continue
            for mod, path in distribution_modules(typeshed_path, distribution, py2=major == 2):
                if mod in seen or mod.startswith("."):
                    continue
                add_files(files, seen, mod, path, args, exclude_list)
                add_configuration(configurations, seen_dist_configs, distribution)

        if files:
            # TODO: remove custom_typeshed after mypy 0.920 is released
            this_code = run_mypy(args, configurations, major, minor, files, custom_typeshed=True)
            code = max(code, this_code)
            runs += 1

    if code:
        print("--- exit status", code, "---")
        sys.exit(code)
    if not runs:
        print("--- nothing to do; exit 1 ---")
        sys.exit(1)


if __name__ == "__main__":
    main()
