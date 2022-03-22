#!/usr/bin/env python3

"""
Print information about third-party package versions.

By default, this prints the latest stub version and the latest version
and release date on PyPI to the console. Optionally, it prints the
status as an HTML page.
"""

from __future__ import annotations

import asyncio
import datetime
import sys
from argparse import ArgumentParser
from asyncio.tasks import Task
from dataclasses import dataclass
from enum import Enum
from html import escape as html_escape
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import quote as quote_url

import aiohttp
import tomli

try:
    from termcolor import colored
except ImportError:

    def colored(s: str, c: str) -> str:
        return s


NOW = datetime.datetime.utcnow()
OLD_YEARS = 2
ANCIENT_YEARS = 5


class OutputFormat(Enum):
    TEXT = "text"
    HTML = "html"


@dataclass
class StubInfo:
    distribution: str
    stub_version: str
    pypi_version: str
    pypi_date: datetime.datetime
    obsolete: bool = False

    @property
    def version_freshness(self) -> VersionFreshness:
        if self.stub_version.endswith(".*"):
            stub_parts = self.stub_version.split(".")[:-1]
            pypi_parts = self.pypi_version.split(".")
            assert len(stub_parts) <= len(pypi_parts)
            if stub_parts == pypi_parts[: len(stub_parts)]:
                return VersionFreshness.FRESH
            elif stub_parts[0] == pypi_parts[0]:
                return VersionFreshness.NEW_MINOR
            else:
                return VersionFreshness.NEW_MAJOR
        else:
            if self.stub_version == self.pypi_version:
                return VersionFreshness.FRESH
            else:
                return VersionFreshness.NEW_MINOR

    @property
    def date_freshness(self) -> DateFreshness:
        days = (NOW - self.pypi_date).days
        if days > 365 * ANCIENT_YEARS:
            return DateFreshness.ANCIENT
        elif days > 265 * OLD_YEARS:
            return DateFreshness.OLD
        else:
            return DateFreshness.FRESH


class VersionFreshness(Enum):
    FRESH = "fresh"
    NEW_MINOR = "new-minor"
    NEW_MAJOR = "new-major"


class DateFreshness(Enum):
    FRESH = "fresh"
    OLD = "old"
    ANCIENT = "ancient"


VERSION_FRESHNESS_COLORS = {
    VersionFreshness.FRESH: "green",
    VersionFreshness.NEW_MINOR: "yellow",
    VersionFreshness.NEW_MAJOR: "red",
}

DATE_FRESHNESS_COLORS = {DateFreshness.FRESH: "green", DateFreshness.OLD: "yellow", DateFreshness.ANCIENT: "red"}


def main() -> None:
    format = parse_args()
    ts_stubs = list(read_typeshed_stubs())
    pypi_stubs = asyncio.run(fetch_pypi_stubs(t[0] for t in ts_stubs))
    stubs = [
        StubInfo(distribution, stub_version, pypi_version, pypi_date, obsolete)
        for (distribution, stub_version, obsolete), (pypi_version, pypi_date) in zip(ts_stubs, pypi_stubs)
    ]
    stubs.sort(key=lambda si: si.distribution)
    if format == OutputFormat.HTML:
        print_stubs_html(stubs)
    else:
        print_stubs_text(stubs)


def parse_args() -> OutputFormat:
    formats = [o.value for o in OutputFormat]
    parser = ArgumentParser()
    parser.add_argument("--output-format", type=str, choices=formats, default=formats[0])
    args = parser.parse_args()
    return OutputFormat(args.output_format)


def read_typeshed_stubs() -> Iterator[tuple[str, str, bool]]:
    """Yield (distribution, version, obsolete) tuples for all third-party stubs in typeshed."""
    for stub_path in Path("stubs").iterdir():
        with (stub_path / "METADATA.toml").open("rb") as f:
            meta = tomli.load(f)
        obsolete = meta.get("obsolete_since") is not None
        yield stub_path.name, meta["version"], obsolete


async def fetch_pypi_stubs(distributions: Iterable[str]) -> tuple[tuple[str, datetime.datetime], ...]:
    conn = aiohttp.TCPConnector(limit_per_host=10)
    async with aiohttp.ClientSession(connector=conn) as session:
        tasks: list[Task[tuple[str, datetime.datetime]]] = []
        for distribution in distributions:
            tasks.append(asyncio.create_task(fetch_pypi_info(session, distribution)))
        return await asyncio.gather(*tasks)


async def fetch_pypi_info(session: aiohttp.ClientSession, distribution: str) -> tuple[str, datetime.datetime]:
    """Return the latest version and release date of a distribution on PyPI."""
    url = f"https://pypi.org/pypi/{quote_url(distribution)}/json"
    async with session.get(url) as response:
        assert response.status == 200
        j = await response.json()
        version = j["info"]["version"]
        date = datetime.datetime.fromisoformat(j["releases"][version][0]["upload_time"])
        return version, date


def print_stubs_text(stubs: Iterable[StubInfo]) -> None:
    dist_len = max(len(st.distribution) for st in stubs) + 2
    stub_v_len = max(len(st.stub_version) for st in stubs) + 2
    pypi_v_len = max(len(st.pypi_version) for st in stubs) + 2
    for stub in stubs:
        distribution_text = color(stub.distribution, "red") if stub.obsolete else stub.distribution
        version_color = VERSION_FRESHNESS_COLORS[stub.version_freshness]
        date_color = DATE_FRESHNESS_COLORS[stub.date_freshness]
        print(distribution_text, end="")
        print(" " * (dist_len - len(stub.distribution)), end="")
        print(color(stub.stub_version, version_color), end="")
        print(" " * (stub_v_len - len(stub.stub_version)), end="")
        print(stub.pypi_version, end="")
        print(" " * (pypi_v_len - len(stub.pypi_version)), end="")
        print(color(stub.pypi_date.date().isoformat(), date_color))


def color(s: str, color: str) -> str:
    if sys.stdout.isatty():
        return colored(s, color)
    else:
        return s


HTML_HEADER = f"""<!DOCTYPE html>
<html>
    <head>
        <title>typeshed Third-Party Stub Status</title>
        <style>
            body {{ font-family: Sans; }}
            th {{ text-align: left; }}
            .fresh {{ color: green; }}
            .old, .new-minor {{ color: orange; }}
            .ancient, .new-major {{ color: red; }}
            .obsolete {{ text-decoration: line-through; }}
        </style>
    </head>
    <body>
        <p>Last update: {NOW.date().isoformat()}</p>
        <table>
            <thead>
                <tr><th>Distribution</th><th>Stub</th><th>PyPI</th><th>Last Release</th></tr>
            </thead>
            <tbody>
"""

HTML_FOOTER = "</tbody></table></body></html>"


def print_stubs_html(stubs: Iterable[StubInfo]) -> None:
    print(HTML_HEADER)
    for stub in stubs:
        print(
            f"""
        <tr>
            <td class="{'obsolete' if stub.obsolete else ''}">{html_escape(stub.distribution)}</td>
            <td class="{stub.version_freshness.value}">{html_escape(stub.stub_version)}</td>
            <td>{html_escape(stub.pypi_version)}</td>
            <td class="{stub.date_freshness.value}">{stub.pypi_date.date().isoformat()}</td>
        </tr>
        """
        )
    print(HTML_FOOTER)


if __name__ == "__main__":
    main()
