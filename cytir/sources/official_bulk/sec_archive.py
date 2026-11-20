#!/usr/bin/env python3
"""One-home validation for the official SEC submissions.zip member grammar."""
from __future__ import annotations

from dataclasses import dataclass
import re
import zipfile


PRIMARY = re.compile(r"CIK([0-9]{10})\.json")
HISTORICAL = re.compile(r"CIK([0-9]{10})-submissions-([0-9]{3})\.json")
PLACEHOLDER_NAME = "placeholder.txt"
PLACEHOLDER_BYTES = b"Placeholder file"


@dataclass(frozen=True)
class SubmissionArchiveMembers:
    primary: tuple[zipfile.ZipInfo, ...]
    historical: tuple[zipfile.ZipInfo, ...]
    placeholders: tuple[zipfile.ZipInfo, ...]

    @property
    def total(self) -> int:
        return len(self.primary) + len(self.historical) + len(self.placeholders)


def validate_submission_archive(
    archive: zipfile.ZipFile,
) -> SubmissionArchiveMembers:
    """Validate every member and distinguish entity roots from filing-history shards.

    SEC's official bulk contains one primary entity JSON per CIK, optional
    ``-submissions-NNN`` history shards for large filers, and an exact 16-byte
    source-owned placeholder.  Only primary JSON members are company records.
    """
    infos = archive.infolist()
    if not infos:
        raise ValueError("SEC submissions ZIP is empty")
    names = [info.filename for info in infos]
    if len(names) != len(set(names)):
        raise ValueError("SEC submissions ZIP has duplicate member names")

    primary: list[zipfile.ZipInfo] = []
    historical: list[zipfile.ZipInfo] = []
    placeholders: list[zipfile.ZipInfo] = []
    primary_ciks: set[str] = set()
    historical_ciks: set[str] = set()
    for info in infos:
        if info.is_dir() or info.flag_bits & 0x1:
            raise ValueError("SEC submissions ZIP has directory/encrypted member")
        match = PRIMARY.fullmatch(info.filename)
        if match:
            primary.append(info)
            primary_ciks.add(match.group(1))
            continue
        match = HISTORICAL.fullmatch(info.filename)
        if match:
            historical.append(info)
            historical_ciks.add(match.group(1))
            continue
        if info.filename == PLACEHOLDER_NAME:
            if placeholders or archive.read(info) != PLACEHOLDER_BYTES:
                raise ValueError("SEC submissions ZIP placeholder is not exact")
            placeholders.append(info)
            continue
        raise ValueError(f"unexpected SEC submissions ZIP member: {info.filename!r}")
    if not primary:
        raise ValueError("SEC submissions ZIP has no primary entity members")
    if not historical_ciks.issubset(primary_ciks):
        raise ValueError("SEC history shard has no primary entity member")
    return SubmissionArchiveMembers(
        primary=tuple(primary),
        historical=tuple(historical),
        placeholders=tuple(placeholders),
    )
