from __future__ import annotations

import sys
import time
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TypeVar


try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:  # pragma: no cover - exercised by runtime environments without tqdm
    _tqdm = None


T = TypeVar("T")


@dataclass
class _SimpleProgressBar(Iterable[T]):
    iterable: Iterable[T]
    total: int | None
    desc: str | None
    leave: bool

    def __iter__(self) -> Iterator[T]:
        total = self.total
        if total is None:
            yield from self.iterable
            return

        stream = sys.stderr
        width = 28
        start = time.monotonic()
        count = 0
        for count, item in enumerate(self.iterable, start=1):
            filled = min(width, int(width * count / max(total, 1)))
            percent = int(100 * count / max(total, 1))
            prefix = f"{self.desc}: " if self.desc else ""
            stream.write(
                f"\r{prefix}[{'#' * filled}{'.' * (width - filled)}] {percent:3d}% "
                f"({count}/{total})"
            )
            stream.flush()
            yield item

        elapsed = time.monotonic() - start
        prefix = f"{self.desc}: " if self.desc else ""
        stream.write(
            f"\r{prefix}[{'#' * width}] 100% ({count}/{total}) {elapsed:.1f}s"
        )
        if self.leave:
            stream.write("\n")
        else:
            stream.write("\r")
        stream.flush()


def progress_iter(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str | None = None,
    enabled: bool = True,
    leave: bool = True,
) -> Iterable[T]:
    if not enabled:
        return iterable
    if _tqdm is not None:
        return _tqdm(
            iterable,
            total=total,
            desc=desc,
            leave=leave,
            dynamic_ncols=True,
        )
    return _SimpleProgressBar(
        iterable=iterable,
        total=total,
        desc=desc,
        leave=leave,
    )
