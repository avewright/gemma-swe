#!/usr/bin/env python3
"""Download the contest data file by file into data/.

Kaggle's bundled zip endpoint currently returns 404 ("no gcs url"), so this
pulls each published file instead. Re-running skips files that already match
the remote size.
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION = "gemma-4-developer-agent"
ROOT = Path(__file__).resolve().parents[1] / "data"
API = f"https://www.kaggle.com/api/v1/competitions/data/download/{COMPETITION}/"


def list_files(api: KaggleApi) -> list[tuple[str, int]]:
    files: list[tuple[str, int]] = []
    token = None
    while True:
        page = api.competition_list_files(COMPETITION, page_token=token, page_size=200)
        for item in page.files or []:
            name = item.name or item.ref
            if not name or ".." in Path(name).parts:
                continue
            files.append((name, int(item.total_bytes or 0)))
        token = page.next_page_token or None
        if not token:
            return files


def download(name: str, size: int, token: str) -> None:
    dest = ROOT / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if size and have == size:
        print(f"skip {name}", flush=True)
        return
    if size and have > size:
        dest.unlink()
        have = 0

    headers = {"Authorization": f"Bearer {token}"}
    meta = requests.get(API + quote(name, safe=""), headers=headers, allow_redirects=False, timeout=60)
    meta.raise_for_status()
    location = meta.headers["Location"]
    meta.close()

    extra = {"Range": f"bytes={have}-"} if have else {}
    response = requests.get(location, headers=extra, stream=True, timeout=(30, 300))
    if have and response.status_code == 200:
        have = 0
    response.raise_for_status()
    mode = "ab" if have and response.status_code == 206 else "wb"
    if mode == "wb":
        have = 0
    with dest.open(mode) as handle:
        for chunk in response.iter_content(1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)
            have += len(chunk)
    response.close()
    got = dest.stat().st_size
    if size and got != size:
        raise RuntimeError(f"{name} size {got} != {size}")
    print(f"got {name} {got}", flush=True)


def main() -> None:
    token = os.environ["KAGGLE_API_TOKEN"]
    api = KaggleApi()
    api.authenticate()
    files = list_files(api)
    total = sum(size for _, size in files)
    print(f"{len(files)} files, {total} bytes", flush=True)
    failed: list[str] = []
    for index, (name, size) in enumerate(files, start=1):
        print(f"[{index}/{len(files)}] {name}", flush=True)
        try:
            download(name, size, token)
        except Exception as exc:
            failed.append(name)
            print(f"fail {name}: {exc}", flush=True)
    print(f"done failed={len(failed)}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
