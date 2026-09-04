#!/usr/bin/env python3
"""Mass clone/update of Git repositories listed in a text file.

Local layout: ``repos/<host>/<owner>/<repository>``
Examples:
    https://github.com/docker/docs.git   -> repos/github.com/docker/docs
    git@github.com:docker/docs.git        -> repos/github.com/docker/docs
    https://gitlab.com/company/docs.git   -> repos/gitlab.com/company/docs

Only the Python standard library is used. ``git`` must be available in PATH.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

SCRIPT_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# URL parsing / normalization
# --------------------------------------------------------------------------- #
_SCP_RE = re.compile(
    r"^(?:(?P<user>[^@]+)@)?(?P<host>[^:/]+):(?P<path>.+)$"
)


def parse_git_url(url: str) -> tuple[str, str, str]:
    """Parse a Git URL into ``(host, owner, repository)``.

    Supports HTTPS (``https://host/owner/repo.git``), the SSH scp-like form
    (``git@host:owner/repo.git``) and explicit SSH (``ssh://git@host/owner/repo.git``).
    """
    url = url.strip()

    if "://" not in url:
        m = _SCP_RE.match(url)
        if not m:
            raise ValueError(f"Cannot parse Git URL: {url!r}")
        host = m.group("host")
        parts = _split_path(m.group("path"))
    else:
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https", "ssh", "git", "file"):
            raise ValueError(f"Unsupported scheme in URL: {url!r}")
        if not parsed.hostname:
            raise ValueError(f"Missing host in URL: {url!r}")
        host = parsed.hostname
        parts = _split_path(parsed.path)

    if len(parts) < 2:
        raise ValueError(f"URL does not contain owner/repository: {url!r}")
    owner, repo = parts[-2], parts[-1]
    return host, owner, repo


def _split_path(path: str) -> list[str]:
    """Split a URL path into non-empty segments, dropping a trailing ``.git``."""
    cleaned = path.strip("/")
    if cleaned.endswith(".git"):
        cleaned = cleaned[: -len(".git")]
    return [seg for seg in cleaned.split("/") if seg]


def normalize_git_url(url: str) -> str:
    """Return a canonical logical id ``host/owner/repository`` for a Git URL.

    Both ``https://github.com/docker/docs.git`` and ``git@github.com:docker/docs.git``
    normalize to ``github.com/docker/docs``, so they are treated as the same repo.
    """
    host, owner, repo = parse_git_url(url)
    return f"{host.lower()}/{owner}/{repo}"


# --------------------------------------------------------------------------- #
# Path / git helpers
# --------------------------------------------------------------------------- #
def get_repo_path(url: str, dest: Path) -> Path:
    """Build the local path ``<dest>/<host>/<owner>/<repository>``."""
    host, owner, repo = parse_git_url(url)
    return dest / host.lower() / owner / repo


def _run_git(args: list[str], cwd: Optional[Path] = None) -> subprocess.CompletedProcess:
    """Run a git command without a shell. Raises on non-zero exit via caller."""
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        capture_output=True,
        text=True,
        check=False,
    )


def is_git_repository(path: Path) -> bool:
    """Return True if ``path`` is the root of a Git working tree."""
    if not path.exists():
        return False
    result = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=path)
    return result.returncode == 0 and result.stdout.strip() == "true"


def get_origin_url(path: Path) -> Optional[str]:
    """Return the ``origin`` remote URL of the repo at ``path`` or None."""
    result = _run_git(["config", "--get", "remote.origin.url"], cwd=path)
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None


def clone_repository(url: str, path: Path) -> None:
    """Clone ``url`` into ``path`` (parent directories are created)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    result = _run_git(["clone", "--depth", "1", "--", url, str(path)])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "git clone failed")


def _detect_default_branch(path: Path) -> str:
    """Detect the default branch name (main/master) from origin/HEAD."""
    result = _run_git(["symbolic-ref", "refs/remotes/origin/HEAD"], cwd=path)
    if result.returncode == 0:
        ref = result.stdout.strip()
        if ref.startswith("refs/remotes/origin/"):
            return ref[len("refs/remotes/origin/"):]
    for candidate in ("main", "master", "dev"):
        result = _run_git(["rev-parse", "--verify", f"origin/{candidate}"], cwd=path)
        if result.returncode == 0:
            return candidate
    return "main"


def update_repository(path: Path) -> None:
    """Update the repo by fetching and resetting to the remote default branch."""
    result = _run_git(["fetch", "origin"], cwd=path)
    if result.returncode != 0:
        msg = result.stderr.strip() or "git fetch failed"
        raise RuntimeError(msg)

    branch = _detect_default_branch(path)
    result = _run_git(["reset", "--hard", f"origin/{branch}"], cwd=path)
    if result.returncode != 0:
        msg = result.stderr.strip() or f"git reset --hard origin/{branch} failed"
        raise RuntimeError(msg)


# --------------------------------------------------------------------------- #
# Per-repository processing
# --------------------------------------------------------------------------- #
def process_repository(url: str, dest: Path) -> tuple[str, str, str]:
    """Clone or update a single repository.

    Returns ``(url, status, detail)`` where status is one of
    ``cloned``, ``updated``, ``conflict``, ``error``.
    """
    try:
        path = get_repo_path(url, dest)
        target = normalize_git_url(url)

        if path.exists():
            if not is_git_repository(path):
                return (
                    url,
                    "conflict",
                    f"path exists but is not a Git repository: {path} "
                    f"(left untouched)",
                )

            origin = get_origin_url(path)
            if origin is None:
                return (
                    url,
                    "conflict",
                    f"existing Git repo at {path} has no origin remote "
                    f"(left untouched)",
                )
            if normalize_git_url(origin) != target:
                return (
                    url,
                    "conflict",
                    f"existing repo at {path} has origin "
                    f"{origin!r}, expected {url!r} (left untouched)",
                )

            update_repository(path)
            return (url, "updated", str(path))

        clone_repository(url, path)
        return (url, "cloned", str(path))

    except Exception as exc:  # noqa: BLE001 - isolate per-repo failures
        return (url, "error", str(exc))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def load_urls(path: Path) -> list[str]:
    """Read repository URLs from ``path``, skipping blanks and ``#`` comments.

    Tolerates a UTF-8 BOM.
    """
    if not path.is_file():
        print(f"[!] List file not found: {path}", file=sys.stderr)
        sys.exit(1)
    urls: list[str] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            urls.append(line)
    return urls


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone/update Git repositories from a list file.",
    )
    parser.add_argument(
        "--file",
        default="repos.txt",
        help="File with one repo URL per line (default: repos.txt)",
    )
    parser.add_argument(
        "--dest",
        default="repos",
        help="Destination root directory (default: repos/)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    list_file = (SCRIPT_DIR / args.file).resolve()
    dest_dir = (SCRIPT_DIR / args.dest).resolve()

    urls = load_urls(list_file)
    if not urls:
        print("[*] No repositories to process.")
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[*] Processing {len(urls)} repos into {dest_dir} "
        f"with {args.workers} workers\n"
    )

    cloned = updated = conflicts = errors = 0
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(process_repository, url, dest_dir): url for url in urls
        }
        for future in as_completed(futures):
            url, status, detail = future.result()
            if status == "cloned":
                cloned += 1
                print(f"[+] cloned  {url}")
            elif status == "updated":
                updated += 1
                print(f"[~] updated {url}")
            elif status == "conflict":
                conflicts += 1
                print(f"[!] conflict {url}: {detail}", file=sys.stderr)
            else:
                errors += 1
                print(f"[!] error   {url}: {detail}", file=sys.stderr)

    print(
        f"\n[*] Done. cloned={cloned} updated={updated} "
        f"conflicts={conflicts} errors={errors}"
    )


if __name__ == "__main__":
    main()
