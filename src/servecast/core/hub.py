"""An LLM's config.json from the Hugging Face hub, read with the user's own token (HF_TOKEN) and cached.

Only config.json is fetched: the cost laws need the geometry, never the weights. A gated model needs a token that has
accepted its licence; the token is read from the environment and never written anywhere.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from servecast.core.paths import cache_dir

REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
URL = "https://huggingface.co/{repo}/resolve/{revision}/config.json"
TIMEOUT_S = 30


class _TokenStaysOnTheHub(urllib.request.HTTPRedirectHandler):
    """Follows the hub's redirects, which stay on its host, and sends the token on only there and only over HTTPS when
    it came over HTTPS: urllib copies every header of a request to the location it is redirected to, whatever the
    host."""

    def redirect_request(self, req: urllib.request.Request, fp: Any, code: int, msg: str, headers: Any,
                         newurl: str) -> urllib.request.Request | None:
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        old, to = urllib.parse.urlsplit(req.full_url), urllib.parse.urlsplit(newurl)
        if new is not None and (to.hostname != old.hostname or (old.scheme == "https" and to.scheme != "https")):
            new.remove_header("Authorization")
        return new


def cached_path(repo_id: str, revision: str = "main") -> Path:
    return cache_dir() / "configs" / repo_id.replace("/", "__") / revision / "config.json"


def fetch_config(repo_id: str, revision: str = "main", *, offline: bool = False) -> dict[str, Any]:
    """The repository's config.json, from the cache or the hub."""
    if not REPO_ID.match(repo_id):
        raise ValueError(f"{repo_id!r} is not a Hugging Face id (<owner>/<name>)")
    path = cached_path(repo_id, revision)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    if offline:
        raise ValueError(f"{repo_id}: config.json is not cached and the hub may not be used; give a local directory "
                         "holding it instead")
    request = urllib.request.Request(URL.format(repo=repo_id, revision=revision))
    token = os.environ.get("HF_TOKEN")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.build_opener(_TokenStaysOnTheHub).open(request, timeout=TIMEOUT_S) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            # the hub answers 401 for a repository that does not exist too, so that its name says nothing
            raise ValueError(f"{repo_id}: the hub refused config.json ({error.code}): no such repository, or a gated "
                             "one, which needs HF_TOKEN set to a token that has accepted its licence; or give a local "
                             "directory holding config.json") from error
        if error.code == 404:
            raise ValueError(f"{repo_id}: no such repository, or no config.json in it") from error
        raise ValueError(f"{repo_id}: the hub answered {error.code}") from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ValueError(f"{repo_id}: the hub could not be reached ({error}); give a local directory holding "
                         "config.json instead") from error
    config = json.loads(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=1) + "\n", encoding="utf-8")
    return config
