"""Where Switchboard's files live.

THE BUG THIS FIXES. Until now, paths were computed like this:

    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    providers_file = PROJECT_ROOT / "providers.yaml"
    database_url = "sqlite:///data/switchboard.db"

"Two folders up from this source file" is the repository root when you are
working in a checkout. Installed with pip, it is somewhere inside
`site-packages` - a directory the user often cannot write to, and which is
replaced wholesale on the next upgrade.

The database was worse, because `data/switchboard.db` is a RELATIVE path. It
resolves against whatever directory the user happened to be standing in. Run
the server from your home folder on Monday and from a project folder on
Tuesday, and Tuesday's Switchboard has no users, no budgets and no history -
with no error, because it helpfully creates a fresh empty database instead.

THE RULE. There are three kinds of file and they belong in three places:

    shipped data   inside the package, read-only      the rule set, samples
    user config    a config directory                 providers.yaml, .env
    user data      a data directory                   the ledger, the router

TWO LAYOUTS, DETECTED RATHER THAN CONFIGURED.

A checkout and an installed copy want different answers, and asking the user
which one they are in would be a question with an obvious answer that they can
still get wrong. So it is detected: if a `providers.yaml` sits next to the
package, that is a deliberately laid-out installation - a git checkout, or the
Docker image at /app - and everything stays there, exactly as before.

Otherwise this is an installed copy, and the operating system's own directories
are used. That keeps a developer's repo working unchanged while making a pip
install behave like every other command-line tool.

WHY NO `platformdirs`. It is a small, well-behaved library that does exactly
this. The rules below are about twenty lines, and this project already made the
same call for the Prometheus exposition format in `metrics.py`: a dependency
that ships in every install, forever, should buy more than twenty lines.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Two directories up from this file. The repository root in a checkout, and
#: somewhere inside site-packages once installed.
PACKAGE_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = PACKAGE_ROOT.parent

APP_NAME = "switchboard"

#: Marker for "config lives beside the package". True in a git checkout and in
#: the Docker image, where providers.yaml is copied to /app.
LAYOUT_MARKER = "providers.yaml"


def is_bundled_layout() -> bool:
    """Is there a providers.yaml sitting next to the package?"""
    return (BUNDLE_ROOT / LAYOUT_MARKER).exists()


def _home() -> Path:
    return Path(os.path.expanduser("~"))


def config_dir() -> Path:
    """Where providers.yaml and .env live.

    Honours SWITCHBOARD_HOME, which overrides everything - useful for running
    two instances side by side, and for tests that must not touch a real one.
    """
    if override := os.environ.get("SWITCHBOARD_HOME"):
        return Path(override)
    if is_bundled_layout():
        return BUNDLE_ROOT

    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(_home() / "AppData" / "Roaming")
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return _home() / "Library" / "Application Support" / APP_NAME
    # Linux and friends: the XDG spec, which most tools follow.
    base = os.environ.get("XDG_CONFIG_HOME") or str(_home() / ".config")
    return Path(base) / APP_NAME


def data_dir() -> Path:
    """Where the ledger and the trained router live.

    Separate from config because they are different things to a user: config is
    edited and worth copying between machines, data is generated and worth
    backing up. Most operating systems draw the same distinction.
    """
    if override := os.environ.get("SWITCHBOARD_HOME"):
        return Path(override) / "data"
    if is_bundled_layout():
        return BUNDLE_ROOT / "data"

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(
            _home() / "AppData" / "Local"
        )
        return Path(base) / APP_NAME
    if sys.platform == "darwin":
        return _home() / "Library" / "Application Support" / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(_home() / ".local" / "share")
    return Path(base) / APP_NAME


def ensure_data_dir() -> Path:
    """Create the data directory if it does not exist, and return it.

    Called before anything writes there. SQLite will not create a missing
    parent directory - it reports "unable to open database file", which reads
    like a permissions problem and sends people looking in the wrong place.
    """
    directory = data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def providers_file() -> Path:
    return config_dir() / LAYOUT_MARKER


def database_path() -> Path:
    return data_dir() / "switchboard.db"


def router_path() -> Path:
    return data_dir() / "router.joblib"


def default_database_url() -> str:
    """An ABSOLUTE sqlite URL.

    Absolute is the whole point. The old default was `sqlite:///data/...`,
    relative to the working directory, so the same command run from two folders
    used two different databases and silently created the second one.

    The four slashes are not a typo: `sqlite:///` is the scheme, and an
    absolute POSIX path starts with another. Windows paths are written with
    forward slashes, which SQLite accepts.
    """
    return "sqlite:///" + database_path().as_posix()


def load_env_files() -> list[str]:
    """Put .env into the process environment, not just into Settings.

    THE BUG THIS FIXES, which shipped from the day providers were added and
    went unnoticed because all development used a local Ollama needing no key.

    pydantic-settings reads .env to populate `Settings` - but PROVIDER keys are
    not Settings fields. `providers.yaml` names an environment variable
    (`api_key_env: GROQ_API_KEY`) and `catalog.py` reads it straight from
    os.environ. Nothing connected the two, so a key written exactly where
    .env.example says to write it never reached the code looking for it, and
    every remote provider reported "no key" forever.

    IT LIVES HERE, not in config.py, and that placement is the second half of
    the fix. The first version put it in config.py - and `catalog.py` does not
    import config, so anything that loaded the catalog without going through
    settings still saw no keys. This module is imported by both, so there is no
    ordering to get wrong.

    `override=False` matters: a variable already set in the real environment
    always wins. That is what makes a one-off `$env:GROQ_API_KEY = "..."` work,
    and what stops a stale .env quietly overriding a deliberate export.

    Load order gives the CONFIG DIRECTORY priority over the working directory.
    An installed copy keeps its keys in the config dir, and almost every
    project on a machine has its own .env - stepping into one of those must not
    silently repoint somebody's gateway.
    """
    from dotenv import load_dotenv

    loaded = []
    for candidate in (config_dir() / ".env", Path(".env")):
        if candidate.is_file() and load_dotenv(candidate, override=False):
            loaded.append(str(candidate))
    return loaded


#: Which .env files were read, for `switchboard where`. Populated at import,
#: so every entry point gets provider keys without having to remember to ask.
LOADED_ENV_FILES = load_env_files()


def describe() -> dict[str, str]:
    """Where everything is, for `switchboard where` and /health."""
    return {
        "layout": "bundled" if is_bundled_layout() else "installed",
        "config_dir": str(config_dir()),
        "data_dir": str(data_dir()),
        "providers_file": str(providers_file()),
        "database": default_database_url(),
    }
