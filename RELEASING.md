# Publishing Switchboard to PyPI

PyPI is the index `pip install` reads. Publishing there is what turns

```powershell
git clone https://github.com/vbvansh/switchboard
cd switchboard
pip install -r requirements.txt
```

into

```powershell
pip install switchboard-router
switchboard init
```

Nothing here is automated on purpose. **A release cannot be taken back** — PyPI
does not allow re-uploading a version number, ever, even to fix a typo. The
worst outcome is not a failed upload; it is a successful one that installs
broken software for everybody who tries it that week.

---

## Before every release

```powershell
python -m switchboard release-check
```

That runs the things that are cheap to check and expensive to get wrong:
version consistency, a clean tree, the test suite, the linter, and a real wheel
build whose contents are inspected. Read its output rather than skimming for
green.

---

## One-time setup

### You need TWO accounts, not one

This is the single most common way a first release goes wrong, so it is the
first thing here.

| | Address | What it is |
|---|---|---|
| **TestPyPI** | [test.pypi.org](https://test.pypi.org) | a practice index, wiped periodically |
| **PyPI** | [pypi.org](https://pypi.org) | the real one, permanent |

They are **entirely separate websites**. Separate accounts, separate passwords,
separate API tokens. A token issued by one is meaningless to the other, and
using the wrong one produces:

```
ERROR HTTPError: 403 Forbidden from https://test.pypi.org/legacy/
      Forbidden
```

— after the file has already uploaded to 100%, which makes it look like a
problem with the package. It is not. It is the wrong token.

### Steps

1. Register at **[test.pypi.org/account/register](https://test.pypi.org/account/register/)**
   and enable two-factor authentication.
2. There: **Account settings → API tokens → Add API token**, scope "Entire
   account". Copy it — it is shown once, and starts with `pypi-`.
3. Register **separately** at **[pypi.org](https://pypi.org)**, enable 2FA, and
   create a second token the same way.
4. Keep them apart. They look identical and are not interchangeable.

```powershell
pip install build twine
```

`build` makes the distribution files. `twine` uploads them and checks them
first.

### Optional: stop retyping the tokens

Create a `.pypirc` file in your home directory — `C:\Users\<you>\.pypirc` on
Windows, `~/.pypirc` elsewhere:

```ini
[distutils]
index-servers =
    pypi
    testpypi

[pypi]
username = __token__
password = pypi-YOUR-REAL-PYPI-TOKEN

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-YOUR-TESTPYPI-TOKEN
```

The username is the literal string `__token__` for both — that is how twine is
told the password is a token rather than a password.

**That file holds two credentials in plain text.** It belongs in your home
directory and nowhere else. Do not put it in a project folder, and do not commit
it.

---

## Releasing

### 1. Decide the version

**`switchboard/__init__.py` holds it** — `pyproject.toml` reads it from there,
so there is one place to change. Three files used to hold three different
answers and nothing noticed until `release-check` compared them.

The convention:

| Change | Bump | Example |
|---|---|---|
| Bug fix, nothing else changes | patch | `0.4.0` → `0.4.1` |
| New feature, old usage still works | minor | `0.4.0` → `0.5.0` |
| Something that breaks existing setups | major | `0.4.0` → `1.0.0` |

A **new database migration is not** a breaking change — `switchboard db upgrade`
handles it and never destroys data. **Renaming a setting or a command is**, and
belongs in the release notes with the old name and the new one.

### 2. Build

```powershell
# Old builds must go, or twine will happily upload them again
Remove-Item -Recurse -Force dist -ErrorAction SilentlyContinue

python -m build
```

Two files appear in `dist/`: a `.whl` (what pip installs) and a `.tar.gz` (the
source, for anyone who needs to build it themselves).

### 3. Check what you are about to publish

```powershell
twine check dist/*
```

That validates the metadata and the README rendering. Then look inside the
wheel, because a missing data file installs cleanly and fails on the user's
machine:

```powershell
python -c "import zipfile,glob; print('\n'.join(zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist()))"
```

Confirm all of these are present:

- `switchboard/alembic.ini` and `switchboard/migrations/versions/*.py` — without
  them an installed copy cannot migrate its own database and **cannot start**
- `switchboard/guardrail_samples.jsonl` — `guardrails calibrate` reads it
- `switchboard/router.joblib`, if you built one with `bench train-broad --save`

And confirm `eval/` and `tests/` are **absent**. A server should not carry the
research harness.

### 4. Publish to TestPyPI first

TestPyPI is a separate index that exists precisely so a first attempt cannot
hurt anybody.

```powershell
twine upload --repository testpypi dist/*
```

Username `__token__`, password your **TestPyPI** token - not the pypi.org
one. If this returns `403 Forbidden`, see
[When an upload is rejected](#when-an-upload-is-rejected).

Then install it into a **fresh** virtual environment — not the one you develop
in, which already has every dependency and will hide a missing one:

```powershell
python -m venv C:\Temp\sbtest
C:\Temp\sbtest\Scripts\Activate.ps1

pip install --index-url https://test.pypi.org/simple/ `
            --extra-index-url https://pypi.org/simple/ `
            switchboard-router

switchboard where
switchboard init --local-only
switchboard serve
```

The `--extra-index-url` matters: TestPyPI does not mirror real packages, so
without it every dependency fails to resolve.

### 5. Publish for real

```powershell
twine upload dist/*
```

Username `__token__`, password the API token including its `pypi-` prefix.

### 6. Tag it

```powershell
git tag -a v0.4.0 -m "Switchboard 0.4.0"
git push origin v0.4.0
```

A tag is how somebody finds the exact source a released version was built from.

---

## After releasing

Verify the real thing from a clean environment:

```powershell
pip install switchboard-router
switchboard init
```

Then bump `switchboard/__init__.py` to the next patch version and commit. Working on a
version number that is already published invites building something and
uploading it under a name that means something else.

---

## If you get it wrong

**You cannot replace a published version.** Yank it and publish a fix:

```powershell
# On pypi.org: Manage → Releases → Yank
# Then:
#   bump the version in switchboard/__init__.py
#   python -m build
#   twine upload dist/*
```

Yanking hides a release from new installs while leaving it available to anyone
who pinned that exact version, so it does not break somebody's running build.

---

## What ships, and what does not

| Ships | Does not |
|---|---|
| `switchboard/` — the server and CLI | `eval/` — the research harness |
| migrations and `alembic.ini` | `tests/` |
| `guardrail_samples.jsonl` | benchmark datasets (6.6 GB, not ours to redistribute) |
| a trained `router.joblib`, if built | `data/` — anybody's ledger |

Runtime dependencies are about a dozen packages. The research stack (pandas,
matplotlib, datasets, fastembed) is an optional extra:

```powershell
pip install "switchboard-router[research]"
```

Those are for *measuring* routing, never for serving it. In the required list
they would land on every user's machine forever.

---

## When an upload is rejected

### `403 Forbidden`

Almost always the wrong token, and almost always because **TestPyPI and PyPI
are separate sites**. Check, in this order:

1. **Is this a TestPyPI token?** A `pypi.org` token returns 403 on
   `test.pypi.org` and vice versa. They look identical. Get one from
   [test.pypi.org/manage/account/token](https://test.pypi.org/manage/account/token/).
2. **Did you paste the whole thing?** The token includes its `pypi-` prefix.
   Terminals do not echo it, so a truncated paste is invisible.
3. **Is the username `__token__`?** Literally that, underscores included. If
   twine only prompted for a token it is already doing this for you.
4. **Is 2FA enabled on that account?** Both indexes require it before they will
   accept an upload.
5. **Is the token scoped to a different project?** A project-scoped token cannot
   create a *new* project. The first upload needs an account-wide one.

The file uploading to 100% before the error is normal and means nothing —
authorisation is checked after the bytes arrive.

### `400 File already exists`

That version number is used and **cannot be reused**, on either index. Bump the
version in `switchboard/__init__.py`, rebuild, upload again.

### `403 The user is not allowed to upload to project`

Somebody else owns that name. `switchboard` is taken on PyPI by an unrelated
project — which is why this one publishes as `switchboard-router`. Both
`switchboard-router` and `switchboard` were free on TestPyPI at the time of
writing, and `switchboard-router` was free on PyPI.

Check before assuming:

```powershell
python -c "import urllib.request; urllib.request.urlopen('https://pypi.org/pypi/switchboard-router/json')"
```

A `404` means available. Anything else means taken.

### The install from TestPyPI cannot find dependencies

```powershell
pip install --index-url https://test.pypi.org/simple/ `
            --extra-index-url https://pypi.org/simple/ `
            switchboard-router
```

`--extra-index-url` is not optional. TestPyPI does not mirror real packages, so
without it fastapi, sqlalchemy and everything else fail to resolve.
