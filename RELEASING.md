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

1. Create an account at [pypi.org](https://pypi.org) and enable two-factor
   authentication. PyPI requires it for publishing.
2. **Account → API tokens → Add API token.** Scope it to this project once the
   project exists; the first upload needs an account-wide token.
3. Install the two tools:

```powershell
pip install build twine
```

`build` makes the distribution files. `twine` uploads them and checks them
first.

---

## Releasing

### 1. Decide the version

`pyproject.toml` holds it. The convention:

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

Then bump `pyproject.toml` to the next patch version and commit. Working on a
version number that is already published invites building something and
uploading it under a name that means something else.

---

## If you get it wrong

**You cannot replace a published version.** Yank it and publish a fix:

```powershell
# On pypi.org: Manage → Releases → Yank
# Then:
#   bump the version in pyproject.toml
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
