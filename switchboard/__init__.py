"""Switchboard - a self-hostable AI model router."""

#: THE single source of the version number.
#
# pyproject.toml reads it from here (`dynamic = ["version"]`) and api.py
# imports it, so there is one place to change and nothing to keep in sync.
# Before this, three files each held their own answer - 0.1.0, 0.3.0 and
# 0.4.0 - and nothing noticed until `switchboard release-check` compared
# them. A wrong version in a published package is not fixable: PyPI never
# lets a version number be reused.
__version__ = "0.4.0"
