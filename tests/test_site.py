"""The public landing page.

Two properties matter here and the rest is decoration:

1. **The page makes no external requests.** It is served by an instance that
   may have no internet, and no third party should learn who visited.
2. **The numbers on it match docs/RESULTS.md.** A landing page that drifts
   away from the measurements is the exact failure this project keeps
   designing against, and it is the easiest one to commit by accident.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from switchboard.site import (
    BASELINE_ACCURACY,
    MEASURED_ACCURACY,
    MEASURED_SAVING_PCT,
    SiteContext,
    render,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def page() -> str:
    return render(SiteContext(version="0.3.0"))


# --- The two rules ----------------------------------------------------------


def test_the_page_makes_no_external_requests(page: str) -> None:
    """No CDN, no font service, no analytics. It has to render on a machine
    with no internet, and nobody outside should learn who looked at it."""
    for forbidden in (
        "cdn.",
        "googleapis",
        "unpkg",
        "jsdelivr",
        "analytics",
        "<script src",
        "<link rel=\"stylesheet\"",
    ):
        assert forbidden not in page, forbidden


def test_only_relative_and_repository_links_are_used(page: str) -> None:
    """Every absolute link should point at the project's own repository. A
    stray third-party link is how a tracking pixel gets in."""
    import re

    for url in re.findall(r'href="(https?://[^"]+)"', page):
        assert "github.com" in url, url


def test_the_headline_numbers_match_the_results_document(page: str) -> None:
    """THE test. If the measured figures change, this fails until the page is
    updated - rather than the page quietly advertising a number nobody can
    reproduce."""
    results = (PROJECT_ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    for value in (f"{MEASURED_ACCURACY}%", f"{BASELINE_ACCURACY}%"):
        assert value in results, f"{value} is on the site but not in RESULTS.md"
        assert value in page


def test_the_measured_costs_appear_on_both(page: str) -> None:
    results = (PROJECT_ROOT / "docs" / "RESULTS.md").read_text(encoding="utf-8")
    for value in ("$6.42", "$15.07"):
        assert value in results
        assert value in page


def test_the_saving_is_consistent_with_the_costs() -> None:
    """57% is not a slogan; it is 1 - 6.42/15.07. Pinned so the marketing
    figure cannot drift away from the arithmetic behind it."""
    implied = 100 * (1 - 6.42 / 15.07)
    assert abs(implied - MEASURED_SAVING_PCT) < 1.0


# --- Honesty ----------------------------------------------------------------


def test_the_limitations_section_is_present(page: str) -> None:
    """A product whose argument is "we measure honestly" cannot have a landing
    page that only lists wins."""
    assert "What it does not do" in page
    assert "does not transfer to short chat prompts" in page
    assert "untested against live APIs" in page


def test_the_calculator_says_it_is_an_illustration(page: str) -> None:
    """A savings calculator that reads as a quote is a lie with a text box."""
    assert "not a quote" in page
    assert "shadow mode" in page


def test_simulated_pricing_is_disclosed(page: str) -> None:
    assert "simulated" in page.lower()


# --- Behaviour --------------------------------------------------------------


def test_demo_mode_is_announced_when_nothing_is_connected() -> None:
    """Better to say "no provider here" than to let a visitor discover it by
    getting a 503 from an endpoint they expected to work."""
    assert "This instance is a demo" in render(
        SiteContext(version="0.3.0", demo_mode=True)
    )


def test_a_configured_instance_does_not_claim_to_be_a_demo() -> None:
    page = render(SiteContext(version="0.3.0", demo_mode=False))
    assert "This instance is a demo" not in page


def test_the_version_is_shown(page: str) -> None:
    assert "0.3.0" in page


def test_the_page_is_valid_enough_to_parse() -> None:
    """Cheap structural check: every section the nav links to must exist, or
    the navigation silently does nothing."""
    page = render(SiteContext(version="0.3.0"))
    for anchor in ("how", "results", "providers", "limits"):
        assert f'id="{anchor}"' in page
        assert f'href="#{anchor}"' in page


def test_the_calculator_script_is_wired_to_the_real_number(page: str) -> None:
    """The placeholder must be substituted. Left in, the calculator would
    silently compute NaN and show a dash forever."""
    assert "SAVING_PCT / 100" not in page
    assert f"var saving = {MEASURED_SAVING_PCT} / 100" in page


# --- Through the API --------------------------------------------------------


def test_the_root_url_serves_the_page(client) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Switchboard" in response.text


def test_the_landing_page_needs_no_api_key(client) -> None:
    """It is a public website. Requiring a key would make it unreachable to
    exactly the people it is written for."""
    assert client.get("/").status_code == 200


# --- Deployment configuration -----------------------------------------------


def test_render_health_check_points_at_liveness() -> None:
    """Pointed at /health/ready, Render would restart the service forever: a
    deployed instance normally has no provider, so readiness is 503 by design.
    This is the single most likely way this deployment silently fails."""
    render_yaml = (PROJECT_ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "healthCheckPath: /health/live" in render_yaml
    assert "healthCheckPath: /health/ready" not in render_yaml


def test_render_binds_to_all_interfaces() -> None:
    """The default 127.0.0.1 is right on a laptop and unreachable in a
    container - and the failure looks like a startup hang, not a config error."""
    render_yaml = (PROJECT_ROOT / "render.yaml").read_text(encoding="utf-8")
    assert "SWITCHBOARD_HOST" in render_yaml
    assert "0.0.0.0" in render_yaml


def test_no_api_keys_are_committed_in_the_blueprint() -> None:
    """render.yaml is in git. A key here would be public forever."""
    render_yaml = (PROJECT_ROOT / "render.yaml").read_text(encoding="utf-8")
    for prefix in ("sk-", "sk-or-", "sk-ant-", "AIza"):
        assert prefix not in render_yaml


def test_the_entrypoint_honours_the_platform_port() -> None:
    """Render, Railway, Fly and Heroku all assign a port through PORT. Ignore
    it and the platform reports a startup timeout with no explanation."""
    entrypoint = (PROJECT_ROOT / "docker" / "entrypoint.sh").read_text(
        encoding="utf-8"
    )
    assert "${PORT:-" in entrypoint
    assert "exec python -m switchboard serve" in entrypoint


def test_the_deployment_guide_exists_and_warns_about_the_free_disk() -> None:
    """Losing every user and every spending record on a redeploy is not
    something anyone should discover for themselves."""
    # Whitespace is collapsed first: the guide is wrapped prose, and a test
    # that breaks when a sentence rewraps is a test people learn to ignore.
    guide = " ".join(
        (PROJECT_ROOT / "DEPLOY.md").read_text(encoding="utf-8").split()
    )
    assert "no persistent disk" in guide
    assert "PostgreSQL" in guide
