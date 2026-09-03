"""Putting a password on the spend page.

`/dashboard` shows spend per developer, by name. It shows no prompt text and no
API keys — that is deliberate and stays true — but "who is spending what" is not
something to hand to anyone who finds the link once the service is on a public
URL.

Open by default, because that is the historical behaviour and right on a laptop.
Set a password before deploying.

HTTP Basic, because a browser prompts for it natively: no login page, no cookie,
no session store, nothing to get wrong. The trade-off is that the password
travels on every request, which is fine behind the HTTPS every deployment path
in DEPLOY.md terminates in front of the app.
"""

from __future__ import annotations

import base64

import pytest


def basic(password: str, user: str = "anyone") -> dict:
    encoded = base64.b64encode(f"{user}:{password}".encode()).decode()
    return {"Authorization": f"Basic {encoded}"}


@pytest.fixture
def locked(client, monkeypatch):
    from switchboard.config import settings

    monkeypatch.setattr(settings, "dashboard_password", "hunter2")
    return client


# --- Open by default --------------------------------------------------------


def test_the_dashboard_is_open_when_no_password_is_set(client) -> None:
    """A laptop install must keep working exactly as before."""
    assert client.get("/dashboard").status_code == 200


def test_health_reports_that_it_is_unprotected(client) -> None:
    """An operator about to deploy should be able to see this without guessing."""
    assert client.get("/health").json()["dashboard_protected"] is False


# --- With a password --------------------------------------------------------


def test_no_credentials_is_refused(locked) -> None:
    assert locked.get("/dashboard").status_code == 401


def test_the_browser_is_told_to_ask(locked) -> None:
    """WWW-Authenticate is what makes the browser show its own password box,
    which is why there is no login page to build."""
    response = locked.get("/dashboard")
    assert "Basic" in response.headers["WWW-Authenticate"]
    assert "Switchboard" in response.headers["WWW-Authenticate"]


def test_the_right_password_gets_in(locked) -> None:
    response = locked.get("/dashboard", headers=basic("hunter2"))
    assert response.status_code == 200
    assert "Switchboard" in response.text


def test_the_wrong_password_does_not(locked) -> None:
    assert locked.get("/dashboard", headers=basic("hunter3")).status_code == 401


def test_any_username_is_accepted(locked) -> None:
    """Only the password is checked. A username to remember would be a second
    secret protecting nothing."""
    for user in ("admin", "alice", ""):
        response = locked.get("/dashboard", headers=basic("hunter2", user))
        assert response.status_code == 200


def test_health_reports_that_it_is_protected(locked) -> None:
    assert locked.get("/health").json()["dashboard_protected"] is True


# --- Things that must not crash it ------------------------------------------


@pytest.mark.parametrize(
    "header",
    [
        "Basic",
        "Basic !!!not-base64!!!",
        "Bearer hunter2",
        "hunter2",
        "",
        "Basic " + base64.b64encode(b"\xff\xfe").decode(),
    ],
)
def test_malformed_credentials_are_refused_not_fatal(locked, header: str) -> None:
    """A 500 here would be a denial of service on the page, and would leak a
    stack trace to whoever sent the malformed header."""
    response = locked.get("/dashboard", headers={"Authorization": header})
    assert response.status_code == 401


def test_a_password_with_a_colon_in_it_works(client, monkeypatch) -> None:
    """Basic auth joins user and password with a colon, so a password
    containing one is the obvious way to split it in the wrong place."""
    from switchboard.config import settings

    monkeypatch.setattr(settings, "dashboard_password", "a:b:c")
    assert client.get("/dashboard", headers=basic("a:b:c")).status_code == 200


# --- What stays open --------------------------------------------------------


def test_metrics_stays_open(locked) -> None:
    """It carries no prompt text, no keys and no user names, and a scrape
    endpoint that needs credentials is one nobody gets round to configuring."""
    assert locked.get("/metrics").status_code == 200


def test_the_health_endpoints_stay_open(locked) -> None:
    """A health check that needs credentials is useless to a load balancer."""
    assert locked.get("/health/live").status_code == 200
    assert locked.get("/health").status_code == 200


def test_the_landing_page_stays_open(locked) -> None:
    """It is a public website. Requiring a password would make it unreachable
    to exactly the people it is written for."""
    assert locked.get("/").status_code == 200


def test_the_api_is_unaffected(locked, auth, provider) -> None:
    """The dashboard password is for humans with browsers. API keys are the
    machine path and must not start needing a second credential."""
    response = locked.post(
        "/v1/chat/completions",
        json={
            "model": "auto",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
        },
        headers=auth,
    )
    assert response.status_code == 200


def test_the_password_is_never_rendered_into_the_page(locked) -> None:
    page = locked.get("/dashboard", headers=basic("hunter2")).text
    assert "hunter2" not in page
