"""Closing the loop: real traffic -> feedback -> a router that fits it.

Three properties are load-bearing here.

1. **A label is never invented.** An unrated request is not a bad one, and no
   code path may turn silence into a training signal.
2. **Training refuses when the data cannot support it.** Thirty positive
   ratings and nothing else must not become a classifier that answers "yes" to
   everything and then wins every routing decision.
3. **Feedback is scoped to the caller.** Ratings are training data, so being
   able to rate someone else's traffic is being able to steer their router.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
import pytest
from sqlalchemy import select

from switchboard import training
from switchboard.ledger.models import RequestLog
from switchboard.ledger.service import UnknownRequest

#: Distinguishes "build the default prompt" from "this row genuinely has no
#: stored text", which is a case the training code must handle.
_KEEP = object()


class Row:
    """A ledger row reduced to what the training code reads."""

    def __init__(
        self,
        prompt: str = "how do I reverse a list in python",
        model: str = "small",
        feedback: str | None = "good",
        created_at: datetime | None = None,
        raw_json: str | None = _KEEP,
    ) -> None:
        self.prompt_json = (
            json.dumps([{"role": "user", "content": prompt}])
            if raw_json is _KEEP
            else raw_json
        )
        self.served_model = model
        self.feedback = feedback
        self.created_at = created_at or datetime(2026, 8, 1)


# --- Reading a prompt back --------------------------------------------------


def test_the_users_text_is_recovered() -> None:
    assert training.prompt_from_json(
        json.dumps([{"role": "user", "content": "hello"}])
    ) == "hello"


def test_the_system_prompt_is_left_out() -> None:
    """It is written by the application, not the person. Identical across every
    example, it would dominate the vocabulary and teach the classifier nothing.
    """
    text = training.prompt_from_json(
        json.dumps(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "fix this bug"},
            ]
        )
    )
    assert text == "fix this bug"


@pytest.mark.parametrize("value", [None, "", "not json", "[1,2,3]", "{}"])
def test_unreadable_prompts_return_empty_rather_than_raising(value) -> None:
    assert training.prompt_from_json(value) == ""


# --- Collecting -------------------------------------------------------------


def test_only_rated_requests_become_examples() -> None:
    """THE rule. An unrated request is not a bad one."""
    examples = training.collect(
        [Row(feedback="good"), Row(feedback=None), Row(feedback="bad")]
    )
    assert len(examples) == 2
    assert {e.correct for e in examples} == {True, False}


def test_requests_without_stored_text_are_dropped() -> None:
    """Training on them as empty strings would teach every classifier that a
    blank question is a normal question."""
    assert training.collect([Row(raw_json=None)]) == []


def test_an_unexpected_rating_is_ignored() -> None:
    assert training.collect([Row(feedback="maybe")]) == []


# --- Readiness --------------------------------------------------------------


def _rows(model: str, good: int, bad: int) -> list[Row]:
    return [Row(prompt=f"{model} good {i}", model=model) for i in range(good)] + [
        Row(prompt=f"{model} bad {i}", model=model, feedback="bad")
        for i in range(bad)
    ]


def test_a_model_with_enough_balanced_ratings_is_usable() -> None:
    readiness = training.assess(_rows("small", 25, 10), [("small", 100)], True)
    assert readiness.models[0].usable


def test_too_few_ratings_is_not_usable() -> None:
    readiness = training.assess(_rows("small", 5, 5), [("small", 100)], True)
    entry = readiness.models[0]
    assert not entry.usable
    assert "20 more rated requests" in entry.blocker()


def test_one_sided_ratings_are_refused_however_many_there_are() -> None:
    """Forty ratings that all say "good" describe a model nobody has seen fail,
    not a model that does not fail. Fitted on them, the classifier answers
    "yes" to everything and wins every routing decision."""
    readiness = training.assess(_rows("small", 40, 0), [("small", 100)], True)
    entry = readiness.models[0]
    assert not entry.usable
    assert "rated bad" in entry.blocker()


def test_two_good_models_are_needed() -> None:
    """A router with one choice is not a router."""
    one = training.assess(_rows("small", 25, 10), [("small", 100)], True)
    assert not one.can_train

    two = training.assess(
        _rows("small", 25, 10) + _rows("big", 25, 10),
        [("small", 100), ("big", 100)],
        True,
    )
    assert two.can_train


def test_prompt_storage_being_off_is_reported_as_the_blocker() -> None:
    """The most common reason someone cannot train, and the least obvious."""
    readiness = training.assess([], [], store_prompts=False)
    assert any("STORE_PROMPTS" in b for b in readiness.blockers())


def test_having_no_ratings_at_all_explains_how_to_get_them() -> None:
    blockers = " ".join(training.assess([], [("small", 50)], True).blockers())
    assert "X-Switchboard-Request-Id" in blockers
    assert "/v1/feedback" in blockers


def test_ratings_without_prompt_text_are_still_counted_as_ratings() -> None:
    """Otherwise the report tells somebody who has been rating for a fortnight
    that nobody has rated anything."""
    readiness = training.assess(
        [Row(raw_json=None) for _ in range(4)], [("small", 10)], True
    )
    assert readiness.total_rated == 4
    assert readiness.with_prompt_text == 0
    assert not readiness.can_train


def test_coverage_does_not_divide_by_zero() -> None:
    assert training.assess([], [], True).coverage_pct == 0.0


def test_the_period_covered_is_reported() -> None:
    rows = [
        Row(created_at=datetime(2026, 8, 1)),
        Row(created_at=datetime(2026, 8, 20)),
    ]
    assert training.assess(rows, [("small", 2)], True).period == (
        "2026-08-01 to 2026-08-20"
    )


# --- Splitting --------------------------------------------------------------


def test_the_split_is_by_prompt_not_by_row() -> None:
    """The same question asked twice - by a retrying script, or two people
    hitting the same problem - must not land in both halves, or the AUC
    measures memory rather than judgement."""
    examples = training.collect(
        [Row(prompt="same question", model=m) for m in ("small", "big")] * 5
        + [Row(prompt=f"unique {i}") for i in range(20)]
    )
    train, test = training.split(examples, test_size=0.5, seed=0)
    assert not ({e.prompt for e in train} & {e.prompt for e in test})


def test_the_split_is_deterministic() -> None:
    examples = training.collect([Row(prompt=f"q{i}") for i in range(30)])
    first = training.split(examples, seed=7)[1]
    second = training.split(examples, seed=7)[1]
    assert [e.prompt for e in first] == [e.prompt for e in second]


# --- Training ---------------------------------------------------------------


def test_each_classifier_only_sees_its_own_requests() -> None:
    """Live data is sparse: each request was answered by one model. Flattening
    it into the dense benchmark shape would write a 0 wherever a model was not
    asked, teaching every classifier that questions someone else handled were
    ones it personally got wrong."""
    examples = training.collect(_rows("small", 6, 6) + _rows("big", 4, 4))
    sets = training.training_sets(examples, ["small", "big"])
    assert len(sets["small"][0]) == 12
    assert len(sets["big"][0]) == 8
    assert sets["small"][1].sum() == 6


def test_a_model_with_no_examples_is_absent_rather_than_empty() -> None:
    sets = training.training_sets(training.collect(_rows("small", 6, 6)), ["big"])
    assert sets == {}


def test_a_predictor_is_produced_and_predicts() -> None:
    examples = training.collect(
        [
            Row(prompt=f"simple question about lists {i}", model="small")
            for i in range(20)
        ]
        + [
            Row(
                prompt=f"prove the halting problem is undecidable {i}",
                model="small",
                feedback="bad",
            )
            for i in range(20)
        ]
        + [
            Row(prompt=f"another easy one {i}", model="big") for i in range(20)
        ]
        + [
            Row(prompt=f"very hard maths proof {i}", model="big", feedback="bad")
            for i in range(20)
        ]
    )
    predictor = training.train(examples, ["small", "big"], features="surface")
    assert set(predictor.models) == {"small", "big"}

    probabilities = predictor.predict_one("simple question about lists")
    assert set(probabilities) == {"small", "big"}
    assert all(0.0 <= p <= 1.0 for p in probabilities.values())


def test_a_one_sided_model_is_dropped_not_made_constant() -> None:
    """`ConstantPredictor` is right for benchmark data over thousands of graded
    questions. On live feedback it would hand a model every decision on the
    strength of a handful of positive ratings."""
    examples = training.collect(_rows("small", 20, 20) + _rows("big", 20, 0))
    predictor = training.train(examples, ["small", "big"], features="surface")
    assert predictor.models == ["small"]


def test_scoring_reports_unscorable_rather_than_inventing_a_number() -> None:
    examples = training.collect(_rows("small", 20, 20))
    predictor = training.train(examples, ["small"], features="surface")
    # Held-out set with only one outcome: AUC is undefined, not 0.5.
    scores = training.score(predictor, training.collect(_rows("small", 3, 0)))
    assert np.isnan(scores["small"]["auc"])


# --- The feedback endpoint --------------------------------------------------


def _chat(text: str = "hello") -> dict:
    return {
        "model": "qwen2.5:3b",
        "messages": [{"role": "user", "content": text}],
        "temperature": 0,
    }


def test_a_response_carries_a_request_id(client, auth) -> None:
    response = client.post("/v1/chat/completions", json=_chat(), headers=auth)
    assert response.headers["X-Switchboard-Request-Id"]


def test_a_streamed_response_carries_one_too(client, auth) -> None:
    """Its ledger row is written when the stream ENDS, so an id generated at
    write time could never reach the client that needs it."""
    response = client.post(
        "/v1/chat/completions", json=_chat() | {"stream": True}, headers=auth
    )
    assert response.headers["X-Switchboard-Request-Id"]


def test_the_id_matches_the_ledger_row(client, auth, database) -> None:
    response = client.post("/v1/chat/completions", json=_chat(), headers=auth)
    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.public_id == response.headers["X-Switchboard-Request-Id"]


def test_feedback_is_recorded(client, auth, database) -> None:
    request_id = client.post(
        "/v1/chat/completions", json=_chat(), headers=auth
    ).headers["X-Switchboard-Request-Id"]

    response = client.post(
        "/v1/feedback",
        json={"request_id": request_id, "rating": "bad", "note": "wrong answer"},
        headers=auth,
    )
    assert response.status_code == 200

    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.feedback == "bad"
    assert row.feedback_note == "wrong answer"
    assert row.feedback_at is not None


def test_a_rating_can_be_changed(client, auth, database) -> None:
    """Someone who rereads an answer and changes their mind is giving better
    information than their first reaction, not worse."""
    request_id = client.post(
        "/v1/chat/completions", json=_chat(), headers=auth
    ).headers["X-Switchboard-Request-Id"]

    for rating in ("good", "bad"):
        client.post(
            "/v1/feedback",
            json={"request_id": request_id, "rating": rating},
            headers=auth,
        )

    with database.session() as session:
        row = session.scalar(select(RequestLog))
    assert row.feedback == "bad"


def test_feedback_needs_a_key(client) -> None:
    response = client.post(
        "/v1/feedback", json={"request_id": "x", "rating": "good"}
    )
    assert response.status_code == 401


def test_you_cannot_rate_someone_elses_request(client, auth, ledger) -> None:
    """Ratings are training data. Rating another team's traffic is steering
    their router."""
    request_id = client.post(
        "/v1/chat/completions", json=_chat(), headers=auth
    ).headers["X-Switchboard-Request-Id"]

    mallory = ledger.create_user("mallory", monthly_budget_usd=10.0)
    response = client.post(
        "/v1/feedback",
        json={"request_id": request_id, "rating": "bad"},
        headers={"Authorization": f"Bearer {mallory.api_key}"},
    )
    assert response.status_code == 404


def test_an_unknown_id_looks_the_same_as_someone_elses(client, auth) -> None:
    """Distinguishing them would turn this endpoint into a way to discover
    other people's request ids."""
    response = client.post(
        "/v1/feedback",
        json={"request_id": "nope-does-not-exist", "rating": "good"},
        headers=auth,
    )
    assert response.status_code == 404
    assert response.json()["error"]["message"] == "No such request."


@pytest.mark.parametrize(
    "body",
    [
        {"rating": "good"},
        {"request_id": "x"},
        {"request_id": "x", "rating": "excellent"},
        {"request_id": "", "rating": "good"},
    ],
)
def test_malformed_feedback_is_rejected(client, auth, body) -> None:
    assert client.post("/v1/feedback", json=body, headers=auth).status_code == 400


def test_feedback_is_counted_in_metrics(client, auth) -> None:
    request_id = client.post(
        "/v1/chat/completions", json=_chat(), headers=auth
    ).headers["X-Switchboard-Request-Id"]
    client.post(
        "/v1/feedback",
        json={"request_id": request_id, "rating": "good"},
        headers=auth,
    )
    assert "switchboard_feedback_total" in client.get("/metrics").text


def test_public_ids_are_unguessable(client, auth) -> None:
    """A sequential id would tell anyone who saw one roughly how many requests
    this instance has served, and invite guessing at other people's."""
    ids = {
        client.post("/v1/chat/completions", json=_chat(), headers=auth).headers[
            "X-Switchboard-Request-Id"
        ]
        for _ in range(5)
    }
    assert len(ids) == 5
    assert all(len(value) >= 16 for value in ids)
    assert not any(value.isdigit() for value in ids)


# --- The ledger side --------------------------------------------------------


def test_rated_requests_come_back_from_the_ledger(ledger, alice) -> None:
    from switchboard.ledger.service import STATUS_OK

    entries = [
        ledger.record(
            user_id=1,
            requested_model="auto",
            served_model="qwen2.5:3b",
            prompt_tokens=10,
            completion_tokens=5,
            tokens_estimated=False,
            latency_ms=100,
            status=STATUS_OK,
            messages=[{"role": "user", "content": f"question {i}"}],
            public_id=f"pid-{i}",
        )
        for i in range(3)
    ]
    assert entries

    ledger.record_feedback(1, "pid-0", "good")
    ledger.record_feedback(1, "pid-2", "bad")

    rated = ledger.rated_requests()
    assert {r.public_id for r in rated} == {"pid-0", "pid-2"}


def test_an_invalid_rating_is_refused_at_the_ledger(ledger, alice) -> None:
    from switchboard.ledger.service import LedgerError

    with pytest.raises(LedgerError):
        ledger.record_feedback(1, "pid-0", "five stars")


def test_rating_a_request_that_does_not_exist_raises(ledger, alice) -> None:
    with pytest.raises(UnknownRequest):
        ledger.record_feedback(1, "no-such-id", "good")


def test_served_counts_include_unrated_requests(ledger, alice) -> None:
    """The readiness report needs the denominator: how much traffic exists,
    not just how much was rated."""
    from switchboard.ledger.service import STATUS_OK

    for i in range(4):
        ledger.record(
            user_id=1,
            requested_model="auto",
            served_model="qwen2.5:3b",
            prompt_tokens=1,
            completion_tokens=1,
            tokens_estimated=False,
            latency_ms=1,
            status=STATUS_OK,
            public_id=f"s-{i}",
        )
    assert ledger.served_counts() == [("qwen2.5:3b", 4)]


# --- The artifact must load where the server runs ---------------------------


def test_the_predictor_ships_with_the_server_not_with_the_research_code() -> None:
    """THE bug this phase found. A joblib pickle records the module a class came
    from. While these lived in `eval/` - which the Docker image deliberately
    does not copy - every trained router failed to unpickle inside a container.
    The failure was caught safely, routing switched itself off, and /health said
    "no router artifact loaded" with nothing pointing at the cause.
    """
    from switchboard.routing.features import FeatureExtractor
    from switchboard.routing.predictor import SuccessPredictor

    for cls in (SuccessPredictor, FeatureExtractor):
        assert cls.__module__.startswith("switchboard."), cls.__module__


def test_a_trained_artifact_references_only_shipped_modules(tmp_path) -> None:
    """Checked on a real trained object rather than on the class, because what
    matters is what joblib actually wrote into the file."""
    from switchboard.routing import artifact as artifact_mod

    examples = training.collect(_rows("small", 20, 20) + _rows("big", 20, 20))
    predictor = training.train(examples, ["small", "big"], features="surface")

    path = artifact_mod.save(
        tmp_path / "r.joblib", predictor, artifact_mod.RouterMetadata()
    )
    loaded, _ = artifact_mod.load(path)

    for obj in (loaded, loaded.extractor):
        assert type(obj).__module__.startswith("switchboard."), type(obj).__module__


def test_an_old_artifact_asks_to_be_retrained_rather_than_crashing(tmp_path) -> None:
    import joblib

    from switchboard.routing import artifact as artifact_mod

    path = tmp_path / "old.joblib"
    joblib.dump({"metadata": {"version": 1}, "predictor": object()}, path)

    with pytest.raises(artifact_mod.ArtifactError) as caught:
        artifact_mod.load(path)
    assert "Retrain" in str(caught.value)


def test_live_metadata_says_where_it_came_from() -> None:
    from switchboard.routing.artifact import RouterMetadata

    described = RouterMetadata(
        source="your ledger",
        label_source="live traffic",
        period="2026-08-01 to 2026-08-30",
        models=["a", "b"],
        n_train_questions=412,
        features="tfidf",
    ).describe()
    assert "your ledger" in described
    assert "412 requests" in described
    assert "2026-08-01" in described


def test_benchmark_metadata_still_reads_as_questions() -> None:
    from switchboard.routing.artifact import RouterMetadata

    described = RouterMetadata(
        source="llmrouterbench",
        benchmark="mmlupro",
        models=["a"],
        n_train_questions=1200,
        features="tfidf",
    ).describe()
    assert "1,200 questions" in described


# --- End to end -------------------------------------------------------------


def test_the_whole_loop(client, auth, ledger, database) -> None:
    """Serve, rate, assess, train. The loop that did not close before."""
    ledger._store_prompts = True

    for index in range(64):
        hard = index % 2 == 0
        text = (
            f"prove that this algorithm terminates, case {index}"
            if hard
            else f"format this json snippet {index}"
        )
        model = "qwen2.5:7b" if hard else "qwen2.5:3b"
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": text}],
                # Distinct temperature per request so the response cache does
                # not collapse them into one ledger row.
                "temperature": index / 100,
            },
            headers=auth,
        )
        client.post(
            "/v1/feedback",
            json={
                "request_id": response.headers["X-Switchboard-Request-Id"],
                # Alternate verdicts so both classes are present per model.
                "rating": "good" if index % 4 < 2 else "bad",
            },
            headers=auth,
        )

    readiness = training.assess(
        ledger.rated_requests(), ledger.served_counts(), store_prompts=True
    )
    assert readiness.total_rated == 64
    assert readiness.with_prompt_text == 64
    assert readiness.can_train, readiness.blockers()

    examples = training.collect(ledger.rated_requests())
    predictor = training.train(
        examples, [m.model for m in readiness.usable_models], features="surface"
    )
    assert len(predictor.models) == 2

    probabilities = predictor.predict_one("prove that this terminates")
    assert set(probabilities) == {"qwen2.5:3b", "qwen2.5:7b"}


def test_training_data_reflects_the_period_it_covers(ledger, alice) -> None:
    from switchboard.ledger.service import STATUS_OK

    base = datetime(2026, 8, 1)
    for index in range(3):
        entry = ledger.record(
            user_id=1,
            requested_model="auto",
            served_model="m",
            prompt_tokens=1,
            completion_tokens=1,
            tokens_estimated=False,
            latency_ms=1,
            status=STATUS_OK,
            messages=[{"role": "user", "content": f"q{index}"}],
            public_id=f"p-{index}",
        )
        assert entry is not None
        with ledger._db.session() as session:
            row = session.scalar(
                select(RequestLog).where(RequestLog.public_id == f"p-{index}")
            )
            row.created_at = base + timedelta(days=index)

    for index in range(3):
        ledger.record_feedback(1, f"p-{index}", "good")

    readiness = training.assess(
        ledger.rated_requests(), ledger.served_counts(), store_prompts=True
    )
    assert readiness.period == "2026-08-01 to 2026-08-03"
