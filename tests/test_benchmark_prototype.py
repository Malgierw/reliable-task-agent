from __future__ import annotations

import inspect

import pytest

from benchmarks.adapters import langgraph_adapter, rta_adapter
from benchmarks.aggregate import aggregate_results
from benchmarks.adapters.rta_adapter import run_rta_worker
from benchmarks.business_store import BenchmarkBusinessStore
from benchmarks.harness import (
    CONFIGURATIONS,
    LABELS,
    PAYLOAD,
    RANKED_SCENARIOS,
    build_manifest,
    run_trial,
    validate_fairness,
)
from benchmarks.identity import expected_effect_identity


def test_shared_business_store_distinguishes_invocations(
    tmp_path,
) -> None:
    store = BenchmarkBusinessStore(tmp_path / "business.sqlite3")
    payload = {"title": "A", "description": "B"}

    for phase in ("initial", "recovery"):
        store.create_ticket(
            payload=payload,
            logical_action_id="logical",
            idempotency_key="stable-key",
            write_mode="idempotent",
            worker_phase=phase,
            process_id=1,
        )

    assert len(store.ticket_rows("logical")) == 1
    assert store.handler_invocation_count("logical") == 2


def test_benchmark_identity_matches_integrated_rta(
    tmp_path,
) -> None:
    manifest = build_manifest(
        "reliable_task_agent_effect_boundary",
        "F0",
        0,
    )
    _, expected_key = expected_effect_identity(
        manifest["run_id"],
        manifest["tool_call_id"],
    )

    result = run_rta_worker(manifest, tmp_path, "initial")

    assert expected_key == manifest["expected_idempotency_key"]
    assert result["actual_idempotency_key"] == expected_key
    assert result["effect_state"] == "COMMITTED"
    assert result["status"] == "SUCCESS"


def test_b_and_c_observe_the_same_neutral_idempotency_key(
    tmp_path,
) -> None:
    b_result = run_trial(
        tmp_path / "b",
        "langgraph_idempotent",
        "F0",
    )
    c_result = run_trial(
        tmp_path / "c",
        "reliable_task_agent_effect_boundary",
        "F0",
    )

    assert b_result["observed_idempotency_keys"] == c_result[
        "observed_idempotency_keys"
    ]
    assert b_result["observed_idempotency_keys"] == [
        b_result["manifest"]["expected_idempotency_key"]
    ]


def test_rta_adapter_uses_explicit_run_id_without_uuid_monkeypatch() -> None:
    source = inspect.getsource(rta_adapter)

    assert "run_id=manifest[\"run_id\"]" in source
    assert "trace_module" not in source
    assert "uuid4" not in source


def test_configuration_labels_and_fair_manifests() -> None:
    assert LABELS["langgraph_checkpoint_only"] == (
        "checkpoint-only experimental baseline, not recommended "
        "LangGraph production practice"
    )
    manifests = [
        build_manifest(configuration, scenario, 0)
        for configuration in CONFIGURATIONS
        for scenario in RANKED_SCENARIOS
    ]

    validate_fairness(manifests)
    for scenario in RANKED_SCENARIOS:
        group = [item for item in manifests if item["scenario"] == scenario]
        assert {tuple(sorted(item["payload"].items())) for item in group} == {
            tuple(sorted(PAYLOAD.items()))
        }
        assert len({item["logical_action_id"] for item in group}) == 1
        assert len({item["run_id"] for item in group}) == 1
        assert len({item["expected_idempotency_key"] for item in group}) == 1
        assert {item["uses_external_model_api"] for item in group} == {False}


def test_both_adapters_use_shared_business_schema() -> None:
    assert "BenchmarkBusinessStore" in inspect.getsource(langgraph_adapter)
    assert "BenchmarkBusinessStore" in inspect.getsource(rta_adapter)


@pytest.mark.parametrize("scenario", ["F1", "F2", "F3"])
def test_rta_faults_use_real_process_kill(tmp_path, scenario) -> None:
    result = run_trial(
        tmp_path / scenario,
        "reliable_task_agent_effect_boundary",
        scenario,
    )

    assert result["initial_process"]["kill_requested"] is True
    assert result["fault_marker"]["pid"] == result["initial_process"][
        "terminated_pid"
    ]
    assert result["metrics"]["recovery_success"] is True
    assert result["metrics"]["business_effect_count"] == 1


def test_rta_ambiguous_handler_failure_recovers_by_reconciliation(
    tmp_path,
) -> None:
    result = run_trial(
        tmp_path / "F4",
        "reliable_task_agent_effect_boundary",
        "F4",
    )

    assert result["initial_result"]["status"] == "ERROR"
    assert result["metrics"]["handler_invocation_count"] == 1
    assert result["metrics"]["reconciliation_count"] == 1
    assert result["metrics"]["terminal_effect_state"] == "COMMITTED"
    assert result["metrics"]["recovery_success"] is True


def test_rta_f5_reconciliation_exception_fails_closed(tmp_path) -> None:
    result = run_trial(
        tmp_path / "F5",
        "reliable_task_agent_effect_boundary",
        "F5",
    )

    assert result["metrics"]["terminal_effect_state"] == "UNKNOWN"
    assert result["metrics"]["checkpoint_status"] == "failed"
    assert result["metrics"]["reconciliation_count"] == 1
    assert result["metrics"]["fail_closed_correctly"] is True
    assert result["metrics"]["recovery_success"] is False


def test_trial_directories_are_isolated_and_not_reused(tmp_path) -> None:
    root = tmp_path / "isolated"
    run_trial(root, "langgraph_idempotent", "F0")

    with pytest.raises(FileExistsError):
        run_trial(root, "langgraph_idempotent", "F0")


def test_aggregation_has_one_row_per_cell_and_raw_counts(tmp_path) -> None:
    result = run_trial(
        tmp_path / "aggregate",
        "langgraph_idempotent",
        "F0",
    )

    summary = aggregate_results([result])

    assert len(summary) == 1
    assert summary[0]["trials"] == 1
    assert summary[0]["successful_trials"] == 1
    assert summary[0]["business_effect_counts"] == [1]
    assert summary[0]["contributing_trial_ids"] == [result["trial_id"]]
