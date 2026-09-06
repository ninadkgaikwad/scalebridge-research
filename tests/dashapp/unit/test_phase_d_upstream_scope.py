from scalebridge.dashapp.services.phase_d.upstream_phase_c import selected_aggregation_count


def _context():
    return {
        "aggregation_run_count": 4,
        "aggregation_rows": [
            {"case_id": "c1", "aggregation_id": "identity", "weight_mode": "equal"},
            {"case_id": "c1", "aggregation_id": "identity", "weight_mode": "floor_area"},
            {"case_id": "c1", "aggregation_id": "custom_v1", "weight_mode": "equal"},
            {"case_id": "c2", "aggregation_id": "identity", "weight_mode": "equal"},
        ],
    }


def test_scope_count_matches_runner_filter_intersection():
    assert selected_aggregation_count(_context()) == 4
    assert selected_aggregation_count(_context(), case_ids=["c1"]) == 3
    assert selected_aggregation_count(_context(), case_ids=["c1"], aggregation_ids=["identity"], weight_modes=["equal"]) == 1
    assert selected_aggregation_count(_context(), max_aggregation_runs=2) == 2
