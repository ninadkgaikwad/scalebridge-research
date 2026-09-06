from scalebridge.dashapp.pages.data_pipeline.phase_c_heat_input.results.page import build_layout


def _walk(component):
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, (list, tuple)):
        for child in children:
            yield from _walk(child)
    else:
        yield from _walk(children)


def _ids(layout):
    return {
        component.id: component
        for component in _walk(layout)
        if isinstance(getattr(component, "id", None), str)
    }


def test_results_page_exposes_model_trajectory_metrics_and_export_contract():
    ids = _ids(build_layout())
    required = {
        "phase-c-results-run",
        "phase-c-results-stage-summary",
        "phase-c-results-availability-summary",
        "phase-c-results-building",
        "phase-c-results-weather",
        "phase-c-results-case",
        "phase-c-results-aggregation",
        "phase-c-results-weight",
        "phase-c-results-zone",
        "phase-c-results-model",
        "phase-c-results-estimator",
        "phase-c-results-dataset-resolution",
        "phase-c-results-load-dataset",
        "phase-c-results-dataset-graph",
        "phase-c-results-dataset-legend",
        "phase-c-results-split",
        "phase-c-results-evaluation-mode",
        "phase-c-results-load-evaluation",
        "phase-c-results-evaluation-graph",
        "phase-c-results-evaluation-legend",
        "phase-c-results-metrics-table",
        "phase-c-results-model-table",
        "phase-c-results-building-phvac-table",
        "phase-c-results-annual-zone",
        "phase-c-results-annual-components",
        "phase-c-results-load-annual",
        "phase-c-results-annual-graph",
        "phase-c-results-annual-legend",
        "phase-c-results-comparison-kind",
        "phase-c-results-comparison-graph",
        "phase-c-results-comparison-legend",
        "phase-c-results-load-inventories",
        "phase-c-results-validation-stage",
        "phase-c-results-load-validation",
        "phase-c-results-dataset-download-format",
        "phase-c-results-dataset-download-plot-data",
        "phase-c-results-dataset-plot-download",
        "phase-c-results-evaluation-download-format",
        "phase-c-results-evaluation-download-plot-data",
        "phase-c-results-evaluation-plot-download",
        "phase-c-results-annual-download-format",
        "phase-c-results-annual-download-plot-data",
        "phase-c-results-annual-plot-download",
        "phase-c-results-comparison-download-format",
        "phase-c-results-comparison-download-plot-data",
        "phase-c-results-comparison-plot-download",
        "phase-c-results-download-summary",
        "phase-c-results-download-model-bundle",
    }
    assert required.issubset(ids)
    assert "phase-c-results-climate" not in ids


def test_result_context_filters_are_multiselect_without_climate_filter():
    ids = _ids(build_layout())
    for component_id in (
        "phase-c-results-building",
        "phase-c-results-weather",
        "phase-c-results-case",
        "phase-c-results-aggregation",
        "phase-c-results-weight",
        "phase-c-results-zone",
        "phase-c-results-model",
        "phase-c-results-estimator",
    ):
        assert ids[component_id].multi is True
    assert "phase-c-results-climate" not in ids


def test_all_main_result_graphs_use_75_25_plot_legend_layout():
    layout = build_layout()
    columns = [component for component in _walk(layout) if component.__class__.__name__ == "Col"]
    for graph_id, legend_id in (
        ("phase-c-results-dataset-graph", "phase-c-results-dataset-legend"),
        ("phase-c-results-evaluation-graph", "phase-c-results-evaluation-legend"),
        ("phase-c-results-annual-graph", "phase-c-results-annual-legend"),
        ("phase-c-results-comparison-graph", "phase-c-results-comparison-legend"),
    ):
        graph_col = next(
            col
            for col in columns
            if any(getattr(child, "id", None) == graph_id for child in _walk(col))
        )
        legend_col = next(
            col
            for col in columns
            if any(getattr(child, "id", None) == legend_id for child in _walk(col))
        )
        assert graph_col.width == 9
        assert legend_col.width == 3


def test_results_page_keeps_metrics_and_makes_time_trajectory_default():
    ids = _ids(build_layout())
    evaluation = ids["phase-c-results-evaluation-plot-kind"]
    assert evaluation.value == "time_series"
    assert {option["value"] for option in evaluation.options} == {
        "time_series",
        "scatter",
        "residual_time_series",
        "residual_distribution",
    }
    comparison_values = {
        option["value"] for option in ids["phase-c-results-comparison-kind"].options
    }
    assert comparison_values == {
        "estimator_metric",
        "coefficient",
        "error_context",
        "split_coverage",
        "availability",
        "building_phvac",
    }
    assert "dataset_preview" not in comparison_values


def test_each_main_plot_has_local_csv_parquet_download_controls():
    ids = _ids(build_layout())
    for prefix in ("dataset", "evaluation", "annual", "comparison"):
        format_component = ids[f"phase-c-results-{prefix}-download-format"]
        assert {option["value"] for option in format_component.options} == {"csv", "parquet"}
        assert format_component.value == "csv"
        assert f"phase-c-results-{prefix}-download-plot-data" in ids
        assert f"phase-c-results-{prefix}-plot-download" in ids


def test_general_artifact_downloads_remain_separate_from_plot_downloads():
    ids = _ids(build_layout())
    assert "phase-c-results-download-summary" in ids
    assert "phase-c-results-download-model-bundle" in ids
    assert "phase-c-results-artifact-export-message" in ids
