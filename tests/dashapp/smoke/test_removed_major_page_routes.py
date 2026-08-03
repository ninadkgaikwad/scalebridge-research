"""Smoke tests for removed major-page routes."""

from scalebridge.dashapp.layout.routing import router


def test_removed_routes_are_not_registered():
    assert "/artifact-lineage" not in router._ROUTES
    assert "/validation-center" not in router._ROUTES


def test_removed_page_builders_are_not_imported():
    assert not hasattr(router, "build_artifact_lineage_page")
    assert not hasattr(router, "build_validation_center_page")


def test_retained_routes_still_exist():
    expected = {
        "/",
        "/campaigns",
        "/data-pipeline",
        "/thermal-modeling",
        "/model-catalog",
        "/simulators",
        "/results-explorer",
        "/publication-studio",
        "/settings",
    }
    assert expected == set(router._ROUTES)
