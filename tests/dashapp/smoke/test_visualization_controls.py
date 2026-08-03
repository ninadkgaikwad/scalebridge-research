"""Smoke tests for retained Visualization controls."""

from dash.development.base_component import Component

from scalebridge.dashapp.pages.settings.visualization.page import build_layout


EXPECTED_IDS = {
    "settings-visual-theme",
    "settings-visual-display-units",
    "settings-visual-publication-units",
    "settings-visual-figure-width",
    "settings-visual-png-dpi",
    "settings-visual-vector-formats",
    "settings-visual-decimals",
    "settings-visual-page-size",
}


def _walk(component):
    """Yield every Dash component below the supplied root."""
    if isinstance(component, Component):
        yield component
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                yield from _walk(child)
        elif children is not None:
            yield from _walk(children)


def _string_ids(component):
    """Collect exact string IDs while ignoring pattern-matching dictionary IDs."""
    return {
        component_id
        for item in _walk(component)
        for component_id in (getattr(item, "id", None),)
        if isinstance(component_id, str) and component_id
    }


def test_visualization_controls_build_and_persist():
    ids = _string_ids(build_layout())
    assert EXPECTED_IDS.issubset(ids)


def test_visualization_page_has_no_badges():
    assert all(
        component.__class__.__name__ != "Badge"
        for component in _walk(build_layout())
    )
