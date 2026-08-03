"""Smoke tests for refined Settings UI behavior."""

from dash.development.base_component import Component

from scalebridge.dashapp.pages.settings.environments.page import build_layout as build_environment
from scalebridge.dashapp.pages.settings.help.page import build_layout as build_help
from scalebridge.dashapp.pages.settings.mlflow.page import build_layout as build_mlflow


def _walk(component):
    if isinstance(component, Component):
        yield component
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                yield from _walk(child)
        elif children is not None:
            yield from _walk(children)


def _string_ids(component):
    return {
        cid
        for item in _walk(component)
        for cid in (getattr(item, "id", None),)
        if isinstance(cid, str) and cid
    }


def test_environment_page_uses_compact_package_wrapper():
    layout = build_environment()
    class_names = {
        getattr(item, "className", "")
        for item in _walk(layout)
        if isinstance(getattr(item, "className", None), str)
    }
    assert "settings-package-sections" in class_names
    assert "settings-package-card-wrap" in class_names


def test_mlflow_page_contains_open_ui_button():
    texts = []
    for item in _walk(build_mlflow()):
        children = getattr(item, "children", None)
        if isinstance(children, str):
            texts.append(children)
    assert any("Open MLflow UI" in text for text in texts)


def test_help_page_no_longer_contains_removed_panels():
    text_blob = " ".join(
        str(getattr(item, "children"))
        for item in _walk(build_help())
        if getattr(item, "children", None) is not None
    )
    assert "Registered Help Entries" not in text_blob
    assert "Settings Help Entries" not in text_blob
    assert "Safety Model" not in text_blob
    assert "Settings Help Registry" not in text_blob
