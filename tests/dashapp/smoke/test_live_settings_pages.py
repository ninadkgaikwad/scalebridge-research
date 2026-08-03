"""Smoke tests for live current-machine Settings pages."""

from dash.development.base_component import Component

from scalebridge.dashapp.pages.settings.environments.page import (
    build_layout as environment,
)
from scalebridge.dashapp.pages.settings.machines.page import (
    build_layout as machine,
)
from scalebridge.dashapp.pages.settings.mlflow.page import (
    build_layout as mlflow,
)
from scalebridge.dashapp.pages.settings.paths.page import (
    build_layout as paths,
)


def walk(component):
    """Yield every Dash component below the supplied root."""
    if isinstance(component, Component):
        yield component
        children = getattr(component, "children", None)
        if isinstance(children, (list, tuple)):
            for child in children:
                yield from walk(child)
        elif children is not None:
            yield from walk(children)


def string_ids(component):
    """Return only hashable string component IDs.

    Dictionary IDs are valid Dash pattern-matching IDs and are intentionally
    ignored by these exact string-ID assertions.
    """
    return {
        component_id
        for item in walk(component)
        for component_id in (getattr(item, "id", None),)
        if isinstance(component_id, str) and component_id
    }


def test_pages_build():
    for builder in (paths, machine, environment, mlflow):
        assert builder() is not None


def test_pages_have_no_badges():
    for builder in (paths, machine, environment, mlflow):
        assert all(
            item.__class__.__name__ != "Badge"
            for item in walk(builder())
        )


def test_profile_editor_is_absent():
    ids = string_ids(machine())
    assert "settings-profile-selector" not in ids
    assert "settings-profile-save-draft" not in ids
    assert "settings-profile-activate" not in ids
