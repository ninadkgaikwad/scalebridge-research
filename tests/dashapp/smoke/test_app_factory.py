"""Smoke test for the Dash application factory."""

from scalebridge.dashapp import create_app


def test_app_factory_builds_layout():
    app = create_app()
    assert app.layout is not None
    assert app.title == "Building-Grid Intelligence Research Studio"
