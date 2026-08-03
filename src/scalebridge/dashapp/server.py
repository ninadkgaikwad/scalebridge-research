"""WSGI-compatible server entry point."""

from .app import create_app

app = create_app()
server = app.server
