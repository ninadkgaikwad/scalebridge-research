"""Run the Building-Grid Intelligence Research Studio."""

from scalebridge.dashapp import create_app
from scalebridge.dashapp.config import DashAppConfig


def main() -> None:
    """Start the Dash development server."""
    config = DashAppConfig.from_environment()
    app = create_app(config)
    app.run(host=config.host, port=config.port, debug=config.debug)


if __name__ == "__main__":
    main()
