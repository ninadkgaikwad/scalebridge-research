"""Layout for Figure Builder."""

from ....layout.placeholders import build_module_placeholder


def build_layout():
    """Build the Figure Builder shell layout."""
    return build_module_placeholder(
        page_id="publication_studio",
        subpage_id="figure_builder",
        title="Figure Builder",
        description='The Figure Builder workspace within Publication Studio. This shell module is ready for integration with the authoritative scientific implementation.',
    )
