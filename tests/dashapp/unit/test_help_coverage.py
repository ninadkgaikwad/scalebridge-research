"""Unit tests for mandatory shell contextual-help coverage."""

from scalebridge.dashapp.help.registry import HELP_ENTRIES
from scalebridge.dashapp.pages.registry import MAJOR_PAGES, SUBPAGES


def test_every_major_page_has_help():
    for page in MAJOR_PAGES:
        assert f"page.{page['id']}" in HELP_ENTRIES


def test_every_subpage_has_help():
    for page in MAJOR_PAGES:
        for subpage in SUBPAGES[page["id"]]:
            assert (
                f"subpage.{page['id']}.{subpage['id']}"
                in HELP_ENTRIES
            )


def test_help_entries_have_required_shell_fields():
    for help_id, entry in HELP_ENTRIES.items():
        assert entry.get("title"), help_id
        assert entry.get("summary"), help_id
        assert entry.get("details"), help_id
