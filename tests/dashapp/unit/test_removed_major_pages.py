"""Regression tests for removed BGIRS major pages."""

from pathlib import Path

from scalebridge.dashapp.pages.registry import MAJOR_PAGES, SUBPAGES


REMOVED_IDS = {"artifact_lineage", "validation_center"}
REMOVED_PATHS = {"/artifact-lineage", "/validation-center"}


def test_removed_pages_are_not_in_major_page_registry():
    registered_ids = {page["id"] for page in MAJOR_PAGES}
    registered_paths = {page["path"] for page in MAJOR_PAGES}

    assert REMOVED_IDS.isdisjoint(registered_ids)
    assert REMOVED_PATHS.isdisjoint(registered_paths)


def test_removed_pages_have_no_subpage_registry_entries():
    assert REMOVED_IDS.isdisjoint(SUBPAGES)


def test_removed_page_directories_are_absent():
    dashapp_root = Path(__file__).resolve().parents[3] / "src" / "scalebridge" / "dashapp"
    assert not (dashapp_root / "pages" / "artifact_lineage").exists()
    assert not (dashapp_root / "pages" / "validation_center").exists()
