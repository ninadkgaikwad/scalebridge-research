"""Unit tests for the shell page registry."""

from scalebridge.dashapp.pages.registry import MAJOR_PAGES, SUBPAGES


def test_major_page_ids_are_unique():
    ids = [page["id"] for page in MAJOR_PAGES]
    assert len(ids) == len(set(ids))


def test_major_paths_are_unique():
    paths = [page["path"] for page in MAJOR_PAGES]
    assert len(paths) == len(set(paths))


def test_every_major_page_has_subpages():
    for page in MAJOR_PAGES:
        assert page["id"] in SUBPAGES
        assert SUBPAGES[page["id"]]
