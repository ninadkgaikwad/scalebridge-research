"""Print the major-page and subpage route inventory."""

from scalebridge.dashapp.pages.registry import MAJOR_PAGES, SUBPAGES


def main() -> None:
    """Print all registered navigation items."""
    for page in MAJOR_PAGES:
        print(f"{page['label']}  ->  {page['path']}")
        for subpage in SUBPAGES[page["id"]]:
            print(f"    {subpage['label']}  [{subpage['id']}]")


if __name__ == "__main__":
    main()
