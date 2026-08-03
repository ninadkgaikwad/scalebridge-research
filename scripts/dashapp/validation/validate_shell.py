"""Validate shell registry and contextual-help coverage."""

from scalebridge.dashapp.help.registry import HELP_ENTRIES
from scalebridge.dashapp.pages.registry import MAJOR_PAGES, SUBPAGES


def main() -> None:
    """Validate every shell page and subpage has registered help."""
    missing: list[str] = []

    for page in MAJOR_PAGES:
        page_id = page["id"]
        page_help = f"page.{page_id}"
        if page_help not in HELP_ENTRIES:
            missing.append(page_help)

        for subpage in SUBPAGES[page_id]:
            help_id = f"subpage.{page_id}.{subpage['id']}"
            if help_id not in HELP_ENTRIES:
                missing.append(help_id)

    if missing:
        raise SystemExit(
            "Shell help validation failed. Missing IDs:\n- "
            + "\n- ".join(missing)
        )

    print("Shell validation passed.")
    print(f"Major pages: {len(MAJOR_PAGES)}")
    print(f"Subpages: {sum(len(items) for items in SUBPAGES.values())}")
    print(f"Registered help entries: {len(HELP_ENTRIES)}")


if __name__ == "__main__":
    main()
