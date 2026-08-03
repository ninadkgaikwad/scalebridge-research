"""Idempotently remove Validation Center and Artifact Lineage from BGIRS."""

from __future__ import annotations

import ast
from pathlib import Path
import pprint
import shutil
import sys


REMOVED_PAGE_IDS = {"artifact_lineage", "validation_center"}
REMOVED_PATHS = {"/artifact-lineage", "/validation-center"}


def repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def replace_literal_assignment(path: Path, assignment_name: str, transform) -> bool:
    """Rewrite a literal top-level assignment when present.

    Returns False when the assignment is absent, allowing the operation to remain
    safe across alternate source layouts and partially applied patches.
    """
    if not path.exists():
        return False

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    target = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(item, ast.Name) and item.id == assignment_name
            for item in node.targets
        ):
            target = node
            break

    if target is None:
        return False

    value = ast.literal_eval(target.value)
    updated = transform(value)
    replacement = (
        f"{assignment_name} = "
        + pprint.pformat(updated, width=100, sort_dicts=False)
    )

    lines = source.splitlines(keepends=True)
    start = target.lineno - 1
    end = target.end_lineno
    new_source = "".join(lines[:start]) + replacement + "\n" + "".join(lines[end:])
    path.write_text(new_source, encoding="utf-8")
    return True


def update_registry(repo: Path) -> None:
    path = repo / "src/scalebridge/dashapp/pages/registry.py"

    major_changed = replace_literal_assignment(
        path,
        "MAJOR_PAGES",
        lambda pages: [
            page for page in pages if page.get("id") not in REMOVED_PAGE_IDS
        ],
    )
    subpages_changed = replace_literal_assignment(
        path,
        "SUBPAGES",
        lambda subpages: {
            key: value
            for key, value in subpages.items()
            if key not in REMOVED_PAGE_IDS
        },
    )

    print(
        "Page registry: "
        f"MAJOR_PAGES={'updated' if major_changed else 'already updated or alternate layout'}, "
        f"SUBPAGES={'updated' if subpages_changed else 'already updated or alternate layout'}"
    )


def update_router(repo: Path) -> None:
    path = repo / "src/scalebridge/dashapp/layout/routing/router.py"
    if not path.exists():
        print(f"Router not found; skipped: {path}")
        return

    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    class Transformer(ast.NodeTransformer):
        def visit_ImportFrom(self, node):
            module = node.module or ""
            if module.endswith(".pages.artifact_lineage"):
                return None
            if module.endswith(".pages.validation_center"):
                return None
            return node

        def visit_Assign(self, node):
            self.generic_visit(node)
            if any(
                isinstance(target, ast.Name) and target.id == "_ROUTES"
                for target in node.targets
            ) and isinstance(node.value, ast.Dict):
                retained = []
                for key, value in zip(node.value.keys, node.value.values):
                    literal_key = (
                        ast.literal_eval(key)
                        if isinstance(key, ast.Constant)
                        else None
                    )
                    if literal_key not in REMOVED_PATHS:
                        retained.append((key, value))
                node.value.keys = [key for key, _ in retained]
                node.value.values = [value for _, value in retained]
            return node

        def visit_FunctionDef(self, node):
            if node.name in {
                "render_artifact_lineage_tab",
                "render_validation_center_tab",
            }:
                return None
            return self.generic_visit(node)

    transformed = Transformer().visit(tree)
    ast.fix_missing_locations(transformed)
    updated = ast.unparse(transformed) + "\n"

    if updated != source:
        path.write_text(updated, encoding="utf-8")
        print("Router updated.")
    else:
        print("Router already updated.")


def remove_page_directories(repo: Path) -> None:
    for relative in (
        "src/scalebridge/dashapp/pages/artifact_lineage",
        "src/scalebridge/dashapp/pages/validation_center",
    ):
        path = repo / relative
        if path.exists():
            shutil.rmtree(path)
            print(f"Removed {relative}")
        else:
            print(f"Already absent: {relative}")


def main() -> int:
    repo = repository_root()
    update_registry(repo)
    update_router(repo)
    remove_page_directories(repo)

    print(
        "Validation Center and Artifact Lineage removal completed. "
        "The help registry was intentionally left unchanged because unused "
        "help entries do not affect navigation or help coverage."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
