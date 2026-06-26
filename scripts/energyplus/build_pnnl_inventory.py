"""Build the ScaleBridge PNNL commercial prototype inventory.

Run this script from any directory after setting:

``SCALEBRIDGE_EXTERNAL_DATA_ROOT``
    Existing shared ``Data`` directory containing prototypes and EPWs.

``SCALEBRIDGE_GENERATED_DATA_ROOT``
    ScaleBridge-owned generated-data directory.

The script scans ASHRAE 90.1-2013 commercial prototypes and writes CSV and JSON
inventories without modifying source files.
"""

from __future__ import annotations

from scalebridge.integration.energyplus.prototypes import (
    build_and_write_pnnl_inventory,
)


def main() -> None:
    """Build the inventory and print its high-level validation summary."""
    result = build_and_write_pnnl_inventory(standard_year=2013)

    print("PNNL commercial prototype inventory completed")
    print(f"Records: {result.record_count}")
    print(f"Status counts: {result.status_counts}")
    print(f"CSV: {result.csv_path}")
    print(f"JSON: {result.json_path}")


if __name__ == "__main__":
    main()
