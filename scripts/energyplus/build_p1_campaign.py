"""Build the locked 64-case P1 campaign without running EnergyPlus."""

from scalebridge.integration.energyplus.p1 import (
    build_p1_case_specs,
    write_p1_campaign_manifest,
)


def main() -> None:
    """Validate source files and write portable P1 campaign manifests."""
    cases = build_p1_case_specs()
    result = write_p1_campaign_manifest(cases)
    print(f"Campaign: {result.campaign_id}")
    print(f"Cases: {result.case_count}")
    print(f"JSON: {result.json_path}")
    print(f"CSV: {result.csv_path}")


if __name__ == "__main__":
    main()
