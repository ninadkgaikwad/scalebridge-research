$ErrorActionPreference = "Stop"

python scripts\dashapp\maintenance\remove_validation_and_artifact_pages.py

if ($LASTEXITCODE -ne 0) {
    throw "Navigation cleanup failed with exit code $LASTEXITCODE."
}

Write-Host "Validation Center and Artifact Lineage removal completed."
