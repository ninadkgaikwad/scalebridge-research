$ErrorActionPreference = "Stop"

$Paths = @(
    "src\scalebridge\dashapp\callbacks\settings",
    "src\scalebridge\dashapp\schemas\settings",
    "src\scalebridge\dashapp\services\settings",
    "config\machines",
    "config\machines_archive",
    "config\machine_profile_history",
    "config\active_machine.json",
    "tests\dashapp\unit\test_machine_profile_schema.py",
    "tests\dashapp\unit\test_machine_profile_store.py",
    "tests\dashapp\unit\test_settings_runtime.py",
    "tests\dashapp\smoke\test_machine_profile_page.py"
)

foreach ($Path in $Paths) {
    if (Test-Path $Path) {
        Remove-Item $Path -Recurse -Force
        Write-Host "Removed $Path"
    }
}

Write-Host "Obsolete machine-profile implementation and tests removed."
