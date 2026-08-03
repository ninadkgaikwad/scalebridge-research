$ErrorActionPreference="Stop"
$Paths=@(
"src\scalebridge\dashapp\callbacks\settings",
"src\scalebridge\dashapp\schemas\settings",
"src\scalebridge\dashapp\services\settings",
"config\machines","config\machines_archive",
"config\machine_profile_history","config\active_machine.json"
)
foreach($Path in $Paths){
    if(Test-Path $Path){Remove-Item $Path -Recurse -Force;Write-Host "Removed $Path"}
}
Write-Host "Machine profile manager cleanup completed."
