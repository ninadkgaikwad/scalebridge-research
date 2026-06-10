$ErrorActionPreference = "Continue"

$Machine = $env:COMPUTERNAME
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutDir = "outputs\environment_reports"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Report = Join-Path $OutDir "windows_gpu_runtime_diagnostic_${Machine}_${Timestamp}.txt"

function Write-Section($Title) {
    "`n============================================================" | Out-File $Report -Append -Encoding UTF8
    $Title | Out-File $Report -Append -Encoding UTF8
    "============================================================" | Out-File $Report -Append -Encoding UTF8
}

"WINDOWS GPU / CUDA / CONDA RUNTIME DIAGNOSTIC" | Out-File $Report -Encoding UTF8
"Machine: $Machine" | Out-File $Report -Append -Encoding UTF8
"Date: $(Get-Date)" | Out-File $Report -Append -Encoding UTF8
"PWD: $(Get-Location)" | Out-File $Report -Append -Encoding UTF8

Write-Section "SYSTEM"
"ComputerName: $env:COMPUTERNAME" | Out-File $Report -Append -Encoding UTF8
"UserName: $env:USERNAME" | Out-File $Report -Append -Encoding UTF8
"OS:" | Out-File $Report -Append -Encoding UTF8
Get-CimInstance Win32_OperatingSystem |
    Select-Object Caption, Version, BuildNumber, OSArchitecture, TotalVisibleMemorySize, FreePhysicalMemory |
    Format-List | Out-String | Out-File $Report -Append -Encoding UTF8

Write-Section "CPU"
Get-CimInstance Win32_Processor |
    Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed |
    Format-List | Out-String | Out-File $Report -Append -Encoding UTF8

Write-Section "GPU FROM WINDOWS"
Get-CimInstance Win32_VideoController |
    Select-Object Name, DriverVersion, AdapterRAM, VideoProcessor |
    Format-List | Out-String | Out-File $Report -Append -Encoding UTF8

Write-Section "NVIDIA-SMI"
"nvidia-smi path:" | Out-File $Report -Append -Encoding UTF8
(Get-Command nvidia-smi -ErrorAction SilentlyContinue | Out-String) | Out-File $Report -Append -Encoding UTF8

"nvidia-smi output:" | Out-File $Report -Append -Encoding UTF8
nvidia-smi 2>&1 | Out-File $Report -Append -Encoding UTF8

"nvidia-smi query:" | Out-File $Report -Append -Encoding UTF8
nvidia-smi --query-gpu=name,driver_version,cuda_version,memory.total,compute_cap --format=csv 2>&1 | Out-File $Report -Append -Encoding UTF8

Write-Section "CUDA TOOLKIT VISIBILITY"
"nvcc path:" | Out-File $Report -Append -Encoding UTF8
(Get-Command nvcc -ErrorAction SilentlyContinue | Out-String) | Out-File $Report -Append -Encoding UTF8

"nvcc version:" | Out-File $Report -Append -Encoding UTF8
nvcc --version 2>&1 | Out-File $Report -Append -Encoding UTF8

"CUDA environment variables:" | Out-File $Report -Append -Encoding UTF8
Get-ChildItem Env: | Where-Object { $_.Name -match "CUDA|CUDNN|NVIDIA" } |
    Sort-Object Name |
    Format-Table -AutoSize | Out-String | Out-File $Report -Append -Encoding UTF8

Write-Section "VISUAL C++ RUNTIME DLLS IN SYSTEM32"
$Dlls = @(
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "msvcp140.dll",
    "concrt140.dll",
    "vcomp140.dll",
    "ucrtbase.dll"
)

foreach ($dll in $Dlls) {
    $p = Join-Path $env:WINDIR "System32\$dll"
    if (Test-Path $p) {
        $item = Get-Item $p
        "$dll FOUND $($item.FullName) version=$($item.VersionInfo.FileVersion)" | Out-File $Report -Append -Encoding UTF8
    } else {
        "$dll MISSING from System32" | Out-File $Report -Append -Encoding UTF8
    }
}

Write-Section "PATH CHECK"
$env:PATH -split ";" | ForEach-Object { $_ } | Out-File $Report -Append -Encoding UTF8

Write-Section "CONDA"
"conda path:" | Out-File $Report -Append -Encoding UTF8
(Get-Command conda -ErrorAction SilentlyContinue | Out-String) | Out-File $Report -Append -Encoding UTF8

"conda info:" | Out-File $Report -Append -Encoding UTF8
conda info 2>&1 | Out-File $Report -Append -Encoding UTF8

"conda env list:" | Out-File $Report -Append -Encoding UTF8
conda env list 2>&1 | Out-File $Report -Append -Encoding UTF8

"conda config --show channels:" | Out-File $Report -Append -Encoding UTF8
conda config --show channels 2>&1 | Out-File $Report -Append -Encoding UTF8

Write-Section "PYTHON DEFAULT"
"python path:" | Out-File $Report -Append -Encoding UTF8
(Get-Command python -ErrorAction SilentlyContinue | Out-String) | Out-File $Report -Append -Encoding UTF8

"python version:" | Out-File $Report -Append -Encoding UTF8
python --version 2>&1 | Out-File $Report -Append -Encoding UTF8

Write-Section "REPO"
git status --short 2>&1 | Out-File $Report -Append -Encoding UTF8
git branch -vv 2>&1 | Out-File $Report -Append -Encoding UTF8
git log --oneline -3 2>&1 | Out-File $Report -Append -Encoding UTF8

Write-Section "SUMMARY"
"Diagnostic complete." | Out-File $Report -Append -Encoding UTF8
"Report file: $Report" | Out-File $Report -Append -Encoding UTF8

Write-Host "Created diagnostic report:"
Write-Host $Report
