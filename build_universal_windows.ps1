param(
    [string]$BasePython = "$env:USERPROFILE\emt-env\Scripts\python.exe",
    [string]$Environment = ".venv-universal",
    [ValidateSet("cu126", "cu128")]
    [string]$CudaRuntime = "cu128",
    [switch]$SkipInstall,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Project
$EnvironmentPath = [System.IO.Path]::GetFullPath((Join-Path $Project $Environment))
$Python = Join-Path $EnvironmentPath "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $BasePython)) {
    throw "Base Python was not found at $BasePython."
}
if (-not (Test-Path -LiteralPath $Python)) {
    & $BasePython -m venv $EnvironmentPath
    if ($LASTEXITCODE -ne 0) {
        throw "Could not create the universal build environment."
    }
}

if (-not $SkipInstall) {
    & $Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Could not update pip." }

    $TorchIndex = "https://download.pytorch.org/whl/$CudaRuntime"
    & $Python -m pip install --upgrade torch torchvision --index-url $TorchIndex
    if ($LASTEXITCODE -ne 0) { throw "Could not install CUDA-enabled PyTorch." }

    & $Python -m pip install -r requirements-desktop.txt -r requirements-build.txt
    if ($LASTEXITCODE -ne 0) { throw "Could not install StimTrace dependencies." }
}

$TorchDetails = & $Python -c "import torch; print(torch.__version__); print(torch.version.cuda or '')"
if ($LASTEXITCODE -ne 0 -or $TorchDetails.Count -lt 2 -or -not $TorchDetails[1]) {
    throw "This environment does not contain a CUDA-enabled PyTorch build."
}

Write-Host "Building with PyTorch $($TorchDetails[0]), CUDA runtime $($TorchDetails[1])."
& (Join-Path $Project "build_windows.ps1") `
    -Python $Python `
    -BundleName "StimTrace-Universal" `
    -WorkPath "build-universal" `
    -SkipInstall `
    -Clean:$Clean

if ($LASTEXITCODE -ne 0) {
    throw "Universal StimTrace build failed."
}

Write-Host "The universal package uses CUDA when available and falls back to CPU."
