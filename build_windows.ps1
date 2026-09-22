param(
    [string]$Python = "$env:USERPROFILE\emt-env\Scripts\python.exe",
    [string]$BundleName = "StimTrace",
    [string]$WorkPath = "build",
    [switch]$SkipInstall,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$Project = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Project

if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found at $Python. Pass -Python with the intended python.exe."
}

& $Python "verify_build.py"
if ($LASTEXITCODE -ne 0) {
    throw "StimTrace build preflight failed. Resolve the reported issue before packaging."
}

if (-not $SkipInstall) {
    & $Python -m pip install -r requirements-build.txt
    if ($LASTEXITCODE -ne 0) {
        throw "Could not install the executable build dependency."
    }
}

$env:STIMTRACE_BUNDLE_NAME = $BundleName
$PyInstallerArguments = @(
    "-m", "PyInstaller", "--noconfirm",
    "--distpath", "dist",
    "--workpath", $WorkPath
)
if ($Clean) {
    $PyInstallerArguments += "--clean"
}
$PyInstallerArguments += "StimTrace.spec"

& $Python @PyInstallerArguments
if ($LASTEXITCODE -ne 0) {
    throw "StimTrace executable build failed."
}

$Executable = Join-Path $Project "dist\$BundleName\StimTrace.exe"
if (-not (Test-Path -LiteralPath $Executable)) {
    throw "Build finished without creating $Executable."
}

# Keep end-user documentation and license terms visible beside the executable.
foreach ($Document in @(
    "QUICK_START.md",
    "LICENSE",
    "NOTICE.md",
    "AUTHORS.md",
    "CITATION.cff"
)) {
    Copy-Item -LiteralPath (Join-Path $Project $Document) `
        -Destination (Join-Path $Project "dist\$BundleName\$Document") `
        -Force
}

$ThirdPartySource = Join-Path $Project "THIRD_PARTY_LICENSES"
$ThirdPartyDestination = Join-Path $Project "dist\$BundleName\THIRD_PARTY_LICENSES"
Copy-Item -LiteralPath $ThirdPartySource -Destination $ThirdPartyDestination -Recurse -Force

Write-Host ""
Write-Host "StimTrace was built successfully:"
Write-Host $Executable
Write-Host ""
Write-Host "Distribute the complete dist\$BundleName folder, not only StimTrace.exe."
