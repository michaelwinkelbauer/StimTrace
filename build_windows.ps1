param(
    [string]$Python = "$env:USERPROFILE\emt-env\Scripts\python.exe",
    [string]$BundleName = "",
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

$ApplicationVersion = (& $Python -c "import app_metadata; print(app_metadata.APPLICATION_VERSION)").Trim()
if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($ApplicationVersion)) {
    throw "Could not read the application version from app_metadata.py."
}
if ([string]::IsNullOrWhiteSpace($BundleName)) {
    $BundleName = "StimTrace-$ApplicationVersion-Windows-x64"
}
$DistributionPath = Join-Path $Project "dist\$BundleName"
if (Test-Path -LiteralPath $DistributionPath) {
    throw "Refusing to overwrite existing release folder: $DistributionPath. Choose a new version or pass a new -BundleName."
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
New-Item -ItemType Directory -Path $ThirdPartyDestination -Force | Out-Null
# Some wheels (notably PyTorch) ship deeply nested vendored test-project
# licenses. Copy the supplied notices while omitting that generated subtree so
# Windows MAX_PATH cannot prevent an otherwise valid release from being built.
Get-ChildItem -LiteralPath $ThirdPartySource -File -Recurse |
    Where-Object { $_.FullName -notmatch "[\\/]licenses[\\/]third_party([\\/]|$)" } |
    ForEach-Object {
        $Relative = $_.FullName.Substring($ThirdPartySource.Length + 1)
        $Destination = Join-Path $ThirdPartyDestination $Relative
        New-Item -ItemType Directory -Path (Split-Path -Parent $Destination) -Force | Out-Null
        Copy-Item -LiteralPath $_.FullName -Destination $Destination -Force
    }

Write-Host ""
Write-Host "StimTrace was built successfully:"
Write-Host $Executable
Write-Host ""
Write-Host "Distribute the complete dist\$BundleName folder, not only StimTrace.exe."
