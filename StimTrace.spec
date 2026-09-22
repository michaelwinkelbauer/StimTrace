# -*- mode: python ; coding: utf-8 -*-
import os
from pathlib import Path
from PyInstaller.utils.win32 import versioninfo

project_dir = Path(SPECPATH)
bundle_name = os.environ.get("STIMTRACE_BUNDLE_NAME", "StimTrace")
metadata_namespace = {}
exec(
    (project_dir / "app_metadata.py").read_text(encoding="utf-8"),
    metadata_namespace,
)
application_name = metadata_namespace["APPLICATION_NAME"]
application_version = metadata_namespace["APPLICATION_VERSION"]
version_numbers = [int(part) for part in application_version.split(".")]
version_quad = tuple((version_numbers + [0, 0, 0, 0])[:4])
windows_version = versioninfo.VSVersionInfo(
    ffi=versioninfo.FixedFileInfo(
        filevers=version_quad,
        prodvers=version_quad,
        mask=0x3F,
        flags=0x0,
        OS=0x40004,
        fileType=0x1,
        subtype=0x0,
        date=(0, 0),
    ),
    kids=[
        versioninfo.StringFileInfo([
            versioninfo.StringTable("040904B0", [
                versioninfo.StringStruct("FileDescription", application_name),
                versioninfo.StringStruct("FileVersion", application_version),
                versioninfo.StringStruct("InternalName", application_name),
                versioninfo.StringStruct("LegalCopyright", "Copyright (c) 2026 ETH Zurich"),
                versioninfo.StringStruct("OriginalFilename", "StimTrace.exe"),
                versioninfo.StringStruct("ProductName", application_name),
                versioninfo.StringStruct("ProductVersion", application_version),
            ]),
        ]),
        versioninfo.VarFileInfo([
            versioninfo.VarStruct("Translation", [0x0409, 1200]),
        ]),
    ],
)
model_candidates = [
    project_dir / "model.pth",
    project_dir.parent / "unet_multitask_center_ellipse_512x512 - Backup.pth",
]
model_path = next((path for path in model_candidates if path.is_file()), None)
if model_path is None:
    raise FileNotFoundError(
        "The bundled segmentation checkpoint was not found. Expected model.pth "
        "in the project folder or the approved backup checkpoint one folder above it."
    )

datas = [
    (str(project_dir / "assets"), "assets"),
    (str(project_dir / "colab_worker.py"), "."),
    (str(project_dir / "stimtrace_colab_worker.ipynb"), "."),
    (str(project_dir / "LICENSE"), "."),
    (str(project_dir / "NOTICE.md"), "."),
    (str(project_dir / "AUTHORS.md"), "."),
    (str(project_dir / "CITATION.cff"), "."),
    (str(project_dir / "THIRD_PARTY_LICENSES"), "THIRD_PARTY_LICENSES"),
    (str(model_path), "."),
]
binaries = []
hiddenimports = []

a = Analysis(
    ["desktop_app.py"],
    pathex=[str(project_dir)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "hf_xet",
        "tensorboard",
        "timm.loss",
        "timm.optim",
        "timm.scheduler",
        "timm.task",
        "torch.utils.tensorboard",
    ],
    noarchive=False,
    optimize=0,
)

# google-api-python-client ships static discovery documents for every Google API.
# StimTrace only builds a Drive v3 client, so retain that document and discard the
# unrelated API schemas collected by the standard PyInstaller hook.
discovery_prefix = "googleapiclient/discovery_cache/documents/"
a.datas[:] = [
    entry
    for entry in a.datas
    if not entry[0].replace("\\", "/").startswith(discovery_prefix)
    or entry[0].replace("\\", "/") == discovery_prefix + "drive.v3.json"
]
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="StimTrace",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    # Keep stdout/stderr pipes available for the bundled local worker while
    # hiding the console window for normal desktop use.
    console="hide-early",
    icon=str(project_dir / "assets" / "stimtrace.ico"),
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version=windows_version,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=bundle_name,
)
