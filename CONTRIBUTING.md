# Contributing to StimTrace

StimTrace is research-use software. Changes to segmentation, tracking, timing,
calibration, or force calculations must describe their scientific rationale and
include focused regression tests where practical.

## Development setup

1. Create and activate a Python virtual environment.
2. Install `requirements-desktop.txt`.
3. Add an approved checkpoint as `model.pth` only when local inference or a
   packaged build is required. Model files are intentionally excluded from Git.
4. When testing cloud workflows, select a Google OAuth Desktop client configuration
   stored outside the repository. Never commit, bundle, or share OAuth client files or
   user tokens.

Run the regression suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

Before opening a pull request, confirm that no recordings, result files,
credentials, tokens, local configuration, or build outputs are staged.

By contributing, you certify that you have the right to submit the contribution
and agree that it may be distributed under the MIT License as part of StimTrace.
Institutional, employment, and collaborator permissions remain the contributor's
responsibility.
