# Third-party licenses

This directory contains the license and notice files supplied by the Python
distributions installed in the environment used to build StimTrace 1.0.0 for
Windows. Each subdirectory is named after the corresponding installed
distribution and version.

The collection is deliberately conservative: it also retains notices for build
tools and transitive packages present in that controlled release environment.
Inclusion here does not imply that every listed package is loaded by every
StimTrace workflow. The direct runtime dependencies are listed in
`requirements-desktop.txt`.

Qt/PySide components are distributed under their own applicable license terms,
including the LGPL option represented by the license files supplied with the
PySide6 distributions. Users may replace or relink the corresponding shared Qt
libraries in the extracted application directory.
