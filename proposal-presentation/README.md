# Proposal Presentation (Beamer)

## Build
- Quick (no bibliography):
  ```bash
  make -C proposal-presentation quick
  ```
- Full (with bibliography via biber):
  ```bash
  make -C proposal-presentation
  ```
  If you add citations in slides, rerun `make` so biber regenerates the `.bbl`.

## Dependencies (Ubuntu)
```bash
sudo apt-get update && sudo apt-get install -y texlive texlive-latex-extra texlive-bibtex-extra biber
```

## Structure
- `main.tex` includes `shared/` preamble and `sections/*.tex`
- `shared/common.tex` sets class, packages, biblatex+biber
- `shared/beamersetup.tex` defines theme and layout
- `shared/definitions.tex` custom colors, math, helper macros
- `shared/titlepage.tex` title frame
- `references.bib` bibliography database
- `Makefile` build rules
