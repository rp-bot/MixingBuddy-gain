# Final Paper: MixingBuddy

This LaTeX project mirrors the `proposal/` setup to author the final paper.

## Project Structure

```
final-paper/
├── main.tex                 # Main LaTeX document
├── references.bib           # Bibliography database
├── Makefile                 # Build automation
├── README.md                # This file
├── figures/                 # Figures (add PDFs/PNGs here)
└── sections/                # Individual section files
    ├── 01_introduction.tex
    ├── 02_motivation.tex
    ├── 03_related_work_min.tex
    ├── 04_proposed_method.tex
    └── 05_proposed_evaluation.tex
```

## Build

```bash
make all     # full compile with bibtex
make quick   # compile without bibliography
make clean   # remove build artifacts
```

## Notes

- Figures are not auto-copied from `proposal/`. Place required assets in `figures/` and update paths in the `.tex` files.
- The bibliography was initialized from `proposal/references.bib`. Add new entries as needed.

