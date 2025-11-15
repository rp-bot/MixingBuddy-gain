# Makefile for LaTeX final paper compilation

# Detect OS
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
else
    DETECTED_OS := $(shell uname -s)
endif

# Main document name (without .tex extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex

# Bibliography processor
BIBTEX = bibtex

# Output directory
OUTDIR = build

# OS-specific commands
ifeq ($(DETECTED_OS),Windows)
    MKDIR = mkdir
    COPY = copy /Y
    RM = del /Q /F
    RMDIR = rmdir /S /Q
    VIEW = start
    # Convert forward slashes to backslashes for Windows paths
    FIXPATH = $(subst /,\,$1)
else
    MKDIR = mkdir -p
    COPY = cp
    RM = rm -f
    RMDIR = rm -rf
    VIEW = xdg-open
    FIXPATH = $1
endif

# Create output directory if it doesn't exist
$(OUTDIR):
ifeq ($(DETECTED_OS),Windows)
	@if not exist $(OUTDIR) $(MKDIR) $(OUTDIR)
else
	$(MKDIR) $(OUTDIR)
endif

# Default target
all: $(MAIN).pdf

# Compile the document
$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib | $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) -interaction=nonstopmode $(MAIN).tex
	-$(BIBTEX) $(OUTDIR)/$(MAIN) || true
	$(LATEX) -output-directory=$(OUTDIR) -interaction=nonstopmode $(MAIN).tex
	$(LATEX) -output-directory=$(OUTDIR) -interaction=nonstopmode $(MAIN).tex
ifeq ($(DETECTED_OS),Windows)
	cmd /c "copy /Y $(OUTDIR)\$(MAIN).pdf ."
else
	$(COPY) $(OUTDIR)/$(MAIN).pdf .
endif
	@echo "Cleaning up any stray auxiliary files..."
ifeq ($(DETECTED_OS),Windows)
	@$(RM) *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz 2>nul || true
else
	@$(RM) *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz
endif

# Quick compilation (without bibliography)
quick: $(MAIN).tex sections/*.tex | $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) -interaction=nonstopmode $(MAIN).tex
ifeq ($(DETECTED_OS),Windows)
	cmd /c "copy /Y $(OUTDIR)\$(MAIN).pdf ."
else
	$(COPY) $(OUTDIR)/$(MAIN).pdf .
endif
	@echo "Cleaning up any stray auxiliary files..."
ifeq ($(DETECTED_OS),Windows)
	@$(RM) *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz 2>nul || true
else
	@$(RM) *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz
endif

# Clean up generated files (keeps PDF)
clean:
	-$(RMDIR) $(OUTDIR)
ifeq ($(DETECTED_OS),Windows)
	-$(RM) *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz 2>nul || true
else
	-$(RM) *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz
endif

# Clean everything including PDF
cleanall: clean
	-$(RM) $(MAIN).pdf

# Clean and rebuild
rebuild: clean all

# View the PDF (requires a PDF viewer)
view: $(MAIN).pdf
	$(VIEW) $(MAIN).pdf

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Compile the complete document with bibliography"
	@echo "  quick    - Quick compilation without bibliography"
	@echo "  clean    - Remove auxiliary files (keeps PDF)"
	@echo "  cleanall - Remove all generated files including PDF"
	@echo "  rebuild  - Clean and rebuild"
	@echo "  view     - Open the PDF in default viewer"
	@echo "  help     - Show this help message"

.PHONY: all quick clean cleanall rebuild view help

