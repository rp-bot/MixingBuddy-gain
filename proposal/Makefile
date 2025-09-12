# Makefile for LaTeX proposal compilation

# Main document name (without .tex extension)
MAIN = main

# LaTeX compiler
LATEX = pdflatex

# Bibliography processor
BIBTEX = bibtex

# Output directory
OUTDIR = build

# Create output directory if it doesn't exist
$(OUTDIR):
	mkdir -p $(OUTDIR)

# Default target
all: $(MAIN).pdf

# Compile the document
$(MAIN).pdf: $(MAIN).tex sections/*.tex references.bib | $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	$(BIBTEX) $(OUTDIR)/$(MAIN)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	cp $(OUTDIR)/$(MAIN).pdf .

# Quick compilation (without bibliography)
quick: $(MAIN).tex sections/*.tex | $(OUTDIR)
	$(LATEX) -output-directory=$(OUTDIR) $(MAIN).tex
	cp $(OUTDIR)/$(MAIN).pdf .

# Clean up generated files
clean:
	rm -rf $(OUTDIR)
	rm -f $(MAIN).pdf
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.fdb_latexmk *.fls *.synctex.gz

# Clean and rebuild
rebuild: clean all

# View the PDF (requires a PDF viewer)
view: $(MAIN).pdf
	xdg-open $(MAIN).pdf

# Help
help:
	@echo "Available targets:"
	@echo "  all      - Compile the complete document with bibliography"
	@echo "  quick    - Quick compilation without bibliography"
	@echo "  clean    - Remove all generated files"
	@echo "  rebuild  - Clean and rebuild"
	@echo "  view     - Open the PDF in default viewer"
	@echo "  help     - Show this help message"

.PHONY: all quick clean rebuild view help
