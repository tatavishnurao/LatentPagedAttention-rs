# Build the PDF from the data: regenerate macros, tables, bibliography and figures, then compile.
pdf:
	python3 gen_latex.py
	latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
clean:
	latexmk -C
