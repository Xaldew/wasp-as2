TMP           := $(CURDIR)/build
TEX_SRCS      := wasp-beamer.tex
SVG_SRCS      := $(wildcard *.svg graphics/*.svg)
PDF_OUT       := $(SVG_SRCS:.svg=.pdf)
LATEXMK       ?= latexmk
LACHECK       ?= lacheck
TEX_FLAGS     ?= --shell-escape
LATEXMK_FLAGS ?= -bibtex -pdf -output-directory=$(TMP) -pdflatex="xelatex $(TEX_FLAGS) %O %S"

.PHONY: all
all : $(TEX_SRCS) $(PDF_OUT)
	$(LATEXMK) $(LATEXMK_FLAGS) $(TEX_SRCS)

.PHONY: check
check : all
	$(LACHECK) $(LACHECK_FLAGS) $(TEX_SRCS)

.PHONY: wc
wc : ${TEX_SRCS}
	detex $< | wc --words

.PHONY: gray
gray : all
	gs  -sOutputFile=$(basename $(TEX_SRCS))_grayscale.pdf \
	-sDEVICE=pdfwrite \
	-sColorConversionStrategy=Gray \
	-dProcessColorModel=/DeviceGray \
	-dCompatibilityLevel=1.4 \
	-dNOPAUSE \
	-dBATCH \
	$(basename $(TEX_SRCS)).pdf

%.pdf : %.svg
	inkscape $< -A $(patsubst %.svg, %.pdf, $<)

.PHONY: clean
clean :
	$(RM) -r $(TMP) $(PDF_OUT)
