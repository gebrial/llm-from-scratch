# Makefile to convert ALL jupyter notebooks into python scripts

# Find all notebooks
NOTEBOOKS := $(wildcard *.ipynb)
NOTEBOOKS := $(filter-out .ipynb_checkpoints%, $(NOTEBOOKS))

# convert all notebook names to .py targets
PY_FILES := $(NOTEBOOKS:.ipynb=.py)

# default target - convert all notebooks to python scripts
all: $(PY_FILES)

# rule to convert a notebook to a python script
%.py: %.ipynb
	jupyter nbconvert --to script "$<"

clean: rm -f $(PY_FILES)

help:
	@echo "Makefile for converting all jupyter notebooks to python scripts"
	@echo "Usage: "
	@echo "  make       - Convert all .ipynb files .py"
	@echo "  make clean - Remove all generated .py files"
	@echo "  make help  - Show this message"
