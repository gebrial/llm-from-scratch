# generate some files
all: custom-tokenizer.py README.md 

# rule to convert a notebook to a python script
%.py: %.ipynb
	jupyter nbconvert --to script "$<"

README.md: README.ipynb
	jupyter nbconvert --to markdown README.ipynb

