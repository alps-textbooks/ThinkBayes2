# Needs jupyter-book, ghp-import, sphinxcontrib-bibtex (see environment-dev.yml)

# Build the Jupyter book version

# make the asciidoc version of the preface
pandoc -t asciidoc --wrap none --markdown-headings=atx preface.md > preface.asciidoc

# if adding chapters, update _toc.yml

# copy notebooks
cp ../soln/chap[0-2]*.ipynb .
# and each chapter again, with its solutions, as the teachers' copy;
# prep_alps.py turns the first copy into the students' edition
for f in ../soln/chap[0-2]*.ipynb; do cp "$f" "$(basename "$f" .ipynb)_soln.ipynb"; done
cp ../soln/redline.ipynb .
cp ../examples/redline_pymc.ipynb .
cp ../examples/vaccine2.ipynb .
cp ../examples/usb.ipynb .
cp ../examples/sister.ipynb .
cp ../examples/bayes_dice.ipynb .
cp ../examples/radiation.ipynb .
cp ../examples/hospital.ipynb .
cp ../examples/hospital_birth_rate.ipynb .
# the one example with solutions gets a teachers' copy too, like the chapters
cp ../examples/hospital_birth_rate.ipynb hospital_birth_rate_soln.ipynb
cp ../examples/ok.ipynb .
cp ../examples/bookstore.ipynb .
cp ../examples/beta_binomial.ipynb .
cp ../examples/zipf.ipynb .
cp ../examples/bread.ipynb .
cp ../examples/cancer.ipynb .
cp ../examples/raven.ipynb .
cp ../examples/frog.ipynb .
cp ../examples/hepatitis.ipynb .
cp ../examples/diabetes.ipynb .
cp ../examples/ordinal.ipynb .
cp ../soln/utils.py .

# add tags to hide the solutions
python prep_notebooks.py

# adapt the copies for the in-browser runtime
python prep_alps.py

# build the HTML version
jb build .

# push it to GitHub
ghp-import -n -p -f _build/html
