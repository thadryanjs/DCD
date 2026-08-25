format:
    uv run black *.py

process-dataset:
    uv run 00_process-dataset.py

book-process:
    rm -f book/00_process-dataset.ipynb
    uv run jupytext --to notebook 00_process-dataset.py
    uv run jupyter nbconvert --to notebook --execute 00_process-dataset.ipynb --inplace
    mkdir -p book
    mv 00_process-dataset.ipynb book/

render-report:
    just book-process
    uv run quarto render book/report.qmd

preview-report:
    just book-process
    uv run quarto preview book/report.qmd
