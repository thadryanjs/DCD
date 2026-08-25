format:
    uv run black *.py

process-dataset:
    uv run 00_process-dataset.py

book-process:
    mkdir -p book
    jupytext --to notebook 00_process-dataset.py
    mv 00_process-dataset.ipynb book/

render-report:
    just book-process
    quarto render book/report.qmd

preview-report:
    just book-process
    quarto preview book/report.qmd
