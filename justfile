format:
    uv run black *.py

setup:
    mkdir -p book

process-dataset:
    just setup
    uv run jupytext --to notebook --execute 00_process-dataset.py
    mv 00_process-dataset.ipynb book/
    touch book/report.qmd

render-report:
    uv run quarto render book/report.qmd

preview-report:
    uv run quarto preview book/report.qmd
