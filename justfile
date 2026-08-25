format:
    uv run black *.py

setup:
    mkdir -p book

process-dataset:
    just setup
    uv run jupytext --to notebook --execute 00_process-dataset.py
    mv 00_process-dataset.ipynb book/
    touch book/report.qmd

compile-report:
    uv run quarto render book/report.qmd --to html

explore-data:
    uv run jupytext --to notebook --execute 01_explore-data.py
    mv 01_explore-data.ipynb book/

model-cross-validation:
    uv run jupytext --to notebook --execute 02_model-cross-validation.py
    mv 02_model-cross-validation.ipynb book/

preview-report:
    uv run quarto preview book/report.qmd

render-report:
    uv run quarto render book/report.qmd
