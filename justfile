format:
    uv run black *.py

setup:
    mkdir -p book

process-data:
    just setup
    uv run jupytext --to notebook --execute 00_process-data.py
    mv 00_process-data.ipynb book/
    touch book/report.qmd

explore-data:
    uv run jupytext --to notebook --execute 01_explore-data.py
    mv 01_explore-data.ipynb book/
    touch book/report.qmd

analyze-data: # install R deps first
    Rscript 02_analyze-data.R

analyze-model:
    uv run jupytext --to notebook --execute 04_analyze-model.py
    mv 04_analyze-model.ipynb book/
    touch book/report.qmd

model:
    uv run jupytext --to notebook --execute 03_model.py
    mv 03_model.ipynb book/
    touch book/report.qmd

# needs fortran
install-r-deps:
    Rscript -e "install.packages(c('arrow', 'tidyverse', 'lme4', 'lmerTest', 'broom.mixed', 'corrplot', 'mice'), repos='https://cloud.r-project.org')"

run-all:
    just process-data
    just explore-data
    # just analyze-data
    just model
    just analyze-model


preview-report:
    uv run quarto preview book/report.qmd

render-report:
    uv run quarto render book/report.qmd

render-report-to-html:
    uv run quarto render book/report.qmd --to html
