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
    uv run jupytext --to notebook --execute 02_analyze-data.R

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
    Rscript -e "install.packages(c('arrow', 'tidyverse', 'lme4', 'lmerTest', 'broom.mixed', 'corrplot', 'mice', 'IRkernel'), repos='https://cloud.r-project.org')"
    Rscript -e "IRkernel::installspec(user = TRUE)"

run-all:
    just process-data
    just explore-data
    just analyze-data
    just model
    just analyze-model

preview-report:
    uv run quarto preview book/report_tech.qmd

render-report:
	uv run quarto render book/report_tech.qmd --to html
	uv run quarto render book/report_med.qmd --to html

render-pdf:
	uv run quarto render book/report_tech.qmd --to pdf
	uv run quarto render book/report_med.qmd --to pdf


render-report-to-html:
    uv run quarto render book/report.qmd --to html
