### DCD pipeline
#
# pixi manages both the Python and R environments. Quarto renders the book

format:
    pixi run ruff format *.qmd || pixi run black *.py

format-r:
    pixi run Rscript -e "styler::style_dir('.')"

# Render one chapter at a time. Quarto caches via freeze, so re-rendering
# only re-executes chapters whose source changed.
process-data:
    pixi run quarto render 00_process-data.qmd

explore-data:
    pixi run quarto render 01_explore-data.qmd

analyze-data:
    pixi run quarto render 02_analyze-data.qmd

model:
    pixi run quarto render 03_model.qmd

analyze-model:
    pixi run quarto render 04_analyze-model.qmd

secondary-check:
    pixi run quarto render 05_secondary-check.qmd

# Full book. Chapter order comes from _quarto.yml, not from this recipe.
render-all:
    just run-all

run-all:
    pixi run quarto render *.qmd

# Force a full recompute, ignoring the freeze cache. Use when data changed
# rather than code.
run-all-fresh:
    rm -rf _freeze
    pixi run quarto render

preview:
    pixi run quarto preview

render-html:
    pixi run quarto render --to html

# Needs the pdf format enabled in _quarto.yml and a tex install:
#   pixi run quarto install tinytex
render-pdf:
    pixi run quarto render --to pdf

bundle:
    mkdir -p reports/ship/plots reports/ship/data reports/ship/html
    cp output/*.png reports/ship/plots/ 2>/dev/null || true
    cp output/*.csv reports/ship/plots/ 2>/dev/null || true
    cp data/processed/*.parquet reports/ship/data/ 2>/dev/null || true
    cp data/processed/*.csv reports/ship/data/ 2>/dev/null || true
    cp *.html reports/ship/html/ 2>/dev/null || true
    cp *.qmd _quarto.yml pixi.toml pixi.lock justfile spec.md reports/ship/
    echo "Bundle created in reports/ship/ (rendered HTML in reports/ship/html/)"

clean:
    mkdir -p reports
    mv output reports/ 2>/dev/null || true
    rm -rf *_files *.html quarto_ipynb .quarto _book

start-kernel:
  pixi run jupyter notebook --no-browser
