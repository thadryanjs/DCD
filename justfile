### DCD pipeline
#
# pixi manages both the Python and R environments. Quarto renders the book

format:
    pixi run ruff format *.qmd || pixi run black *.py

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
run-all:
    pixi run quarto render

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
    mkdir -p shipment/plots shipment/data shipment/book
    cp output/*.png shipment/plots/ 2>/dev/null || true
    cp output/*.csv shipment/plots/ 2>/dev/null || true
    cp data/processed/*.parquet shipment/data/ 2>/dev/null || true
    cp data/processed/*.csv shipment/data/ 2>/dev/null || true
    cp -r _book/* shipment/book/
    cp *.qmd _quarto.yml pixi.toml pixi.lock justfile spec.md shipment/
    echo "Bundle created in shipment/ (rendered book in shipment/book/)"

clean:
    mkdir -p reports
    mv _book reports/ 2>/dev/null || true
    mv output reports/ 2>/dev/null || true
    rm -rf _freeze .quarto shipment

start-kernel:
  pixi run jupyter notebook --no-browser
