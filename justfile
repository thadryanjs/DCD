### DCD pipeline
#
# pixi manages both the Python and R environments. Quarto renders the book

format:
    pixi run ruff format *.qmd || pixi run black *.py

format-r:
    pixi run Rscript -e "styler::style_dir('.', recursive = FALSE)"

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

# Standalone update deck (not a book chapter). Quarto's --to pdf forces the
# LaTeX writer (continuous doc); quarto docs specify the revealjs print
# stylesheet as THE pdf path. This drives it headlessly: ?print-pdf view +
# virtual-time-budget lets the deck finish initializing before printing.
summary-update:
    pixi run quarto render 05_summary-update.qmd --to revealjs
    chromium --headless --disable-gpu --no-pdf-header-footer \
        --virtual-time-budget=15000 \
        --print-to-pdf=05_summary-update.pdf "file://$(pwd)/05_summary-update.html?print-pdf"

# Full book. Chapter order comes from _quarto.yml, not from this recipe.
render-all:
    just run-all

# Pipeline order is explicit here — NOT inferred from filenames or the glob.
run-all: process-data explore-data analyze-data model analyze-model

# Force a full recompute, ignoring the freeze cache. Use when data changed
# rather than code, or when an upstream step's results changed.
run-all-fresh:
    rm -rf _freeze
    just run-all

preview:
    pixi run quarto preview

render-html:
    pixi run quarto render --to html

# Needs the pdf format enabled in _quarto.yml and a tex install:
#   pixi run quarto install tinytex
render-pdf:
    pixi run quarto render --to pdf

# Share bundle: ONLY artifacts cleared to ship right now (prelim — nothing
# reviewed by clinical staff). Everything else stays internal until cleared;
# add files here only once they are safe to share.
bundle:
    rm -rf reports/ship
    mkdir -p reports/ship
    cp variable-map.md reports/ship/
    cp output/cv_auc_boxplot_first_look.png reports/ship/
    cp output/final_performance_summary_first_look.csv reports/ship/
    cp output/cv_metrics_per_fold_first_look.csv reports/ship/
    cp 05_summary-update.pdf reports/ship/ 2>/dev/null || true
    echo "Share bundle created in reports/ship/ (colname map, prelim AUC data, deck PDF)"

clean:
    mkdir -p reports
    mv output reports/ 2>/dev/null || true
    rm -rf *_files *.html *.quarto_ipynb* __pycache__ .quarto _book _freeze

start-kernel:
  pixi run jupyter notebook --no-browser
