
## Project
- This report used a previous report about Ophthalmology as a template and agents keep trying to stick name of it all over the place. It's a template. We're not cloning the name. This project is about DCD and that should be the only name reflected in naming conventions.
- Don't run the report preview - I usually have one going so this only invites confusion and mistakes.
- Don't execute code unless I say to - I usually run and selectively share output.
- We're using `uv`. This means run commands need to be written `uv run {whatever}` not just `{whatever}`. This applies to writing the justfile commands as well.

## Data & ML Constraints
- **Patient-Level Leakage**: The dataset contains multiple observations per patient (`alias_filled`). Standard random splits cause severe data leakage (memorization of patients).
- **Requirement**: Always use `StratifiedGroupKFold` or `GroupShuffleSplit` with `groups=df["alias_filled"]` to ensure all observations for a single patient stay within the same fold.

## Style
- Use `just format` which calls `black` - sane defaults don't overthink it.
- Don't use ALL_CAPS for constants it's pointless and loud and in certain types of scripts it makes half the script uppercase.
- Don't use type hints unless there is a meaningful indication to add this complexity, ie
    - There is a type that impacts a calculation or could result in a bad transformation
    - Don't bother with it on something like "read_file" where passing number will result in a harmless error - type checking this is a marginal improvement over just getting a FileNotFound error.
- We're using `jupytext` to have clean scripts with good git diffs while also being able to compile to jupyter books. If you're an agent, ask about a skill for this. Some basics:
    - Any prints should occur at the end of a cell and start a new cell - so three smalls cells to instead of one larger one to print three things. It swill then read code/output/code/output/code/output not code/code/code/output/output/output.
    - Don't do this. Agents love it but it makes things worse not better if we're using jupyter because natural lightweight delineation happens with the cells already.
        `print("\nMissingness Comparison: Positive vs Negative Class")`
        `print("=" * 70)`

- All the imports go at the top! No exceptions!
- Don't change file names without asking me. If it's a slightly different name than expected it's because I did it on purpose.

