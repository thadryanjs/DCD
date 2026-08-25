

## Style
- Use `just format` which calls `black` - sane defaults don't overthink it.
- Don't use ALL_CAPS for constants it's pointless and loud and in certain types of scripts it makes half the script uppercase.
- Don't use type hints unless there is a meaningful indication to add this complexity, ie
    - There is a type that impacts a calculation or could result in a bad transformation
    - Don't bother with it on something like "read_file" where passing number will result in a harmless error - type checking this is a marginal improvement over just getting a FileNotFound error.
- We're using `jupytext` to have clean scripts with good git diffs while also being able to compile to jupyter books. If you're an agent, ask about a skill for this. Some basics:
    - Any prints should occur at the end of a cell and start a new cell - so three smalls cells to instead of one larger one to print three things. It swill then read code/output/code/output/code/output not code/code/code/output/output/output.

## Project
- We're using `uv`. This means run commands need to be written `uv run {whatever}` not just `{whatever}`. This applies to writing the justfile commands as well.
