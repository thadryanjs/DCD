import re
with open('03_model.qmd', 'r') as f:
    content = f.read()
blocks = re.findall(r'```\{python\}\s*(.*?)\s*```', content, re.DOTALL)
with open('run_model.py', 'w') as f:
    f.write('\n\n'.join(blocks))
print(f"Extracted {len(blocks)} blocks")
