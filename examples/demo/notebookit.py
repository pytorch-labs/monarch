import argparse

import nbformat as nbf


def parse_python_to_ipynb(input_file, output_file, include_script=False):
    # Read the input Python file
    with open(input_file, "r") as f:
        lines = f.readlines()
    # Create a new Jupyter Notebook
    nb = nbf.v4.new_notebook()
    cells = []
    # Track whether we're inside a markdown block
    context = [nbf.v4.new_code_cell]
    in_quote = False
    contents = []
    for line in lines:
        sline = line.strip()
        if sline.startswith('"""') or sline.startswith('# """'):
            if contents:
                body = "".join(contents).strip()
                if body and not body.isspace():
                    r = context[-1](body)
                    if r is not None:
                        cells.append(r)
                contents.clear()
            if in_quote:
                context.pop()
            else:
                if sline == '"""script':
                    context.append(
                        nbf.v4.new_markdown_cell if include_script else lambda x: None
                    )
                elif sline == '"""py':
                    context.append(nbf.v4.new_code_cell)
                elif sline == '# """':
                    context.append(lambda x: None)
                else:
                    context.append(nbf.v4.new_markdown_cell)
            in_quote = not in_quote
        else:
            contents.append(line)
    assert not in_quote
    if contents:
        cells.append(nbf.v4.new_code_cell("".join(contents).strip()))
    nb["cells"] = cells
    # Write the notebook to a file
    with open(output_file, "w") as f:
        nbf.write(nb, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Python script to a Jupyter Notebook."
    )
    parser.add_argument("input_file", help="The input Python file to convert.")
    parser.add_argument("output_file", help="The output Jupyter Notebook file.")
    parser.add_argument(
        "--all", action="store_true", help="Include script comments as markdown."
    )
    args = parser.parse_args()
    parse_python_to_ipynb(args.input_file, args.output_file, args.all)
