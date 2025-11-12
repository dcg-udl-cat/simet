import pathlib

import mkdocs_gen_files

PACKAGE = "simet"

nav = mkdocs_gen_files.Nav()
root = pathlib.Path(PACKAGE)

for path in sorted(root.rglob("*.py")):
    if path.name == "__init__.py":
        continue
    mod_path = path.with_suffix("").as_posix().replace("/", ".")
    doc_path = (
        pathlib.Path("reference", *path.parts[1:]).with_suffix(".md")
        if path.parts[0] == PACKAGE
        else pathlib.Path("reference", *path.parts).with_suffix(".md")
    )

    nav_parts = [p for p in path.relative_to(root).with_suffix("").parts]
    relative_doc_path = doc_path.relative_to("reference")
    nav[*nav_parts] = relative_doc_path.as_posix()

    with mkdocs_gen_files.open(doc_path, "w") as fd:
        title = "# " + mod_path
        print(title, file=fd)
        print("\n::: " + mod_path, file=fd)

# write navigation index
with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())

# Create reference/index.md landing
with mkdocs_gen_files.open("reference/index.md", "w") as fd:
    fd.write("# API Reference\n\n")
    fd.write("This API is generated from Google-style docstrings.\n\n")
    fd.write('--8<-- "reference/SUMMARY.md"\n')
