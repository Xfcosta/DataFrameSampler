# Publication

This folder contains the JMLR-style LaTeX manuscript for DataFrameSampler.
The manuscript uses the vendored official `jmlr2e.sty` style file and regular
JMLR article structure.

Build with:

```bash
cd publication
latexmk -pdf main.tex
```

The JMLR metadata in `main.tex` intentionally contains `TODO` placeholders for
paper id, dates, editor, email, funding, and competing-interest disclosure.
Those fields should be finalized before submission.
