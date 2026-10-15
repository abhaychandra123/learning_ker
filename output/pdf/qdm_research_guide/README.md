# QDM research playbook

An editable LaTeX guide for a 4–6 month investigation of joint standoff estimation and near-plane magnetic-field reconstruction.

## Build

From this directory:

```sh
python3 generate_figures.py
tectonic qdm_research_guide.tex
```

Alternatively, use an ordinary LaTeX distribution:

```sh
pdflatex qdm_research_guide.tex
pdflatex qdm_research_guide.tex
```

The bibliography is self-contained; no BibTeX pass is required. Figures are original TikZ diagrams and analytical plots. The included figure PDFs allow building without running Python; Python with NumPy and Matplotlib is needed only to regenerate them.

## Scientific status

- This is a proposed research plan, not a report of completed new experiments.
- Pilot sizes, hyperparameters, and schedules are starting configurations, not optimality claims.
- The user confirmed an extensive existing COMSOL database and access to the real dataset used in Bathla et al. The external files were not supplied or inspected. Their usage protocols are conditional on the published description and require a local audit when the hard disk is available.
- The confirmed primary output is the near-plane magnetic field; current density is an auxiliary physical representation.
- Existing project scripts and saved results are retained; this guide does not implement or validate the proposed models.
- Conference dates are a dated snapshot and must be rechecked before submission.
