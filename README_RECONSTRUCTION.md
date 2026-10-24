# Reconstruct a selected standoff with the trained model

`scripts/reconstruct_combined.py` is the reusable inference/report command. It
does not train a model. Change the command-line flags to select geometry, input
standoff, target standoff and component.

## Reconstruct the existing 10 µm COMSOL input

Copy `scripts/reconstruct_combined.py` into the project's `scripts/` folder on
the professor's machine. The existing `combined_lista.py`, `torch_inverse.py`,
`iterative.py` and `propagator.py` must also be present. Use the existing Python
environment containing NumPy, Matplotlib and PyTorch.

On that machine:

```bash
cd "$HOME/learning_ker"

"$HOME/qdm-env/bin/python" scripts/reconstruct_combined.py \
  --run runs/combined_full_200 \
  --geometry L1_C1 \
  --standoff 10 \
  --target-standoff 0.5 \
  --device cuda \
  --include-saved-baselines \
  --output runs/reconstruction_L1_C1_10um
```

This loads `best.pt`, the saved 10 → 0.5 µm operator, the actual COMSOL input
and its reference. It runs ConvLISTA once per component and saves a complete
report. `--include-saved-baselines` adds the original ISTA/FISTA reconstructions
without rerunning their potentially long solves. Their nonconvergence flags
remain visible. Omit this flag for reference/input/ConvLISTA comparisons only.

To reconstruct at 6 µm instead, change `--standoff 10` to `--standoff 6` and
choose a new output directory. To use the other geometry, change
`--geometry L1_C1` to `--geometry IC_4um`. IC_2um's available pair is 2 → 1 µm.
All heights are absolute standoffs in µm, not propagation distances.

List valid saved pairs without loading PyTorch or a checkpoint:

```bash
"$HOME/qdm-env/bin/python" scripts/reconstruct_combined.py \
  --run runs/combined_full_200 --list
```

`--target-standoff` can be omitted when the selected geometry has one saved
reference height. `--component Bz` processes only Bz; the default `all` processes
Bx, By and Bz independently with the same learned weights. `--device auto`
chooses CUDA if available, otherwise CPU. GPU memory is bounded to one component
at a time; CPU report generation holds only the selected pair and requested
saved baselines. Existing output directories are refused.

## Required learned-model files

Keep these together in the run directory:

- `best.pt`: selected learned weights, model configuration, normalization scale.
- `setup.json`: geometry grids, height pairs and operator metadata.
- `physics/*.npy`: saved pair-specific operators; checksums are verified.
- `comparison/*.npz`: original COMSOL inputs/references and saved predictions.
- `comparison.csv`: original convergence metadata, required when adding saved baselines.

`last.pt` is for resuming training and is not used by this command. The checkpoint
must match the run configuration and scale. Fields are divided by the saved
scale before inference and multiplied back afterwards. No new normalization
is estimated from the supplied image. Fixed pair operators are separate from
the shared learned weights.

The downloaded local `runs/combined_full_200` currently has **no `best.pt` or
`last.pt`, and its `physics/` directory is empty**. Fresh inference should run
on the training machine, or those files must be copied from it first. The
command never silently substitutes an untrained model or different operator.

## Output report and arrays

| File | Contents |
|---|---|
| `index.html` | Clickable report with metrics and full-size figures |
| `figures/*_independent.png`, `*_shared.png` | Reference, blurred input, reconstruction and signed error, with labelled display scales |
| `figures/*_baselines_*.png` | Optional saved classical comparison panels |
| `reconstruction_uT.npy` | Float32 `[selected_components, ny, nx]` reconstruction in µT |
| `fields_uT.npz` | Input, reconstruction, optional reference and classical arrays |
| `metrics.csv`, `metrics.json` | Per-component RMSE, range NRMSE, optional baseline stopping metadata |
| `provenance.json` | Heights, grid, mode, source hashes, model/operator details and units |
| `status.json` | Report completion/failure and count of nonconverged baseline rows |

Components appear in Bx/By/Bz order for `all`, or the single requested component.
The default `--color-scale both` shows individually scaled fields, each with its
own µT colourbar, followed by an expandable shared-scale view for amplitude
comparison. Use `--color-scale independent` or `--color-scale shared` for only
one view. Limits are symmetric about zero using each field's maximum absolute
value; no percentile clipping is applied. Signed residuals always have their
own scale. Colours in independently scaled panels cannot be compared as equal
amplitudes. Display scaling never alters arrays or metrics.

The last panel is **reconstruction minus reference**, not another reconstructed
image. Sharp structures remaining in that error map indicate missed features.
`plot_diagnostics.json` and the HTML report additionally show field min/max and
relative L2 error, `100 * norm(prediction-reference) / norm(reference)`, undefined
for zero-energy or absent references. Range NRMSE is not a percentage accuracy.
Metrics use full-resolution arrays and physical µT:

```text
RMSE = sqrt(mean((reconstruction − reference)²))
range NRMSE (%) = 100 × RMSE / (max(reference) − min(reference))
```

A zero-range reference has undefined NRMSE (JSON null / blank CSV cell).
The report also scores the input against the same reference. Scores can differ
slightly from the original GPU CSV due to float64 CPU metric accumulation.
Fresh inference seconds are synchronized individual forward-call measurements,
not the original warmed three-repeat benchmark. Reused predictions have no
new inference time.

Redraw a completed report without PyTorch, a checkpoint or fresh inference:

```bash
python3 scripts/reconstruct_combined.py \
  --replot runs/reconstruction_local_10um --color-scale both
```

This updates figures, HTML, `plot_settings.json` and `plot_diagnostics.json` in
place. Original arrays, metrics, status and inference provenance are retained.

The model **does not generate ground truth**. For an existing COMSOL pair the
actual lower-height reference is loaded from the archive. These remain
same-sample results from the earlier training experiment, not a new independent
test. Saved baselines that hit a safety limit stay marked NOT CONVERGED even
when report generation completes successfully.

## View saved reconstructions without a checkpoint

This explicit mode recreates the figures and metrics from archived predictions;
it does not run the model and does not require PyTorch. Operator files can be
absent because no operator is applied; their verification status is recorded.

```bash
python3 scripts/reconstruct_combined.py \
  --run runs/combined_full_200 \
  --geometry L1_C1 --standoff 10 --target-standoff 0.5 \
  --saved-results --include-saved-baselines \
  --output runs/reconstruction_L1_C1_10um_preview
```

This report has already been generated locally at
`runs/reconstruction_L1_C1_10um_preview/index.html`.

## Supply a new calibrated field

Use `--input your_field_uT.npy`. Supported shapes are `[3,ny,nx]` in Bx/By/Bz
order, or `[ny,nx]` with a single `--component`. The array must match the selected
geometry's grid dimensions, pixel spacing, coordinates and vector convention.
NPY does not carry those physical metadata, so the caller must ensure the grid
and µT units are correct. A screenshot/PNG is not a calibrated field input.

Provide `--reference target_uT.npy` only if the actual target field is known
and aligned on that grid. Without it, the command writes input/reconstruction
figures and leaves accuracy metrics unavailable. It never borrows the original
COMSOL reference for a new input.

By default, the command uses the saved operator for the selected height pair.
For a new height, supply an input and explicitly select `--operator analytic`:

```bash
"$HOME/qdm-env/bin/python" scripts/reconstruct_combined.py \
  --run runs/combined_full_200 --geometry L1_C1 \
  --standoff 7.5 --target-standoff 0.5 \
  --input my_Bz_at_7p5um_uT.npy --component Bz \
  --operator analytic --device cuda \
  --output runs/reconstruction_new_Bz_7p5um
```

The analytical operator uses the selected grid, `Δz = input − target`, and
the checkpoint experiment's padding (32 pixels in this trained run). It does
not use the synthetic generator's 256-pixel default. Substituting an analytical
operator for this joint-fit-trained model is a new inference condition whose
accuracy needs validation. Choosing a saved operator for a new measurement
also does not establish that the fitted operator transfers to it. Saved
classical baselines cannot be included for a different input or operator.

## Validation

Run the tests in the trained environment:

```bash
"$HOME/qdm-env/bin/python" -m unittest discover \
  -s tests -p 'test_reconstruct_combined.py' -v
```

Local validation: three tests passed for saved-array/metric preservation,
nonconvergence flags, input/reference separation, component selection, invalid
height/source combinations and operator checksum handling. The fourth test
checks actual checkpoint inference and physical normalization against a known
identity-operator result; it was skipped locally because this Python has no
PyTorch. A full-resolution saved-results report with all components and four
classical baselines was generated and checked. No fresh GPU inference has been
run locally; that check remains for the trained environment.
