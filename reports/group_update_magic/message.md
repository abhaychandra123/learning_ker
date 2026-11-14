Hi all, an update on the standoff reconstruction (3 µm → 0.5 µm).

I implemented MAGIC-UNet from Reed et al. 2024, "Machine Learning for Improved Current Density Reconstruction from 2D Vector Magnetic Images" (arXiv 2407.14553), and tested it on our COMSOL data.

What it is:
- A U-Net that takes the measured field (Bx, By, Bz) and predicts the current (Jx, Jy) that produced it. The field at 0.5 µm is then computed from that current with Biot–Savart, so the physics step is exact.
- An encoder shrinks the 64×64 image three times (64 → 128 → 256 → 512 channels, three conv–BatchNorm–ReLU blocks per level). Two separate decoders grow it back, one for Jx and one for Jy, with skip connections.
- 15.5 million parameters. The layer graph matches the authors' released trained model exactly, checked layer by layer.
- The network is never told the standoff. Each training image gets a random standoff (3 ± 0.6 µm), as in the paper.

Training:
- 153,600 crops of 64×64 pixels (0.2 µm pixels) from 2,400 simulated current layouts. COMSOL was never used for training or tuning.
- Paper recipe: 50 epochs, batch 32, RMSprop, MSE.
- About 4 h on one RTX A6000, plus about 1.7 h to generate the data. Larger images are processed in overlapping 64×64 tiles.

Results: COMSOL, 3 → 0.5 µm, mean over 15 pairs (5 coil configurations × Bx/By/Bz). Relative L2 = ‖output − target‖ / ‖target‖; NRMSE = RMSE / (max − min of the target). Lower is better.
- MAGIC-UNet + one data-consistency step (our addition): L2 24.3%, NRMSE 1.74% (L2 over 3 seeds: 24.3 / 24.5 / 24.5%)
- FISTA-1000, learned dictionary: L2 26.7%, NRMSE 1.92%
- MAGIC-UNet as in the paper: L2 27.0%, NRMSE 1.93% (L2 over 3 seeds: 27.0 / 27.2 / 26.7%)
- FISTA-Net: L2 33.3%, NRMSE 2.41%
- CSC-LISTA, 64 layers: L2 33.8%, NRMSE 2.43%
- Tikhonov: L2 49.3%, NRMSE 3.53%
- The 3 µm input itself, for reference: L2 74.1%, NRMSE 5.29%

FISTA, FISTA-Net, CSC-LISTA and Tikhonov are given the 3 µm standoff; MAGIC-UNet is not.

Data-consistency step (our addition, not in the paper): we blur the network's 0.5 µm output back up to 3 µm, compare it with the actual measurement, and apply one Tikhonov correction for the mismatch. Why: the network fills in lost fine detail better than FISTA, but nothing in it forces agreement with the measured data. Blurred back up, its output missed the real 3 µm input by 17–33%, so it was worst exactly in the frequencies we did measure. The correction uses a height the network estimates from its own current, and its strength was chosen on simulated data.

Attached:
- Side-by-side Bz maps for L1_C1 and IC_4um: target, the 3 µm input, every reconstruction, and the error maps, each labelled with L2 and NRMSE
- Line cuts through the middle of each crop, with L2 and NRMSE along the line

L1_C1, Bz. Full image: L2 / NRMSE. Line cut: L2 / NRMSE.
- Input (3 µm): 76.6% / 4.52%. Line: 77.5% / 18.97%
- MAGIC-UNet + data consistency: 26.3% / 1.56%. Line: 24.2% / 5.91%
- MAGIC-UNet: 28.5% / 1.68%. Line: 26.4% / 6.45%
- FISTA-1000: 29.8% / 1.76%. Line: 29.0% / 7.10%
- FISTA-Net: 38.2% / 2.26%. Line: 39.2% / 9.59%
- CSC-LISTA: 38.0% / 2.24%. Line: 38.0% / 9.30%

IC_4um, Bz. Full image: L2 / NRMSE. Line cut: L2 / NRMSE.
- Input (3 µm): 73.7% / 6.26%. Line: 74.3% / 12.91%
- MAGIC-UNet + data consistency: 28.7% / 2.43%. Line: 25.5% / 4.44%
- MAGIC-UNet: 32.5% / 2.76%. Line: 32.4% / 5.63%
- FISTA-1000: 29.3% / 2.49%. Line: 30.4% / 5.29%
- FISTA-Net: 42.6% / 3.62%. Line: 42.5% / 7.39%
- CSC-LISTA: 35.9% / 3.05%. Line: 31.9% / 5.54%

(Line-cut NRMSE is larger than full-image NRMSE because it is all signal: the line crosses the wires, while most of the full image is empty space, which pulls the full-image RMSE down.)

Caveats:
- With the correction, MAGIC-UNet beats FISTA-1000 on all three seeds for the C1-family coils. On IC_4um the two are not distinguishable: one seed of three is below FISTA.
- The paper's network on its own ties FISTA-1000. The extra 2–3 points come from the data-consistency step.
- The jagged steps in the COMSOL target are mesh noise. About 16% of the target norm is above the frequency where it is mesh noise, and no method can reproduce that part.
