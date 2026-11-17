# Group update: the 10 µm COMSOL mystery, solved

Attach: `resolve_vs_old_maps.png` (the three versions of the coil) and `resolved_reconstruction_L1_C1.png` (MAGIC-UNet on both).

---

Hi everyone! A small detective story from the last few days that I think is worth sharing 🕵️

**The puzzle.** For the 10 µm → 0.5 µm work I generate training data with my own field simulator: a thin current sheet, Biot–Savart, the same physics COMSOL solves. But at 10 µm it was off by 60–130% (65 / 61 / 134% for Bx / By / Bz on L1_C1). Same coil, same physics, yet completely different fields. That didn't make sense to me, and I couldn't let it go.

**Digging in.** Thanks to Hari for sending me the COMSOL model files. Going through the exported fields frequency by frequency, I found that the COMSOL field fell off faster with height than physics allows, as if an extra decay of about 0.1 rad/µm had been added on top. It also failed Ampère's law in the air above the coil by 27–52%, where exact physics gives about 1%. So the question became: is it my simulator, or the export?

**The test.** Ritaban re-solved the same model at a much tighter solver tolerance (1e-8; the original used the default). With the re-solve:
- the extra decay disappears;
- at 10 µm the original export is 42 / 39 / 58% off the re-solve;
- **my simulator is 3 / 3 / 9% off it, with nothing fitted** (first image: original | re-solve | simulator).

So the simulator was right all along, and the original export was off.

**Why only at 10 µm?** An iterative solver that stops early leaves a smooth, slowly varying error. At 0.5 µm the field is mostly fine detail and that error barely matters (only 2% of its energy is in broad structure). At 10 µm, 68% of the field is broad structure, exactly where the leftover error sits.

**The payoff.** Same network (MAGIC-UNet + data consistency), same coil, only the input changed (second image):

| 10 → 0.5 µm, L1_C1 | Original export | Re-solved |
|---|---|---|
| Relative L2 error (mean of Bx, By, Bz) | 60.4% | **26.2%** |
| Height the network infers (true 10 µm) | 8.25 µm | 9.72 µm |

On the original input it recovered the coil at less than half its strength. On the re-solved input the amplitude is right and only the sharpest edges are missing, and those just aren't in a 10 µm field. (At 3 µm the change is small, 23.9% → 22.0%, as you'd expect.)

**Caveats, honestly:**
- This is one coil shape and one training seed.
- FISTA and the other baselines are still being scored on the re-solved data, so I'm **not** claiming yet that MAGIC-UNet beats them on COMSOL. Results in a day or so.
- The 15.8% number I mentioned before is on simulated layouts; 26.2% is the first real COMSOL check at 10 µm.
- Only L1_C1 has been re-solved so far. The other configurations probably need the same tight tolerance before we trust them at large standoffs.

The lesson I'm taking away: when a careful model and a trusted reference disagree, check the reference too. Happy to walk anyone through the plots!
