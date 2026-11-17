# Follow-up for the group: every method on the re-solved coil

Attach: `resolved_methods_L1_C1.png` (copied into this folder).

---

Quick follow-up to the COMSOL story: all the other methods have now been scored on the re-solved coil too, and the result surprised me 👀

| L1_C1, 10 → 0.5 µm (mean of Bx, By, Bz) | Original export | Re-solved |
|---|---|---|
| MAGIC-UNet (+ data consistency) | 60.4% | **26.2%** |
| FISTA, 3,000 iterations (depth chosen at 10 µm, 3 µm dictionary) | 49.8% | 47.2% |
| FISTA-Net (3 µm settings) | 90.6% | 90.6% |
| CSC-LISTA (3 µm settings) | 99.6% | 98.0% |

On the original export FISTA was ahead. On the correctly solved field **the order reverses**: MAGIC-UNet improves by 34 points, FISTA by only 2.6. FISTA now sits close to its score on simulated data (48.8%); MAGIC-UNet is still about 10 points above its own (15.8%). At 3 µm everything moves only a little (MAGIC-UNet 22.0%, FISTA 25.8%).

To be careful about what this shows:
- It is one coil shape and one training seed, so it's a strong hint, not a general result. The other COMSOL configurations still need re-solving at the tight tolerance.
- At 10 µm the comparison is **not like-for-like**: only MAGIC-UNet was adapted to the larger scale (bigger pixels, so it sees a 44.8 µm patch). FISTA-Net and CSC-LISTA ran with their 3 µm settings and basically hand back the blurred input. Their numbers say "the 3 µm setup doesn't transfer", not "the architecture can't work".
- I haven't tested why FISTA barely moves.
