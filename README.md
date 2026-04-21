# Tact_gen
Visual to Machinable Tactile

---

## Manufacturing Postprocessing

### Why postprocessing instead of retraining

The diffusion model learns the aesthetic mapping from visual texture to tactile
relief. Imposing hard machining constraints during training would complicate the
loss function and reduce model generality. Instead, a deterministic postprocessor
corrects the heightfield after generation — similar to how a slicer corrects a
CAD model after design.

### Pipeline stages

```
Input image
  └─ Diffusion (DDPM)            → heightfield.npy
  └─ Machining filter (ADC)      → heightfield_machinable.npy
  └─ Geometry                    → tactile.stl
  └─ Mockup (preview)            → preview.obj
  └─ Fabrication check           → PASS / FAIL report
```

### Machining filter: ADC-style quantisation

Rather than blurring slopes away, the filter **quantises heights into discrete
terrace levels** (like an ADC). Relief becomes a stack of flat plateaus joined
by near-vertical risers. Two benefits:

- **Plateaus have 0° slope** — trivially machinable.
- **Risers are ~90°** — machinable by the side of a ball-end mill. A 3-axis
  mill handles vertical walls fine; only undercuts (>90°) fail.

Pipeline inside the filter:

1. Normalise to [0, 1]
2. Downsample if needed to stay under the Fusion face budget
3. Mild Gaussian to prune sub-tool-scale noise
4. Morphological opening to kill grooves narrower than the tool diameter
5. Hard ADC quantisation into N discrete levels (no riser softening)

### Hard manufacturability constraint: 6 mm tool diameter

Default tool is a **6 mm diameter ball-end mill** (radius = 3 mm). Any groove
or concavity narrower than 6 mm cannot be machined; morphological opening
suppresses them automatically.

### Default parameters

| Parameter | Default | Notes |
|---|---|---|
| `tool_radius_mm` | 3.0 | Radius = diameter / 2 |
| `physical_size_mm` | 50.0 | Part XY footprint in mm |
| `max_height_mm` | 5.0 | Maximum relief depth in mm |
| `max_slope_deg` | 45.0 | Plateau-only slope target |
| `terrace_steps` | 0 (auto) | Auto derives from size / tool diameter |
| `face_limit` | 500 000 | Fusion 360 CAM stability limit |

### Running the app

```bash
python src/app.py
```

- **Tab 3 — Diffusion:** Generates raw heightfield via the trained DDPM.
- **Tab 3.5 — Machining Filter:** ADC quantisation. Shows plateau fraction,
  plateau slope, raw slope (with risers), and full JSON report.
- **Tab 4 — Geometry:** Heightfield → watertight STL.
- **Tab 5 — Mockup:** 256×256 OBJ preview with z-exaggeration.
- **Tab 6 — Fabrication Check:** Mesh slope / watertightness / face count /
  GRBL workspace check. Slope check masks near-vertical risers automatically.

### Running tests

```bash
pytest tests/test_machining_filter.py -v
```
