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
  └─ Preprocessing
  └─ Tactile mapping
  └─ Diffusion → heightfield_raw.npy
  └─ Fabrication corrector (memory-RAG) → heightfield_corrected.npy
  └─ Machining filter (deterministic)   → heightfield_machinable.npy
  └─ Geometry → tactile.stl
  └─ Fabrication check
```

### Hard manufacturability constraint: 6 mm tool diameter

The default milling tool is a **6 mm diameter ball-end mill** (radius = 3 mm).

- Any groove, channel, concavity, or recessed surface feature **narrower than
  6 mm** cannot be physically machined — the tool cannot fit inside.
- The machining filter suppresses these features automatically using
  **morphological opening** on the inverted heightfield with a disk footprint
  equal to the tool radius in pixels.
- Fabrication reports flag sub-6 mm feature violations in physical millimeters.

### Default parameters (6 mm ball-end mill)

| Parameter | Default | Notes |
|---|---|---|
| `tool_radius_mm` | 3.0 | Radius = diameter / 2 |
| `physical_size_mm` | 50.0 | Part XY footprint in mm |
| `max_height_mm` | 5.0 | Maximum relief depth in mm |
| `max_slope_deg` | 45.0 | 3-axis machining limit |
| `face_limit` | 500 000 | Fusion 360 CAM stability limit |
| `gaussian_sigma_px` | auto | tool_radius_mm / pixel_size_mm |

### Running the app

```bash
python src/app.py
```

- **Tab 3 — Diffusion:** Check **Apply machining filter** to run the filter
  immediately after generation. Side-by-side previews show raw vs filtered.
- **Tab 6 — Fabrication Check:** Use the **Heightfield source** dropdown
  (`auto` / `raw` / `machinable`) to compare before and after filtering.
  The machining filter report is shown as JSON.

### Running the CLI pipeline

```bash
python src/agent.py path/to/image.jpg
# Machining filter is ON by default (apply_machining_filter=True).
# Outputs:
#   outputs/heightfields/heightfield_raw.npy
#   outputs/heightfields/heightfield_machinable.npy
#   outputs/heightfields/machining_filter_report.json
#   outputs/stl_fabrication/tactile.stl
```

### Running tests

```bash
pytest tests/test_machining_filter.py -v
```

Tests cover: slope reduction, face-budget logic, shape validity, geometry
compatibility, report field population, JSON serialization, and narrow-recess
suppression (the 6 mm constraint).
