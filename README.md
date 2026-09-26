# ComfyUI-PortraitDatasetTools

Measurement-driven nodes for building **face-centric LoRA training sets**.

Point a workflow at a folder of source photos and let it decide, per image,
what that image actually needs: reject it, super-resolve it, colourise it,
fix a colour cast, restore its skin texture — or leave it alone. Every
decision is a *measurement*, and every expensive step sits behind a lazy
branch so it only runs when the measurement says so.

The companion workflow is in [`example_workflows/`](example_workflows/).

---

## Why this exists

Curating a face dataset by hand doesn't scale, and the usual "run everything
through an upscaler" approach quietly makes things worse: it wastes hours on
images that were already fine, and it manufactures plausible-looking detail on
images that had none to begin with — which then gets baked into the LoRA.

These nodes measure first. The thresholds below aren't guesses; each one comes
from surveying real datasets, and the notes say what was measured and where the
number breaks down.

---

## The nodes

### Measurement

| node | what it measures | why you care |
|---|---|---|
| **Portrait Quality Score** | face size in native px, clarity (variance of Laplacian), skin grain, chroma | the main gate: accept / super-resolve / reject |
| **Skin Texture Check** | skin texture on a finished crop | gates a restoration branch, and flags what's still marginal afterwards |
| **Image Chroma Check** | mean Lab chroma + circular hue spread | detects B&W, sepia and toned monochrome |
| **Face Similarity Score** | ArcFace cosine vs a reference folder | identity checking |

Everything is measured **on the face region at a normalised scale**, not on the
whole frame — so a sharp face against bokeh passes, a sharp background behind a
soft face fails, and a 219px source is comparable with a 6590px one.

### Decision & safety

| node | what it does |
|---|---|
| **Lazy Image Select** | picks one of two images and *only executes the branch it picks* — this is what makes conditional pipelines cheap |
| **Face Identity Guard** | compares a generative pass's output to its input by ArcFace and hands back the original if the face drifted |

### Processing

| node | what it does |
|---|---|
| **Face Align Crop** | derotates to eye level using 5-point landmarks, then takes a square crop — rotation happens *before* cropping so corners are never clipped |
| **Skin Tone Match** | fixes a colour cast by shifting Lab a/b so the *skin* lands on a target. **L is untouched**, so exposure and detail survive |
| **Upscale If Smaller** | runs an upscale model only when the image is under a size floor |

### Plumbing

| node | what it does |
|---|---|
| **Multi-Dir Image Batch** | walks many directories by index, one image per queued run, and **stops cleanly at the end** |
| **Save Image With Name** | saves under the source file's stem plus an optional suffix |

---

## Install

**ComfyUI-Manager:** search for *Portrait Dataset Tools*.

**Manual:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BillyBGit/ComfyUI-PortraitDatasetTools
pip install -r ComfyUI-PortraitDatasetTools/requirements.txt
```

Requires `insightface`, `opencv-python`, `numpy`, `Pillow`, `onnxruntime`.
On first run InsightFace downloads its `buffalo_l` models (~300 MB) to
`~/.insightface/models/`.

---

## What the numbers mean

Measured on a 1024 crop where the face is ~800px:

| metric | range | reading |
|---|---|---|
| **clarity** | ~20 | visibly soft |
| | ~60 | acceptable |
| | 150+ | crisp |
| **grain / texture** | < 0.15 | *no sensor noise at all* — an illustration, an airbrushed export, or AI output |
| | 0.4–0.7 | waxy; a super-res pass invented this |
| | 1.2–2.5 | real photographic pores |
| **chroma** | < 3 | black and white |
| | 3–8 with hue spread < 0.45 | sepia / toned monochrome |
| | 7+ | genuine colour |
| **skin a / b** | a ≈ 10–12 | stable across datasets |
| | b ≈ 9–15 | **dataset-specific — measure yours** |

### Three things worth knowing before you trust a number

**Texture can't tell pores from dither.** A metric rise is necessary but not
sufficient. In testing, three separate things scored *higher* than real skin
while looking obviously wrong at 1:1 — a VAE that sprayed uniform dither, an
edit model that laid down a regular dot lattice, and a base model that simply
retained the input's chroma noise. **Always check at 1:1 before believing a
threshold change.**

**Face pixel size beats face-area percentage.** Since you're cropping to the
head anyway, what predicts the final crop is how many *native* pixels the face
has — not how much of the frame it fills. A 4000px landscape with a 300px face
is worse material than a 1200px headshot.

**Skin `b` is dataset-specific.** Setting `Skin Tone Match` to a target from
someone else's dataset will shift every face toward that dataset's tone. Run
the survey on your own set and read the median off it.

---

## Notes on specific nodes

**Face Align Crop** corrects roll only. Yaw and pitch are genuine 3D pose and a
2D rotation can't fix them. It pads-and-retries on tight edge-to-edge crops,
because face detectors miss faces that fill the frame.

**Lazy Image Select** relies on ComfyUI's lazy-input mechanism, which needs a
real output cache. **Do not run it with `--cache-none`**: the executor then
can't tell what has already run and re-executes a branch's whole upstream chain
every time it's requested.

**Multi-Dir Image Batch** rejects an exhausted batch at *validation* time, not
execution time. That matters — failing during execution only fails that run,
and the frontend's auto-queue re-submits from the client regardless, so the
index climbs forever. A prompt rejected at validation never enters the queue.
Never point an output folder at the directory being scanned: the loader
re-globs each run, so the batch would grow forever.

**Face Identity Guard** returns the *reference* when either face can't be read.
No measurement, no substitution — the conservative direction.

---

## Licence

MIT. Third-party models (InsightFace `buffalo_l`, and anything your workflow
loads) carry their own licences — check them before commercial use.
