"""ComfyUI-PortraitDatasetTools

Nodes for building face-centric LoRA training sets: measure a portrait, decide
what it needs, and route it through only the work it actually needs.

The design idea is that every decision is a MEASUREMENT, not a guess, and every
expensive step sits behind a lazy branch so it runs only when the measurement
says so. See README.md for the full pipeline and the numbers behind the
default thresholds.

Licence: MIT. See LICENSE.
"""

import itertools
import os

import torch

_face_align_counter = itertools.count()


class FaceSimilarityScore:
    """
    Compares a candidate face image against a folder of reference images using
    InsightFace (ArcFace buffalo_l).

    approach="mean_embedding"  — average all reference embeddings into one centroid,
                                  then a single cosine similarity against it.
    approach="average_scores"  — cosine similarity against every reference embedding
                                  individually, then average those N scores.

    Reference embeddings are cached in memory keyed by (folder, mtime, file-count)
    so they are only recomputed when the folder contents change.

    First run downloads the buffalo_l model (~300 MB) to ~/.insightface/models/.
    """

    _app = None
    _ref_cache: dict = {}   # (folder_str, mtime, n_files) → {"embeddings": [...], "mean": ndarray}

    EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    @classmethod
    def _get_app(cls):
        if cls._app is None:
            from insightface.app import FaceAnalysis
            cls._app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            cls._app.prepare(ctx_id=0, det_size=(640, 640))
            print("[FaceSimilarityScore] InsightFace buffalo_l model ready")
        return cls._app

    @classmethod
    def _largest_embedding(cls, bgr):
        """
        Return the normalised ArcFace embedding of the largest face in a BGR image,
        or None if no face is found.

        Tight face crops (face fills the whole frame, no margin) defeat the SCRFD
        detector — its anchors expect the face to be a region within a larger scene.
        So on a raw miss we retry with a black border padded around the image, which
        recovers detection for edge-to-edge portrait crops.
        """
        import numpy as np
        import cv2

        app = cls._get_app()

        def best_face(img):
            faces = app.get(img)
            if not faces:
                return None
            return max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

        face = best_face(bgr)
        if face is None:
            # Retry with padding for tight/edge-to-edge crops.
            h, w = bgr.shape[:2]
            m = int(max(h, w) * 0.5)
            padded = cv2.copyMakeBorder(bgr, m, m, m, m, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            face = best_face(padded)
            if face is None:
                return None

        e = face.embedding
        return e / np.linalg.norm(e)

    @classmethod
    def _get_ref_data(cls, reference_folder):
        """Return (embeddings_list, mean_embedding), computing only on cache miss."""
        import numpy as np
        import cv2
        from pathlib import Path

        ref_dir = Path(reference_folder)
        if not ref_dir.is_dir():
            print(f"[FaceSimilarityScore] reference_folder not found: {reference_folder}")
            return [], None

        img_files = sorted(p for p in ref_dir.iterdir() if p.suffix.lower() in cls.EXTS)
        cache_key = (str(ref_dir), ref_dir.stat().st_mtime, len(img_files))

        if cache_key in cls._ref_cache:
            entry = cls._ref_cache[cache_key]
            print(f"[FaceSimilarityScore] cache hit — {len(entry['embeddings'])} reference embeddings")
            return entry["embeddings"], entry["mean"]

        embeddings = []
        n_missed = 0
        for p in img_files:
            img = cv2.imread(str(p))
            if img is None:
                continue
            emb = cls._largest_embedding(img)
            if emb is not None:
                embeddings.append(emb)
            else:
                n_missed += 1
                print(f"[FaceSimilarityScore] no face in reference {p.name}")

        if not embeddings:
            print("[FaceSimilarityScore] No faces detected in reference folder")
            return [], None

        mean_emb = np.mean(embeddings, axis=0)
        mean_emb /= np.linalg.norm(mean_emb)

        cls._ref_cache[cache_key] = {"embeddings": embeddings, "mean": mean_emb}
        print(f"[FaceSimilarityScore] computed + cached {len(embeddings)} embeddings from {reference_folder}")
        return embeddings, mean_emb

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference_folder": ("STRING", {"default": "/path/to/reference/images"}),
                "candidate_image":  ("IMAGE",),
                "approach": (["mean_embedding", "average_scores", "any_match"], {
                    "tooltip": (
                        "mean_embedding: average all reference embeddings into one centroid, "
                        "single comparison (fast, standard). "
                        "average_scores: compare to every reference individually and average "
                        "the N scores (more robust to outlier reference images). "
                        "any_match: KEEP if similarity >= threshold against ANY single reference "
                        "(most permissive — best-match score is reported)."
                    )
                }),
                "threshold":     ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                                            "tooltip": "Similarity ≥ this → KEEP. 0.45 is a good starting point."}),
                "keep_folder":   ("STRING", {"default": "vix_keep"}),
                "review_folder": ("STRING", {"default": "vix_review"}),
            }
        }

    RETURN_TYPES  = ("FLOAT", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES  = ("similarity", "passes_threshold", "output_folder", "score_label")
    FUNCTION      = "compare"
    CATEGORY      = "image/face"

    def compare(self, reference_folder, candidate_image, approach, threshold, keep_folder, review_folder):
        import numpy as np
        import cv2

        def tensor_to_bgr(t):
            arr = (t.cpu().numpy() * 255).astype(np.uint8)
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        ref_embeddings, ref_mean = self._get_ref_data(reference_folder)

        if not ref_embeddings:
            return (0.0, False, review_folder, "NO_REF_FACES")

        cand_emb = self._largest_embedding(tensor_to_bgr(candidate_image[0]))
        if cand_emb is None:
            print("[FaceSimilarityScore] No face detected in candidate")
            return (0.0, False, review_folder, "sim_0.000_NO_FACE")

        if approach == "mean_embedding":
            similarity = float(np.clip(np.dot(ref_mean, cand_emb), 0.0, 1.0))
        elif approach == "average_scores":
            scores = [float(np.clip(np.dot(e, cand_emb), 0.0, 1.0)) for e in ref_embeddings]
            similarity = float(np.mean(scores))
        else:  # any_match
            scores = [float(np.clip(np.dot(e, cand_emb), 0.0, 1.0)) for e in ref_embeddings]
            similarity = float(max(scores))

        passes = similarity >= threshold
        folder = keep_folder if passes else review_folder
        label  = f"sim_{similarity:.3f}_{'KEEP' if passes else 'REVIEW'}"

        print(f"[FaceSimilarityScore] approach={approach} sim={similarity:.3f} "
              f"threshold={threshold} → {folder}")
        return (similarity, passes, folder, label)


class FaceAlignCrop:
    """
    Derotates a face to eye-level before cropping, so faces shot at a tilted
    camera/head angle (roll) come out upright instead of at whatever angle they
    were photographed at. Uses the same InsightFace (buffalo_l) detector as
    FaceSimilarityScore — its 5-point landmarks (left eye, right eye, nose,
    mouth corners) give a roll angle from the eye line, and the whole frame is
    rotated around the eye midpoint before the square crop is taken, so the
    crop never clips face content that a rotate-after-crop approach would lose
    at the corners.

    Only in-plane rotation (roll) is corrected — yaw/pitch (3/4 or angled
    up/down shots) is genuine 3D pose and can't be fixed by a 2D rotation.

    Drop-in replacement for NudeNetDetect+NudeNetCrop in a "face" branch: same
    (crops, labels, scores) output shape, so it can feed straight into the
    same ImageUpscaleWithModel → resize → save chain.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_size": (
                    ["native", "512", "768", "1024"],
                    {
                        "default": "native",
                        "tooltip": (
                            "native = return crop at original pixel dimensions "
                            "(recommended before upscaling). "
                            "512/768/1024 = resize to square."
                        ),
                    },
                ),
                "padding_pct": (
                    "FLOAT",
                    {"default": 0.15, "min": 0.0, "max": 0.5, "step": 0.05,
                     "tooltip": "Expand the aligned face box by this fraction of width/height on each side."},
                ),
            },
            "optional": {
                "save_path": (
                    "STRING",
                    {"default": "", "tooltip": "Directory to save cropped JPEGs directly. Empty = no save here."},
                ),
                "filename_prefix": ("STRING", {"default": "face_aligned"}),
                "clamp_before_upscale": (
                    "INT",
                    {"default": 512, "min": 0, "max": 2048, "step": 64,
                     "tooltip": "Only applies in native mode. If the crop's longest side exceeds this, scale it down before returning. 0 = disabled."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("crops", "labels", "scores")
    FUNCTION = "align_crop"
    CATEGORY = "image/face"

    def align_crop(
        self,
        image,
        target_size,
        padding_pct=0.15,
        save_path="",
        filename_prefix="face_aligned",
        clamp_before_upscale=512,
    ):
        import json

        import cv2
        import numpy as np
        from PIL import Image as PILImage
        from comfy_execution.graph_utils import ExecutionBlocker

        if image is None:
            return (ExecutionBlocker(None), json.dumps([]), json.dumps([]))

        rgb = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        img_h, img_w = rgb.shape[:2]
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        app = FaceSimilarityScore._get_app()

        def best_face(im):
            faces = app.get(im)
            if not faces:
                return None
            return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        face = best_face(bgr)
        offset_x, offset_y = 0, 0
        if face is None:
            # Tight edge-to-edge crops defeat SCRFD's anchors — pad and retry,
            # same fallback as FaceSimilarityScore._largest_embedding.
            m = int(max(img_h, img_w) * 0.5)
            padded = cv2.copyMakeBorder(bgr, m, m, m, m, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            face = best_face(padded)
            if face is None:
                print("[FaceAlignCrop] No face detected — passing through blocked.")
                return (ExecutionBlocker(None), json.dumps([]), json.dumps([]))
            offset_x, offset_y = m, m

        kps = face.kps.copy().astype(np.float64)   # [left_eye, right_eye, nose, mouth_l, mouth_r]
        bbox = face.bbox.copy().astype(np.float64)  # [x1, y1, x2, y2]
        if offset_x or offset_y:
            kps[:, 0] -= offset_x
            kps[:, 1] -= offset_y
            bbox[0] -= offset_x
            bbox[2] -= offset_x
            bbox[1] -= offset_y
            bbox[3] -= offset_y

        left_eye, right_eye = kps[0], kps[1]
        dY = float(right_eye[1] - left_eye[1])
        dX = float(right_eye[0] - left_eye[0])
        angle = float(np.degrees(np.arctan2(dY, dX)))
        eyes_center = (float((left_eye[0] + right_eye[0]) / 2.0), float((left_eye[1] + right_eye[1]) / 2.0))

        M = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)
        # Reflect rather than fill black — rotating a rectangle around an interior point
        # always leaves gaps opposite the direction it swung; a face near the original
        # photo's edge pushes that gap into the crop's padding margin. There's no real
        # data beyond the photo edge either way, so mirror what's there rather than
        # showing a stark black wedge.
        rotated_bgr = cv2.warpAffine(
            bgr, M, (img_w, img_h), flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        # Map the original (unrotated) bbox corners through the same rotation so the
        # crop box tracks the now-upright face rather than the original tilted region.
        corners = np.array([
            [bbox[0], bbox[1]], [bbox[2], bbox[1]],
            [bbox[2], bbox[3]], [bbox[0], bbox[3]],
        ], dtype=np.float64)
        corners_h = np.concatenate([corners, np.ones((4, 1))], axis=1)
        new_corners = corners_h @ M.T
        nx1, ny1 = new_corners[:, 0].min(), new_corners[:, 1].min()
        nx2, ny2 = new_corners[:, 0].max(), new_corners[:, 1].max()

        pad_x = (nx2 - nx1) * padding_pct
        pad_y = (ny2 - ny1) * padding_pct
        x1, y1 = nx1 - pad_x, ny1 - pad_y
        x2, y2 = nx2 + pad_x, ny2 + pad_y

        # Force square, centered on the box — the whole point of the ask.
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        half = max(x2 - x1, y2 - y1) / 2.0
        x1, x2 = cx - half, cx + half
        y1, y2 = cy - half, cy + half

        x1i, y1i = int(round(max(0, x1))), int(round(max(0, y1)))
        x2i, y2i = int(round(min(img_w, x2))), int(round(min(img_h, y2)))
        if x2i <= x1i:
            x2i = x1i + 1
        if y2i <= y1i:
            y2i = y1i + 1

        crop_bgr = rotated_bgr[y1i:y2i, x1i:x2i]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = PILImage.fromarray(crop_rgb, mode="RGB")

        native = target_size == "native"
        size = None if native else int(target_size)
        if size is not None:
            crop_pil = crop_pil.resize((size, size), PILImage.LANCZOS)
        elif clamp_before_upscale > 0:
            cw, ch = crop_pil.size
            longest = max(cw, ch)
            if longest > clamp_before_upscale:
                scale = clamp_before_upscale / longest
                crop_pil = crop_pil.resize(
                    (max(1, int(cw * scale)), max(1, int(ch * scale))), PILImage.LANCZOS,
                )

        save_dir = save_path.strip()
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
            idx = next(_face_align_counter)
            fname = f"{filename_prefix}_{idx:04d}.jpg"
            crop_pil.save(os.path.join(save_dir, fname), format="JPEG", quality=95)
            print(f"[FaceAlignCrop] Saved {fname}")

        crop_tensor = torch.from_numpy(np.array(crop_pil).astype(np.float32) / 255.0).unsqueeze(0)
        score = round(float(face.det_score), 4)
        print(f"[FaceAlignCrop] roll_angle={angle:.2f}deg  det_score={score}  output={crop_pil.size}")

        return (crop_tensor, json.dumps(["FACE_ALIGNED"]), json.dumps([score]))


class PortraitQualityScore:
    """
    Scores the largest face in an image for LoRA-training suitability and
    emits routing signals for a refine-or-reject graph.

    Everything is measured on the face region at a normalised scale (see
    portrait_quality.py) so a 219px and a 6590px source get comparable
    numbers. Thresholds were set from the survey of the first dataset run
    through this (tools/portrait_quality_survey.py) -- rerun that on a new
    set before trusting the defaults.

    Routing outputs:
      image_pass    the image if accepted, else ExecutionBlocker -- hang the
                    crop/save chain off this and it only runs on keepers
      image_reject  the image if NOT accepted, else blocked -- hang the
                    _rejected/ save off this
      needs_refine  True when it failed but is worth a super-res pass; feed
                    LazyImageSelect so the SR branch only executes then
      sr_short_edge short-edge target for that SR pass (lands the face at
                    ~sr_face_target_px, clamped so the 16GB card copes)

    accepted_in / hard_reject_in let a chain of these pass a verdict down
    without rescoring: an image that passed stage 1 goes straight through
    stage 2 and 3; a hard reject never triggers an SR pass.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "hard_floor_face_px": ("INT", {"default": 200, "min": 0, "max": 4096,
                    "tooltip": "Face short edge below this is rejected outright, no refine (0 = off). Set on the FIRST stage only."}),
                "min_face_px": ("INT", {"default": 500, "min": 0, "max": 4096,
                    "tooltip": "Face short edge needed to pass."}),
                "min_clarity": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 10000.0, "step": 1.0,
                    "tooltip": "Variance of Laplacian on the face at normalised scale. ~20 = soft, ~60 = acceptable, 150+ = crisp."}),
                "max_grain": ("FLOAT", {"default": 2.5, "min": 0.0, "max": 50.0, "step": 0.1,
                    "tooltip": "Noise sigma on flat skin at normalised scale. <0.7 clean, >2.5 visibly grainy."}),
                "sr_face_target_px": ("INT", {"default": 800, "min": 256, "max": 2048,
                    "tooltip": "Face size the super-res pass should aim for."}),
                "sr_max_short_edge": ("INT", {"default": 2048, "min": 512, "max": 4096, "step": 16,
                    "tooltip": "Cap on the SR short edge. 2048 is the proven SeedVR2 7b setting on 16GB."}),
                "stage": ("STRING", {"default": "s1", "tooltip": "Label for the log / CSV row. Must end in the stage number (s1, s2, s3) for 'force' to work."}),
                "force": (["off", "accept", "refine_1x", "refine_2x"], {"default": "off",
                    "tooltip": "Manual override, ignores all thresholds. accept = keep as-is at stage 1; "
                               "refine_1x = one SR pass then keep; refine_2x = two SR passes then keep. "
                               "Wire one primitive into every stage so they agree."}),
            },
            "optional": {
                "accepted_in": ("BOOLEAN", {"default": False, "forceInput": True}),
                "hard_reject_in": ("BOOLEAN", {"default": False, "forceInput": True}),
                "filename": ("STRING", {"default": "", "forceInput": True}),
                "csv_path": ("STRING", {"default": "", "tooltip": "Append one row per scoring here (empty = off)."}),
                "min_source_grain": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Stage 1 only. Skin noise below this = no sensor noise at all: illustration, "
                               "airbrushed export, AI image. Hard reject, no SR (it would only produce wax). 0 = off."}),
                "min_texture_after_sr": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Stages 2+. Skin texture below this after super-res = waxy (the source had no real "
                               "pores to recover). Rejected with reason. 0 = off."}),
                "review_texture_below": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Stages 2+. Texture between min_texture_after_sr and this = accepted but the suffix "
                               "output becomes '_rev' so you can eyeball those files. 0 = off."}),
                "suffix_in": ("STRING", {"default": "", "forceInput": True,
                    "tooltip": "Previous stage's suffix output; carried through on passthrough so a '_rev' raised at s2 reaches the filename."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "BOOLEAN", "BOOLEAN", "BOOLEAN", "INT", "STRING", "STRING")
    RETURN_NAMES = ("image_pass", "image_reject", "accepted", "hard_reject", "needs_refine", "sr_short_edge", "report", "suffix")
    FUNCTION = "score"
    CATEGORY = "image/analysis"

    def score(self, image, hard_floor_face_px, min_face_px, min_clarity, max_grain,
              sr_face_target_px, sr_max_short_edge, stage, force="off",
              accepted_in=False, hard_reject_in=False, filename="", csv_path="",
              min_source_grain=0.15, min_texture_after_sr=0.6, review_texture_below=0.85, suffix_in=""):
        import csv
        import cv2
        import numpy as np
        from comfy_execution.graph_utils import ExecutionBlocker
        from . import portrait_quality as pq

        block = ExecutionBlocker(None)

        # Verdict already decided upstream -- pass it through untouched.
        if accepted_in:
            return (image, block, True, False, False, sr_max_short_edge, f"[{stage}] passthrough: accepted upstream", suffix_in or "")
        if hard_reject_in:
            return (block, image, False, True, False, sr_max_short_edge, f"[{stage}] passthrough: hard-rejected upstream", suffix_in or "")

        rgb = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        m = pq.analyse(bgr, FaceSimilarityScore._get_app())
        m.pop("embedding", None)
        accepted, hard, reasons = pq.verdict(m, min_face_px, min_clarity, max_grain, hard_floor_face_px)
        digits = "".join(ch for ch in stage if ch.isdigit())
        n = int(digits) if digits else 1
        suffix = ""
        g = m.get("grain", float("nan"))
        if m.get("face_found") and g == g:
            # Texture sanity. Immerkaer "grain" on skin doubles as a pore/texture
            # measure: real photographs never read < ~0.15 (sensor noise), and
            # skin that comes out of super-res under ~0.6 is wax -- the source had
            # nothing real to recover (survey of 345 SR'd faces, 2026-09-22).
            if n == 1 and min_source_grain > 0 and g < min_source_grain:
                accepted, hard = False, True
                reasons = [f"source grain {g} < {min_source_grain}: no sensor noise (illustration / airbrushed / AI)"]
            elif n >= 2 and min_texture_after_sr > 0 and g < min_texture_after_sr:
                accepted, hard = False, True
                reasons = [f"texture {g} < {min_texture_after_sr} after SR: waxy, nothing real to recover"]
            elif n >= 2 and accepted and review_texture_below > 0 and g < review_texture_below:
                suffix = "_rev"
                reasons = [f"review: texture {g} < {review_texture_below}"]
        if force != "off" and m.get("face_found"):
            # Manual override. Stage number comes from the label's trailing digit.
            want = {"accept": 0, "refine_1x": 1, "refine_2x": 2}[force]
            accepted, hard = (n > want), False
            reasons = [f"forced {force} (stage {n})"]
        needs_refine = (not accepted) and (not hard)
        sr = pq.sr_short_edge(m, sr_face_target_px, lo=1024, hi=sr_max_short_edge)

        state = "ACCEPT" if accepted else ("HARD-REJECT" if hard else "REFINE")
        metrics = "  ".join(f"{k}={m[k]}" for k in ("face_px", "face_area_pct", "clarity", "grain", "chroma") if k in m)
        report = f"[{stage}] {state}  {metrics}  sr={sr}" + (f"  because: {'; '.join(reasons)}" if reasons else "")
        print(f"[PortraitQualityScore] {filename or ''} {report}")

        if csv_path.strip():
            path = os.path.abspath(os.path.expanduser(csv_path.strip()))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            new = not os.path.exists(path)
            cols = ["file", "stage", "state", "img_w", "img_h", "face_px", "face_area_pct",
                    "det_score", "roll_deg", "clarity", "grain", "chroma", "hue_spread", "sr_short_edge", "reasons"]
            row = {k: m.get(k, "") for k in cols}
            row.update(file=filename, stage=stage, state=state, sr_short_edge=sr, reasons="; ".join(reasons))
            with open(path, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols)
                if new:
                    w.writeheader()
                w.writerow(row)

        return (image if accepted else block, block if accepted else image,
                accepted, hard, needs_refine, sr, report, suffix)


class LazyImageSelect:
    """
    Picks one of two images by a boolean -- and only EXECUTES the branch it
    picks. Uses ComfyUI's lazy-input mechanism, so an expensive upstream
    (SeedVR2, a diffusion pass) wired to the unchosen socket never runs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "condition": ("BOOLEAN", {"forceInput": True}),
                "on_true": ("IMAGE", {"lazy": True}),
                "on_false": ("IMAGE", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "select"
    CATEGORY = "utils"

    def check_lazy_status(self, condition, on_true=None, on_false=None):
        if condition:
            return ["on_true"] if on_true is None else []
        return ["on_false"] if on_false is None else []

    def select(self, condition, on_true=None, on_false=None):
        return (on_true if condition else on_false,)


class UpscaleIfSmaller:
    """
    Runs an upscale model only when the image's short edge is below
    min_short_edge. Lets one crop chain serve both a 3000px source (just
    downsample to 1024) and a 600px one (4x model first, then downsample)
    without paying the model cost on the big ones.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "upscale_model": ("UPSCALE_MODEL",),
                "min_short_edge": ("INT", {"default": 1024, "min": 64, "max": 8192}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN")
    RETURN_NAMES = ("image", "was_upscaled")
    FUNCTION = "run"
    CATEGORY = "image/upscaling"

    def run(self, image, upscale_model, min_short_edge):
        h, w = image.shape[1], image.shape[2]
        if min(h, w) >= min_short_edge:
            return (image, False)
        from comfy_extras.nodes_upscale_model import ImageUpscaleWithModel
        out = ImageUpscaleWithModel.execute(upscale_model, image)
        img = out.args[0] if hasattr(out, "args") else out[0]
        print(f"[UpscaleIfSmaller] {w}x{h} -> {img.shape[2]}x{img.shape[1]}")
        return (img, True)


class SaveImageWithName:
    """
    Saves images as PNG under the ORIGINAL file's stem (extension stripped)
    plus an optional suffix, into an explicit folder. Made for dataset
    pipelines where traceability back to the source file matters more than
    ComfyUI's prefix_00001 numbering.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "folder": ("STRING", {"default": ""}),
                "filename": ("STRING", {"default": "", "forceInput": True}),
                "suffix": ("STRING", {"default": ""}),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save(self, images, folder, filename, suffix="", overwrite=True):
        import numpy as np
        from PIL import Image as PILImage

        raw = folder.strip()
        if not raw:
            raise ValueError("[SaveImageWithName] folder is empty -- refusing to write to the working directory")
        folder = os.path.abspath(os.path.expanduser(raw))
        if folder == os.path.sep:
            raise ValueError("[SaveImageWithName] refusing to write to the filesystem root")
        os.makedirs(folder, exist_ok=True)

        stem = os.path.splitext(os.path.basename(filename.strip()))[0] or "image"
        paths = []
        for i in range(images.shape[0]):
            name = f"{stem}{suffix}" + (f"_{i}" if images.shape[0] > 1 else "") + ".png"
            path = os.path.join(folder, name)
            if os.path.exists(path) and not overwrite:
                base, n = path[:-4], 1
                while os.path.exists(f"{base}_{n}.png"):
                    n += 1
                path = f"{base}_{n}.png"
            arr = (images[i].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
            PILImage.fromarray(arr).save(path, compress_level=4)
            paths.append(path)
        print(f"[SaveImageWithName] wrote {len(paths)} -> {paths[0]}")
        return (paths[0],)


class ImageChromaCheck:
    """
    Detects black-and-white / sepia / tinted-monochrome images so a
    colourisation branch (DDColor) can be run only on those. Measures mean
    Lab chroma and circular hue spread over the whole image -- meant to sit
    on a head crop, after FaceAlignCrop.

    Outputs a ready-made filename suffix ("_col" when monochrome) so
    colourised training images stay identifiable on disk.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "max_chroma": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 50.0, "step": 0.1,
                    "tooltip": "Below this mean chroma = monochrome. B&W scans <3, first real colour ~7. 0 disables detection."}),
                "sepia_max_chroma": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 50.0, "step": 0.1,
                    "tooltip": "Up to this chroma still counts as monochrome IF the hue is uniform (sepia / toned prints)."}),
                "sepia_hue_spread": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 3.0, "step": 0.01,
                    "tooltip": "Circular hue spread below this = one dominant tint."}),
                "suffix_if_monochrome": ("STRING", {"default": "_col"}),
            },
            "optional": {
                "suffix_in": ("STRING", {"default": "", "forceInput": True,
                    "tooltip": "Prepended to the suffix output, so upstream tags (e.g. '_rev') survive."}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("is_monochrome", "chroma", "hue_spread", "suffix", "report")
    FUNCTION = "check"
    CATEGORY = "image/analysis"

    def check(self, image, max_chroma, sepia_max_chroma, sepia_hue_spread, suffix_if_monochrome, suffix_in=""):
        import cv2
        import numpy as np
        from . import portrait_quality as pq

        rgb = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        chroma, spread = pq.chroma_stats(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        mono = max_chroma > 0 and pq.is_monochrome(chroma, spread, max_chroma, sepia_max_chroma, sepia_hue_spread)
        report = f"{'MONOCHROME -> colourise' if mono else 'colour'}  chroma={chroma}  hue_spread={spread}"
        print(f"[ImageChromaCheck] {report}")
        return (bool(mono), chroma, spread, (suffix_in or "") + (suffix_if_monochrome if mono else ""), report)


class SkinToneMatch:
    """
    Normalises a portrait's colour cast by anchoring on SKIN, not on
    whole-image statistics. Measures mean Lab a/b on the cheek + forehead
    patches (same InsightFace landmarks the quality gate uses), then shifts
    a/b globally so that skin lands on the target; L is never touched, so
    exposure and detail are unchanged.

    Target = the same skin patches of an optional reference image, or the
    canonical values when no reference is wired. Made for DDColor outputs
    that came out as one orange/teal wash (chroma high, hue spread low) --
    a whole-image Reinhard match also drags luminance and background,
    this doesn't.

    only_if_hue_spread_below: apply only to images whose colour is a single
    wash (hue spread < this); naturally multi-hued photos pass through
    unchanged. 0 = always apply.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_a": ("FLOAT", {"default": 11.0, "min": -60.0, "max": 60.0, "step": 0.5,
                    "tooltip": "Skin Lab a (red-green) when no reference. MEASURE YOUR SET: the survey "
                               "script prints skin_a. typical medians 10-12 across sets; a is stable."}),
                "target_b": ("FLOAT", {"default": 14.0, "min": -60.0, "max": 60.0, "step": 0.5,
                    "tooltip": "Skin Lab b (yellow-blue) when no reference. THIS IS THE DATASET-SPECIFIC ONE: "
                               "measure YOUR set with tools/survey.py. Medians seen: 14.3 (warm/European), 9.2 (East Asian). Too high = skin goes yellow."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "only_if_hue_spread_below": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 3.0, "step": 0.01}),
            },
            "optional": {
                "reference": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "STRING")
    RETURN_NAMES = ("image", "applied", "report")
    FUNCTION = "match"
    CATEGORY = "image/postprocess"

    @staticmethod
    def _skin_mask(bgr, app):
        import numpy as np
        from . import portrait_quality as pq
        face, (ox, oy) = pq.largest_face(app, bgr)
        if face is None:
            return None
        k = face.kps.astype(np.float64).copy()
        k[:, 0] -= ox
        k[:, 1] -= oy
        le, re, no, lm, rm = k
        ipd = float(np.linalg.norm(re - le)) or 1.0
        half = int(max(6, 0.22 * ipd))
        m = np.zeros(bgr.shape[:2], bool)
        for c in ((le + lm) / 2 + (le - no) * 0.25, (re + rm) / 2 + (re - no) * 0.25, (le + re) / 2 - np.array([0.0, 0.75 * ipd])):
            x, y = int(c[0]), int(c[1])
            m[max(0, y - half):y + half, max(0, x - half):x + half] = True
        return m if m.any() else None

    def match(self, image, target_a, target_b, strength, only_if_hue_spread_below, reference=None):
        import cv2
        import numpy as np
        from . import portrait_quality as pq

        app = FaceSimilarityScore._get_app()
        rgb = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        chroma, spread = pq.chroma_stats(bgr)
        if only_if_hue_spread_below > 0 and spread >= only_if_hue_spread_below:
            return (image, False, f"skipped: hue_spread {spread} >= {only_if_hue_spread_below} (not a single-hue cast)")

        mask = self._skin_mask(bgr, app)
        if mask is None:
            return (image, False, "skipped: no face for skin anchor")
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        cur_a, cur_b = float(lab[..., 1][mask].mean() - 128), float(lab[..., 2][mask].mean() - 128)

        src = "canonical"
        if reference is not None:
            rbgr = cv2.cvtColor((reference[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            rmask = self._skin_mask(rbgr, app)
            if rmask is not None:
                rlab = cv2.cvtColor(rbgr, cv2.COLOR_BGR2LAB).astype(np.float32)
                target_a, target_b = float(rlab[..., 1][rmask].mean() - 128), float(rlab[..., 2][rmask].mean() - 128)
                src = "reference"

        da, db = (target_a - cur_a) * strength, (target_b - cur_b) * strength
        lab[..., 1] += da
        lab[..., 2] += db
        out = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        report = (f"applied: skin ab ({cur_a:.1f},{cur_b:.1f}) -> ({target_a:.1f},{target_b:.1f}) [{src}]  "
                  f"shift ({da:+.1f},{db:+.1f})  cast chroma={chroma} hue_spread={spread}")
        print(f"[SkinToneMatch] {report}")
        return (torch.from_numpy(out.astype(np.float32) / 255.0).unsqueeze(0), True, report)


class SkinTextureCheck:
    """
    Measures skin texture on a finished head crop and says whether it needs
    (or still needs) help. Companion to ImageChromaCheck: same shape, same
    suffix-chaining, so a workflow can gate a lazy SUPIR branch on it.

    Use it TWICE in a chain:
      pre  min_texture 0.85, suffix_if_low "_sup"  -> is_low drives the SUPIR branch
      post min_texture 0.70, suffix_if_low "_rev"  -> anything STILL low is flagged
                                                      for a human to look at
    So "_sup" = SUPIR ran and fixed it, "_sup_rev" = SUPIR ran and it is still
    marginal, no tag = it was fine to begin with.

    Reference points on a 1024 crop (face ~800px): wax/illustration below 0.55,
    SUPIR lifts a soft crop to ~0.7-1.1, real photographic pores 1.2-2.5.
    The measure cannot tell pores from dither -- see the artifacts note in
    3-REALISM_PASS_v2 -- so treat it as a gate, not a verdict.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "min_texture": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 5.0, "step": 0.01,
                    "tooltip": "Below this counts as lacking texture. 0 disables (always reports OK)."}),
                "suffix_if_low": ("STRING", {"default": "_sup"}),
            },
            "optional": {
                "suffix_in": ("STRING", {"default": "", "forceInput": True,
                    "tooltip": "Prepended to the suffix output so upstream tags survive."}),
                "label": ("STRING", {"default": "texture", "tooltip": "Shown in the log line."}),
            },
        }

    RETURN_TYPES = ("BOOLEAN", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("is_low", "texture", "suffix", "report")
    FUNCTION = "check"
    CATEGORY = "image/analysis"

    def check(self, image, min_texture, suffix_if_low, suffix_in="", label="texture"):
        import cv2
        import numpy as np
        from . import portrait_quality as pq

        rgb = (image[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        m = pq.analyse(bgr, FaceSimilarityScore._get_app())
        g = m.get("grain", float("nan"))
        if not m.get("face_found") or g != g:
            report = f"[{label}] no face / unmeasurable -- treated as OK"
            print(f"[SkinTextureCheck] {report}")
            return (False, 0.0, suffix_in or "", report)
        low = min_texture > 0 and g < min_texture
        report = f"[{label}] texture={g} {'LOW -> ' + suffix_if_low if low else 'ok'} (threshold {min_texture})"
        print(f"[SkinTextureCheck] {report}")
        return (bool(low), float(g), (suffix_in or "") + (suffix_if_low if low else ""), report)


class FaceIdentityGuard:
    """
    Safety net for an automated generative pass: compares the candidate to the
    image it came from by ArcFace embedding and, if the face has drifted too
    far, hands back the ORIGINAL instead.

    A restoration pass should not change who the person is. SUPIR measured
    0.94 on the tuned crop, but a batch left running unattended will meet
    inputs nothing was tuned on -- odd crops, heavy makeup, profile angles --
    and this is what stops a bad one reaching the training set silently.

    Returns the original unchanged when either face cannot be read, which is
    the conservative direction: no measurement, no substitution.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The candidate (e.g. the SUPIR result)."}),
                "reference": ("IMAGE", {"tooltip": "What it came from (e.g. the crop before SUPIR)."}),
                "min_similarity": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Below this, fall back to the reference. ArcFace cosine: >0.9 is "
                               "comfortably the same person; a good SUPIR pass measures ~0.94."}),
                "suffix_if_reverted": ("STRING", {"default": "_noid"}),
            },
            "optional": {
                "suffix_in": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    RETURN_TYPES = ("IMAGE", "BOOLEAN", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = ("image", "passed", "similarity", "suffix", "report")
    FUNCTION = "guard"
    CATEGORY = "image/analysis"

    def guard(self, image, reference, min_similarity, suffix_if_reverted, suffix_in=""):
        import cv2
        import numpy as np
        from . import portrait_quality as pq

        app = FaceSimilarityScore._get_app()

        def embed(t):
            rgb = (t[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            face, _ = pq.largest_face(app, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            return getattr(face, "normed_embedding", None) if face is not None else None

        a, b = embed(image), embed(reference)
        if a is None or b is None:
            report = "no face in one side -- keeping the reference (conservative)"
            print(f"[FaceIdentityGuard] {report}")
            return (reference, False, 0.0, (suffix_in or "") + suffix_if_reverted, report)

        sim = float(np.dot(a, b))
        ok = sim >= min_similarity
        report = (f"identity {sim:.3f} {'ok' if ok else f'< {min_similarity} -- REVERTED to the original'}")
        print(f"[FaceIdentityGuard] {report}")
        return (image if ok else reference, ok, sim,
                (suffix_in or "") + ("" if ok else suffix_if_reverted), report)


class MultiDirImageBatch:
    """Load Image Batch variant that accepts multiple directories (one per line).

    Maps a global incrementing index across all listed directories in order, so an
    entire multi-directory dataset can be processed without manually changing the
    path or resetting the counter between directories.

    Connect a PrimitiveNode (set to 'increment') to the index input.
    When the index reaches the total image count the node raises IndexError,
    stopping the workflow exactly as WAS Load Image Batch does at end-of-batch.
    """

    CATEGORY = "nudenet"
    RETURN_TYPES = ("IMAGE", "STRING", "INT", "STRING")
    RETURN_NAMES = ("image", "filename_text", "total_count", "current_dir")
    FUNCTION = "load"

    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif", ".gif"}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directories": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "One absolute directory path per line. "
                            "Images are indexed sequentially across all directories in the listed order."
                        ),
                    },
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 999999,
                        "step": 1,
                        "tooltip": (
                            "Global image index across all directories. "
                            "Connect a PrimitiveNode set to 'increment'."
                        ),
                    },
                ),
                "pattern": (
                    "STRING",
                    {
                        "default": "*",
                        "multiline": False,
                        "tooltip": "Glob pattern for filenames within each directory, e.g. '*.jpg' or '*'.",
                    },
                ),
            }
        }

    @classmethod
    def _collect(cls, directories, pattern):
        """The image list a given (directories, pattern) resolves to."""
        import glob
        from pathlib import Path
        dirs = [os.path.expanduser(d.strip()) for d in directories.strip().splitlines() if d.strip()]
        out = []
        for d in dirs:
            if not os.path.isdir(d):
                continue
            out.extend(f for f in sorted(glob.glob(os.path.join(d, pattern)))
                       if Path(f).suffix.lower() in cls._IMAGE_EXTS)
        return dirs, out

    @classmethod
    def VALIDATE_INPUTS(cls, directories, index, pattern, **kwargs):
        """Reject an exhausted batch at VALIDATION time, not execution time.

        Raising during execution only fails that run, and the frontend's
        auto-queue ("instant" mode) re-submits from the client regardless of
        anything the server does to its own queue -- so the index climbs
        forever with nothing left to process. A prompt rejected at validation
        never enters the queue at all, which is what actually stops it.
        """
        try:
            dirs, images = cls._collect(directories, pattern)
        except Exception:
            return True          # let execution report the real problem
        if not dirs:
            return "[MultiDirImageBatch] No directories specified."
        if not images:
            return f"[MultiDirImageBatch] No images found across {len(dirs)} director(y/ies)."
        if index >= len(images):
            return (f"[MultiDirImageBatch] BATCH COMPLETE: index {index} >= {len(images)} images. "
                    f"Turn auto-queue off and reset the index to 0 for the next run.")
        return True

    def load(self, directories, index, pattern):
        import glob
        from pathlib import Path

        dirs = [os.path.expanduser(d.strip()) for d in directories.strip().splitlines() if d.strip()]
        if not dirs:
            raise ValueError("[MultiDirImageBatch] No directories specified.")

        all_images = []
        for d in dirs:
            if not os.path.isdir(d):
                logger.warning("[MultiDirImageBatch] Directory not found, skipping: %s", d)
                continue
            matches = sorted(glob.glob(os.path.join(d, pattern)))
            imgs = [f for f in matches if Path(f).suffix.lower() in self._IMAGE_EXTS]
            all_images.extend(imgs)

        total = len(all_images)
        if total == 0:
            raise ValueError(
                f"[MultiDirImageBatch] No images found across {len(dirs)} "
                f"director{'y' if len(dirs) == 1 else 'ies'}."
            )

        if index >= total:
            # Raising only fails THIS run; every run still queued behind it would
            # fail the same way, one after another, while the index keeps
            # climbing. Drop the pending queue first so the batch actually ends.
            dropped = self._wipe_pending_queue()
            raise IndexError(
                f"[MultiDirImageBatch] Index {index} >= total {total}: batch complete "
                f"({len(dirs)} director{'y' if len(dirs) == 1 else 'ies'} processed). "
                f"Cleared {dropped} pending run{'s' if dropped != 1 else ''} from the queue."
            )

        if index == 0:
            logger.info("[MultiDirImageBatch] %d images across %d director%s -- queue exactly %d runs",
                        total, len(dirs), "y" if len(dirs) == 1 else "ies", total)

        img_path = all_images[index]
        pil_img = Image.open(img_path).convert("RGB")
        img_tensor = torch.from_numpy(
            np.array(pil_img).astype(np.float32) / 255.0
        ).unsqueeze(0)

        filename = os.path.basename(img_path)
        current_dir = os.path.dirname(img_path)
        logger.debug("[MultiDirImageBatch] index=%d/%d  file=%s", index, total - 1, img_path)

        return (img_tensor, filename, total, current_dir)

    @staticmethod
    def _wipe_pending_queue():
        """Remove every prompt still waiting in the server queue. Returns how
        many were dropped (0 if not running inside a ComfyUI server)."""
        try:
            from server import PromptServer
            q = PromptServer.instance.prompt_queue
        except Exception:
            return 0
        with q.mutex:
            pending = len(q.queue)
        if pending:
            q.wipe_queue()
        return pending
