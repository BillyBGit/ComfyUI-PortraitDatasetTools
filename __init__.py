"""ComfyUI-PortraitDatasetTools — node registration."""

from .nodes import (
    FaceAlignCrop,
    FaceIdentityGuard,
    FaceSimilarityScore,
    ImageChromaCheck,
    LazyImageSelect,
    MultiDirImageBatch,
    PortraitQualityScore,
    SaveImageWithName,
    SkinTextureCheck,
    SkinToneMatch,
    UpscaleIfSmaller,
)

NODE_CLASS_MAPPINGS = {
    "MultiDirImageBatch":   MultiDirImageBatch,
    "PortraitQualityScore": PortraitQualityScore,
    "FaceAlignCrop":        FaceAlignCrop,
    "ImageChromaCheck":     ImageChromaCheck,
    "SkinToneMatch":        SkinToneMatch,
    "SkinTextureCheck":     SkinTextureCheck,
    "FaceIdentityGuard":    FaceIdentityGuard,
    "FaceSimilarityScore":  FaceSimilarityScore,
    "LazyImageSelect":      LazyImageSelect,
    "UpscaleIfSmaller":     UpscaleIfSmaller,
    "SaveImageWithName":    SaveImageWithName,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MultiDirImageBatch":   "Multi-Dir Image Batch (queue-driven, stops at the end)",
    "PortraitQualityScore": "Portrait Quality Score (face size / clarity / grain gate)",
    "FaceAlignCrop":        "Face Align Crop (derotate to eye level, square crop)",
    "ImageChromaCheck":     "Image Chroma Check (B&W / sepia detector)",
    "SkinToneMatch":        "Skin Tone Match (fix a colour cast, anchored on skin)",
    "SkinTextureCheck":     "Skin Texture Check (gate a restoration branch)",
    "FaceIdentityGuard":    "Face Identity Guard (revert a pass that drifted the face)",
    "FaceSimilarityScore":  "Face Similarity Score (ArcFace vs a reference folder)",
    "LazyImageSelect":      "Lazy Image Select (only runs the chosen branch)",
    "UpscaleIfSmaller":     "Upscale If Smaller (model upscale only when needed)",
    "SaveImageWithName":    "Save Image With Name (keeps the source stem)",
}

WEB_DIRECTORY = None
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
