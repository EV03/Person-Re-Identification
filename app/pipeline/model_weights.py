"""Published reference weights; no project-specific fine-tuning.

MSMT17 combineall checkpoint linked by the authors' official Model Zoo.
The SHA-256 identifies the downloaded tensor file, not an accuracy claim.
"""

DEFAULT_CHECKPOINT_PATH = "data/models/osnet_x1_0_msmt17.pth"
DEFAULT_CHECKPOINT_SHA256 = "48df972f72887b95cf3b43b3a07c3a7d2398381aea0f9cae64a7ef11d512b727"
DEFAULT_CHECKPOINT_URL = "https://drive.google.com/uc?id=1IosIFlLiulGIjwW3H8uMRmx3MzPwf86x"
DEFAULT_CHECKPOINT_PROVENANCE = {
    "architecture": "osnet_x1_0", "training_dataset": "MSMT17 (combineall=True)",
    "project_fine_tuning": False,
    "model_zoo": "https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO",
    "download": DEFAULT_CHECKPOINT_URL,
    "original_filename": "osnet_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
}
