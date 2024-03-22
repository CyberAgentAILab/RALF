GEO_KEYS = ["center_x", "center_y", "width", "height"]
RETRIEVED_KEYS = [
    "image",
    "saliency",
    "center_x",
    "center_y",
    "width",
    "height",
    "label",
    "mask",
]
DUMMY_LAYOUT = {
    "label": 0,
    "center_x": 0.5,
    "center_y": 0.5,
    "width": 0.05,
    "height": 0.05,
}
DUMMY_LAYOUT = {k: [v] for k, v in DUMMY_LAYOUT.items()}

PRECOMPUTED_WEIGHT_DIR = "./cache/PRECOMPUTED_WEIGHT_DIR"
