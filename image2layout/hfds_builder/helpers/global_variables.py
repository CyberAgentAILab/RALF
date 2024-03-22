import datasets as ds

WIDTH = 513
HEIGHT = 750
WIDTH_RESIZE_IMAGE = 240
HEIGHT_RESIZE_IMAGE = 350
MAX_SEQ_LENGTH = 10

HFDS_FEATURES = ds.Features(
    {
        "id": ds.Value("string"),
        "image_width": ds.Value("int32"),
        "image_height": ds.Value("int32"),
        "image": ds.Image(),
        "saliency": ds.Image(),
        "label": ds.Sequence(ds.Value("string")),
        "center_x": ds.Sequence(ds.Value("float32")),
        "center_y": ds.Sequence(ds.Value("float32")),
        "width": ds.Sequence(ds.Value("float32")),
        "height": ds.Sequence(ds.Value("float32")),
    }
)

DISCRETE_FIELDS = [
    "label",
]

EMPTY_DATA = {  # type: ignore
    "center_x": [],
    "center_y": [],
    "width": [],
    "height": [],
    "label": [],
}
