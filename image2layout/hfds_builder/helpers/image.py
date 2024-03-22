import io
import logging

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def write_image_to_bytes(image: Image.Image) -> bytes:
    with io.BytesIO() as f:
        # image.save(f, format="JPEG")
        image.save(f, "png", icc_profile=None)
        return f.getvalue()


def get_image_object(image: Image, size: tuple[int, int]):  # type: ignore
    image = image.resize(size)
    return [("bytes", write_image_to_bytes(image)), ("path", "png")]


CLS_COLOR_DICT = {1: "green", 2: "red", 3: "orange"}


def draw(image: Image.Image, example: dict) -> Image.Image:
    drawn_outline = image.copy()
    drawn_fill = image.copy()
    draw_ol = ImageDraw.ImageDraw(drawn_outline)
    draw_f = ImageDraw.ImageDraw(drawn_fill)

    W, H = example["image_width"], example["image_height"]

    for i in range(len(example["label"])):
        width, height = example["width"][i] * W, example["height"][i] * H
        left = example["center_x"][i] * W - width / 2.0
        top = example["center_y"][i] * H - height / 2.0
        xy = (left, top, left + width, top + height)

        label = example["label"][i] + 1
        if label > 0:
            color = CLS_COLOR_DICT[label]
            # TODO: suppress the following error
            # https://github.com/python-pillow/Pillow/blob/main/src/PIL/ImagePalette.py#L145-L147
            draw_ol.rectangle(xy, fill=None, outline=color, width=5)
            draw_f.rectangle(xy, fill=color)

    drawn_outline = drawn_outline.convert("RGBA")
    drawn_fill = drawn_fill.convert("RGBA")
    drawn_fill.putalpha(int(256 * 0.3))
    drawn = Image.alpha_composite(drawn_outline, drawn_fill)
    return drawn
