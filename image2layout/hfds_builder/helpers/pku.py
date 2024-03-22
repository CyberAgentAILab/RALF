import logging
import os

import pandas as pd

from .global_variables import HEIGHT, MAX_SEQ_LENGTH, WIDTH
from .util import Coordinates, Element, Sample

logger = logging.getLogger(__name__)


PKU_ID_NAME_MAPPING = {0: "text", 1: "logo", 2: "underlay"}
PKU_CSV_FILES = {"train": "train_csv_9973.csv", "test": "test_csv_905.csv"}
PKU_NG_KEYS = [
    # invalid coordinate (extremely out of bounds)
    "train/183.png",
    "train/208.png",
    "train/827.png",
    # invalid label id
    "train/1478.png",
    "train/1739.png",
    "train/4038.png",
    "train/5821.png",
    "train/8145.png",
    "train/8433.png",
]


def _get_label(num: int) -> str:
    """
    Since PKU's label is 1-indxed, we need to convert it to 0-indexed.
    """
    label_id = num - 1
    assert label_id >= 0 and label_id < len(PKU_ID_NAME_MAPPING), label_id
    return PKU_ID_NAME_MAPPING[label_id]


def read_pku(dataset_root: str, max_seq_length: int = MAX_SEQ_LENGTH) -> list[Sample]:
    logger.info("Loading PKU dataset ...")
    sample_list: list[Sample] = []
    for split in PKU_CSV_FILES:
        csv_path = os.path.join(dataset_root, "annotation", PKU_CSV_FILES[split])
        df = pd.read_csv(csv_path)
        is_test = len(df.columns) == 1  # no annotation part in csv

        for i, (key, sub_df) in enumerate(df.groupby(by="poster_path")):
            if key in PKU_NG_KEYS:
                continue

            name = key.split("/")[-1]
            id_, _ = name.split(".")

            key = f"test/{key}" if split == "test" else key
            assert len(key.split("/")) == 2  # assume f"{split}/{id}.png"

            image_info = {
                "id": str(id_),
                "identifier": key,
                "image_width": WIDTH,
                "image_height": HEIGHT,
                "split": split,
            }

            elements = []
            if not is_test:
                for _, row in sub_df.iterrows():
                    label = _get_label(int(row.cls_elem))
                    coordinates = Coordinates.load_from_pku_ltrb(
                        box=eval(row.box_elem),
                        global_width=WIDTH,
                        global_height=HEIGHT,
                    )
                    if coordinates.has_valid_area():
                        elements.append(Element(label=label, coordinates=coordinates))

                N = len(elements)
                if N == 0 or (max_seq_length and N > max_seq_length):
                    continue
            data = Sample(**image_info, elements=elements)
            sample_list.append(data)

    logger.info("Done.")
    return sample_list
