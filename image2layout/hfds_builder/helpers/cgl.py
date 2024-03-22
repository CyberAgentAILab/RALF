import json
import logging
import os
from collections import defaultdict

from .global_variables import MAX_SEQ_LENGTH
from .util import Coordinates, Element, Sample

logger = logging.getLogger(__name__)

# Note: for ease of pre-processing, layout_imgs_6w_1 and layout_imgs_6w_2 are merged

CGL_ID_NAME_MAPPING = {
    1: "logo",
    2: "text",
    3: "underlay",
    4: "embellishment",
}  # 5 (highlighted text) is not used

CGL_JSON_FILES = {
    "train": "layout_train_6w_fixed_v2.json",
    "validation": "layout_test_6w_fixed_v2.json",
    "test": "yinhe.json",
}
CGL_NG_KEYS: list[str] = []


def read_cgl(dataset_root: str, max_seq_length: int = MAX_SEQ_LENGTH) -> list[Sample]:
    logger.info("Loading CGL dataset ...")
    sample_list: list[Sample] = []

    for split in CGL_JSON_FILES:
        json_path = os.path.join(dataset_root, "annotation", CGL_JSON_FILES[split])
        with open(json_path, "r") as f:
            json_data = json.load(f)

        prefix = "test" if split == "test" else "train"
        # gather all the information based on image id as key
        image_info_dict = {}
        for ann in json_data["images"]:
            image_id = ann["id"]
            image_info_dict[image_id] = {
                "id": ann["file_name"].split(".")[0],
                "image_width": ann["width"],
                "image_height": ann["height"],
                # note: identifier will be removed later after splitting
                "identifier": f"{prefix}/{ann['file_name']}",
                "split": split,
            }

        object_info_dict = defaultdict(list)
        for anns in json_data["annotations"]:
            for ann in anns:
                cat_id = ann["category_id"]
                image_info = image_info_dict[ann["image_id"]]
                if cat_id in CGL_ID_NAME_MAPPING:
                    label = CGL_ID_NAME_MAPPING[cat_id]
                else:
                    continue

                coordinates = Coordinates.load_from_cgl_ltwh(
                    ltwh=ann["bbox"],
                    global_width=image_info["image_width"],
                    global_height=image_info["image_height"],
                )
                if coordinates.has_valid_area():
                    element = Element(label=label, coordinates=coordinates)
                    object_info_dict[ann["image_id"]].append(element)

        for id_, image_info in image_info_dict.items():
            if split == "test":
                # note: test split of CGL does not have annotation
                objects_ann = []
            else:
                objects_ann = object_info_dict[id_]
                N = len(objects_ann)
                if N == 0 or (max_seq_length and N > max_seq_length):
                    continue

            # make sure to cast since all ids are in int data format in CGL dataset
            image_info["id"] = str(image_info["id"])  # type: ignore

            sample = Sample(**image_info, elements=objects_ann)  # type: ignore
            sample_list.append(sample)

    logger.info("Done.")
    return sample_list
