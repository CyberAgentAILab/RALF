import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from .design_seq import box_cxcywh_to_xyxy, reorder

# PKU10: {'logo': 0, 'text': 1, 'underlay': 2, "bg": 3}
# PKU  : {'logo': 2, 'text': 1, 'underlay': 3, "bg": 0}
conversion_dict_pku = {
    0: 2,
    2: 3,
    1: 1,
    3: 0,
}
mapping_pku = torch.tensor(
    [conversion_dict_pku[i] for i in range(len(conversion_dict_pku))]
)

# {'embellishment': 0, 'logo': 1, 'text': 2, 'underlay': 3, 'bg': 4}
# PKUのSort順に合わせる. {"text": 1, "logo": 2, "deco": 3, "bg": 0}であればOK
conversion_dict_cgl = {
    0: 4,
    2: 2,
    1: 1,
    3: 3,
    4: 0,
}
mapping_cgl = torch.tensor(
    [conversion_dict_cgl[i] for i in range(len(conversion_dict_cgl))]
)


def preprocess_layout(
    batch: dict[str, Tensor],
    max_elem: int = 32,
    num_classes: int = 4,
    use_reorder: bool = False,
) -> dict[str, Tensor]:
    # Layout
    batch["label"][batch["mask"] == False] = num_classes - 1
    label_onehot = F.one_hot(
        batch["label"], num_classes=num_classes
    )  # [bs, max_elem, 4]
    boxes_cxcywh = torch.stack(
        [
            batch["center_x"],
            batch["center_y"],
            batch["width"],
            batch["height"],
        ],
        dim=-1,
    )
    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    if use_reorder:
        layout = []
        for idx, (box_xyxy, box_cxcywh, _label_cate, _mask) in enumerate(
            zip(boxes_xyxy, boxes_cxcywh, batch["label"], batch["mask"])
        ):

            if num_classes == 4:  # PKU dataset
                mapped_label_cate = mapping_pku[_label_cate.cpu()]
            elif num_classes == 5:  # CGL dataset
                mapped_label_cate = mapping_cgl[_label_cate.cpu()]
                # Need a padding for bbox
                pad_box = torch.zeros_like(box_xyxy)[..., 0:1]
                box_xyxy = torch.cat([box_xyxy, pad_box], dim=-1)

            if isinstance(box_xyxy, Tensor):
                box_xyxy = box_xyxy.detach().cpu()

            mapped_label_cate = list(
                map(float, mapped_label_cate.detach().cpu().numpy().tolist())
            )
            order = reorder(mapped_label_cate, box_xyxy, "xyxy", max_elem)

            label = np.zeros((max_elem, 2, num_classes))
            for i in range(len(order)):
                idx = order[i]
                label[i][0][int(_label_cate[idx])] = 1
                label[i][1] = box_xyxy[idx]
                if label[i][1][0] > label[i][1][2] or label[i][1][1] > label[i][1][3]:
                    label[i][1][:2], label[i][1][2:] = label[i][1][2:], label[i][1][:2]
                label[i][1] = box_xyxy_to_cxcywh_with_pad(torch.tensor(label[i][1]))
            for i in range(len(order), max_elem):
                label[i][0][0] = 1
            _layout = torch.tensor(label).float()
            layout.append(_layout)

        layout = torch.stack(layout, dim=0)  # [bs, max_elem, 2, 4]
    else:
        if label_onehot.size(-1) == 5:  # CGL dataset
            pad_box = torch.zeros_like(boxes_cxcywh)[..., 0:1]
            boxes_cxcywh = torch.cat([boxes_cxcywh, pad_box], dim=-1)
        layout = torch.stack([label_onehot, boxes_cxcywh], dim=2)
    batch["layout"] = layout

    # Input image
    batch["image_saliency"] = torch.cat(
        [batch["image"], batch["saliency"]], dim=1
    )  # [bs, 4, H, W]

    return batch


def box_xyxy_to_cxcywh_with_pad(x: Tensor) -> Tensor:
    use_pad = False
    if x.size(-1) > 4:
        use_pad = True
        pad = x[..., 4:]
        x0, y0, x1, y1 = x[..., :4].unbind(-1)
    else:
        x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    out = torch.stack(b, dim=-1)
    if use_pad:
        out = torch.cat([out, pad], dim=-1)
    return out


def box_xyxy_to_cxcywh(x: Tensor) -> Tensor:
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def random_init_layout(
    batch_size: int,
    seq_length: int,
    coef: list[float],
    use_reorder: bool,
    num_classes: int,
) -> Tensor:
    cls_1 = torch.tensor(
        np.random.choice(
            num_classes, size=(batch_size, seq_length, 1), p=np.array(coef) / sum(coef)
        )
    )
    cls = torch.zeros((batch_size, seq_length, num_classes))  # [bs, 32, 4]
    cls.scatter_(-1, cls_1, 1)  # one-hot vector
    box_xyxy = torch.normal(0.5, 0.15, size=(batch_size, seq_length, 1, 4))
    box = box_xyxy_to_cxcywh(box_xyxy)  # [bs, 32, 1, 4]
    if num_classes != box.size(-1):
        # Need a padding
        dim_diff = int(abs(num_classes - box.size(-1)))
        pad_box = torch.zeros_like(box)[..., 0:dim_diff]
        box = torch.cat([box, pad_box], dim=-1)

    init_layout = torch.concat([cls.unsqueeze(2), box], dim=2)

    if use_reorder:
        i_, j_ = cls.shape[:2]  # max_elem, 4
        for i in range(i_):
            for j in range(j_):
                if init_layout[i][j][0][0] == 1:
                    init_layout[i][j][1] = torch.zeros_like(init_layout[i][j][1])

        for i in range(i_):
            order = reorder(
                init_layout[i, :, 0].detach().cpu(),
                init_layout[i, :, 1].detach().cpu(),
                "cxcywh",
            )
            tmp = init_layout[i, :, 1].clone()
            for j in range(j_):
                init_layout[i][j][1] = tmp[int(order[j])]

    return init_layout
