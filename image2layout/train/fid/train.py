import argparse
import logging
import os
import shutil
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from image2layout.train.config.dataset import dataset_config_factory
from image2layout.train.data import collate_fn, get_dataset
from image2layout.train.fid.data import BBOX_KEYS, generate_fake_and_real
from image2layout.train.fid.model import FIDNetV3
from image2layout.train.helpers.rich_utils import CONSOLE, get_progress
from image2layout.train.helpers.visualizer import get_colors, save_image
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"  # to see full tracelog for hydra


def save_checkpoint(state, is_best, out_dir):
    out_path = Path(out_dir) / "checkpoint.pth.tar"
    torch.save(state, out_path)

    if is_best:
        best_path = Path(out_dir) / "model_best.pth.tar"
        shutil.copyfile(out_path, best_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="pku")
    parser.add_argument(
        "--data-dir", type=str, default="~/datasets/image_conditioned_layout_generation"
    )
    parser.add_argument("--out-dir", type=str, default="tmp/jobs/pku_16/fid_weights")
    parser.add_argument("--batch-size", type=int, default=64, help="input batch size")
    parser.add_argument(
        "--iteration",
        type=int,
        default=int(2e5),
        help="number of iterations to train for",
    )
    parser.add_argument("--weight-decay", type=float, default=0.001)
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="learning rate, default=3e-4"
    )
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--seed", type=int, default=0, help="manual seed")
    parser.add_argument("--max_epoch", type=int, default=1600)
    args = parser.parse_args()
    CONSOLE.print(args)

    transforms = [
        "sort_lexicographic",
    ]
    dataset_cfg = OmegaConf.structured(dataset_config_factory(args.dataset))
    dataset_cfg.data_dir = os.path.join(args.data_dir, args.dataset)

    args.max_seq_length = dataset_cfg.max_seq_length

    dataset, features = get_dataset(
        dataset_cfg=dataset_cfg,
        transforms=transforms,
        remove_column_names=["image", "saliency"],
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(out_dir / "logs"))

    kwargs = {
        "num_workers": min(16, os.cpu_count()),
        "pin_memory": True,
        "collate_fn": partial(collate_fn, max_seq_length=dataset_cfg.max_seq_length),
        "persistent_workers": True,
        "drop_last": False,
    }

    train_dataloader = DataLoader(
        dataset["train"], shuffle=True, batch_size=args.batch_size, **kwargs
    )
    val_dataloader = DataLoader(
        dataset["val"],
        shuffle=False,
        batch_size=len(dataset["val"]),
        **kwargs,
    )
    test_dataloader = DataLoader(
        dataset["test"],
        shuffle=False,
        batch_size=len(dataset["test"]),
        **kwargs,
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    num_label = features["label"].feature.num_classes
    model = FIDNetV3(num_label=num_label, max_bbox=args.max_seq_length).to(device)

    # setup optimizer
    if args.optimizer == "adam":
        optimizer_class = optim.Adam
    elif args.optimizer == "adamw":
        optimizer_class = optim.AdamW
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

    optimizer = optimizer_class(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    criterion_bce = nn.BCEWithLogitsLoss(reduction="none")
    criterion_label = nn.CrossEntropyLoss(reduction="none")
    criterion_bbox = nn.MSELoss(reduction="none")

    iteration = 0
    best_loss = 1e8
    if args.max_epoch is not None:
        max_epoch = args.max_epoch
        CONSOLE.print(
            f"Training samples = {len(dataset['train'])} * {args.max_epoch} = {len(dataset['train']) * args.max_epoch}"
        )
    else:
        max_epoch = args.iteration * args.batch_size / len(dataset["train"])
        CONSOLE.print(f"{args.iteration} * {args.batch_size} / {len(dataset['train'])}")
        max_epoch = torch.ceil(torch.tensor(max_epoch)).int().item()
    for epoch in range(max_epoch):
        model.train()
        train_loss = {
            "Loss_BCE": [],
            "Loss_Label": [],
            "Loss_BBox": [],
        }

        train_data_len = len(dataset["train"])

        pbar = get_progress(
            train_dataloader,
            f"Epoch {epoch}/{max_epoch}",
            True,
        )
        for i, batch in enumerate(pbar):
            batch = generate_fake_and_real(batch)
            batch = {
                k: v.to(device) for (k, v) in batch.items() if isinstance(v, Tensor)
            }
            model.zero_grad()

            bbox = torch.stack([batch[key] for key in BBOX_KEYS], dim=-1)
            label, is_real, mask = batch["label"], batch["is_real"], batch["mask"]
            logit, logit_cls, bbox_pred = model(batch)

            loss_bce = criterion_bce(logit, is_real)
            loss_label = criterion_label(logit_cls[mask], label[mask])
            loss_bbox = criterion_bbox(bbox_pred[mask], bbox[mask]).sum(-1)
            loss = loss_bce.mean() + loss_label.mean() + 10 * loss_bbox.mean()
            loss.backward()

            optimizer.step()

            loss_bce_mean = loss_bce.mean().item()
            train_loss["Loss_BCE"].append(loss_bce.mean().item())
            loss_label_mean = loss_label.mean().item()
            train_loss["Loss_Label"].append(loss_label.mean().item())
            loss_bbox_mean = loss_bbox.mean().item()
            train_loss["Loss_BBox"].append(loss_bbox.mean().item())

            if i % 50 == 0:
                log_prefix = (
                    f"[{epoch}/{max_epoch}][{i}/{train_data_len // args.batch_size}]"
                )
                log = f"Loss: {loss.item():E}\tBCE: {loss_bce_mean:E}\t"
                log += f"Label: {loss_label_mean:E}\tBBox: {loss_bbox_mean:E}"
                CONSOLE.print(f"{log_prefix}\t{log}")

            iteration += 1

        for key in train_loss.keys():
            train_loss[key] = sum(train_loss[key]) / len(train_loss[key])

        model.eval()
        with torch.no_grad():

            tag_scalar_dict = {
                "train": {**train_loss},
                "test": {
                    "Loss_BCE": [],
                    "Loss_Label": [],
                    "Loss_BBox": [],
                },
                "val": {
                    "Loss_BCE": [],
                    "Loss_Label": [],
                    "Loss_BBox": [],
                },
            }

            for split, loader in zip(
                ["test", "val"], [test_dataloader, val_dataloader]
            ):

                for i, batch in enumerate(loader):
                    vis_H = int(batch["image_height"][0]) // 4
                    vis_W = int(batch["image_width"][0]) // 4

                    batch = generate_fake_and_real(batch)
                    batch = {
                        k: v.to(device)
                        for (k, v) in batch.items()
                        if isinstance(v, Tensor)
                    }

                    bbox = torch.stack([batch[key] for key in BBOX_KEYS], dim=-1)
                    label, is_real, mask = (
                        batch["label"],
                        batch["is_real"],
                        batch["mask"],
                    )
                    logit, logit_cls, bbox_pred = model(batch)

                    loss_bce = criterion_bce(logit, is_real)
                    loss_label = criterion_label(logit_cls[mask], label[mask])
                    loss_bbox = criterion_bbox(bbox_pred[mask], bbox[mask]).sum(-1)

                    tag_scalar_dict[split]["Loss_BCE"].append(loss_bce.mean().item())
                    tag_scalar_dict[split]["Loss_Label"].append(
                        loss_label.mean().item()
                    )
                    tag_scalar_dict[split]["Loss_BBox"].append(loss_bbox.mean().item())

                    if i == 0 and epoch % 10 == 0:
                        data = []
                        colors = get_colors(num_label)
                        data.append(("gt", (label, bbox.clamp(0.0, 1.0), mask)))
                        data.append(
                            (
                                "pred",
                                (
                                    logit_cls.argmax(dim=-1),
                                    bbox_pred.clamp(0.0, 1.0),
                                    mask,
                                ),
                            )
                        )
                        for key, (batch_labels, batch_bboxes, batch_masks) in data:
                            B = batch_labels.size(0)
                            save_image(
                                batch_images=torch.ones((B, 4, vis_H, vis_W)),
                                batch_bboxes=batch_bboxes,
                                batch_labels=batch_labels,
                                batch_masks=batch_masks,
                                colors=colors,
                                use_grid=True,
                                out_path=out_dir / f"{key}_samples_{split}_{epoch}.png",
                            )

                for key in tag_scalar_dict[split].keys():
                    tag_scalar_dict[split][key] = sum(
                        tag_scalar_dict[split][key]
                    ) / len(tag_scalar_dict[split][key])

        for split, _loss_dict in tag_scalar_dict.items():
            for k, v in _loss_dict.items():
                writer.add_scalar(f"{split}/{k}", v, iteration)

        # do checkpointing
        val_loss = sum(tag_scalar_dict["val"].values())
        is_best = val_loss < best_loss
        best_loss = min(val_loss, best_loss)

        if is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "best_loss": best_loss,
                    "optimizer": optimizer.state_dict(),
                },
                is_best,
                out_dir,
            )


if __name__ == "__main__":
    main()
