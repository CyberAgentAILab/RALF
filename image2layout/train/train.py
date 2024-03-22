import logging
import os
import random
import sys
import time
from collections import defaultdict
from functools import partial

import datasets as ds
import hydra
import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
from fsspec.core import url_to_fs
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.distributed import destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image

from .config import init_train_config_store
from .data import collate_fn, get_dataset
from .helpers.distrubuted import DDPWrapper, ddp_setup
from .helpers.io import save_model
from .helpers.layout_tokenizer import init_layout_tokenizer
from .helpers.random_retrieval_dataset_wrapper import RandomRetrievalDatasetWrapper
from .helpers.retrieval_dataset_wrapper import RetrievalDatasetWrapper
from .helpers.rich_utils import get_progress
from .helpers.task import get_condition
from .helpers.util import set_seed
from .helpers.visualizer import render
from .models.common.base_model import (
    ConditionalInputsForDiscreteLayout,
    RetrievalAugmentedConditionalInputsForDiscreteLayout,
)
from .schedulers import requires_metrics

logger = logging.getLogger(__name__)
os.environ["HYDRA_FULL_ERROR"] = "1"  # to see full tracelog for hydra

torch.autograd.set_detect_anomaly(True)
ds.disable_caching()
cudnn.benchmark = True
cs = init_train_config_store()


@hydra.main(config_path="config", config_name="main", version_base="1.2")
def main(cfg: DictConfig) -> None:
    assert torch.cuda.is_available()

    world_size = torch.cuda.device_count()
    if world_size > 1 and not cfg.debug:
        logger.info(f"Launch distributed training using {world_size} GPUs")
        cfg.use_ddp = True
        mp.spawn(main_worker, nprocs=world_size, args=(world_size, cfg))
    else:
        logger.info("Launch normal training using 1 GPU")
        cfg.use_ddp = False
        # Simply call main_worker function
        main_worker(0, world_size, cfg)


def main_worker(rank: int, world_size: int, cfg: DictConfig) -> None:
    # set global setting
    set_seed(cfg.seed)
    os.environ["OMP_NUM_THREADS"] = str(cfg.training.num_workers)

    if cfg.use_ddp:
        logger.info(f"Initialize DDP for rank {rank}")
        ddp_setup(rank, world_size)

    job_dir = cfg.job_dir
    fs, _ = url_to_fs(job_dir)
    if not fs.exists(job_dir):
        fs.mkdir(job_dir)

    # Build writer
    if rank == 0:
        writer = SummaryWriter(os.path.join(job_dir, "logs"))
        # dump training setting
        with fs.open(os.path.join(job_dir, "config.yaml"), "wb") as file_obj:
            file_obj.write(OmegaConf.to_yaml(cfg).encode("utf-8"))
    else:
        writer = None

    # Build dataset
    max_seq_length = cfg.dataset.max_seq_length
    if max_seq_length < 0:
        max_seq_length = None
    dataset, features = get_dataset(
        dataset_cfg=cfg.dataset,
        transforms=list(cfg.data.transforms),
    )
    dataset_splits: list[str] = list(dataset.keys())

    # Build dataloader
    batch_size = cfg.training.batch_size
    loaders: dict[str, torch.utils.data.DataLoader] = {}
    use_retrieval_augment = "RetrievalAugmented" in cfg.generator._target_
    collate_fn_partial = partial(collate_fn, max_seq_length=max_seq_length)
    for split in dataset_splits:
        num_workers = 0 if cfg.debug else cfg.training.num_workers
        if cfg.use_ddp:
            _sampler = DistributedSampler(
                dataset[split], num_replicas=world_size, rank=rank, drop_last=True
            )
        else:
            _sampler = None

        if cfg.use_ddp:
            shuffle = False
        else:
            if split == "train":
                shuffle = True
            else:
                shuffle = False

        current_dataset = dataset[split]
        # TODO: Add ablation study for changing training data size
        if split == "train" and cfg.training.num_trainset is not None:
            logger.info(
                f"Cahnge {split} dataset size from {len(current_dataset)} to {cfg.training.num_trainset}."
            )
            # See https://huggingface.co/docs/datasets/v1.11.0/package_reference/main_classes.html?highlight=select#datasets.Dataset.select
            _random_indices = random.sample(
                range(len(current_dataset)), cfg.training.num_trainset
            )
            print(f"{len(_random_indices)=}")
            current_dataset = current_dataset.select(_random_indices)
            print(f"{len(current_dataset)=}")
            assert len(current_dataset) == cfg.training.num_trainset

        if use_retrieval_augment:

            if cfg.generator.random_retrieval:
                DatasetWrapperClass = RandomRetrievalDatasetWrapper
            else:
                DatasetWrapperClass = RetrievalDatasetWrapper
            logger.info(f"{DatasetWrapperClass=}")

            _dataset = DatasetWrapperClass(
                dataset_name=cfg.dataset.name,
                dataset=current_dataset,
                db_dataset=dataset["train"],
                split=split,
                top_k=cfg.generator.top_k,
                max_seq_length=max_seq_length,
                retrieval_backbone=cfg.generator.retrieval_backbone,
                random_retrieval=cfg.generator.random_retrieval,
                saliency_k=cfg.generator.saliency_k,
                inference_num_saliency=8,
            )
        else:
            _dataset = current_dataset

        loaders[split] = torch.utils.data.DataLoader(
            _dataset,
            shuffle=shuffle,
            num_workers=num_workers,
            batch_size=batch_size,
            pin_memory=True,
            collate_fn=collate_fn_partial,
            sampler=_sampler,
            persistent_workers=True if num_workers > 0 else False,
            drop_last=False,
        )
        logger.info(
            f"Create {split} loader with {len(loaders[split])} batches (shuffle={shuffle}), "
            f"i.e. {len(loaders[split])}x{batch_size}="
            f"{len(loaders[split]) * batch_size} samples"
        )

    # Build generator
    models = {}
    kwargs = {"features": features}
    if cfg.data.tokenization:
        kwargs["tokenizer"] = init_layout_tokenizer(
            tokenizer_cfg=cfg.tokenizer,
            dataset_cfg=cfg.dataset,
            label_feature=features["label"].feature,
        )
        use_sorted_seq = "shuffle" not in cfg.data.transforms
        kwargs["tokenizer"].use_sorted_seq = use_sorted_seq
    else:
        kwargs["max_seq_length"] = max_seq_length

    # RetrievalAugmentedAutoreg
    if use_retrieval_augment:
        kwargs["db_dataset"] = dataset["train"]
        kwargs["dataset_name"] = cfg.dataset.name
        kwargs["max_seq_length"] = max_seq_length

    logger.info(f"Instantiate: {cfg.generator['_target_']}")
    models["gen"] = instantiate(cfg.generator)(**kwargs)
    models["gen"].compute_stats()

    # Load or create cache
    if use_retrieval_augment:
        for split in dataset_splits:
            if cfg.generator.random_retrieval:
                continue
            assert (
                loaders[split].dataset.table_idx is not None
            ), f"table_idx is None in {split}."

    if cfg.use_ddp:
        models["gen"] = DDPWrapper(models["gen"].to(rank), device_ids=[rank])
    else:
        models["gen"] = models["gen"].to(rank)

    lr = cfg.training.lr
    optimizers: dict[str, torch.optim.Optimizer] = {}
    logger.info(f"Build generator's optimizer {cfg.optimizer['_target_']=}")
    # note: cfg.optimizer["weight_decay"] is not globally set.
    # weight_decay for each parameter is set by optim_groups (confirmed)
    optimizers["gen"] = instantiate(cfg.optimizer)(
        params=models["gen"].optim_groups(
            base_lr=lr,
            weight_decay=cfg.optimizer.weight_decay,
            custom_lr={"encoder.extractor.body": lr * 0.1},
        ),
    )
    logger.info(f"Build generator's scheduler {cfg.scheduler['_target_']=}")
    scheduler = instantiate(cfg.scheduler)(
        optimizer=optimizers["gen"],
        epochs=cfg.training.epochs,
        network="generator",
        dataset=cfg.dataset.name,
    )

    if not cfg.discriminator.get("is_dummy", False):
        models["dis"] = instantiate(cfg.discriminator)(features=features).to(rank)
        models["dis"].set_argmax(models["gen"].use_reorder)
        lr_dis = lr * models["dis"].LR_MULT
        logger.info(
            f"Build discriminator's optimizer {cfg.optimizer['_target_']=} with lr={lr_dis}"
        )
        optimizers["dis"] = instantiate(cfg.optimizer)(
            params=models["dis"].optim_groups(
                base_lr=lr_dis,
                weight_decay=cfg.optimizer.weight_decay,
                custom_lr={"encoder.extractor.body": lr_dis * 0.1},
            ),
        )
        logger.info(f"Build discriminator's scheduler {cfg.scheduler['_target_']=}")
        scheduler_dis = instantiate(cfg.scheduler)(
            optimizer=optimizers["dis"],
            epochs=cfg.training.epochs,
            network="discriminator",
        )

    best_val_loss = float("Inf")
    for epoch in range(1, cfg.training.epochs + 1):
        models["gen"].update_per_epoch(
            epoch, cfg.training.warmup_dis_epoch, cfg.training.epochs
        )

        # see warning of the following section to understand why its necessary
        # https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        if cfg.use_ddp:
            loaders["train"].sampler.set_epoch(epoch)

        start_time = time.time()

        train_loss_dict: dict[str, float] = train(
            models=models,
            data_loader=loaders["train"],
            optimizers=optimizers,
            writer=writer,
            cfg=cfg,
            epoch=epoch,
            rank=rank,
        )

        val_loss_dict: dict[str, float] = evaluate(
            models=models,
            data_loader=loaders["val"],
            cfg=cfg,
        )
        val_loss: float = val_loss_dict["total"]

        # some scheduler requires observation for update
        if requires_metrics(scheduler):
            scheduler.step(metrics=val_loss)
        else:
            scheduler.step()
        if "dis" in models:
            if requires_metrics(scheduler_dis):
                scheduler_dis.step(metrics=val_loss)
            else:
                scheduler_dis.step()

        if rank == 0:

            logger.info(
                "Epoch %d/%d: elapsed = %.1fs, train_loss = %.4f, val_loss = %.4f, "
                "learning_rate = %g"
                % (
                    epoch,
                    cfg.training.epochs,
                    time.time() - start_time,
                    train_loss_dict["G/total"],
                    val_loss,
                    scheduler.get_last_lr()[0],
                )
            )

            writer.add_scalar("gen_lr", scheduler.get_last_lr()[0], epoch)
            for k, v in train_loss_dict.items():
                writer.add_scalar(f"train_on_epoch/{k}", v, epoch)
            for k, v in val_loss_dict.items():
                writer.add_scalar(f"val_on_epoch/{k}", v, epoch)

            if "dis" in models:
                writer.add_scalar("dis_lr", scheduler_dis.get_last_lr()[0], epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # save_model(models["gen"], job_dir, best_or_final="best", prefix="gen")

            if (
                epoch % cfg.training.plot_generated_samples_epoch_interval == 0
                or cfg.debug
            ):

                # Determine "cond_type" for sampling
                cond_type = "uncond"
                if "AuxilaryTask" in type(models["gen"]).__name__:
                    if not models["gen"].use_multitask:
                        cond_type = models["gen"].auxilary_task

                logger.info(f"Sample layouts with cond_type={cond_type}")

                for _split in ["train", "val"]:
                    batch = next(iter(loaders[_split]))
                    cond, batch = get_condition(
                        batch=batch,
                        cond_type=cond_type,
                        tokenizer=kwargs.get("tokenizer", None),
                    )
                    if isinstance(
                        cond, ConditionalInputsForDiscreteLayout
                    ) or isinstance(
                        cond, RetrievalAugmentedConditionalInputsForDiscreteLayout
                    ):
                        cond = cond.to(rank)
                        sapmpling_batch_size = cond.image.size(0)
                    else:
                        cond = {
                            k: v.to(rank)
                            for k, v in cond.items()
                            if isinstance(v, torch.Tensor)
                        }
                        if use_retrieval_augment:
                            cond["retrieved"] = batch["retrieved"]

                        sapmpling_batch_size = cond["image"].size(0)

                    outputs = models["gen"].sample(
                        batch_size=sapmpling_batch_size,
                        cond=cond,
                        sampling_cfg=cfg.sampling,
                        cond_type=cond_type,
                        return_violation=False,
                    )
                    outputs["image"] = batch["image"].cpu()
                    for _set, _outputs in zip(["gt", "pred"], [batch, outputs]):
                        layout = render(
                            prediction=_outputs,
                            label_feature=models["gen"].features["label"].feature,
                        )

                        if (
                            cfg.run_on_local
                            and epoch % cfg.training.save_vis_epoch == 0
                        ):
                            out_path = os.path.join(
                                job_dir, f"layout_{_split}_{epoch}_{_set}.png"
                            )
                            logger.info(f"Save sampled layout to {out_path}")
                            save_image(layout, out_path, normalize=False)
                        else:
                            writer.add_image(
                                f"sampled_results_{_split}_{_set}",
                                layout,
                                epoch,
                            )

            if (epoch) % cfg.training.save_tmp_model_epoch == 0:
                save_model(
                    models["gen"],
                    job_dir,
                    best_or_final=f"epoch{epoch}",
                    prefix="gen",
                )

        if cfg.debug:
            logger.info("Debug mode, break training loop after 1 epoch")
            break

    if rank == 0:
        save_model(models["gen"], job_dir, best_or_final="final", prefix="gen")

    if cfg.use_ddp:
        destroy_process_group()


def train(
    models: dict[str, torch.nn.Module],
    data_loader: torch.utils.data.DataLoader,
    optimizers: dict[str, torch.optim.Optimizer],
    writer: SummaryWriter,
    cfg: DictConfig,
    epoch: int,
    rank: int,
) -> dict[str, float]:
    """Single training epoch."""
    for key in models:
        models[key].train()

    steps = 0
    pbar = get_progress(
        data_loader,
        f"Train [{epoch}/{cfg.training.epochs}]",
        run_on_local=cfg.run_on_local and not cfg.debug,
    )

    log_losses = defaultdict(list)

    for inputs in pbar:
        inputs, targets = models["gen"].preprocess(inputs)
        inputs = {
            k: v.to(rank) if torch.is_tensor(v) else v for (k, v) in inputs.items()
        }
        targets = {
            k: v.to(rank) if torch.is_tensor(v) else v for (k, v) in targets.items()
        }

        models["gen"].zero_grad()
        if "dis" in models:
            outputs_gen, losses = models["gen"].train_loss(
                inputs, targets, models["dis"]
            )
        else:
            outputs_gen, losses = models["gen"].train_loss(inputs, targets)

        loss = sum(losses.values())
        loss.backward()  # type: ignore
        if cfg.training.clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                models["gen"].parameters(), cfg.training.clip_max_norm
            )
        optimizers["gen"].step()
        steps += 1

        log_losses["G/total"].append(loss.cpu().item())
        for k, v in losses.items():
            log_losses[f"G/{k}"].append(v.cpu().item())

        if "dis" in models:
            models["dis"].zero_grad()

            _, losses = models["gen"].train_dis_loss(
                inputs, targets, outputs_gen, models["dis"]
            )
            loss = sum(losses.values())
            loss.backward()  # type: ignore
            if cfg.training.clip_max_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    models["dis"].parameters(), cfg.training.clip_max_norm
                )
            optimizers["dis"].step()

            log_losses["D/total"].append(loss.cpu().item())
            for k, v in losses.items():
                log_losses[f"D/{k}"].append(v.cpu().item())

        if rank == 0:
            iter_count = len(data_loader) * (epoch - 1) + steps
            if cfg.debug:
                break

        torch.cuda.empty_cache()

    for k, v in log_losses.items():
        log_losses[k] = sum(v) / len(v)

    return log_losses


@torch.no_grad()  # type: ignore
def evaluate(
    models: dict[str, torch.nn.Module],
    data_loader: torch.utils.data.DataLoader,
    cfg: DictConfig,
) -> dict[str, float]:
    """Evaluate the model."""
    for key in models:
        models[key].eval()
    log_losses = defaultdict(list)

    for inputs in data_loader:
        inputs, targets = models["gen"].preprocess(inputs)
        inputs = {
            k: v.to(models["gen"].device) if torch.is_tensor(v) else v
            for (k, v) in inputs.items()
        }
        targets = {
            k: v.to(models["gen"].device) if torch.is_tensor(v) else v
            for (k, v) in targets.items()
        }

        _, losses = models["gen"].train_loss(inputs, targets, test=True)
        for k, v in losses.items():
            log_losses[k].append(v.cpu().item())

        if cfg.debug:
            break
        torch.cuda.empty_cache()

    for k, v in log_losses.items():
        log_losses[k] = sum(v) / len(v)

    log_losses["total"] = sum([v for k, v in log_losses.items() if "loss" in k])

    return log_losses


def _filter_args_for_ai_platform() -> None:
    """
    This is to filter out "--job-dir <JOB_DIR>" which is passed from AI Platform training command,
    """
    key = "--job_dir"
    if key in sys.argv:
        logger.warning(f"{key} removed")
        arguments = sys.argv
        ind = arguments.index(key)
        sys.argv = [a for (i, a) in enumerate(arguments) if i not in [ind, ind + 1]]

    key = "--job-dir"
    for i, arg in enumerate(sys.argv):
        if len(arg) >= len(key) and arg[: len(key)] == key:
            sys.argv = [a for (j, a) in enumerate(sys.argv) if i != j]


if __name__ == "__main__":
    _filter_args_for_ai_platform()
    main()
