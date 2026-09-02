"""Train and evaluate a parameter-matched PyTorch TopoBox-3D model."""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
from datetime import datetime, timezone

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

from topobox3d.pde_dataset import (
    COCHAIN_NORMALIZATION,
    ModelBatchCollator,
    TopoBoxPDEDataset,
    make_model_dataloader,
    mass_weighted_relative_mse,
)

from .metrics import sample_metrics
from .model_registry import (
    BFLOAT16_MODEL_NAMES,
    MATCHED_CONFIGS,
    MODEL_OUTPUT_NAMES,
    TORCH_MODEL_NAMES,
    build_torch_model,
    count_parameters,
    forward_torch_model,
    move_supervision,
)


CONFIG_NAMES = (
    "non_harmonic",
    "weak_harmonic",
    "balanced",
    "strong_harmonic",
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def autocast_context(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


class ConfigCyclingSampler(Sampler[int]):
    """Visit every geometry once per epoch and cycle its initial condition."""

    def __init__(self, dataset, seed: int):
        if len(dataset) % len(CONFIG_NAMES):
            raise ValueError("Config-cycling requires four items per geometry")
        self.geometry_count = len(dataset) // len(CONFIG_NAMES)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        order = list(range(self.geometry_count))
        random.Random(self.seed + self.epoch).shuffle(order)
        for geometry_index in order:
            config_index = (
                self.epoch + geometry_index + self.seed
            ) % len(CONFIG_NAMES)
            yield geometry_index * len(CONFIG_NAMES) + config_index

    def __len__(self):
        return self.geometry_count


def make_loader(args, split: str, shuffle: bool):
    if split == "train" and not args.all_configs_per_epoch:
        dataset = TopoBoxPDEDataset(
            args.geometry_root,
            args.solution_root,
            protocol=args.protocol,
            split=split,
            degrees=(args.degree,),
            configs=CONFIG_NAMES,
            cache_derived=args.ram_cache,
            cache_adjacency=args.ram_cache and args.model == "mgn-lite",
        )
        if args.ram_cache:
            stats = dataset.preload()
            print(
                json.dumps(
                    {
                        "event": "ram_cache_loaded",
                        "split": split,
                        **stats,
                    }
                ),
                flush=True,
            )
        sampler = ConfigCyclingSampler(dataset, args.seed)
        loader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=ModelBatchCollator(args.model),
            persistent_workers=args.num_workers > 0,
        )
        return dataset, loader, sampler
    dataset, loader = make_model_dataloader(
        args.geometry_root,
        args.solution_root,
        protocol=args.protocol,
        split=split,
        model_name=args.model,
        degrees=(args.degree,),
        configs=CONFIG_NAMES,
        shuffle=shuffle,
        num_workers=args.num_workers,
    )
    if args.ram_cache:
        # ``make_model_dataloader`` preserves the public dataset API; enable
        # the same in-process caches for validation and test datasets here.
        dataset.cache_derived = True
        dataset.cache_adjacency = args.model == "mgn-lite"
        stats = dataset.preload()
        print(
            json.dumps(
                {
                    "event": "ram_cache_loaded",
                    "split": split,
                    **stats,
                }
            ),
            flush=True,
        )
    return dataset, loader, None


def train_epoch(model, loader, optimizer, device, args) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    count = 0
    started = time.perf_counter()
    for step, batch in enumerate(loader):
        if args.max_train_samples and step >= args.max_train_samples:
            break
        target, mass = move_supervision(batch, device)
        with autocast_context(args.amp):
            prediction = forward_torch_model(args.model, model, batch, device)
            loss = mass_weighted_relative_mse(prediction, target, mass)
            scaled_loss = loss / args.accumulate_steps
        scaled_loss.backward()
        should_step = (
            (step + 1) % args.accumulate_steps == 0
            or step + 1 == len(loader)
            or (
                args.max_train_samples
                and step + 1 == args.max_train_samples
            )
        )
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        total_loss += float(loss.detach().cpu())
        count += 1
    elapsed = time.perf_counter() - started
    return {
        "relative_mse": total_loss / max(count, 1),
        "samples": count,
        "seconds": elapsed,
        "samples_per_second": count / max(elapsed, 1e-9),
    }


@torch.inference_mode()
def evaluate(
    model,
    loader,
    device,
    args,
    records_path: Path | None = None,
) -> dict[str, object]:
    model.eval()
    values: dict[str, list[float]] = defaultdict(list)
    by_geometry: dict[str, list[float]] = defaultdict(list)
    count = 0
    started = time.perf_counter()
    writer = records_path.open("w", encoding="utf-8") if records_path else None
    try:
        for step, batch in enumerate(loader):
            if args.max_eval_samples and step >= args.max_eval_samples:
                break
            target, mass = move_supervision(batch, device)
            with autocast_context(args.amp):
                prediction = forward_torch_model(args.model, model, batch, device)
            metrics = sample_metrics(
                prediction,
                target,
                mass,
                batch["target_harmonic_basis"],
            )
            for key in (
                "relative_l2",
                "relative_mse",
                "harmonic_relative",
                "nonharmonic_relative",
            ):
                values[key].append(float(metrics[key]))
            by_geometry[batch["geometry_id"]].append(float(metrics["relative_l2"]))
            if writer:
                record = {
                    "geometry_id": batch["geometry_id"],
                    "protocol": batch["protocol"],
                    "split": batch["split"],
                    "degree": int(batch["degree"]),
                    "config_name": batch["config_name"],
                    "beta1": int(batch["beta1"]),
                    "beta2": int(batch["beta2"]),
                    "realized_energy_fractions": (
                        batch["realized_energy_fractions"].cpu().tolist()
                    ),
                    **metrics,
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    finally:
        if writer:
            writer.close()
    elapsed = time.perf_counter() - started
    geometry_means = [float(np.mean(items)) for items in by_geometry.values()]
    summary: dict[str, object] = {
        key: {
            "mean": float(np.mean(items)) if items else math.nan,
            "median": float(np.median(items)) if items else math.nan,
        }
        for key, items in values.items()
    }
    summary["geometry_clustered_relative_l2"] = {
        "mean": float(np.mean(geometry_means)) if geometry_means else math.nan,
        "median": float(np.median(geometry_means)) if geometry_means else math.nan,
        "geometry_count": len(geometry_means),
    }
    summary.update(
        {
            "samples": count,
            "seconds": elapsed,
            "samples_per_second": count / max(elapsed, 1e-9),
        }
    )
    return summary


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=TORCH_MODEL_NAMES)
    parser.add_argument("--protocol", required=True, choices=("A", "B", "C", "D"))
    parser.add_argument("--degree", required=True, type=int, choices=(0, 1, 2))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help=(
            "Deprecated compatibility option. Validation patience is reported "
            "but never stops fixed-budget training."
        ),
    )
    parser.add_argument("--validate-every", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--accumulate-steps", type=int, default=1)
    parser.add_argument(
        "--all-configs-per-epoch",
        action="store_true",
        help="Use all four initial conditions instead of the default cycling sampler.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--ram-cache",
        action="store_true",
        help="Preload each active split and cache derived simplex arrays in RAM.",
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument(
        "--geometry-root", type=Path, default=Path("data/TopoBox-3D/packed")
    )
    parser.add_argument(
        "--solution-root", type=Path, default=Path("data/TopoBox-3D-HodgeHeat")
    )
    parser.add_argument("--output-root", type=Path, default=Path("runs/topobox3d"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ram_cache and args.num_workers:
        raise ValueError("--ram-cache requires --num-workers 0 to avoid duplication")
    if not torch.cuda.is_available():
        raise RuntimeError("Formal TopoBox training requires a CUDA GPU")
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    args.amp = bool(args.amp and args.model in BFLOAT16_MODEL_NAMES)
    run_dir = (
        args.output_root
        / MODEL_OUTPUT_NAMES[args.model]
        / f"protocol_{args.protocol}"
        / f"k{args.degree}"
        / f"seed_{args.seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    run_started = time.time()
    model = build_torch_model(args.model, args.degree).to(device)
    parameter_count = count_parameters(model)
    run_config = {
        "model": args.model,
        "protocol": args.protocol,
        "degree": args.degree,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "model_config": MATCHED_CONFIGS[args.model],
        "training_config": vars(args),
        "cochain_normalization": COCHAIN_NORMALIZATION,
        "device": torch.cuda.get_device_name(device),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(run_dir / "run_config.json", run_config)
    write_json_atomic(
        run_dir / "progress.json",
        {
            "status": "initializing",
            "epoch": 0,
            "epochs": args.epochs,
            "best_validation_relative_mse": None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1), eta_min=args.learning_rate * 0.05
    )
    checkpoint_path = run_dir / "best.pt"
    last_path = run_dir / "last.pt"
    start_epoch = 1
    best_validation = math.inf
    stale_validations = 0
    if args.evaluate_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
        source = checkpoint_path
        payload = torch.load(source, map_location=device, weights_only=False)
        if (
            args.degree == 2
            and payload.get("cochain_normalization") != COCHAIN_NORMALIZATION
        ):
            raise RuntimeError(
                "This k=2 checkpoint predates cochain normalization and is "
                "not evaluation-compatible; retrain it from scratch."
            )
        model.load_state_dict(payload["model"])
    elif args.resume and last_path.exists():
        payload = torch.load(last_path, map_location=device, weights_only=False)
        checkpoint_epochs = int(payload.get("args", {}).get("epochs", args.epochs))
        if checkpoint_epochs != args.epochs:
            raise RuntimeError(
                "Cannot resume a checkpoint created for "
                f"{checkpoint_epochs} epochs with --epochs {args.epochs}. "
                "The cosine learning-rate horizon would be inconsistent. "
                "Start a fresh run without --resume (or use --force in the "
                "MGN-lite suite runner)."
            )
        if (
            args.degree == 2
            and payload.get("cochain_normalization") != COCHAIN_NORMALIZATION
        ):
            raise RuntimeError(
                "This k=2 checkpoint predates cochain normalization and "
                "cannot be resumed; rerun without --resume."
            )
        model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        start_epoch = int(payload["epoch"]) + 1
        best_validation = float(payload["best_validation"])
        stale_validations = int(payload.get("stale_validations", 0))

    history_path = run_dir / "history.json"
    history: list[dict] = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if start_epoch > 1 and history_path.exists()
        else []
    )
    if not args.evaluate_only:
        train_dataset, train_loader, train_sampler = make_loader(
            args, "train", True
        )
        validation_dataset, validation_loader, _ = make_loader(
            args, "validation", False
        )
        try:
            for epoch in range(start_epoch, args.epochs + 1):
                if train_sampler is not None:
                    train_sampler.set_epoch(epoch - 1)
                training = train_epoch(
                    model, train_loader, optimizer, device, args
                )
                scheduler.step()
                event: dict[str, object] = {
                    "epoch": epoch,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "train": training,
                }
                if epoch % args.validate_every == 0 or epoch == args.epochs:
                    validation = evaluate(
                        model, validation_loader, device, args
                    )
                    event["validation"] = validation
                    score = float(validation["relative_mse"]["mean"])
                    improved = score < best_validation
                    if improved:
                        best_validation = score
                        stale_validations = 0
                        save_checkpoint(
                            checkpoint_path,
                            {
                                "model": model.state_dict(),
                                "epoch": epoch,
                                "best_validation": best_validation,
                                "parameter_count": parameter_count,
                                "args": vars(args),
                                "cochain_normalization": COCHAIN_NORMALIZATION,
                            },
                        )
                    else:
                        stale_validations += 1
                history.append(event)
                save_checkpoint(
                    last_path,
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "epoch": epoch,
                        "best_validation": best_validation,
                        "stale_validations": stale_validations,
                        "parameter_count": parameter_count,
                        "args": vars(args),
                        "cochain_normalization": COCHAIN_NORMALIZATION,
                    },
                )
                history_path.write_text(
                    json.dumps(history, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                write_json_atomic(
                    run_dir / "progress.json",
                    {
                        "status": "training",
                        "epoch": epoch,
                        "epochs": args.epochs,
                        "best_validation_relative_mse": (
                            best_validation
                            if math.isfinite(best_validation)
                            else None
                        ),
                        "stale_validations": stale_validations,
                        "last_event": event,
                        "elapsed_seconds": time.time() - run_started,
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
                print(json.dumps(event, ensure_ascii=False), flush=True)
        finally:
            train_dataset.close()
            validation_dataset.close()
            train_dataset.clear_memory_cache()
            validation_dataset.clear_memory_cache()

    best = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    completed_epoch = (
        int(history[-1]["epoch"]) if history else int(best["epoch"])
    )
    all_summaries = {}
    for split in ("validation", "test_iid", "test_ood"):
        dataset, loader, _ = make_loader(args, split, False)
        try:
            all_summaries[split] = evaluate(
                model,
                loader,
                device,
                args,
                run_dir / f"{split}.jsonl",
            )
        finally:
            dataset.close()
            dataset.clear_memory_cache()
    result = {
        "model": args.model,
        "protocol": args.protocol,
        "degree": args.degree,
        "seed": args.seed,
        "parameter_count": parameter_count,
        "cochain_normalization": COCHAIN_NORMALIZATION,
        "completed_epoch": completed_epoch,
        "best_epoch": int(best["epoch"]),
        "best_validation_relative_mse": float(best["best_validation"]),
        "splits": all_summaries,
    }
    write_json_atomic(run_dir / "summary.json", result)
    write_json_atomic(
        run_dir / "progress.json",
        {
            "status": "completed",
            "epoch": completed_epoch,
            "epochs": args.epochs,
            "best_epoch": int(best["epoch"]),
            "best_validation_relative_mse": float(best["best_validation"]),
            "elapsed_seconds": time.time() - run_started,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
