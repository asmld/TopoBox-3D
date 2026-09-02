"""Wait for parallel generation, then audit, validate, and visualize automatically."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from .audit_hodge_heat_dataset import audit
from .rebuild_hodge_heat_index import rebuild
from .validate_pde_dataset import validate
from .visualize_hodge_heat_dataset import render


EXPECTED_SHARDS = 212
EXPECTED_GEOMETRIES = 5280


def _finished_shards(root: Path) -> list[Path]:
    return sorted(
        path for path in root.glob("protocol_*/*/shard_*.h5")
        if not path.name.endswith(".tmp.h5")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry-root", type=Path, default=Path("data/TopoBox-3D")
    )
    parser.add_argument(
        "--solution-root", type=Path, default=Path("data/TopoBox-3D-HodgeHeat")
    )
    parser.add_argument(
        "--visualization-output",
        type=Path,
        default=Path("outputs/TopoBox-3D-HodgeHeat-representatives"),
    )
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stable-polls", type=int, default=4)
    args = parser.parse_args()

    stable = 0
    previous_count = -1
    while stable < args.stable_polls:
        shards = _finished_shards(args.solution_root)
        temporary = list(args.solution_root.glob("protocol_*/*/shard_*.tmp.h5"))
        count = len(shards)
        if (
            count == EXPECTED_SHARDS
            and not temporary
            and count == previous_count
        ):
            stable += 1
        else:
            stable = 0
        previous_count = count
        print(
            f"WAIT shards={count}/{EXPECTED_SHARDS} temporary={len(temporary)} "
            f"stable={stable}/{args.stable_polls}",
            flush=True,
        )
        if stable < args.stable_polls:
            time.sleep(args.poll_seconds)

    records = rebuild(args.solution_root, args.geometry_root)
    if len(records) != EXPECTED_GEOMETRIES:
        raise RuntimeError(
            f"Expected {EXPECTED_GEOMETRIES} indexed geometries, got {len(records)}"
        )
    report = audit(args.solution_root, require_complete=True, deep=True)
    audit_path = args.solution_root / "audit_report.json"
    audit_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if report["errors"]:
        raise RuntimeError(
            f"Deep audit found {len(report['errors'])} errors; see {audit_path}"
        )

    adapter_errors = []
    for protocol in "ABCD":
        for split in ("train", "validation", "test_iid", "test_ood"):
            adapter_errors.extend(
                validate(
                    args.geometry_root / "packed",
                    args.solution_root,
                    protocol,
                    split,
                )
            )
    if adapter_errors:
        raise RuntimeError(
            "Model adapter validation failed:\n" + "\n".join(adapter_errors[:100])
        )

    visualization = render(
        args.geometry_root,
        args.solution_root,
        args.visualization_output,
        "balanced",
    )
    completion = {
        "status": "complete",
        "geometry_count": len(records),
        "pde_sample_count": len(records) * 3 * 4,
        "shard_count": EXPECTED_SHARDS,
        "deep_audit_error_count": 0,
        "model_adapter_validation": "passed",
        "representative_visualizations": visualization,
        "completed_at_unix": time.time(),
    }
    (args.solution_root / "COMPLETION.json").write_text(
        json.dumps(completion, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"COMPLETE geometries={len(records)} pde_samples={len(records) * 12}",
        flush=True,
    )


if __name__ == "__main__":
    main()
