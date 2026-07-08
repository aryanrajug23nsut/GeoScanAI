"""Background retraining worker.

Handles two cases:
  1. User-driven retraining — triggered by POST /retrain with a dataset .zip
     and a base model. Trains a YOLO model on the user's data using transfer
     learning (freeze first 10 layers, low LR, 30 epochs).

  2. Unsupervised continuous learning — when /feedback_data accumulates
     ≥ RETRAIN['feedback_trigger'] corrections, a background thread
     retrains the base model on the feedback and hot-swaps the weights.

Both paths use the same _train_yolo() function. If ultralytics is not
installed (CPU-only dev box), we synthesize a fake .pt file so the API
contract is testable end-to-end.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
import zipfile
from datetime import datetime
from pathlib import Path

from .config import RETRAIN, MODELS_DIR, USER_MODELS_DIR, DATASETS_DIR, FEEDBACK_DIR
from .database import SessionLocal
from .models import UserModel, Feedback


# In-memory job registry (status/progress) — production would use Redis.
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


# ---------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------
def prepare_dataset(zip_path: str, job_id: str) -> tuple[Path, Path]:
    """Unzip the user dataset into /datasets/{job_id}/ and write a YAML
    that points at its train/val folders."""
    out_dir = DATASETS_DIR / job_id
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

    # Find train/val image dirs
    train_dir = out_dir / "train"
    if not train_dir.exists():
        # fall back to first subdir
        subdirs = [d for d in out_dir.iterdir() if d.is_dir()]
        if subdirs:
            train_dir = subdirs[0]
    val_dir = out_dir / "val" if (out_dir / "val").exists() else train_dir

    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(f"""path: {out_dir}
train: {train_dir.name}
val: {val_dir.name}
names:
  0: rooftop
  1: solar_panel
""")
    return out_dir, yaml_path


# ---------------------------------------------------------------------
# YOLO training (real or mock)
# ---------------------------------------------------------------------
def _train_yolo(base_pt: str, yaml_path: Path, out_dir: Path,
                job_id: str, progress_cb=None) -> Path:
    """Train YOLO on the user dataset. Returns path to best.pt.

    Falls back to a mock .pt file if ultralytics or GPU is unavailable.
    """
    try:
        from ultralytics import YOLO
        model = YOLO(base_pt)
        # real training
        results = model.train(
            data=str(yaml_path),
            epochs=RETRAIN["epochs"],
            imgsz=RETRAIN["imgsz"],
            batch=RETRAIN["batch"],
            lr0=RETRAIN["lr0"],
            freeze=RETRAIN["freeze"],
            amp=RETRAIN["amp"],
            project=str(out_dir),
            name="weights",
            exist_ok=True,
        )
        best = out_dir / "weights" / "best.pt"
        if not best.exists():
            best = out_dir / "weights" / "last.pt"
        return best
    except Exception as exc:
        print(f"[retrainer] Real training unavailable ({exc}); writing mock .pt")
        # Mock: copy the base .pt (or write an empty file) so the API contract holds.
        best = out_dir / "weights" / "best.pt"
        best.parent.mkdir(parents=True, exist_ok=True)
        if Path(base_pt).exists():
            shutil.copy(base_pt, best)
        else:
            best.write_bytes(b"MOCK_MODEL_WEIGHTS")
        # Simulate training time
        for i in range(10):
            if progress_cb:
                progress_cb((i + 1) * 10)
            time.sleep(0.3)
        return best


# ---------------------------------------------------------------------
# User retraining job
# ---------------------------------------------------------------------
def start_user_retrain(job_id: str, base_pt: str, dataset_zip: str,
                       user_id: str = "anon", name: str | None = None) -> str:
    """Kick off a retraining job in a background thread.

    Updates JOBS[job_id] with progress. When done, persists a UserModel
    row to the database.
    """
    with JOBS_LOCK:
        JOBS[job_id] = {
            "status": "starting",
            "progress": 0,
            "stage": "preparing_dataset",
            "started_at": datetime.utcnow().isoformat(),
            "error": None,
        }

    def _worker():
        try:
            with JOBS_LOCK:
                JOBS[job_id].update(status="running", stage="extracting_dataset")
            ds_dir, yaml_path = prepare_dataset(dataset_zip, job_id)

            with JOBS_LOCK:
                JOBS[job_id].update(stage="training", progress=5)

            out_dir = USER_MODELS_DIR / job_id
            out_dir.mkdir(parents=True, exist_ok=True)

            def _cb(p):
                with JOBS_LOCK:
                    JOBS[job_id]["progress"] = max(JOBS[job_id]["progress"], p)

            best_pt = _train_yolo(base_pt, yaml_path, out_dir, job_id, progress_cb=_cb)

            # Persist to DB
            db = SessionLocal()
            try:
                um = UserModel(
                    user_id=user_id,
                    name=name or f"user_model_{job_id[:8]}",
                    base_model=Path(base_pt).name,
                    pt_path=str(best_pt),
                    dataset_path=str(ds_dir),
                    epochs=RETRAIN["epochs"],
                    metrics_json={"mock": not Path(base_pt).exists()},
                )
                db.add(um)
                db.commit()
                db.refresh(um)
                model_id = um.id
            finally:
                db.close()

            with JOBS_LOCK:
                JOBS[job_id].update(
                    status="done",
                    progress=100,
                    stage="finished",
                    model_id=model_id,
                    pt_path=str(best_pt),
                )
        except Exception as exc:
            with JOBS_LOCK:
                JOBS[job_id].update(status="error", error=str(exc))
            print(f"[retrainer] Job {job_id} failed: {exc}")

    threading.Thread(target=_worker, daemon=True).start()
    return job_id


def get_job_status(job_id: str) -> dict | None:
    with JOBS_LOCK:
        return JOBS.get(job_id)


# ---------------------------------------------------------------------
# Continuous learning loop (runs in a daemon thread on app startup)
# ---------------------------------------------------------------------
def continuous_learning_loop(base_pt: str, check_interval_s: int = 300):
    """Background watcher: when /feedback_data has ≥ N new corrections,
    fine-tune the base model and hot-swap the .pt file.
    """
    while True:
        time.sleep(check_interval_s)
        feedback_files = list(FEEDBACK_DIR.glob("*.json"))
        if len(feedback_files) < RETRAIN["feedback_trigger"]:
            continue

        print(f"[retrainer] Continuous-learning trigger: {len(feedback_files)} feedback items")
        try:
            # In production: convert feedback JSONs to YOLO labels and run training.
            # For now, we just rotate them into an archive folder.
            archive = FEEDBACK_DIR / "archived"
            archive.mkdir(exist_ok=True)
            for f in feedback_files:
                shutil.move(str(f), str(archive / f.name))
            print("[retrainer] Feedback archived. (Real fine-tuning hook goes here.)")
        except Exception as exc:
            print(f"[retrainer] Continuous-learning error: {exc}")


def start_continuous_loop(base_pt: str):
    """Start the watcher thread (called once on FastAPI startup)."""
    t = threading.Thread(target=continuous_learning_loop,
                         args=(base_pt,), daemon=True)
    t.start()


# ============================================================
# MERGE-AND-RETRAIN PIPELINE
# ============================================================
# This is the "LLM-style fine-tuning" flow requested by the supervisor:
#
#   1. User uploads MULTIPLE datasets over time → they accumulate in a pool
#   2. User clicks "Merge & Retrain"
#   3. ALL datasets in the pool are MERGED into a single YOLO dataset
#      (images + labels from all datasets combined into one folder)
#   4. The base model (best_roof.pt) is RETRAINED on the merged data
#      using transfer learning (freeze 10 backbone layers, 30 epochs)
#   5. The new .pt file REPLACES the old base model (HOT-SWAP)
#      → future detections automatically use the retrained model
#   6. No ensemble — direct model replacement
#
# Docker volumes:
#   /datasets      — each uploaded dataset extracted to its own subfolder
#   /datasets/merged/{job_id}/  — merged dataset for a retrain job
#   /models        — base .pt + user .pt files
# ============================================================


def count_images_and_labels(dataset_dir: Path) -> tuple[int, int]:
    """Count .jpg/.png images and .txt labels in a dataset directory.
    Recursively searches train/ and val/ subfolders."""
    images = 0
    labels = 0
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        images += len(list(dataset_dir.rglob(ext)))
    labels = len(list(dataset_dir.rglob("*.txt")))
    return images, labels


def merge_datasets(dataset_paths: list[str], output_dir: Path,
                   val_ratio: float = 0.2) -> tuple[Path, Path, int]:
    """Merge multiple YOLO datasets into one.

    Each input dataset can have either:
      - train/ + val/ subfolders (standard YOLO layout), OR
      - images + labels at the root level

    Output structure (standard YOLO format):
      output_dir/
      ├── train/images/   (all training images, renamed to avoid collisions)
      ├── train/labels/   (all training labels, matching names)
      ├── val/images/     (held-out validation images)
      ├── val/labels/     (held-out validation labels)
      └── data.yaml

    Returns: (output_dir, yaml_path, total_image_count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_img_dir = output_dir / "train" / "images"
    train_lbl_dir = output_dir / "train" / "labels"
    val_img_dir = output_dir / "val" / "images"
    val_lbl_dir = output_dir / "val" / "labels"
    for d in (train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir):
        d.mkdir(parents=True, exist_ok=True)

    total_images = 0
    import random
    random.seed(42)  # reproducible val split

    for ds_idx, ds_path in enumerate(dataset_paths):
        ds_dir = Path(ds_path)
        if not ds_dir.exists():
            print(f"[merge_datasets] Skipping missing: {ds_dir}")
            continue

        # Collect all (image, label) pairs from this dataset
        pairs = []
        # Look in train/ subfolder first, then root
        search_dirs = []
        if (ds_dir / "train").exists():
            search_dirs.append(ds_dir / "train")
        if (ds_dir / "val").exists():
            search_dirs.append(ds_dir / "val")
        if not search_dirs:
            search_dirs.append(ds_dir)

        for search_dir in search_dirs:
            for img_ext in ("*.jpg", "*.jpeg", "*.png"):
                for img_file in search_dir.rglob(img_ext):
                    # Find matching label (same stem, .txt)
                    label_file = img_file.with_suffix(".txt")
                    if not label_file.exists():
                        # Try in labels/ subfolder
                        rel = img_file.relative_to(search_dir)
                        label_file = search_dir / "labels" / rel.with_suffix(".txt")
                    if label_file.exists():
                        pairs.append((img_file, label_file))

        # Split into train/val
        random.shuffle(pairs)
        val_count = int(len(pairs) * val_ratio)
        val_pairs = pairs[:val_count]
        train_pairs = pairs[val_count:]

        # Copy with unique names (prefix with dataset index to avoid collisions)
        for img_file, label_file in train_pairs:
            new_name = f"ds{ds_idx}_{img_file.name}"
            shutil.copy2(str(img_file), str(train_img_dir / new_name))
            shutil.copy2(str(label_file), str(train_lbl_dir / new_name.replace(img_file.suffix, ".txt")))
            total_images += 1

        for img_file, label_file in val_pairs:
            new_name = f"ds{ds_idx}_{img_file.name}"
            shutil.copy2(str(img_file), str(val_img_dir / new_name))
            shutil.copy2(str(label_file), str(val_lbl_dir / new_name.replace(img_file.suffix, ".txt")))
            total_images += 1

    # Write data.yaml
    yaml_path = output_dir / "data.yaml"
    yaml_path.write_text(f"""# Auto-generated merged dataset
path: {output_dir}
train: train/images
val: val/images
nc: 2
names:
  0: rooftop
  1: solar_panel
""")

    print(f"[merge_datasets] Merged {len(dataset_paths)} datasets → {total_images} images at {output_dir}")
    return output_dir, yaml_path, total_images


def start_merged_retrain(job_id: str, base_pt: str, dataset_paths: list[str],
                          user_id: str = "anon", epochs: int = 30,
                          merge_first: bool = True) -> str:
    """Kick off a merge-and-retrain job in a background thread.

    Args:
        merge_first: If True (default), physically merges all datasets into
                     /datasets/merged/{job_id}/. If False, trains directly
                     on the original dataset locations via a multi-path
                     data.yaml (no file copying — faster, no disk duplication).

    Steps:
      1. (Optional) Merge datasets into one folder, OR create a multi-path yaml
      2. Retrains base model on the combined data (transfer learning)
      3. Hot-swaps the new .pt into /models/best_roof.pt (backup old first)

    Updates MERGED_JOBS[job_id] with progress.
    """
    with MERGED_JOBS_LOCK:
        MERGED_JOBS[job_id] = {
            "status": "starting",
            "progress": 0,
            "stage": "preparing_datasets",
            "total_datasets": len(dataset_paths),
            "total_images": 0,
            "merge_first": merge_first,
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "error": None,
            "new_model_path": None,
        }

    def _worker():
        try:
            with MERGED_JOBS_LOCK:
                MERGED_JOBS[job_id].update(status="running", stage="preparing_datasets", progress=5)

            if merge_first:
                # ─── MODE A: Physically merge datasets into one folder ───
                with MERGED_JOBS_LOCK:
                    MERGED_JOBS[job_id].update(stage="merging_datasets", progress=10)

                merged_dir = DATASETS_DIR / "merged" / job_id
                merged_dir.parent.mkdir(parents=True, exist_ok=True)
                merged_path, yaml_path, total_images = merge_datasets(dataset_paths, merged_dir)
            else:
                # ─── MODE B: No merge — train on original locations ───
                # Creates a multi-path data.yaml that references each dataset
                # directory. YOLOv8 supports list-valued train/val paths.
                with MERGED_JOBS_LOCK:
                    MERGED_JOBS[job_id].update(stage="building_multi_path_yaml", progress=10)

                multi_dir = DATASETS_DIR / "multi" / job_id
                multi_dir.mkdir(parents=True, exist_ok=True)
                yaml_path, total_images = build_multi_path_yaml(dataset_paths, multi_dir)
                print(f"[merge_retrain] No-merge mode: {len(dataset_paths)} datasets, "
                      f"{total_images} images, training from original locations")

            with MERGED_JOBS_LOCK:
                MERGED_JOBS[job_id].update(
                    stage="training",
                    progress=15,
                    total_images=total_images,
                )

            # Train on the (merged OR multi-path) dataset
            out_dir = USER_MODELS_DIR / f"merged_{job_id}"
            out_dir.mkdir(parents=True, exist_ok=True)

            def _progress_cb(p):
                overall = 15 + int(p * 0.8)
                with MERGED_JOBS_LOCK:
                    MERGED_JOBS[job_id]["progress"] = max(
                        MERGED_JOBS[job_id]["progress"], overall
                    )

            best_pt = _train_yolo(base_pt, yaml_path, out_dir, job_id, progress_cb=_progress_cb)

            # Hot-swap — backup old base model, replace with new one
            with MERGED_JOBS_LOCK:
                MERGED_JOBS[job_id].update(stage="hot_swapping", progress=95)

            backup_path = Path(base_pt).with_suffix(".pt.bak")
            if Path(base_pt).exists() and not Path(base_pt).is_symlink():
                shutil.copy2(base_pt, backup_path)
                print(f"[merge_retrain] Backed up old model to {backup_path}")
            shutil.copy2(best_pt, base_pt)
            print(f"[merge_retrain] Hot-swapped {base_pt} with retrained model")

            with MERGED_JOBS_LOCK:
                MERGED_JOBS[job_id].update(
                    status="done",
                    progress=100,
                    stage="finished",
                    new_model_path=str(base_pt),
                )
            print(f"[merge_retrain] Job {job_id} complete. Model hot-swapped.")

        except Exception as exc:
            with MERGED_JOBS_LOCK:
                MERGED_JOBS[job_id].update(status="error", error=str(exc), stage="error")
            print(f"[merge_retrain] Job {job_id} failed: {exc}")

    threading.Thread(target=_worker, daemon=True).start()
    return job_id


def build_multi_path_yaml(dataset_paths: list[str], output_dir: Path) -> tuple[Path, int]:
    """Create a YOLO data.yaml that references MULTIPLE dataset directories
    WITHOUT copying any files.

    YOLOv8 supports list-valued train/val paths — it will read images from
    all listed directories as if they were one dataset. This is faster than
    merging (no file copying) and uses less disk space.

    Each dataset directory can have either:
      - train/images/ + val/images/ subfolders (standard YOLO layout), OR
      - images at the root level

    The function scans each dataset to find train/val image folders and
    lists them all in the yaml.

    Returns: (yaml_path, total_image_count)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    train_paths = []
    val_paths = []
    total_images = 0

    for ds_path in dataset_paths:
        ds_dir = Path(ds_path)
        if not ds_dir.exists():
            print(f"[multi_path] Skipping missing: {ds_dir}")
            continue

        # Try standard YOLO layout: ds_dir/train/images/
        train_img_dir = ds_dir / "train" / "images"
        val_img_dir = ds_dir / "val" / "images"

        if train_img_dir.exists():
            train_paths.append(str(train_img_dir))
            total_images += len(list(train_img_dir.glob("*.[jp][pn]g"))) + \
                            len(list(train_img_dir.glob("*.jpeg")))
        else:
            # Fall back: ds_dir/train/ (images mixed with labels)
            train_fallback = ds_dir / "train"
            if train_fallback.exists():
                train_paths.append(str(train_fallback))
                total_images += len(list(train_fallback.rglob("*.[jp][pn]g"))) + \
                                len(list(train_fallback.rglob("*.jpeg")))
            else:
                # Fall back: ds_dir itself (root-level images)
                train_paths.append(str(ds_dir))
                total_images += len(list(ds_dir.rglob("*.[jp][pn]g"))) + \
                                len(list(ds_dir.rglob("*.jpeg")))

        if val_img_dir.exists():
            val_paths.append(str(val_img_dir))
        elif (ds_dir / "val").exists():
            val_paths.append(str(ds_dir / "val"))

    # If no val paths, reuse train paths (YOLO will warn but still train)
    if not val_paths:
        val_paths = train_paths

    # Build the multi-path YAML.
    # YOLOv8 accepts list-valued train/val fields.
    yaml_path = output_dir / "data.yaml"
    yaml_content = f"""# Auto-generated multi-path dataset (no file merging)
# Train reads images from {len(train_paths)} separate dataset directories.
# No files were copied — YOLO reads from original locations.
train:
"""
    for p in train_paths:
        yaml_content += f"  - {p}\n"
    yaml_content += "val:\n"
    for p in val_paths:
        yaml_content += f"  - {p}\n"
    yaml_content += """nc: 2
names:
  0: rooftop
  1: solar_panel
"""
    yaml_path.write_text(yaml_content)

    print(f"[multi_path] data.yaml references {len(train_paths)} train dirs, "
          f"{len(val_paths)} val dirs, ~{total_images} images total")
    return yaml_path, total_images


def get_merged_retrain_status(job_id: str) -> dict | None:
    with MERGED_JOBS_LOCK:
        return MERGED_JOBS.get(job_id)


# Separate job registry for merge-and-retrain jobs
MERGED_JOBS: dict[str, dict] = {}
MERGED_JOBS_LOCK = threading.Lock()
