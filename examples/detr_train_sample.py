#!/usr/bin/env python3
"""Train DETR using only VOC_ROOT/JPEGImages and VOC_ROOT/Annotations.

All Annotations/*.xml are listed, shuffled with SEED, and split by VAL_FRACTION
into train vs validation. No other folders are used for training data.

Edit the CONFIG block below (no command-line arguments).
"""

from __future__ import annotations

import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForObjectDetection


# =============================================================================
# CONFIG — change paths and hyperparameters here
# =============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

# Parent folder that contains only JPEGImages/ and Annotations/ (for training).
VOC_ROOT = REPO_ROOT / "dataset/VOC2012_train_val/VOC2012_train_val"

VAL_FRACTION = 0.15  # fraction of xmls for validation after shuffle(SEED)

MODEL_NAME = "facebook/detr-resnet-50"
# Fast DetrImageProcessor uses a different .pad() API and breaks batch collation below.
USE_FAST_IMAGE_PROCESSOR = False

EPOCHS = 10
BATCH_SIZE = 2
LR = 1e-5
NUM_WORKERS = 0
LOG_EVERY = 50
SEED = 42
SKIP_DIFFICULT = True

OUTPUT_DIR = REPO_ROOT / "checkpoints/detr-voc"
# Checkpoints: OUTPUT_DIR/last (every epoch) and OUTPUT_DIR/best (lowest val_loss, or train_loss if no val)
CKPT_LAST_SUBDIR = "last"
CKPT_BEST_SUBDIR = "best"

# After training: run inference on a folder of test images (no XML). Set to None to skip.
TEST_IMAGES_DIR: Optional[Path] = None  # e.g. REPO_ROOT / "dataset/my_test_images"
PREDICT_OUTPUT_DIR = REPO_ROOT / "predictions/detr-test"
SCORE_THRESHOLD = 0.5

# Set True to only run test inference (no training). Loads this folder (use "best" or "last").
PREDICT_ONLY = False
CHECKPOINT_DIR = OUTPUT_DIR / CKPT_BEST_SUBDIR


# =============================================================================


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_detr_image_processor(model_name_or_path: str | Path) -> AutoImageProcessor:
    return AutoImageProcessor.from_pretrained(
        str(model_name_or_path),
        use_fast=USE_FAST_IMAGE_PROCESSOR,
    )


def save_model_checkpoint(
    model: AutoModelForObjectDetection,
    processor: AutoImageProcessor,
    checkpoint_dir: Path,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    processor.save_pretrained(checkpoint_dir)


def list_image_ids_from_annotations(ann_dir: Path) -> List[str]:
    """All VOC image stems that have an XML in Annotations/."""
    stems = sorted({p.stem for p in ann_dir.glob("*.xml")})
    if not stems:
        raise ValueError(f"No *.xml files found in {ann_dir}")
    return stems


def random_train_val_split(
    ids: List[str],
    val_fraction: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("VAL_FRACTION must be between 0 and 1 (e.g. 0.15)")
    if len(ids) < 2:
        return list(ids), []
    rng = random.Random(seed)
    order = ids.copy()
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_fraction)))
    n_val = min(n_val, len(order) - 1)
    val_ids = order[:n_val]
    train_ids = order[n_val:]
    return train_ids, val_ids


def parse_voc_xml(
    xml_path: Path,
    label2id: Dict[str, int],
    skip_difficult: bool = True,
) -> Tuple[str, List[Dict[str, Any]]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename = root.findtext("filename")
    if filename is None:
        raise ValueError(f"Missing <filename> in {xml_path}")

    annotations: List[Dict[str, Any]] = []
    for obj in root.findall("object"):
        cls_name = obj.findtext("name")
        bnd = obj.find("bndbox")
        if cls_name is None or bnd is None:
            continue
        cls_name = cls_name.strip()
        if cls_name not in label2id:
            continue
        if skip_difficult:
            diff = obj.findtext("difficult")
            if diff is not None and diff.strip() == "1":
                continue

        xmin = int(float(bnd.findtext("xmin", "0")))
        ymin = int(float(bnd.findtext("ymin", "0")))
        xmax = int(float(bnd.findtext("xmax", "0")))
        ymax = int(float(bnd.findtext("ymax", "0")))
        w = max(1, xmax - xmin)
        h = max(1, ymax - ymin)

        annotations.append(
            {
                "bbox": [xmin, ymin, w, h],
                "category_id": label2id[cls_name],
                "area": float(w * h),
                "iscrowd": 0,
            }
        )
    return filename, annotations


def discover_labels_from_ids(
    ann_dir: Path,
    image_ids: List[str],
) -> List[str]:
    labels: Set[str] = set()
    for stem in image_ids:
        xml_path = ann_dir / f"{stem}.xml"
        if not xml_path.exists():
            continue
        root = ET.parse(xml_path).getroot()
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name:
                labels.add(name.strip())
    if not labels:
        raise ValueError(
            f"No object labels found in annotations for the given train ids under {ann_dir}"
        )
    return sorted(labels)


def find_image_path(jpeg_dir: Path, stem: str, filename_from_xml: str) -> Path:
    """Resolve image file under JPEGImages."""
    direct = jpeg_dir / filename_from_xml
    if direct.exists():
        return direct
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = jpeg_dir / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No image for stem {stem!r} (xml said filename={filename_from_xml!r}) in {jpeg_dir}"
    )


class VocDetrDataset(Dataset):
    """VOC layout: single JPEGImages + Annotations, split by image id list."""

    def __init__(
        self,
        processor: AutoImageProcessor,
        jpeg_dir: Path,
        ann_dir: Path,
        image_ids: List[str],
        label2id: Dict[str, int],
        skip_difficult: bool = True,
    ):
        self.processor = processor
        self.jpeg_dir = jpeg_dir
        self.ann_dir = ann_dir
        self.image_ids = list(image_ids)
        self.label2id = label2id
        self.skip_difficult = skip_difficult

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        stem = self.image_ids[idx]
        xml_path = self.ann_dir / f"{stem}.xml"
        if not xml_path.exists():
            raise FileNotFoundError(f"Missing annotation: {xml_path}")

        filename, annotations = parse_voc_xml(
            xml_path, self.label2id, skip_difficult=self.skip_difficult
        )
        image_path = find_image_path(self.jpeg_dir, stem, filename)
        image = Image.open(image_path).convert("RGB")

        target = {"image_id": idx, "annotations": annotations}
        encoded = self.processor(images=image, annotations=target, return_tensors="pt")
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"][0],
        }


def make_collate_fn(processor: AutoImageProcessor):
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]
        encoding = processor.pad(pixel_values, return_tensors="pt")
        return {
            "pixel_values": encoding["pixel_values"],
            "pixel_mask": encoding["pixel_mask"],
            "labels": labels,
        }

    return collate_fn


@torch.no_grad()
def evaluate(
    model: AutoModelForObjectDetection,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0
    for batch in dataloader:
        pixel_values = batch["pixel_values"].to(device)
        pixel_mask = batch["pixel_mask"].to(device)
        labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]
        outputs = model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        steps += 1
    model.train()
    return total_loss / max(1, steps)


def run_test_inference(
    model: AutoModelForObjectDetection,
    processor: AutoImageProcessor,
    test_images_dir: Path,
    output_dir: Path,
    device: torch.device,
    threshold: float,
) -> None:
    """Run DETR on images without ground truth; save JSON predictions."""
    model.eval()
    test_images_dir = test_images_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}
    paths = sorted(
        p for p in test_images_dir.iterdir() if p.suffix in exts and p.is_file()
    )
    if not paths:
        print(f"No images found in {test_images_dir}")
        return

    id2label = model.config.id2label
    results: List[Dict[str, Any]] = []

    for img_path in paths:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]], device=device)
        raw = processor.post_process_object_detection(
            outputs, threshold=threshold, target_sizes=target_sizes
        )[0]

        entry = {
            "file": img_path.name,
            "scores": raw["scores"].cpu().tolist(),
            "labels": raw["labels"].cpu().tolist(),
            "label_names": [id2label[int(l)] for l in raw["labels"].cpu().tolist()],
            "boxes": raw["boxes"].cpu().tolist(),
        }
        results.append(entry)

    out_json = output_dir / "test_predictions.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {len(results)} predictions to {out_json}")


def train_loop() -> Optional[Path]:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    voc_root = VOC_ROOT.expanduser().resolve()
    jpeg_dir = voc_root / "JPEGImages"
    ann_dir = voc_root / "Annotations"

    if not jpeg_dir.is_dir():
        raise FileNotFoundError(f"JPEGImages not found: {jpeg_dir}")
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"Annotations not found: {ann_dir}")

    all_ids = list_image_ids_from_annotations(ann_dir)
    train_ids, val_ids = random_train_val_split(all_ids, VAL_FRACTION, SEED)

    class_names = discover_labels_from_ids(ann_dir, train_ids)
    label2id = {name: idx for idx, name in enumerate(class_names)}
    id2label = {idx: name for name, idx in label2id.items()}
    print(f"Train images: {len(train_ids)}, Val images: {len(val_ids)}")
    print(f"Classes ({len(class_names)}): {class_names}")

    processor = load_detr_image_processor(MODEL_NAME)
    model = AutoModelForObjectDetection.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    model.train()

    train_dataset = VocDetrDataset(
        processor=processor,
        jpeg_dir=jpeg_dir,
        ann_dir=ann_dir,
        image_ids=train_ids,
        label2id=label2id,
        skip_difficult=SKIP_DIFFICULT,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=make_collate_fn(processor),
    )
    val_loader: Optional[DataLoader] = None
    if val_ids:
        val_dataset = VocDetrDataset(
            processor=processor,
            jpeg_dir=jpeg_dir,
            ann_dir=ann_dir,
            image_ids=val_ids,
            label2id=label2id,
            skip_difficult=SKIP_DIFFICULT,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            collate_fn=make_collate_fn(processor),
        )
    else:
        print("Warning: empty validation split; val_loss will be skipped.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    output_root = OUTPUT_DIR.expanduser().resolve()
    last_dir = output_root / CKPT_LAST_SUBDIR
    best_dir = output_root / CKPT_BEST_SUBDIR
    best_metric = float("inf")
    best_epoch = -1
    metric_name = "val_loss" if val_loader is not None else "train_loss"

    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            pixel_values = batch["pixel_values"].to(device)
            pixel_mask = batch["pixel_mask"].to(device)
            labels = [{k: v.to(device) for k, v in t.items()} for t in batch["labels"]]

            outputs = model(
                pixel_values=pixel_values,
                pixel_mask=pixel_mask,
                labels=labels,
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if step % LOG_EVERY == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch + 1}/{EPOCHS} "
                    f"Step {step}/{len(train_loader)} Loss {loss.item():.4f}"
                )

        train_avg = epoch_loss / len(train_loader)
        if val_loader is not None:
            val_avg = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch + 1}: train_loss={train_avg:.4f} val_loss={val_avg:.4f}"
            )
            metric_for_best = val_avg
        else:
            print(f"Epoch {epoch + 1}: train_loss={train_avg:.4f} val_loss=n/a")
            metric_for_best = train_avg

        save_model_checkpoint(model, processor, last_dir)
        print(f"  Saved checkpoint-last → {last_dir}")

        if metric_for_best < best_metric:
            best_metric = metric_for_best
            best_epoch = epoch + 1
            save_model_checkpoint(model, processor, best_dir)
            print(
                f"  Saved checkpoint-best ({metric_name}={best_metric:.4f}) → {best_dir}"
            )

    state_path = output_root / "checkpoint_metrics.json"
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch,
                "best_metric": best_metric,
                "metric_name": metric_name,
                "last_dir": str(last_dir),
                "best_dir": str(best_dir),
            },
            f,
            indent=2,
        )
    print(f"Training done. last={last_dir} best={best_dir} ({metric_name}={best_metric:.4f} @ epoch {best_epoch})")
    print(f"Wrote {state_path}")
    return best_dir


def run_predict_only() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = CHECKPOINT_DIR.expanduser().resolve()
    if not TEST_IMAGES_DIR:
        raise ValueError("Set TEST_IMAGES_DIR when using PREDICT_ONLY=True")
    processor = load_detr_image_processor(ckpt)
    model = AutoModelForObjectDetection.from_pretrained(ckpt)
    model.to(device)
    run_test_inference(
        model,
        processor,
        TEST_IMAGES_DIR,
        PREDICT_OUTPUT_DIR,
        device,
        threshold=SCORE_THRESHOLD,
    )


def main() -> None:
    if PREDICT_ONLY:
        run_predict_only()
        return

    out = train_loop()
    if TEST_IMAGES_DIR is not None and out is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        processor = load_detr_image_processor(out)
        model = AutoModelForObjectDetection.from_pretrained(out)
        model.to(device)
        run_test_inference(
            model,
            processor,
            TEST_IMAGES_DIR,
            PREDICT_OUTPUT_DIR,
            device,
            threshold=SCORE_THRESHOLD,
        )


if __name__ == "__main__":
    main()
