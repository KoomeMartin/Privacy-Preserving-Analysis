"""
scripts/build_data.py

One-stop data preparation for the UCF-Crime experiment.
Run this ONCE before any training notebook.

Steps
─────
1. Wipe & recreate PROCESSED_DIR
2. Merge seq_*.npz files per video → [32, 1024] .npy  (Cell 3 logic)
3. Write train.list / test.list  (anomaly-first ordering)
4. Build ucf_gt.npy  (frame-level: 285 test videos × 32 × 16 = 145,920)
5. Build ucf_prompt.npy via CLIP  ([14, 512])

Can be imported or run as __main__.
"""

import re
import shutil
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger("vad_unified")


# ── helpers ───────────────────────────────────────────────────────────────────

def _natural_key(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", p.stem)]


def _load_seq(fp: Path, feat_dim: int):
    """Load one seq file → 2-D array [N_windows, feat_dim]."""
    try:
        if fp.suffix == ".npz":
            z   = np.load(fp)
            arr = z[list(z.keys())[0]]
        elif fp.suffix == ".npy":
            arr = np.load(fp, allow_pickle=True)
            if arr.dtype == object:
                obj = arr.item()
                arr = obj[list(obj.keys())[0]] if isinstance(obj, dict) else obj
        else:
            return None
        return np.array(arr, dtype=np.float32).reshape(-1, feat_dim)
    except Exception:
        return None


def _segment_pool(feat: np.ndarray, n_seg: int) -> np.ndarray:
    """Average-pool [T, D] → [n_seg, D]; NN-upsample if T < n_seg."""
    T = len(feat)
    if T == n_seg:
        return feat
    idx = np.round(np.linspace(0, T - 1, n_seg)).astype(int)
    if T >= n_seg:
        out   = np.zeros((n_seg, feat.shape[1]), dtype=np.float32)
        bounds = np.round(np.linspace(0, T, n_seg + 1)).astype(int)
        for i in range(n_seg):
            s, e = bounds[i], bounds[i + 1]
            out[i] = feat[s:e].mean(0) if s < e else feat[s]
        return out
    else:
        return feat[idx]


# ── Step 1+2: feature preprocessing ──────────────────────────────────────────

def preprocess_features(dataset_root, processed_dir, num_segments=32, feat_dim=1024):
    """
    Merge per-window seq_*.npz → one [32,1024] .npy per video.

    Parameters
    ----------
    dataset_root  : Path  raw splitted-feature dataset root
    processed_dir : Path  output directory
    """
    dataset_root  = Path(dataset_root)
    processed_dir = Path(processed_dir)

    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True)
    logger.info(f"Cleared and recreated: {processed_dir}")

    # auto-detect feature extension
    sample_dir  = next((dataset_root / "train" / "anomaly").iterdir())
    feat_ext    = None
    for ext in [".npz", ".npy"]:
        if list(sample_dir.glob(f"*{ext}")):
            feat_ext = ext
            break
    if feat_ext is None:
        raise FileNotFoundError("No .npz or .npy feature files found.")
    logger.info(f"Detected feature extension: {feat_ext}")

    errors = []
    total  = 0
    for split in ["train", "test"]:
        for label in ["anomaly", "normal"]:
            src = dataset_root / split / label
            dst = processed_dir / split / label
            if not src.exists():
                continue
            dst.mkdir(parents=True, exist_ok=True)
            for vid_dir in tqdm(sorted(src.iterdir()),
                                desc=f"{split}/{label}", leave=False):
                if not vid_dir.is_dir():
                    continue
                out_path = dst / (vid_dir.name + ".npy")
                seq_files = sorted(vid_dir.glob(f"*{feat_ext}"), key=_natural_key)
                if not seq_files:
                    errors.append(f"no files: {vid_dir}")
                    continue
                windows = [w for f in seq_files if (w := _load_seq(f, feat_dim)) is not None]
                if not windows:
                    errors.append(f"unloadable: {vid_dir}")
                    continue
                stacked = np.concatenate(windows, axis=0)   # [T, 1024]
                pooled  = _segment_pool(stacked, num_segments)  # [32, 1024]
                np.save(out_path, pooled)
                total += 1

    logger.info(f"Processed {total} videos. Errors: {len(errors)}")
    if errors:
        for e in errors[:5]:
            logger.warning(f"  {e}")

    # quick shape sanity check
    sample = np.load(next((processed_dir / "train" / "anomaly").glob("*.npy")))
    assert sample.shape == (num_segments, feat_dim), \
        f"Bad shape: {sample.shape}, expected ({num_segments}, {feat_dim})"
    logger.info(f"Sample shape OK: {sample.shape}")
    return processed_dir


# ── Step 3: list files ────────────────────────────────────────────────────────

def build_list_files(processed_dir, lists_dir):
    """
    Write train.list and test.list.
    Anomaly paths come first, then normal — required by MGFN's dataset
    index splitting (list[:N_abn] = anomaly, list[N_abn:] = normal).
    """
    processed_dir = Path(processed_dir)
    lists_dir     = Path(lists_dir)
    lists_dir.mkdir(parents=True, exist_ok=True)

    stats = {}
    for split in ["train", "test"]:
        abn = sorted((processed_dir / split / "anomaly").glob("*.npy"))
        nrm = sorted((processed_dir / split / "normal").glob("*.npy"))
        out = lists_dir / f"{split}.list"
        with open(out, "w") as f:
            for p in abn + nrm:
                f.write(str(p) + "\n")
        stats[split] = (len(abn), len(nrm))
        logger.info(f"  {split}.list: {len(abn)} anomaly + {len(nrm)} normal = {len(abn)+len(nrm)}")

    return lists_dir / "train.list", lists_dir / "test.list", stats


# ── Step 4: ground-truth array ────────────────────────────────────────────────

def build_gt(processed_dir, lists_dir, num_segments=32, frames_per_seg=16):
    """
    Build ucf_gt.npy: frame-level binary labels for all test videos.
    Shape = (n_test × num_segments × frames_per_seg,)
    """
    processed_dir = Path(processed_dir)
    lists_dir     = Path(lists_dir)

    test_abn = sorted((processed_dir / "test" / "anomaly").glob("*.npy"))
    test_nrm = sorted((processed_dir / "test" / "normal").glob("*.npy"))
    test_vids = test_abn + test_nrm

    gt = []
    for vid_path in test_vids:
        label = 1 if "anomaly" in str(vid_path) else 0
        gt.extend([label] * (num_segments * frames_per_seg))

    gt_array = np.array(gt, dtype=np.int8)
    gt_path  = lists_dir / "ucf_gt.npy"
    np.save(gt_path, gt_array)

    logger.info(
        f"GT: shape={gt_array.shape}  "
        f"({len(test_abn)} abn + {len(test_nrm)} nrm = {len(test_vids)} test videos)"
    )
    return gt_path


# ── Step 5: CLIP prompt embeddings ────────────────────────────────────────────

def build_clip_prompts(lists_dir, ucf_classes, device="cuda"):
    """
    Encode 14 UCF-Crime class names with CLIP ViT-B/32.
    Saves [14, 512] float32 array to ucf_prompt.npy.
    """
    import torch
    import clip as openai_clip

    lists_dir = Path(lists_dir)
    model, _  = openai_clip.load("ViT-B/32", device=device)
    model.eval()

    prompts = [f"A surveillance video of {c}" for c in ucf_classes]
    with torch.no_grad():
        tokens = openai_clip.tokenize(prompts).to(device)
        feats  = model.encode_text(tokens)                  # [14, 512]
        feats  = feats / feats.norm(dim=-1, keepdim=True)
        feats  = feats.cpu().numpy().astype(np.float32)

    prompt_path = lists_dir / "ucf_prompt.npy"
    np.save(prompt_path, feats)
    logger.info(f"CLIP prompts: shape={feats.shape}  saved → {prompt_path}")
    return prompt_path


# ── Convenience wrapper ───────────────────────────────────────────────────────

def build_all(dataset_root, processed_dir, lists_dir,
              num_segments=32, feat_dim=1024,
              ucf_classes=None, clip_device="cuda"):
    """
    Run all five build steps and return a dict of output paths.
    Suitable for calling from a notebook cell.
    """
    from configs.base import UCF_CLASSES
    if ucf_classes is None:
        ucf_classes = UCF_CLASSES

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    logger.info("=" * 60)
    logger.info("Step 1+2: Feature preprocessing")
    preprocess_features(dataset_root, processed_dir, num_segments, feat_dim)

    logger.info("Step 3: List files")
    train_list, test_list, _ = build_list_files(processed_dir, lists_dir)

    logger.info("Step 4: Ground truth")
    gt_path = build_gt(processed_dir, lists_dir, num_segments)

    logger.info("Step 5: CLIP prompts")
    prompt_path = build_clip_prompts(lists_dir, ucf_classes, clip_device)

    logger.info("=" * 60)
    logger.info("Data build complete.")

    return {
        "processed_dir": str(processed_dir),
        "train_list":    str(train_list),
        "test_list":     str(test_list),
        "gt_path":       str(gt_path),
        "prompt_path":   str(prompt_path),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root",  required=True)
    parser.add_argument("--processed_dir", required=True)
    parser.add_argument("--lists_dir",     required=True)
    parser.add_argument("--num_segments",  type=int, default=32)
    parser.add_argument("--feat_dim",      type=int, default=1024)
    args = parser.parse_args()
    result = build_all(args.dataset_root, args.processed_dir,
                       args.lists_dir, args.num_segments, args.feat_dim)
    for k, v in result.items():
        print(f"  {k}: {v}")
