"""
configs/mgfn_cfg.py
All hyperparameters for MGFN on UCF-Crime.

MGFN expects features of shape [B, n_crops, T, C+1] where the +1 channel
is the L2 magnitude of each feature vector.  With our single-crop
[32, 1024] features we simulate n_crops=1 in the dataset loader.
"""
from configs.base import *

def get_mgfn_config(overrides: dict = None) -> dict:
    cfg = dict(
        model_name      = "mgfn",
        dataset         = "ucf-crime",
        metrics         = "AUC",

        # ── Paths ──────────────────────────────────────────────────────────
        feat_prefix     = str(PROCESSED_DIR),
        train_list      = str(TRAIN_LIST),
        test_list       = str(TEST_LIST),
        gt              = str(GT_PATH),
        save_dir        = str(RESULTS_DIR / "mgfn") + "/",
        logs_dir        = str(RESULTS_DIR / "mgfn_train.log"),
        ckpt_path       = None,

        # ── Feature ────────────────────────────────────────────────────────
        # MGFN original uses 2048-dim (RGB+flow) UCF 10-crop features.
        # We use 1024-dim single-crop.  Magnitude channel is appended by
        # the dataset, making the input channel = feat_dim + 1.
        feat_dim        = FEAT_DIM,   # 1024
        n_crops         = 1,          # single-crop (original: 10)
        seg_length      = NUM_SEGMENTS,  # 32

        # ── Model architecture ──────────────────────────────────────────────
        # Dims tuple: (stage1_dim, stage2_dim, stage3_dim=output_dim)
        dims            = (64, 128, 1024),
        depths          = (3, 3, 2),
        mgfn_types      = ("gb", "fb", "fb"),  # gb=GLANCE, fb=FOCUS
        dim_head        = 64,
        ff_repe         = 4,
        dropout         = 0.7,
        attention_dropout = 0.0,
        mag_ratio       = 0.1,        # weight of magnitude branch

        # ── Training ───────────────────────────────────────────────────────
        # LR schedule: the original uses a step list.
        # We use a flat LR + cosine decay for simplicity.
        lr              = 1e-3,
        max_epoch       = 100,
        batch_size      = 16,         # pairs: 16 normal + 16 abnormal
        workers         = 4,
        seed            = 42,

        # ── Loss ───────────────────────────────────────────────────────────
        contrastive_margin = 200.0,
        lambda_smooth   = 8e-4,
        lambda_sparse   = 8e-3,
        top_k           = 3,          # MSNSD top-k selection

        # ── Inference ──────────────────────────────────────────────────────
        test_bs         = 1,
    )
    if overrides:
        cfg.update(overrides)
    return cfg
