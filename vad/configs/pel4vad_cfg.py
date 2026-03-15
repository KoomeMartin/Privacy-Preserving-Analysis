"""
configs/pel4vad_cfg.py
All hyperparameters for PEL4VAD on UCF-Crime.
"""
from configs.base import *

def get_pel4vad_config(overrides: dict = None) -> dict:
    cfg = dict(
        model_name   = "pel4vad",
        dataset      = "ucf-crime",
        metrics      = "AUC",

        # ── Paths (filled in by notebooks at runtime) ──────────────────────
        feat_prefix  = str(PROCESSED_DIR),
        train_list   = str(TRAIN_LIST),
        test_list    = str(TEST_LIST),
        token_feat   = str(PROMPT_PATH),
        gt           = str(GT_PATH),
        save_dir     = str(RESULTS_DIR / "pel4vad") + "/",
        logs_dir     = str(RESULTS_DIR / "pel4vad_train.log"),
        ckpt_path    = None,   # set after training

        # ── Feature ────────────────────────────────────────────────────────
        feat_dim     = FEAT_DIM,       # 1024
        max_seqlen   = 200,            # padding target during training

        # ── TCA (Temporal Context Aggregation) ─────────────────────────────
        win_size     = 9,
        gamma        = 0.6,
        bias         = 0.2,
        norm         = True,

        # ── Causally-Connected (CC) classifier ─────────────────────────────
        t_step       = 9,

        # ── Model head dims ────────────────────────────────────────────────
        head_num     = 1,
        hid_dim      = 128,
        out_dim      = 300,

        # ── Training ───────────────────────────────────────────────────────
        lr           = 5e-4,
        dropout      = 0.1,
        train_bs     = 128,
        max_epoch    = 50,
        workers      = 4,
        temp         = 0.09,   # logit scale temperature
        lamda        = 1,      # weight of prompt-contrastive loss
        seed         = 9,

        # ── Inference / test ───────────────────────────────────────────────
        test_bs      = 1,      # MUST stay 1 — see evaluator.py for reason
        smooth       = "slide",
        kappa        = 7,
    )
    if overrides:
        cfg.update(overrides)
    return cfg
