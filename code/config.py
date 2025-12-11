# code/config.py

# ==========================================
# GLOBAL SETTINGS
# ==========================================
SYSTEM_CONFIG = {
    "seed": 1337,
    "work_dir": "../report_src",  # Where to save plots
    "data_dir": "../datasets",  # Root data folder containing subfolders
}

# DELIVERABLE 1: TINY SHAKESPEARE SWEEPS

# 1. Linear Sweep (Varying Context Length)
LINEAR_SWEEP = {
    "name": "Linear_Context_Sweep",
    "model_type": "linear",
    "base": {
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 5,
        "n_embd": 64,
    },
    "sweep_param": "block_size",
    "sweep_values": [8, 16, 32, 64, 128],
}

# 2. MLP Sweep (Varying Hidden Dimension)
MLP_SWEEP = {
    "name": "MLP_HiddenDim_Sweep",
    "model_type": "mlp",
    "base": {
        "block_size": 32,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 5,
        "n_layers": 3,
        "n_embd": 64,
    },
    "sweep_param": "hidden_size",
    "sweep_values": [64, 128, 256, 512],
}

# 3. Transformer Sweep (Varying Number of Heads)
TRANSFORMER_SWEEP = {
    "name": "Transformer_Heads_Sweep",
    "model_type": "transformer",
    "base": {
        "block_size": 64,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 5,
        "n_embd": 128,  # Must be divisible by n_head
        "n_layers": 2,
    },
    "sweep_param": "n_head",
    "sweep_values": [2, 4, 8],  # 128 must be divisible by these
}

# DELIVERABLE 2: WORD LEVEL (PTB / WIKI)
D2_SETTINGS = {"batch_size": 32, "epochs": 5, "lr": 5e-4, "block_size": 64}
