# code/config.py

SYSTEM_CONFIG = {
    "seed": 6783,
    "work_dir": "../report_src",  # Where to save plots/text
    "data_dir": "../datasets",  # Root data folder
}


# Linear Predictor (Context Length)
LINEAR_CONTEXT_SWEEP = {
    "name": "Linear_Context_Sweep",
    "model_type": "linear",
    "base": {
        "batch_size": 128,
        "lr": 1e-3,
        "epochs": 5,
        "n_embd": 64,
        "optimizer": "adamw",
    },
    "sweep_param": "block_size",
    "sweep_values": [8, 32, 64, 128, 256],
}

# MLP Experiments (Width & Depth)

# Varying Width (Hidden Size)
MLP_WIDTH_SWEEP = {
    "name": "MLP_Width_Sweep",
    "model_type": "mlp",
    "base": {
        "block_size": 32,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 5,
        "n_layers": 3,
        "n_embd": 64,
        "optimizer": "adamw",
    },
    "sweep_param": "hidden_size",
    "sweep_values": [64, 128, 256, 512],
}

#  Varying Depth (Number of Layers)
MLP_DEPTH_SWEEP = {
    "name": "MLP_Depth_Sweep",
    "model_type": "mlp",
    "base": {
        "block_size": 32,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 5,
        "hidden_size": 128,
        "n_embd": 64,
        "optimizer": "adamw",
    },
    "sweep_param": "n_layers",
    "sweep_values": [2, 4, 8],
}

# Attention-Only Model (No FeedForward)
ATTENTION_SWEEP = {
    "name": "AttentionOnly_Heads_Sweep",
    "model_type": "attention",
    "base": {
        "block_size": 64,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 5,
        "n_embd": 128,
        "n_layers": 2,
        "optimizer": "adamw",
        "dropout": 0.2,
    },
    "sweep_param": "n_head",
    "sweep_values": [2, 4, 8],
}

# Transformer Experiments (Heads & Depth)

# Varying Heads
TRANSFORMER_HEADS_SWEEP = {
    "name": "Transformer_Heads_Sweep",
    "model_type": "transformer",
    "base": {
        "block_size": 64,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 5,
        "n_embd": 128,
        "n_layers": 2,
        "optimizer": "adamw",
        "dropout": 0.2,
    },
    "sweep_param": "n_head",
    "sweep_values": [2, 4, 8],
}

# Varying Depth
TRANSFORMER_DEPTH_SWEEP = {
    "name": "Transformer_Depth_Sweep",
    "model_type": "transformer",
    "base": {
        "block_size": 64,
        "batch_size": 64,
        "lr": 5e-4,
        "epochs": 5,
        "n_embd": 64,
        "n_head": 4,
        "optimizer": "adamw",
        "dropout": 0.2,
    },
    "sweep_param": "n_layers",
    "sweep_values": [1, 2, 4, 6],
}

# Optimization Experiments

# Optimizer Choice (AdamW vs SGD)
OPTIMIZER_SWEEP = {
    "name": "Optimizer_Sweep",
    "model_type": "transformer",
    "base": {
        "block_size": 64,
        "batch_size": 64,
        "lr": 1e-3,
        "epochs": 5,
        "n_embd": 64,
        "n_head": 4,
        "n_layers": 2,
        "dropout": 0.2,
    },
    "sweep_param": "optimizer",
    "sweep_values": ["adamw", "sgd"],
}

# Learning Rate Sweep
LR_SWEEP = {
    "name": "LearningRate_Sweep",
    "model_type": "transformer",
    "base": {
        "block_size": 64,
        "batch_size": 64,
        "epochs": 5,
        "n_embd": 64,
        "n_head": 4,
        "n_layers": 2,
        "optimizer": "adamw",
        "dropout": 0.2,
    },
    "sweep_param": "lr",
    "sweep_values": [1e-2, 1e-3, 5e-4, 1e-4],
}
