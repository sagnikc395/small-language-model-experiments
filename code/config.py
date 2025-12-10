# config of the experiments
from utils import PTB_DIR, WT2_DIR


EXPERIMENT_CONFIG_DELIVERABLE_1 = {
    "linear": [
        {"name": "ctx_32", "context_len": 32, "lr": 1e-3},
        {"name": "ctx_64", "context_len": 64, "lr": 1e-3},
        {"name": "ctx_128", "context_len": 128, "lr": 1e-3},
    ],
    "mlp": [
        # Varying depth/width
        {"name": "dim_128", "hidden_dims": [128, 128], "context_len": 64, "lr": 1e-3},
        {"name": "dim_256", "hidden_dims": [256, 256], "context_len": 64, "lr": 1e-3},
        {"name": "dim_512", "hidden_dims": [512, 512], "context_len": 64, "lr": 1e-3},
    ],
    "self_attention": [
        # Varying Heads (embed_dim 128 is divisible by 2, 4, 8)
        {
            "name": "head_2",
            "embed_dim": 128,
            "num_heads": 2,
            "context_len": 64,
            "lr": 1e-3,
        },
        {
            "name": "head_4",
            "embed_dim": 128,
            "num_heads": 4,
            "context_len": 64,
            "lr": 1e-3,
        },
        {
            "name": "head_8",
            "embed_dim": 128,
            "num_heads": 8,
            "context_len": 64,
            "lr": 1e-3,
        },
    ],
    "transformer": [
        # Varying Depth
        {
            "name": "layer_1",
            "num_layers": 1,
            "embed_dim": 128,
            "num_heads": 4,
            "mlp_hidden": 256,
            "context_len": 128,
            "lr": 5e-4,
        },
        {
            "name": "layer_2",
            "num_layers": 2,
            "embed_dim": 128,
            "num_heads": 4,
            "mlp_hidden": 256,
            "context_len": 128,
            "lr": 5e-4,
        },
        {
            "name": "layer_4",
            "num_layers": 4,
            "embed_dim": 128,
            "num_heads": 4,
            "mlp_hidden": 256,
            "context_len": 128,
            "lr": 5e-4,
        },
    ],
}

EXPERIMENT_CONFIG_DELIVERABLE_2 = [
    {
        "name": "PTB",
        "path": PTB_DIR,
        "vocab": 10000,
        "prompts": ["the school announced that"],
    },
    {
        "name": "WikiText-2",
        "path": WT2_DIR,
        "vocab": 20000,
        "prompts": ["The history of machine learning beigns"],
    },
    {
        "name": "WikiText-2",
        "path": WT2_DIR,
        "vocab": 15000,
        "prompts": ["During the time of World War 2"],
    },
]
