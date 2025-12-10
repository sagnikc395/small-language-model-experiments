# config of the experiments

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
