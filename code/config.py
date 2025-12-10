# config of the experiments
from architectures import MLP, LinearRegressionModel, SelfAttentionLM, TransformerLM
from utils import PTB_DIR, WT2_DIR
from typing import Literal

OptimizerType = Literal["adam", "rmsprop", "sgd", "sgd-momentum"]
ArchitectureType = Literal["linear", "mlp", "transformer", "self_attention"]

EXPERIMENT_CONFIG_DELIVERABLE_1 = {
    # 1a. Linear: Vary Context Length
    "linear_ctx": [
        {
            "name": "lin_ctx_32",
            "model_type": "linear",
            "context_len": 32,
            "activation": "identity",
            "lr": 1e-3,
            "varying_param": 32,
            "param_label": "Context Length",
        },
        {
            "name": "lin_ctx_64",
            "model_type": "linear",
            "context_len": 64,
            "activation": "identity",
            "lr": 1e-3,
            "varying_param": 64,
            "param_label": "Context Length",
        },
        {
            "name": "lin_ctx_128",
            "model_type": "linear",
            "context_len": 128,
            "activation": "identity",
            "lr": 1e-3,
            "varying_param": 128,
            "param_label": "Context Length",
        },
    ],
    # 1b. Linear: Vary Activation Function (Fixed Context Length 64)
    "linear_act": [
        {
            "name": "lin_identity",
            "model_type": "linear",
            "context_len": 64,
            "activation": "identity",
            "lr": 1e-3,
            "varying_param": "Identity",
            "param_label": "Activation",
        },
        {
            "name": "lin_relu",
            "model_type": "linear",
            "context_len": 64,
            "activation": "relu",
            "lr": 1e-3,
            "varying_param": "ReLU",
            "param_label": "Activation",
        },
        {
            "name": "lin_tanh",
            "model_type": "linear",
            "context_len": 64,
            "activation": "tanh",
            "lr": 1e-3,
            "varying_param": "Tanh",
            "param_label": "Activation",
        },
    ],
    # 2. MLP: Vary Hidden Dimensions
    "mlp": [
        {
            "name": "mlp_128",
            "model_type": "mlp",
            "hidden_dims": [128, 128],
            "context_len": 64,
            "lr": 1e-3,
            "varying_param": 128,
            "param_label": "Hidden Dim",
        },
        {
            "name": "mlp_256",
            "model_type": "mlp",
            "hidden_dims": [256, 256],
            "context_len": 64,
            "lr": 1e-3,
            "varying_param": 256,
            "param_label": "Hidden Dim",
        },
        {
            "name": "mlp_512",
            "model_type": "mlp",
            "hidden_dims": [512, 512],
            "context_len": 64,
            "lr": 1e-3,
            "varying_param": 512,
            "param_label": "Hidden Dim",
        },
    ],
    # 3. Self-Attention: Vary Heads
    "self_attention": [
        {
            "name": "heads_2",
            "model_type": "self_attention",
            "num_heads": 2,
            "embed_dim": 128,
            "context_len": 64,
            "lr": 1e-3,
            "varying_param": 2,
            "param_label": "Num Heads",
        },
        {
            "name": "heads_4",
            "model_type": "self_attention",
            "num_heads": 4,
            "embed_dim": 128,
            "context_len": 64,
            "lr": 1e-3,
            "varying_param": 4,
            "param_label": "Num Heads",
        },
        {
            "name": "heads_8",
            "model_type": "self_attention",
            "num_heads": 8,
            "embed_dim": 128,
            "context_len": 64,
            "lr": 1e-3,
            "varying_param": 8,
            "param_label": "Num Heads",
        },
    ],
    # 4. Transformer: Vary Layers
    "transformer": [
        {
            "name": "layers_1",
            "model_type": "transformer",
            "num_layers": 1,
            "embed_dim": 128,
            "num_heads": 4,
            "mlp_hidden": 256,
            "context_len": 64,
            "lr": 5e-4,
            "varying_param": 1,
            "param_label": "Num Layers",
        },
        {
            "name": "layers_2",
            "model_type": "transformer",
            "num_layers": 2,
            "embed_dim": 128,
            "num_heads": 4,
            "mlp_hidden": 256,
            "context_len": 64,
            "lr": 5e-4,
            "varying_param": 2,
            "param_label": "Num Layers",
        },
        {
            "name": "layers_4",
            "model_type": "transformer",
            "num_layers": 4,
            "embed_dim": 128,
            "num_heads": 4,
            "mlp_hidden": 256,
            "context_len": 64,
            "lr": 5e-4,
            "varying_param": 4,
            "param_label": "Num Layers",
        },
    ],
    # 5. Optimization: Vary Optimizer
    "optimization": [
        {
            "name": "opt_adam",
            "model_type": "transformer",
            "optimizer": "adam",
            "lr": 1e-3,
            "context_len": 64,
            "varying_param": "Adam",
            "param_label": "Optimizer",
        },
        {
            "name": "opt_sgd",
            "model_type": "transformer",
            "optimizer": "sgd",
            "lr": 1e-3,
            "context_len": 64,
            "varying_param": "SGD",
            "param_label": "Optimizer",
        },
        {
            "name": "opt_rmsprop",
            "model_type": "transformer",
            "optimizer": "rmsprop",
            "lr": 1e-3,
            "context_len": 64,
            "varying_param": "RMSProp",
            "param_label": "Optimizer",
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


def get_model(tokenizer, ctx, conf):
    """
    init models using the 'model_type' key in the config
    """

    m_type = conf["model_type"]
    vocab_size = tokenizer.vocab_size

    match m_type:
        case "linear":
            return LinearRegressionModel(
                vocab_size, ctx, activation=conf.get("activation", "identity")
            )

        case "mlp":
            return MLP(vocab_size, ctx, conf["hidden_dims"])

        case "self_attention":
            return SelfAttentionLM(
                vocab_size, ctx, conf["embed_dim"], conf["num_heads"]
            )

        case "transformer":
            return TransformerLM(
                vocab_size,
                ctx,
                embed_dim=conf.get("embed_dim", 128),
                num_heads=conf.get("num_heads", 4),
                mlp_hidden=conf.get("mlp_hidden", 256),
                num_layers=conf.get("num_layers", 2),
            )
        case _:
            raise ValueError(f"Unknown model_type {m_type}")
