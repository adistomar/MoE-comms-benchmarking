# Copyright (c) 2026. MoE dispatch/combine decode benchmark — model presets.
"""
Nemotron-3 model presets: the MoE shape each benchmark run models (--model).

Values are copied from each model's Hugging Face config.json (pulled 2026-10-14):
  super: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16/blob/main/config.json
  ultra: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16/blob/main/config.json

Both are hybrid Mamba/attention/MoE stacks. The layer pattern uses Super's
hybrid_override_pattern alphabet (M = Mamba, E = MoE, * = attention); Ultra's config lists the
same thing as layers_block_type (mamba / moe / attention), converted here. Only E blocks
dispatch and combine, so a decode step does one dispatch->combine per E block.

Both are latent MoEs: tokens are projected from hidden_size down to moe_latent_size before
dispatch and back up after combine, so the rows the collectives move are moe_latent_size wide
(ModelSpec.dispatch_hidden). Each routed expert is a moe_latent_size -> moe_intermediate_size ->
moe_latent_size MLP with squared ReLU (mlp_hidden_act = relu2), which run.py --expert vllm runs.

Kept free of torch so plot_results.py can import it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str                # --model value
    name: str               # display name (plot titles, logs)
    hf_repo: str
    layer_pattern: str      # one char per block: M = Mamba, E = MoE, * = attention
    hidden_size: int        # model hidden size (router input)
    moe_latent_size: int    # dispatched / combined row width
    moe_intermediate_size: int  # routed-expert FFN width (Megatron's moe_ffn_hidden_size)
    num_experts: int        # n_routed_experts
    topk: int               # num_experts_per_tok

    @property
    def num_blocks(self) -> int:
        return len(self.layer_pattern)

    @property
    def num_moe_layers(self) -> int:
        """MoE blocks per decode step = dispatch->combine pairs per step."""
        return self.layer_pattern.count("E")

    @property
    def dispatch_hidden(self) -> int:
        """Row width (elements) that dispatch / combine move."""
        return self.moe_latent_size


MODELS = {
    "super": ModelSpec(
        key="super",
        name="Nemotron-3-Super-120B-A12B",
        hf_repo="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16",
        layer_pattern="MEMEMEM*EMEMEMEM*EMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*EMEMEMEMEM*"
                      "EMEMEMEM*EMEMEMEME",
        hidden_size=4096,
        moe_latent_size=1024,
        moe_intermediate_size=2688,
        num_experts=512,
        topk=22,
    ),
    "ultra": ModelSpec(
        key="ultra",
        name="Nemotron-3-Ultra-550B-A55B",
        hf_repo="nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        layer_pattern="MEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEM*EMEMEM*"
                      "EMEMEMEM*EMEMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME",
        hidden_size=8192,
        moe_latent_size=2048,
        moe_intermediate_size=5120,
        num_experts=512,
        topk=22,
    ),
}
DEFAULT_MODEL = "ultra"

# Block counts from the configs (num_hidden_layers / len(layers_block_type)): guards the
# copied patterns against transcription errors.
assert MODELS["super"].num_blocks == 88 and MODELS["super"].num_moe_layers == 40
assert MODELS["ultra"].num_blocks == 108 and MODELS["ultra"].num_moe_layers == 48


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise ValueError(f"unknown model '{key}' (choose from {sorted(MODELS)})")
    return MODELS[key]
