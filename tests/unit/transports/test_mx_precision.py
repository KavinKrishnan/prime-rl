import torch
from torch import nn

from prime_rl.utils.mx_precision import wire_dtype_overrides


def test_models_without_policy_keep_default_transport_dtype():
    model = nn.Linear(2, 2)
    assert wire_dtype_overrides(model, model.state_dict()) == {}


def test_model_policy_receives_exact_state_dict_names():
    class Model(nn.Module):
        @staticmethod
        def keep_in_fp32_for_weight_transfer(name):
            return name.endswith("selection_bias") or name.endswith("scale")

    tensors = {
        "model._orig_mod.layers.0.mlp.router.selection_bias": torch.tensor([1.000123]),
        "model.norm.scale": torch.tensor(1.000123),
        "model.layers.0.mlp.experts.gate_proj": torch.ones(2, 3),
    }
    assert wire_dtype_overrides(Model(), tensors) == {
        "model._orig_mod.layers.0.mlp.router.selection_bias": torch.float32,
        "model.norm.scale": torch.float32,
    }
