# utils.py
from typing import Dict, Any, Tuple, List
from collections import OrderedDict
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax.training import train_state
import optax
import yaml
import os

from mlp import MLP
from param_schemes import SCHEMES # pyright: ignore[reportMissingImports]
from data import resolve_dataset_path  # noqa: F401  (re-exported for main.py)


# ----------------------------
# Formatting
# ----------------------------

def load_yaml_as_dict(file_path):
    """Load YAML data from a file and return it as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file)
    
    # Ensure it returns a dictionary (use empty dict if None or not a dict)
    if not isinstance(data, dict):
        data = {} if data is None else {"value": data}
    
    return data

def extract_after_char(s, start, stop):
    return s[s.rfind(start)+1:s.rfind(stop)]


def format_widths(widths: Tuple[int, ...]) -> str:
    return "x".join(str(int(w)) for w in widths)


def parameter_summary(params: Dict[str, Any]) -> Tuple[Dict[str, int], int]:
    per_layer: Dict[str, int] = {}
    total = 0
    for layer_name, layer_params in params.items():
        leaves = jax.tree_util.tree_leaves(layer_params)
        layer_total = 0
        for leaf in leaves:
            arr = np.asarray(leaf)
            layer_total += int(arr.size)
        per_layer[layer_name] = layer_total
        total += layer_total
    return per_layer, total

def tree_to_python(tree: Any) -> Any:
    """Convert a pytree of arrays into nested Python lists/numbers for JSON dumping."""
    def convert(x):
        if isinstance(x, (int, float, str, bool)) or x is None:
            return x
        if isinstance(x, np.generic):
            return x.item()
        if hasattr(x, "tolist"):
            return x.tolist()
        return x
    return jax.tree_util.tree_map(convert, tree)

def _sorted_nodes(nodes_map: Dict[str, int]) -> OrderedDict:
    return OrderedDict(
        sorted(
            ((k, int(v)) for k, v in nodes_map.items()),
            key=lambda kv: int(kv[0].split("_")[1])
        )
    )

# ----------------------------
# Data
# ----------------------------

def collect_files_with_ending(directory: Path, ending: str) -> List[Path]:
    matches: List[Path] = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(ending):
                matches.append(Path(root) / filename)
    return matches

# ----------------------------
# Loss
# ----------------------------

def first_layer_term_matrix(params, first_layer_activation_matrix):
    if "Dense_1" not in params or "kernel" not in params["Dense_1"]:
        raise ValueError("Could not find Dense_1/kernel needed to compute first-layer term matrix.")

    second_layer_kernel = params["Dense_1"]["kernel"]
    if second_layer_kernel.ndim != 2:
        raise ValueError(
            "Expected Dense_1/kernel to have shape (width_0, output_dim) so the saved term matrix can be computed. "
            f"Got shape {second_layer_kernel.shape}."
        )

    if second_layer_kernel.shape[1] == 1:
        return first_layer_activation_matrix * second_layer_kernel[:, 0].astype(jnp.float32)

    return (
        first_layer_activation_matrix[:, :, None]
        * second_layer_kernel.astype(jnp.float32)[None, :, :]
    )

def mse_loss(params, apply_fn, xb, yb, return_layer_act=False):
    if return_layer_act:
        preds, acts, first_layer_activation_matrix = apply_fn({"params": params}, xb, capture_layer_acts=True)
        return (
            jnp.mean((preds - yb) ** 2),
            jnp.mean(acts, axis=0),
            first_layer_term_matrix(params, first_layer_activation_matrix),
        )
    else:
        preds = apply_fn({"params": params}, xb)
        return jnp.mean((preds - yb) ** 2)
    
# def cross_entropy_loss(params, apply_fn, xb, yb, return_layer_act=False):
#     # x: (N, C), y: (N,)
#     if return_layer_act:
#         preds, acts = apply_fn({"params": params}, xb, capture_layer_acts=True)
#     else:
#         preds = apply_fn({"params": params}, xb)
    
#     log_sum_exp = jnp.log(jnp.sum(jnp.exp(preds), axis=1))
#     rows = jnp.arange(preds.shape[0])
#     loss = -(preds[rows, yb.astype(jnp.int32)] - log_sum_exp)

#     if return_layer_act:
#         return jnp.mean(loss, axis=0), jnp.mean(acts, axis=0)
#     else: 
#         return jnp.mean(loss, axis=0)

def cross_entropy_loss(params, apply_fn, xb, yb, return_layer_act=False):
    def prepare_classification_labels(logits, labels):
        if logits.ndim != 2:
            raise ValueError(
                "Cross-entropy loss expects model outputs with shape (batch_size, num_classes). "
                f"Got shape {logits.shape}."
            )

        num_classes = int(logits.shape[-1])
        labels_array = jnp.asarray(labels)

        if labels_array.ndim == 1:
            return jax.nn.one_hot(labels_array.astype(jnp.int32), num_classes, dtype=jnp.float32)
        if labels_array.ndim == 2 and labels_array.shape[-1] == 1:
            flat_labels = labels_array[:, 0].astype(jnp.int32)
            return jax.nn.one_hot(flat_labels, num_classes, dtype=jnp.float32)
        if labels_array.ndim == 2 and labels_array.shape[-1] == num_classes:
            return labels_array.astype(jnp.float32)

        raise ValueError(
            "Unsupported classification target shape. Expected integer class indices with "
            f"shape (batch_size,) or one-hot targets with shape (batch_size, {num_classes}), "
            f"but got {labels_array.shape}."
        )

    if return_layer_act:
        preds, acts, first_layer_activation_matrix = apply_fn({"params": params}, xb, capture_layer_acts=True)
        labels_onehot = prepare_classification_labels(preds, yb)
        return (
            jnp.mean(optax.safe_softmax_cross_entropy(preds, labels=labels_onehot), axis=0),
            jnp.mean(acts, axis=0),
            first_layer_term_matrix(params, first_layer_activation_matrix),
        )
    else:
        preds = apply_fn({"params": params}, xb)
        labels_onehot = prepare_classification_labels(preds, yb)
        return jnp.mean(optax.safe_softmax_cross_entropy(preds, labels=labels_onehot), axis=0)


# ----------------------------
# Grad norms (optional instrumentation)
# Currently unused
# ----------------------------

def grad_norms_by_layer(grads):
    flat = jax.tree_util.tree_flatten_with_path(grads)
    norms = {}
    for path, leaf in flat[0]:
        if leaf is None:
            continue
        key = "/".join([str(p) for p in path])
        norms[key] = jnp.linalg.norm(leaf)
    return norms

# ----------------------------
# Optimizer utilities
# ----------------------------
class TrainState(train_state.TrainState):
    lr_mults: Any  # pytree of LR multipliers (same structure as variables)

def decay_mask(params):
    # True where we want weight decay, False where we don't
    return jax.tree_util.tree_map(lambda p: p.ndim > 1, params)

def make_optimizer(base_lr: float, wd: float, lr_mults_pytree):
    # A transform that multiplies grads by the LR multipliers
    def scale_by_lr_mults(lr_mults):
        def init_fn(_): return optax.EmptyState()
        def update_fn(updates, state, params=None):
            scaled = jax.tree.map(lambda g, m: g * m, updates, lr_mults)
            return scaled, state
        return optax.GradientTransformation(init_fn, update_fn)

    tx = optax.chain(
        optax.add_decayed_weights(wd, mask=decay_mask),
        scale_by_lr_mults(lr_mults_pytree),
        optax.sgd(learning_rate=base_lr, momentum=None, nesterov=False),
    )
    return tx

def identity(x):
    return x

ACTS = {
    "identity": identity,
    "tanh": jax.nn.tanh,
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "sigmoid" : jax.nn.sigmoid
}


def create_state(
    rng,
    widths,
    activations,
    param_scheme_name: str,
    lr: float,
    wd: float,
    sample_x,
    kernel_dims,
    nodes_per_layer,
    base_layer_widths, 
    base_kernel_dims,
):
    
    acts = [ACTS[act] for act in activations] 
    scheme_template = SCHEMES[param_scheme_name]
    scheme = (
        scheme_template
              .with_layer_widths(nodes_per_layer)
              .with_base_layer_widths(base_layer_widths)
              .with_base_kernel_dims(base_kernel_dims)
              )
              
    model = MLP(widths=tuple(widths), acts=acts, param_scheme=scheme, kernel_dims=kernel_dims, base_kernel_dims=base_kernel_dims)
    variables = model.init(rng, sample_x)
    params = variables["params"]
    lr_mults_vars = scheme.lr_multiplier_pytree(kernel_dims)
    tx = make_optimizer(lr, wd, lr_mults_vars["params"])  # apply multipliers to param leaves
    state = TrainState(
        step=0,
        apply_fn=model.apply,
        params=params,
        tx=tx,
        opt_state=tx.init(params),
        lr_mults=lr_mults_vars
    )
    return state, model
