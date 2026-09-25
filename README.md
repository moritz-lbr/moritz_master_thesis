# Universal Feature Learning Dynamics Revealed by Self-Averaging Observables

Code accompanying the Master's thesis (For the sake of clarity, the repository has been reorganised using Claude.)

> **Moritz Rautenberg**, *Universal Feature Learning Dynamics Revealed by Self-Averaging Observables*,
> Master's thesis, Faculty of Physics, Ludwig-Maximilians-Universität München, 2026.
> Supervised by Pascal de Jong, Daniel Pals and Prof. Dr. Steffen Rulands.
> The thesis is included in this repository: [`Moritz_Rautenberg_Masterthesis.pdf`](Moritz_Rautenberg_Masterthesis.pdf).

The thesis compares the standard parametrization (SP) and the maximal-update
parametrization (µP) of a one-hidden-layer MLP, `f(x) = W¹ σ(W⁰ x)`, trained with plain
mini-batch SGD on CIFAR-10 (and SVHN) at hidden widths `n = 512 … 4096`. This repository
contains everything needed to

1. train the networks (JAX/Flax, [`src/`](src)),
2. post-process the saved weight trajectories ([`training_analysis/`](training_analysis)), and
3. create every data-driven figure of the thesis ([`visualizations/`](visualizations)) as well as
   the schematic illustrations ([`explanation_figures/`](explanation_figures)).

---

## Contents

- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Quickstart](#quickstart)
- [Reproducing the thesis](#reproducing-the-thesis)
- [Figure overview](#figure-overview)
- [Experiment configuration](#experiment-configuration)
- [Output of a training run](#output-of-a-training-run)
- [Getting started with µP](#getting-started-with-µp)

## Repository structure

```text
.
├── Moritz_Rautenberg_Masterthesis.pdf   the thesis
├── pyproject.toml / pixi.lock           environment definition (pixi)
├── src/                                 training code
│   ├── run_experiment.py                  entry point: trains all width configs of an experiment
│   ├── main.py                            trains one network and writes its logs
│   ├── train.py                           SGD loop, loss/weight snapshots (zarr)
│   ├── mlp.py                             Flax MLP with parametrization-aware initialization
│   ├── param_schemes.py                   abc-parametrizations: SP ("standard") and µP ("muP")
│   ├── config.py                          config dataclass
│   ├── data.py                            dataset loaders (CIFAR-10, SVHN)
│   └── utils.py                           losses, optimizer with per-layer learning rates, state
├── datasets/
│   └── download.py                      downloads CIFAR-10 and SVHN into datasets/
├── experiments/                         experiment configs (Hydra), see below
│   ├── cifar10/{mup,sp}/configs/          the CIFAR-10 runs of the thesis
│   ├── svhn/{mup,sp}/configs/             the SVHN runs
│   └── quickstart/{cifar10,svhn}_{mup,sp}/configs/   small runs for testing (minutes on a CPU)
├── training_analysis/
│   └── compute_metrics.py               layer-wise weight-update norms from the saved weights
├── visualizations/                      scripts that create the thesis figures
│   ├── analyze_acts.py                    Fig. 3.1
│   ├── plot_observables.py                Figs. 5.1–5.3
│   ├── plot_update_norms.py               update norms incl. deviations/breakdown times (supplementary)
│   ├── analyze_superposition.py           Figs. 5.7, 5.8 (+ caches for mp_spectra_corr.py)
│   ├── mp_spectra_corr.py                 Figs. 5.9–5.12
│   ├── run_utils.py                       shared helpers (run discovery, legends, metrics)
│   └── superposition_utils.py             shared helpers (zarr snapshots, evaluation, styling)
├── explanation_figures/                 schematic figures (Figs. 3.2, 4.1, 5.4–5.6), no data needed
├── scripts/
│   ├── quickstart.sh                      download → train → all figures, in one command
│   └── make_figures.sh                    all thesis figures from trained job directories
└── slurm/
    ├── submit_training.sh / train.sbatch  one SLURM array task per width config
    └── make_figures.sbatch                figure pipeline as a SLURM job
```

## Setup

The environment is managed with [pixi](https://pixi.sh), which installs Python, JAX/Flax/Optax and
all other dependencies from [`pyproject.toml`](pyproject.toml) into a project-local environment
(`.pixi/`). The exact versions used for the thesis are pinned in [`pixi.lock`](pixi.lock).

1. Install pixi (once):

   ```bash
   curl -fsSL https://pixi.sh/install.sh | bash
   ```

2. Clone the repository and install the environment:

   ```bash
   git clone https://github.com/moritz-lbr/moritz_master_thesis.git
   cd moritz_master_thesis
   pixi install              # CPU environment ("default"), enough for the quickstart
   pixi install -e gpu       # optional: NVIDIA GPU with CUDA 12 (used for the thesis runs)
   ```

   Three environments are defined: `default` (CPU), `gpu` (CUDA 12) and `gpu13` (CUDA 13).
   Commands run in the default environment unless `-e <env>` is given, e.g.
   `pixi run -e gpu train ...`.

3. Check the installation:

   ```bash
   pixi run python -c "import jax; print(jax.devices())"
   ```

> **Tip for clusters with a small home quota:** pixi caches packages in `~/.cache/rattler`.
> Redirect the caches before installing, e.g.
> `export PIXI_CACHE_DIR=/scratch/$USER/.pixi-cache XDG_CACHE_HOME=/scratch/$USER/.cache`.

All commands below are run from the repository root.

## Quickstart

The quickstart trains one-hidden-layer MLPs of width `n = 128, 256, 512` under µP and SP on
CIFAR-10 and SVHN (a few epochs each) and runs the complete analysis pipeline on them:

```bash
pixi run quickstart            # CIFAR-10 and SVHN (~30 min on a CPU, plus the download)
pixi run quickstart cifar10    # CIFAR-10 only
```

The same steps can also be run one at a time:

```bash
# 1. Download the data (~170 MB CIFAR-10, ~250 MB SVHN) into datasets/
pixi run download-data cifar10 svhn

# 2. Train all widths of an experiment. Each config in <experiment>/configs is trained in turn.
pixi run train --path experiments/quickstart/cifar10_mup --output_dir experiments/quickstart/cifar10_mup/logs/job-quickstart
pixi run train --path experiments/quickstart/cifar10_sp  --output_dir experiments/quickstart/cifar10_sp/logs/job-quickstart

# 3. Create all figures from the two job directories
pixi run figures --mup experiments/quickstart/cifar10_mup/logs/job-quickstart \
                 --sp  experiments/quickstart/cifar10_sp/logs/job-quickstart \
                 --out figures/quickstart/cifar10
```

(and likewise with `svhn_mup` / `svhn_sp`). The results are organized as follows:

```text
experiments/quickstart/cifar10_mup/logs/job-quickstart/
├── scheme-muP_widths-128x10/        one folder per trained network (see "Output of a training run")
├── scheme-muP_widths-256x10/
└── scheme-muP_widths-512x10/

figures/quickstart/cifar10/
├── fig3_1_activation_deltas_{mup,sp}.png           Fig. 3.1
├── fig5_1_update_norms_mup.png                     Fig. 5.1
├── fig5_2_test_loss_mup.png                        Fig. 5.2
├── fig5_3_{update_norms,test_loss}_sp.png          Fig. 5.3
├── fig5_7_superposition_mup_{step0,best_test_loss}.png   Fig. 5.7
├── fig5_8_superposition_sp_{step0,best_test_loss}.png    Fig. 5.8
├── fig5_9_eigenvalues_mup_{step0,best_test_loss}.png     Fig. 5.9
├── fig5_10_eigenvalues_sp_{step0,best_test_loss}.png     Fig. 5.10
├── fig5_11_relative_reconstruction_error_{mup,sp}.png    Fig. 5.11
├── fig5_12_reconstruction_best_test_loss_{mup,sp}.png    Fig. 5.12
└── details/{mup,sp}/                               every output of the individual scripts
    ├── analyze_acts/                                 activation changes
    ├── update_norms/<task>/<job>/                    update norms, deviations, breakdown times
    └── superposition/<task>/<job>/                   superposition matrices (all saved steps),
                                                      eigenvalue spectra, reconstructions, caches
```

The quickstart networks are small and trained only briefly, so the figures show the
qualitative behaviour at best. The thesis figures need the full runs described next.

## Reproducing the thesis

### 1. Training

The configs in [`experiments/cifar10`](experiments/cifar10) reproduce the CIFAR-10 runs of the
thesis: widths `n = 512·2^{k/2}, k = 0…6` (512 … 4096), base width 128, batch size 512, a fixed
batch order (`batch_seed: 0`) shared by all widths, losses and weights saved every
`τ = 100` SGD steps, cross-entropy loss, no biases, no weight decay:

| experiment | parametrization | learning rate | epochs |
|---|---|---|---|
| `experiments/cifar10/mup` | µP | η = 0.005 (with the µP per-layer rescaling) | 250 |
| `experiments/cifar10/sp`  | SP | η = 0.005 | 250 |
| `experiments/svhn/{mup,sp}` | µP / SP | η = 0.01 | 500 |

The thesis runs were trained on NVIDIA A40 GPUs with one SLURM array task per width. Install the
GPU environment once before submitting (otherwise all array tasks try to install it simultaneously):

```bash
pixi install -e gpu
bash slurm/submit_training.sh experiments/cifar10/mup     # -> experiments/cifar10/mup/logs/job-<id>/
bash slurm/submit_training.sh experiments/cifar10/sp      # -> experiments/cifar10/sp/logs/job-<id>/
```

Adapt the `#SBATCH` partition/GPU lines in [`slurm/train.sbatch`](slurm/train.sbatch) to your
cluster. `PIXI_ENV=default bash slurm/submit_training.sh ...` runs on CPU nodes instead; additional
`sbatch` options can be appended. Without SLURM, `pixi run -e gpu train --path experiments/cifar10/mup`
trains all widths sequentially.

The loss curves in Figs. 5.2 and 5.3 are averages over **five independent initializations**:
submit each experiment five times (the network initialization is random, the batch order fixed).

> **Resources:** one width takes up to ~1 h on an A40. Because the weights are saved every 100 SGD
> steps, one job (7 widths) needs ~25 GB of disk space.

### 2. Figures

```bash
pixi run figures --mup experiments/cifar10/mup/logs/job-<id1> ... experiments/cifar10/mup/logs/job-<id5> \
                 --sp  experiments/cifar10/sp/logs/job-<id1>  ... experiments/cifar10/sp/logs/job-<id5> \
                 --out figures/thesis --thesis
```

or, recommended for the full-size data, as a SLURM job with more memory:

```bash
sbatch slurm/make_figures.sbatch --mup <mup-job-dirs> --sp <sp-job-dirs> --out figures/thesis --thesis
```

`--thesis` fixes the axis limits of the loss plots to those used in the thesis. All jobs given are
averaged in the loss/update-norm figures; the weight-based analyses use the first job of each
parametrization. Intermediate results (weight metrics, spectra, reconstructions) are cached, so
re-running is fast; use `--force` to recompute.

[`scripts/make_figures.sh`](scripts/make_figures.sh) simply chains the individual scripts, which
can also be called directly (all accept `--help`):

```bash
pixi run metrics --log-dir <job-dirs>                                             # weight_metrics/ per run
pixi run python visualizations/plot_observables.py update-norms --log-dirs <job-dirs> --output fig5_1.png
pixi run python visualizations/plot_observables.py test-loss    --log-dirs <job-dirs> --output fig5_2.png --limits thesis
pixi run python visualizations/analyze_acts.py          --log-dir   <job-dir> --output-dir figures/analyze_acts
pixi run python visualizations/analyze_superposition.py --input-dir <job-dir> --output-dir figures/superposition --best-input-space
pixi run python visualizations/mp_spectra_corr.py       --input-dir <job-dir> --output-dir figures/superposition
```

`mp_spectra_corr.py` reads the caches written by `analyze_superposition.py`, so both need the same
`--output-dir`.

### 3. Schematic figures

```bash
pixi run explanation-figures      # writes the PNGs into explanation_figures/
```

## Figure overview

| Thesis figure | Content | Script | Output (in `--out`) |
|---|---|---|---|
| 3.1 | Change of hidden pre-/postactivations vs. width (SP / µP) | `visualizations/analyze_acts.py` | `fig3_1_activation_deltas_{sp,mup}.png` |
| 3.2 | Network used in the experiments | `explanation_figures/mlp_figures.py` | `explanation_figures/MLP_z.png` |
| 4.1 | Weights connected to one hidden neuron | `explanation_figures/mlp_figures.py` | `explanation_figures/MLP_z_neuron_connections.png` |
| 5.1 | Raw and rescaled layer-wise update norms, µP | `visualizations/plot_observables.py update-norms` | `fig5_1_update_norms_mup.png` |
| 5.2 | Test loss averaged over seeds, µP | `visualizations/plot_observables.py test-loss` | `fig5_2_test_loss_mup.png` |
| 5.3 | Update norms and test loss, SP | `visualizations/plot_observables.py` | `fig5_3_update_norms_sp.png`, `fig5_3_test_loss_sp.png` |
| 5.4 | Pixels select columns of W⁰ | `explanation_figures/hidden_representation.py` | `explanation_figures/hidden_representations.png` |
| 5.5 | Overlap of hidden representations | `explanation_figures/overlap_hidden_representation.py` | `explanation_figures/Overlap_hidden_representations.png` |
| 5.6 | From overlaps to the superposition matrix | `explanation_figures/superposition.py` | `explanation_figures/from_hidden_rep_to_superposition.png` |
| 5.7 / 5.8 | Superposition matrix before training and at the best test loss, µP / SP (largest width) | `visualizations/analyze_superposition.py` | `fig5_7_*`, `fig5_8_*` |
| 5.9 / 5.10 | Eigenvalue spectra vs. decorrelated null spectrum, µP / SP | `visualizations/mp_spectra_corr.py` | `fig5_9_*`, `fig5_10_*` |
| 5.11 | Relative performance of the PCA-reconstructed model | `visualizations/mp_spectra_corr.py` | `fig5_11_relative_reconstruction_error_{mup,sp}.png` |
| 5.12 | Best test loss of the reconstructed model vs. k/k_det | `visualizations/mp_spectra_corr.py` | `fig5_12_reconstruction_best_test_loss_{mup,sp}.png` |

Figures 2.1–2.5 are taken from the literature (see the thesis).

## Experiment configuration

An experiment is a folder with a `configs/` subfolder (the same layout as the original working
repository). `master_config.yaml` holds the settings shared by all widths; each
`<scheme>_config<N>.yaml` inherits from it through Hydra and sets the hidden width:

```yaml
# experiments/cifar10/mup/configs/master_config.yaml
simulation_parameters:
  network:
    base_layer_widths: {Dense_0: 128, Dense_1: 10}   # base width of the µP rescaling
    activations_per_layer: {Dense_0: relu, Dense_1: identity}
    param_scheme: muP                               # muP | standard
  training:
    lr: 0.005
    wd: 0.0
    task: datasets/cifar-10-batches-py              # dataset folder
    batch_size: 512
    batch_seed: 0                                   # fixed batch order ('random' to reshuffle)
    epochs: 250
    save_loss_frequency: 100                        # save losses + weights every τ SGD steps
    weight_monitoring: True                         # write weights.zarr (needed for the analysis)
    loss_type: "cross_entropy"                      # cross_entropy | mse
```

```yaml
# experiments/cifar10/mup/configs/mup_config1.yaml
defaults:
  - master_config@_master
  - _self_
simulation_parameters:
  network:
    activations_per_layer: ${_master.simulation_parameters.network.activations_per_layer}
    nodes_per_layer: {Dense_0: 512, Dense_1: 10}    # hidden width n and output dimension
    param_scheme: ${_master.simulation_parameters.network.param_scheme}
    base_layer_widths: ${_master.simulation_parameters.network.base_layer_widths}
  training: ${_master.simulation_parameters.training}
```

Width configs may also override individual training settings by replacing
`training: ${_master...}` with an explicit `training:` block (e.g. a width-dependent `lr`).
To create a new experiment, copy one of the folders in `experiments/` and edit the configs.
The parametrizations are defined in [`src/param_schemes.py`](src/param_schemes.py)
(the `a`, `b`, `c` exponents of Table 3.1 in the thesis).

### Further datasets

[`src/data.py`](src/data.py) ships loaders for CIFAR-10 and SVHN, the datasets used in the thesis.
Further benchmark datasets, for example from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu), can be added with a loader that
returns `(x_train, y_train, x_test, y_test)`, a detection function and a branch in
`load_dataset` (see the module docstring). The dataset folder below `datasets/` also needs a
`dataset_overview.yaml` with the input and output dimension, as written by
[`datasets/download.py`](datasets/download.py).

## Output of a training run

Each trained network writes a folder `<experiment>/logs/<job>/scheme-<muP|standard>_widths-<n>x<p>/`:

| file | content |
|---|---|
| `simulation_config.yaml` | resolved network and training settings, dataset info, parameter counts |
| `training_log.json` | train/test loss and mean layer activations at every saved step (`final_metrics.history`), final parameters |
| `weights.zarr/` | weight matrices `Dense_0/kernel`, `Dense_1/kernel` at initialization and every `save_loss_frequency` steps (float16), plus first-layer terms for 5 test inputs (`logits`) |
| `weight_metrics/<layer>.json` | written by `training_analysis/compute_metrics.py`: squared update norms between saved steps, cosine similarities, path lengths |

## Getting started with µP

The maximal-update parametrization and the infinite-width analysis used in the thesis originate
from the Tensor Programs series by Greg Yang and collaborators. Good entry points are:

- [microsoft/mup](https://github.com/microsoft/mup): the reference implementation of µP for
  PyTorch, with an introduction to µP and µTransfer (hyperparameter transfer across widths).
- The Tensor Programs series:
  - G. Yang, *Tensor Programs I: Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes*, [arXiv:1910.12478](https://arxiv.org/abs/1910.12478)
  - G. Yang, *Tensor Programs II: Neural Tangent Kernel for Any Architecture*, [arXiv:2006.14548](https://arxiv.org/abs/2006.14548)
  - G. Yang, *Tensor Programs III: Neural Matrix Laws*, [arXiv:2009.10685](https://arxiv.org/abs/2009.10685)
  - G. Yang, E. J. Hu, *Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks*, [arXiv:2011.14522](https://arxiv.org/abs/2011.14522): introduces the abc-parametrizations and µP
  - G. Yang, E. J. Hu, I. Babuschkin, S. Sidor, X. Liu, D. Farhi, N. Ryder, J. Pachocki, W. Chen, J. Gao, *Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer*, [arXiv:2203.03466](https://arxiv.org/abs/2203.03466)
  - G. Yang, E. Littwin, *Tensor Programs IVb: Adaptive Optimization in the Infinite-Width Limit*, [arXiv:2308.01814](https://arxiv.org/abs/2308.01814)
  - G. Yang, D. Yu, C. Zhu, S. Hayou, *Tensor Programs VI: Feature Learning in Infinite-Depth Neural Networks*, [arXiv:2310.02244](https://arxiv.org/abs/2310.02244)
```
