import argparse
from pathlib import Path
from datetime import datetime
from hydra import initialize_config_dir, compose
from omegaconf import OmegaConf
from main import run_experiment


def assemble(config_path: Path) -> dict:
    """Compose a width config with its master_config.yaml via Hydra and resolve it."""
    config_dir = config_path.resolve().parent
    name = config_path.name

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=name)
        resolved = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(resolved, dict):
            resolved.pop("_master", None)
    return resolved

def run_experiments(input_path: Path, output_path: Path = None) -> int:

    print("\nNote that the first entry of the weight log file contains the weights at initialization before the training has started.")
    print("\nNote that the total number of training steps displayed in the progress bar refers to the number of training steps that are saved during training.")
    print("The actual total number of training steps conducted in this training process can be obtained by multiplying the total number of steps displayed in the\n" \
    "progress bar with the save_loss_frequency assigned in the training config, as only every save_loss_frequency step losses and weights are saved.")
    print(f"\nRunning training for {input_path.name}") 
    
    if str(input_path).endswith(".yaml"):
        output_dir = output_path or Path("./logs")
        run_experiment(assemble(input_path), output_dir)

    else:
        configs = input_path.joinpath("configs")
        config_files = sorted(configs.glob("*.yaml"))
        config_files = [f for f in config_files if f.name != "master_config.yaml"]

        if output_path:
            output_dir = output_path
        else:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            output_dir = input_path.joinpath(f"logs/run_{timestamp}")

        for path in config_files:
            print(f"\nRunning training for {path.name}")
            run_experiment(assemble(path), output_dir)
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the networks of an experiment. --path is either a single width config "
            "(e.g. experiments/cifar10/mup/configs/mup_config1.yaml) or an experiment directory "
            "containing a configs/ folder, in which case all configs are trained sequentially."
        )
    )

    parser.add_argument("--path", type=Path, help="Config file or experiment directory", required=True)
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory to store run outputs.")

    args = parser.parse_args()

    return run_experiments(args.path, args.output_dir)

if __name__ == "__main__":
    main()