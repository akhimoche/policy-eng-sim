# Section 0: Standard library imports
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Section 1: Load experiment data
def load_experiment_data(experiment_name: str):
    """
    Load all data from a completed experiment.
    
    Args:
        experiment_name: Name of the experiment folder in data/
        
    Returns:
        Dictionary containing loaded data and metadata
    """
    data_dir = ROOT_DIR / "data" / experiment_name
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Experiment '{experiment_name}' not found in data/")
    
    # Load configuration
    config_file = data_dir / "experiment_config.json"
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Load cumulative welfare data
    cumulative_file = data_dir / "cumulative_welfare_all_runs.npy"
    cumulative_data = np.load(cumulative_file)
    
    # Load individual runs
    individual_runs = []
    run_files = sorted(data_dir.glob("run_*.npy"))
    for run_file in run_files:
        individual_runs.append(np.load(run_file))
    
    return {
        "config": config,
        "cumulative_welfare": cumulative_data,
        "individual_runs": individual_runs,
        "experiment_name": experiment_name
    }


def compute_mean_std_cumulative(cumulative_runs: np.ndarray, per_capita: bool = False, num_players: int = 1):
    """Return mean and std across runs for cumulative welfare."""
    mean_curve = np.mean(cumulative_runs, axis=0)
    std_curve = np.std(cumulative_runs, axis=0)
    if per_capita and num_players > 0:
        mean_curve = mean_curve / num_players
        std_curve = std_curve / num_players
    return mean_curve, std_curve


def overlay_plot(experiment_names, shade_std: bool = True, save_path: Path | None = None, per_capita: bool = False):
    """Plot mean cumulative welfare curves for multiple experiments on one figure."""
    datasets = []
    for exp in experiment_names:
        data = load_experiment_data(exp)
        cfg = data["config"]["config"]
        num_players = int(cfg.get("num_players", 1))
        mean_curve, std_curve = compute_mean_std_cumulative(
            data["cumulative_welfare"],
            per_capita=per_capita,
            num_players=num_players
        )
        label = f"{cfg['env_name']} | {cfg['norm_type']} (ε={cfg['epsilon']}, n={num_players})"
        datasets.append({
            "label": label,
            "mean": mean_curve,
            "std": std_curve,
            "timesteps": np.arange(len(mean_curve)),
            "name": data["experiment_name"],
        })

    if not datasets:
        print("No experiments provided.")
        return

    # Align by shortest length
    min_len = min(len(d["mean"]) for d in datasets)
    for d in datasets:
        d["mean"] = d["mean"][:min_len]
        d["std"] = d["std"][:min_len]
        d["timesteps"] = d["timesteps"][:min_len]

    # Plot
    plt.ioff()
    fig, ax = plt.subplots(figsize=(10, 6))

    # Distinct colors cycle
    color_cycle = plt.cm.tab10.colors

    for i, d in enumerate(datasets):
        color = color_cycle[i % len(color_cycle)]
        ax.plot(d["timesteps"], d["mean"], label=d["label"], color=color, linewidth=2)
        if shade_std:
            ax.fill_between(
                d["timesteps"],
                d["mean"] - d["std"],
                d["mean"] + d["std"],
                color=color,
                alpha=0.2,
            )

    ax.set_xlabel("Timestep", fontsize=12)
    y_label = "Cumulative Reward per Capita" if per_capita else "Cumulative Social Welfare"
    title = "Per-Capita Cumulative Reward Comparison" if per_capita else "Cumulative Welfare Comparison Across Experiments"
    ax.set_ylabel(y_label, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=10)

    plt.tight_layout()

    # Save
    if save_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = "per_capita_" if per_capita else ""
        save_path = ROOT_DIR / "data" / f"results_overlay_{suffix}{timestamp}.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Overlay plot saved to: {save_path}")

    plt.show()

# Section 2: Plot experiment results
def plot_experiment_results(data: dict, save_fig: bool = True):
    """
    Create visualizations for experiment results.
    
    Args:
        data: Dictionary from load_experiment_data()
        save_fig: If True, save plots to the experiment folder
    """
    config = data["config"]
    cumulative_data = data["cumulative_welfare"]
    individual_runs = data["individual_runs"]
    
    # Calculate statistics
    mean_cumulative = np.mean(cumulative_data, axis=0)
    std_cumulative = np.std(cumulative_data, axis=0)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: Cumulative welfare with confidence bands
    ax1 = axes[0]
    timesteps = np.arange(len(mean_cumulative))
    
    # Plot individual runs (transparent)
    for run in cumulative_data:
        ax1.plot(timesteps, run, alpha=0.2, color='gray', linewidth=0.5)
    
    # Plot mean with std band
    ax1.plot(timesteps, mean_cumulative, color='blue', linewidth=2, label='Mean')
    ax1.fill_between(
        timesteps, 
        mean_cumulative - std_cumulative, 
        mean_cumulative + std_cumulative,
        alpha=0.3, 
        color='blue',
        label='±1 std'
    )
    
    ax1.set_xlabel('Timestep', fontsize=12)
    ax1.set_ylabel('Cumulative Social Welfare', fontsize=12)
    ax1.set_title(
        f'Cumulative Welfare: {config["config"]["norm_type"]} (ε={config["config"]["epsilon"]})',
        fontsize=14
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Instantaneous rewards (mean ± std)
    ax2 = axes[1]
    individual_array = np.array(individual_runs)
    mean_rewards = np.mean(individual_array, axis=0)
    std_rewards = np.std(individual_array, axis=0)
    
    ax2.plot(timesteps, mean_rewards, color='green', linewidth=2, label='Mean')
    ax2.fill_between(
        timesteps,
        mean_rewards - std_rewards,
        mean_rewards + std_rewards,
        alpha=0.3,
        color='green',
        label='±1 std'
    )
    
    ax2.set_xlabel('Timestep', fontsize=12)
    ax2.set_ylabel('Reward per Step', fontsize=12)
    ax2.set_title('Instantaneous Team Reward', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_fig:
        save_path = ROOT_DIR / "data" / data["experiment_name"] / "results_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()


def list_experiments():
    """List all available experiments in the data/ folder."""
    data_dir = ROOT_DIR / "data"
    
    if not data_dir.exists():
        print("No data/ folder found. Run an experiment first!")
        return []
    
    experiments = sorted(d.name for d in data_dir.iterdir() if d.is_dir())
    
    if not experiments:
        print("No experiments found in data/. Run experiment.py first!")
        return []
    
    print(f"\n{'='*60}")
    print("AVAILABLE EXPERIMENTS:")
    print(f"{'='*60}")
    for i, exp in enumerate(experiments, 1):
        print(f"  {i}. {exp}")
    print(f"{'='*60}\n")
    
    return experiments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze a single experiment or overlay multiple experiments.")
    parser.add_argument(
        "experiments",
        nargs="*",
        help="Experiment directory names under data/. If omitted, will prompt a numbered list.")
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="Overlay multiple experiments (mean cumulative welfare)")
    parser.add_argument(
        "--per-capita",
        action="store_true",
        help="Overlay per-capita cumulative welfare (divide by num_players)")
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Disable ±1 std shading on overlay plots")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional output path for the overlay image")

    args = parser.parse_args()

    experiments = list_experiments()
    if not experiments:
        print("Run experiment.py first to generate data!")
        sys.exit(1)

    if args.overlay:
        exps = args.experiments
        if not exps:
            print(f"\n{'='*60}")
            print("AVAILABLE EXPERIMENTS:")
            print(f"{'='*60}")
            for i, exp in enumerate(experiments, 1):
                print(f"  {i}. {exp}")
            print(f"{'='*60}\n")
            selection = input("Enter comma-separated numbers to select experiments (e.g., 1,3,4): ").strip()
            if not selection:
                print("No selection made.")
                sys.exit(1)
            try:
                indices = [int(s) - 1 for s in selection.split(',')]
                exps = [experiments[i] for i in indices if 0 <= i < len(experiments)]
            except ValueError:
                print("Invalid selection.")
                sys.exit(1)

        save_path = Path(args.save) if args.save else None
        overlay_plot(exps, shade_std=not args.no_std, save_path=save_path, per_capita=args.per_capita)
    else:
        # Interactive mode with explicit single vs overlay choice
        if not args.experiments:
            print("Choose analysis mode:")
            print("  1. Analyze single experiment")
            print("  2. Analyze multiple experiments (and overlay the results)")
            print()
            mode_choice = input("Your choice (1 or 2): ").strip()
            if mode_choice == "2":
                print(f"\n{'='*60}")
                print("AVAILABLE EXPERIMENTS:")
                print(f"{'='*60}")
                for i, exp in enumerate(experiments, 1):
                    print(f"  {i}. {exp}")
                print(f"{'='*60}\n")
                selection = input("Enter comma-separated numbers to select experiments (e.g., 1,3,4): ").strip()
                if not selection:
                    print("No selection made.")
                    sys.exit(1)
                try:
                    indices = [int(s.strip()) - 1 for s in selection.split(',')]
                    exps = [experiments[i] for i in indices if 0 <= i < len(experiments)]
                except ValueError:
                    print("Invalid selection.")
                    sys.exit(1)
                overlay_plot(exps, shade_std=True, save_path=None, per_capita=False)
                sys.exit(0)

        # Single experiment mode
        selected_experiment = None
        if args.experiments:
            if len(args.experiments) != 1:
                print("Please provide exactly one experiment for single-plot mode.")
                sys.exit(1)
            selected_experiment = args.experiments[0]
        else:
            print("Which experiment do you want to analyze?")
            print("  • Enter a number (1, 2, 3...)")
            print("  • Or type 'latest' for the most recent")
            print()
            choice = input("Your choice: ").strip()
            if choice.lower() == 'latest':
                selected_experiment = experiments[-1]
                print(f"\n→ Loading latest experiment: {selected_experiment}\n")
            elif choice.isdigit():
                idx = int(choice) - 1  # User sees 1-indexed list
                if 0 <= idx < len(experiments):
                    selected_experiment = experiments[idx]
                    print(f"\n→ Loading experiment {choice}: {selected_experiment}\n")
                else:
                    print(f"\n❌ Error: Choice {choice} is out of range (1-{len(experiments)})")
                    sys.exit(1)
            else:
                print(f"\n❌ Error: Invalid input '{choice}'. Please enter a number or 'latest'")
                sys.exit(1)

        data = load_experiment_data(selected_experiment)
        print(f"Loaded {len(data['individual_runs'])} runs")
        print(f"Timesteps: {len(data['individual_runs'][0])}")
        print(f"\nGenerating plots...")
        plot_experiment_results(data, save_fig=True)

