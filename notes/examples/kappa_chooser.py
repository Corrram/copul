import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import copul
import os
import time

def generate_sample_data(n_obs=10_000, seed=None):
    """Generate sample data with a known correlation structure."""
    if seed is not None:
        np.random.seed(seed)
    
    factor_cols = [f"F{i}" for i in range(1, 2)]
    factor_data = np.random.normal(size=(n_obs, 2))
    
    all_weights = np.array([1, 1])
    stock = (factor_data @ all_weights) / 2
    return pd.DataFrame(
        np.column_stack((factor_data[:, :-1], stock)), columns=factor_cols + ["Stock"]
    )

def estimate_xi_with_kappa(sample_data, kappa, true_xi=0.3098):
    """Estimate xi using BivCheckPi and BivCheckMin with a specific kappa value."""
    results = {}
    
    # Measure CheckPi method
    ccop_pi = copul.BivCheckPi.from_data(sample_data, kappa=kappa)
    lb = ccop_pi.xi()
    results["check_pi"] = lb
    results["error_check_pi"] = abs(lb - true_xi)
    
    # Measure CheckMin method
    ccop_min = copul.BivCheckMin.from_data(sample_data, kappa=kappa)
    ub = ccop_min.xi()
    results["check_min"] = ub
    results["error_check_min"] = abs(ub - true_xi)
    
    # Calculate bin count based on the kappa value
    n_samples = len(sample_data)
    n_features = 2  # Bivariate data (F1 and Stock)
    bin_count = np.ceil(n_samples ** (2 * kappa / n_features))
    results["n_bins"] = bin_count
    
    return results

def run_kappa_experiment(sample_sizes, kappa_values, n_samples_per_size=30, true_xi=0.3098, seed=42):
    """Run experiments with different kappa values for multiple sample sizes."""
    all_results = []
    
    for size in tqdm(sample_sizes, desc="Testing sample sizes"):
        size_results = []
        
        for sample_idx in tqdm(range(n_samples_per_size), desc=f"Samples for size {size}", leave=False):
            # Generate new data with deterministic seed variation
            sample_seed = seed + (size * 100) + sample_idx
            sample_data = generate_sample_data(n_obs=size, seed=sample_seed)
            
            # Test each kappa value
            for kappa in kappa_values:
                # Get estimates for this kappa
                results = estimate_xi_with_kappa(sample_data, kappa, true_xi)
                
                # Add metadata
                results["kappa"] = kappa
                results["sample_size"] = size
                results["sample_id"] = sample_idx
                
                size_results.append(results)
            
            # Report progress every 5 samples
            if (sample_idx + 1) % 5 == 0:
                print(f"Completed {sample_idx + 1}/{n_samples_per_size} samples for size {size}")
        
        all_results.extend(size_results)
        print(f"Finished all {n_samples_per_size} samples for size {size}")
    
    return pd.DataFrame(all_results)

def plot_kappa_sensitivity(results_df, true_xi=0.3098, save_path=None):
    """Create a plot showing how kappa affects xi estimates at different sample sizes."""
    # Calculate standard deviations for error bars
    std_results = results_df.groupby(["sample_size", "kappa"]).agg({
        "check_pi": ["mean", "std"],
        "check_min": ["mean", "std"],
        "n_bins": "mean"
    })
    
    # Flatten multi-level column index
    std_results.columns = ['_'.join(col).strip() for col in std_results.columns.values]
    std_results = std_results.reset_index()
    
    # Get unique sample sizes
    sample_sizes = sorted(results_df["sample_size"].unique())
    
    # We'll use the standard deviation results
    
    # Create plot with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
    
    # Use a color palette that distinguishes sample sizes well
    colors = sns.color_palette("viridis", len(sample_sizes))
    
    # Plot BivCheckPi results (left subplot)
    for i, size in enumerate(sample_sizes):
        size_data = std_results[std_results["sample_size"] == size]
        
        # Plot the mean line
        ax1.plot(
            size_data["kappa"],
            size_data["check_pi_mean"],
            "o-",
            color=colors[i],
            label=f"n = {size}",
            linewidth=2,
            markersize=8,
        )
        
        # Add error bands using standard deviation
        ax1.fill_between(
            size_data["kappa"],
            size_data["check_pi_mean"] - size_data["check_pi_std"],
            size_data["check_pi_mean"] + size_data["check_pi_std"],
            color=colors[i],
            alpha=0.2
        )
        
        # Annotate the first and last points with bin counts
        first_point = size_data.iloc[0]
        last_point = size_data.iloc[-1]
        ax1.annotate(
            f"{first_point['n_bins_mean']:.0f} bins",
            xy=(first_point["kappa"], first_point["check_pi_mean"]),
            xytext=(-10, -15),
            textcoords="offset points",
            fontsize=9,
            color=colors[i]
        )
        ax1.annotate(
            f"{last_point['n_bins_mean']:.0f} bins",
            xy=(last_point["kappa"], last_point["check_pi_mean"]),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=9,
            color=colors[i]
        )
    
    # Plot BivCheckMin results (right subplot)
    for i, size in enumerate(sample_sizes):
        size_data = std_results[std_results["sample_size"] == size]
        
        # Plot the mean line
        ax2.plot(
            size_data["kappa"],
            size_data["check_min_mean"],
            "o-",
            color=colors[i],
            label=f"n = {size}",
            linewidth=2,
            markersize=8,
        )
        
        # Add error bands using standard deviation
        ax2.fill_between(
            size_data["kappa"],
            size_data["check_min_mean"] - size_data["check_min_std"],
            size_data["check_min_mean"] + size_data["check_min_std"],
            color=colors[i],
            alpha=0.2
        )
        
        # Annotate the first and last points with bin counts
        first_point = size_data.iloc[0]
        last_point = size_data.iloc[-1]
        ax2.annotate(
            f"{first_point['n_bins_mean']:.0f} bins",
            xy=(first_point["kappa"], first_point["check_min_mean"]),
            xytext=(-10, -15),
            textcoords="offset points",
            fontsize=9,
            color=colors[i]
        )
        ax2.annotate(
            f"{last_point['n_bins_mean']:.0f} bins",
            xy=(last_point["kappa"], last_point["check_min_mean"]),
            xytext=(10, -15),
            textcoords="offset points",
            fontsize=9,
            color=colors[i]
        )
    
    # Add true value reference line
    ax1.axhline(
        y=true_xi,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"True Value ({true_xi:.4f})",
    )
    ax2.axhline(
        y=true_xi,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"True Value ({true_xi:.4f})",
    )
    
    # Set titles and labels
    ax1.set_title("BivCheckPi Estimates vs. Kappa", fontsize=16)
    ax2.set_title("BivCheckMin Estimates vs. Kappa", fontsize=16)
    
    for ax in [ax1, ax2]:
        ax.set_xlabel("Kappa Value", fontsize=14)
        ax.set_ylabel("Xi Estimate", fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12, loc='best')
        
        # Add explanation text about shaded areas
        if ax == ax1:
            ax.text(0.98, 0.02, 
                   "Shaded areas represent ±1 standard deviation\nacross 30 random samples",
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Overall title
    plt.suptitle("Effect of Kappa Parameter on Xi Estimates at Different Sample Sizes", fontsize=18)
    
    plt.tight_layout()
    
    # Save or display the plot
    if save_path:
        # Create directory if it doesn't exist
        directory = os.path.dirname(save_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()

if __name__ == "__main__":
    # Set seed for reproducibility
    seed = 42
    np.random.seed(seed)
    
    # Parameters for the experiment
    sample_sizes = [100, 1000, 10000]
    kappa_values = np.linspace(0.1, 0.7, 20)  # 11 values from 1.0 to 2.0
    n_samples_per_size = 30  # 30 samples per size for robust averaging
    true_xi = 0.3098
    
    # Output directory
    output_dir = "images"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"Running kappa sensitivity experiment:")
    print(f"- Sample sizes: {sample_sizes}")
    print(f"- Kappa values: {kappa_values}")
    print(f"- Samples per size: {n_samples_per_size}")
    
    # Run the experiment
    results_df = run_kappa_experiment(
        sample_sizes=sample_sizes,
        kappa_values=kappa_values,
        n_samples_per_size=n_samples_per_size,
        true_xi=true_xi,
        seed=seed
    )
    
    # Create the plot
    kappa_plot_path = os.path.join(output_dir, f"kappa_sensitivity_seed_{seed}.png")
    plot_kappa_sensitivity(results_df, true_xi, save_path=kappa_plot_path)
    
    print("\nExperiment complete!")
    print(f"Plot saved to {kappa_plot_path}")