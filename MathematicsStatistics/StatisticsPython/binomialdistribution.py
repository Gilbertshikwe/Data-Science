import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

def plot_binomial_distribution(n, p, x_range, title, xlabel, ylabel, filename):
    binomial_pmf = binom.pmf(x_range, n, p)
    plt.figure()
    plt.bar(x_range, binomial_pmf, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

def main():
    # Example 1: Coin Flipping
    n_coin = 10
    p_coin = 0.5
    x_range_coin = np.arange(0, n_coin + 1)
    plot_binomial_distribution(n_coin, p_coin, x_range_coin,
                               'Binomial Distribution (n=10, p=0.5)',
                               'Number of Heads', 'Probability',
                               'coin_flipping_binomial.png')

    # Example 2: Quality Control
    n_qc = 100
    p_qc = 0.05
    x_range_qc = np.arange(0, 11)  # We consider up to 10 defective products for visualization
    plot_binomial_distribution(n_qc, p_qc, x_range_qc,
                               'Binomial Distribution (n=100, p=0.05)',
                               'Number of Defective Products', 'Probability',
                               'quality_control_binomial.png')

if __name__ == "__main__":
    main()
