import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

def plot_bernoulli_distribution(p, title, xlabel, ylabel, labels, filename):
    x = np.array([0, 1])
    pmf = bernoulli.pmf(x, p)
    plt.figure()
    plt.bar(x, pmf, width=0.1, edgecolor='k', alpha=0.7)
    plt.xticks([0, 1], labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

def main():
    # Example 1: Coin Toss
    p_coin = 0.5
    plot_bernoulli_distribution(p_coin,
                                'Bernoulli Distribution (p=0.5)',
                                'Outcome', 'Probability',
                                ['Tails (0)', 'Heads (1)'],
                                'coin_bernoulli.png')

    # Example 2: Quality Control
    p_defective = 0.1
    plot_bernoulli_distribution(p_defective,
                                'Bernoulli Distribution (p=0.1)',
                                'Outcome', 'Probability',
                                ['Not Defective (0)', 'Defective (1)'],
                                'quality_control_bernoulli.png')

    # Example 3: Email Click-through Rate
    p_click = 0.03
    plot_bernoulli_distribution(p_click,
                                'Bernoulli Distribution (p=0.03)',
                                'Outcome', 'Probability',
                                ['No Click (0)', 'Click (1)'],
                                'email_clickthrough_bernoulli.png')

    # Example 4: Loan Default
    p_default = 0.05
    plot_bernoulli_distribution(p_default,
                                'Bernoulli Distribution (p=0.05)',
                                'Outcome', 'Probability',
                                ['No Default (0)', 'Default (1)'],
                                'loan_default_bernoulli.png')

if __name__ == "__main__":
    main()