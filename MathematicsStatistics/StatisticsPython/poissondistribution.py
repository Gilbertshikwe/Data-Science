import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

def plot_poisson_distribution(lambda_val, k_range, title, xlabel, ylabel, filename):
    poisson_pmf = poisson.pmf(k_range, lambda_val)
    plt.figure()
    plt.bar(k_range, poisson_pmf, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

def main():
    # Example 1: Number of Emails Received
    lambda_email = 5
    k_range_email = np.arange(0, 15)
    plot_poisson_distribution(lambda_email, k_range_email,
                              'Poisson Distribution (lambda=5)',
                              'Number of Emails', 'Probability',
                              'emails_poisson.png')

    # Example 2: Number of Phone Calls
    lambda_calls = 10
    k_range_calls = np.arange(0, 21)
    plot_poisson_distribution(lambda_calls, k_range_calls,
                              'Poisson Distribution (lambda=10)',
                              'Number of Phone Calls', 'Probability',
                              'calls_poisson.png')

    # Example 3: Number of Cars Passing Through a Toll Booth
    lambda_cars = 20
    k_range_cars = np.arange(0, 41)
    plot_poisson_distribution(lambda_cars, k_range_cars,
                              'Poisson Distribution (lambda=20)',
                              'Number of Cars', 'Probability',
                              'cars_poisson.png')

    # Example 4: Number of Customer Arrivals at a Bank
    lambda_customers = 3
    k_range_customers = np.arange(0, 10)
    plot_poisson_distribution(lambda_customers, k_range_customers,
                              'Poisson Distribution (lambda=3)',
                              'Number of Customers', 'Probability',
                              'customers_poisson.png')

if __name__ == "__main__":
    main()
