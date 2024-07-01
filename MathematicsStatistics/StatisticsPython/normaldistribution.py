import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def plot_histogram(data, title, xlabel, ylabel, filename):
    plt.figure()
    plt.hist(data, bins=30, edgecolor='k', alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.close()

def plot_pdf(x, pdf, title, xlabel, ylabel, filename):
    plt.figure()
    plt.plot(x, pdf, label='Normal Distribution', color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)
    plt.close()

def plot_pdf_with_shaded_areas(x, pdf, mean, std_dev, filename):
    plt.figure()
    plt.plot(x, pdf, label='Normal Distribution', color='blue')
    for num_std in [1, 2, 3]:
        plt.fill_between(x, pdf, where=(x > mean - num_std * std_dev) & (x < mean + num_std * std_dev),
                         color='blue', alpha=0.2 * num_std, label=f'{num_std} std dev')
    plt.title('Normal Distribution with 68-95-99.7 Rule')
    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.savefig(filename)
    plt.close()

def main():
    # Example 1: Generating Normal Distribution Data
    mean = 0
    std_dev = 1
    data = np.random.normal(mean, std_dev, 1000)
    plot_histogram(data, 'Histogram of Normally Distributed Data', 'Value', 'Frequency', 'histogram.png')

    # Example 2: Visualizing the Probability Density Function (PDF)
    x = np.linspace(-4, 4, 1000)
    pdf = norm.pdf(x, mean, std_dev)
    plot_pdf(x, pdf, 'Normal Distribution PDF', 'Value', 'Probability Density', 'pdf.png')

    # Example 3: The 68-95-99.7 Rule
    plot_pdf_with_shaded_areas(x, pdf, mean, std_dev, 'shaded_areas.png')

    # Real-Life Example: Exam Scores
    mean_exam = 75
    std_dev_exam = 10
    exam_scores = np.random.normal(mean_exam, std_dev_exam, 1000)
    plot_histogram(exam_scores, 'Histogram of Exam Scores', 'Score', 'Frequency', 'exam_scores.png')

if __name__ == "__main__":
    main()