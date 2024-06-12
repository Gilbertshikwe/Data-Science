# sea.py

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the example dataset for tips
tips = sns.load_dataset("tips")
# Load the example dataset for flights
flights = sns.load_dataset("flights")

# Line Plot
def line_plot():
    plt.figure(figsize=(10, 6))
    sns.lineplot(x='month', y='passengers', data=flights)
    plt.title('Line Plot of Monthly Passengers Over Time')
    plt.show()

# Scatter Plot
def scatter_plot():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='total_bill', y='tip', data=tips, hue='time', style='time', size='size')
    plt.title('Scatter Plot of Tips vs. Total Bill')
    plt.show()

# Box Plot
def box_plot():
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker')
    plt.title('Box Plot of Total Bill by Day')
    plt.show()

# Point Plot
def point_plot():
    plt.figure(figsize=(10, 6))
    sns.pointplot(x='day', y='total_bill', data=tips, hue='smoker', markers=['o', 'x'])
    plt.title('Point Plot of Total Bill by Day')
    plt.show()

# Count Plot
def count_plot():
    plt.figure(figsize=(10, 6))
    sns.countplot(x='day', data=tips, hue='sex')
    plt.title('Count Plot of Days')
    plt.show()

# Violin Plot
def violin_plot():
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', split=True)
    plt.title('Violin Plot of Total Bill by Day')
    plt.show()

# Swarm Plot
def swarm_plot():
    plt.figure(figsize=(10, 6))
    sns.swarmplot(x='day', y='total_bill', data=tips, hue='sex')
    plt.title('Swarm Plot of Total Bill by Day')
    plt.show()

# Bar Plot
def bar_plot():
    plt.figure(figsize=(10, 6))
    sns.barplot(x='day', y='total_bill', data=tips, hue='sex')
    plt.title('Bar Plot of Total Bill by Day')
    plt.show()

# KDE Plot
def kde_plot():
    plt.figure(figsize=(10, 6))
    sns.kdeplot(tips['total_bill'], fill=True)
    plt.title('KDE Plot of Total Bill')
    plt.show()

if __name__ == "__main__":
    line_plot()
    scatter_plot()
    box_plot()
    point_plot()
    count_plot()
    violin_plot()
    swarm_plot()
    bar_plot()
    kde_plot()
