import matplotlib.pyplot as plt
import numpy as np

def line_plot():
    # Example: Average Temperature Over a Week
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    temperatures = [22, 24, 19, 23, 25, 27, 26]

    plt.plot(days, temperatures, marker='o', linestyle='-', color='b')
    plt.title('Average Temperature Over a Week')
    plt.xlabel('Days')
    plt.ylabel('Temperature (°C)')
    plt.grid(True)
    plt.show()

def customized_line_plot():
    # Example: Sales Data Over Two Weeks
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    sales_week1 = [150, 200, 170, 220, 250, 300, 280]
    sales_week2 = [180, 210, 160, 230, 240, 310, 290]

    plt.plot(days, sales_week1, label='Week 1', color='r', linestyle='--', marker='o')
    plt.plot(days, sales_week2, label='Week 2', color='b', linestyle='-', marker='x')

    plt.title('Sales Data Over Two Weeks')
    plt.xlabel('Days')
    plt.ylabel('Sales ($)')
    plt.legend()
    plt.grid(True)
    plt.show()

def bar_plot():
    # Example: Number of Products Sold in Different Categories
    categories = ['Electronics', 'Clothing', 'Groceries', 'Books', 'Toys']
    products_sold = [120, 150, 200, 90, 80]

    plt.bar(categories, products_sold, color='skyblue')
    plt.title('Number of Products Sold in Different Categories')
    plt.xlabel('Categories')
    plt.ylabel('Number of Products Sold')
    plt.grid(axis='y')
    plt.show()

def scatter_plot():
    # Example: Height vs. Weight of Individuals
    heights = [150, 160, 165, 170, 175, 180, 185]
    weights = [55, 60, 62, 65, 70, 75, 80]
    sizes = [50, 60, 70, 80, 90, 100, 110]  # marker size
    colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']

    plt.scatter(heights, weights, s=sizes, c=colors, alpha=0.5)
    plt.title('Height vs. Weight of Individuals')
    plt.xlabel('Height (cm)')
    plt.ylabel('Weight (kg)')
    plt.grid(True)
    plt.show()

def histogram():
    # Example: Distribution of Exam Scores
    np.random.seed(0)
    scores = np.random.normal(75, 10, 1000)

    plt.hist(scores, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Exam Scores')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def pie_chart():
    # Example: Market Share of Different Companies
    companies = ['Company A', 'Company B', 'Company C', 'Company D']
    market_share = [25, 35, 20, 20]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0, 0)  # explode 1st slice

    plt.pie(market_share, explode=explode, labels=companies, colors=colors, autopct='%1.1f%%',
            shadow=True, startangle=140)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Market Share of Different Companies')
    plt.show()

if __name__ == "__main__":
    line_plot()
    customized_line_plot()
    bar_plot()
    scatter_plot()
    histogram()
    pie_chart()

