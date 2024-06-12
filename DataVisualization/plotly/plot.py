import plotly.graph_objects as go #type:ignore
import numpy as np

# Example data
stock_prices = {'Date': ['2024-06-01', '2024-06-02', '2024-06-03'],
                'Price': [100, 110, 105]}

sales_data = {'Product': ['Product A', 'Product B', 'Product C'],
              'Sales': [150, 200, 180]}

temperature_data = {'Day': [1, 2, 3, 4, 5],
                    'Temperature': [20, 22, 21, 25, 24],
                    'Humidity': [60, 65, 63, 68, 70]}

expenses_data = {'Category': ['Food', 'Utilities', 'Transportation', 'Entertainment'],
                 'Amount': [500, 300, 200, 100]}

# Line Chart
def line_chart():
    fig = go.Figure(data=go.Scatter(x=stock_prices['Date'], y=stock_prices['Price'], mode='lines+markers'))
    fig.update_layout(title='Stock Prices Over Time', xaxis_title='Date', yaxis_title='Price')
    fig.show()

# Bar Chart
def bar_chart():
    fig = go.Figure(data=go.Bar(x=sales_data['Product'], y=sales_data['Sales']))
    fig.update_layout(title='Sales Performance by Product', xaxis_title='Product', yaxis_title='Sales')
    fig.show()

# Scatter Plot
def scatter_plot():
    fig = go.Figure(data=go.Scatter(x=temperature_data['Temperature'], y=temperature_data['Humidity'], mode='markers'))
    fig.update_layout(title='Temperature vs Humidity', xaxis_title='Temperature', yaxis_title='Humidity')
    fig.show()

# Pie Chart
def pie_chart():
    fig = go.Figure(data=go.Pie(labels=expenses_data['Category'], values=expenses_data['Amount']))
    fig.update_layout(title='Monthly Expenses Distribution')
    fig.show()

# Histogram
def histogram():
    np.random.seed(0)
    x = np.random.randn(1000)
    fig = go.Figure(data=[go.Histogram(x=x)])
    fig.update_layout(title='Histogram of Random Data', xaxis_title='Value', yaxis_title='Frequency')
    fig.show()

# 3D Plot
def three_d_plot():
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
    fig.update_layout(title='3D Plot of sin(sqrt(x^2 + y^2))')
    fig.show()

def main():
    line_chart()
    bar_chart()
    scatter_plot()
    pie_chart()
    histogram()
    three_d_plot()

if __name__ == "__main__":
    main()
