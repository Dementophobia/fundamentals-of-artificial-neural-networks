import numpy as np
import matplotlib.pyplot as plt

def generate_water_stats():
    cov  = np.array([[1.2, 1], [1, 1]])
    mean = np.array([3.2, 3.5])

    values = np.random.multivariate_normal(mean=mean, cov=cov, size=100)
    
    return values[:, 0]*5, values[:, 1]

def plot_stats_without_lobf(x, y):
    plt.scatter(x, y)
    plt.title("Fake water consumption study")
    plt.xlabel("Temperature in °C")
    plt.ylabel("Liters of water")
    plt.show()

def plot_stats_with_lobf(x, y, a, b):
    lobf = a*x + b
    plt.scatter(x, y)
    plt.plot(x, lobf)
    plt.title("Fake water consumption study with line of best fit")
    plt.xlabel("Temperature in °C")
    plt.ylabel("Liters of water")
    plt.show()

def calculate_lobf(x, y):
    denom = x.mean() * x.sum() - x.dot(x)
    
    a = (x.sum() * y.mean() - x.dot(y)) / denom
    b = (x.dot(y) * x.mean() - y.mean() * x.dot(x)) / denom
    
    return a, b

x, y = generate_water_stats()
plot_stats_without_lobf(x, y)
a, b = calculate_lobf(x, y)
plot_stats_with_lobf(x, y, a, b)

print(f"Line of best fit: y = {a:.2f}*x + {b:.2f}")
