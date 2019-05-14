import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def add_column_with_ones(x):
    return np.hstack((np.ones((x.shape[0], 1)).astype(int), x))

def get_sample_data():
    x = np.array([[91, 30], [46, 10], [131, 19], [3, 5], [107, 13]])
    y = np.array([5.82, 1.8, 5.99, 1.79, 4.04])
    
    return x, y

def generate_water_stats():
    sport = np.random.randint(200, size=100)
    temp  = np.random.randint(35,  size=100)
    
    x = np.stack((sport, temp)).transpose()
    y = sport/60 + temp/10 + 1 + np.random.randn(100)/2
    
    return x, y

def plot_data(x_dp, y_dp, z_dp, w=None):
    ax = plt.figure().gca(projection='3d')
    
    if isinstance(w, np.ndarray):
        x, y = np.meshgrid(np.linspace(0, 200, 10), np.linspace(0, 35, 10))
        z = w[1]*x + w[2]*y + w[0]
    
        surf = ax.plot_wireframe(x, y, z)

    ax.scatter(x_dp, y_dp, z_dp)

    plt.title("Fake water consumption study")
    ax.set_xlabel("Minutes of sports")
    ax.set_ylabel("Temperature in °C")
    ax.set_zlabel("Liters of water")
    plt.show()

def calculate_pobf(x, y):
    x = add_column_with_ones(x)
    x_trans = x.transpose()

    w_solve = np.linalg.solve(x_trans.dot(x), x_trans.dot(y))
    print(f"Calculated with solve-function:\t{w_solve}")

    w_asis = np.linalg.inv(x_trans.dot(x)).dot(x_trans.dot(y))
    print(f"Calculated 'as-is':\t\t{w_asis}")
    
    return w_solve

np.set_printoptions(precision=2)

x, y = generate_water_stats()
plot_data(x[:,0], x[:,1], y)

w = calculate_pobf(x, y)
plot_data(x[:,0], x[:,1], y, w)