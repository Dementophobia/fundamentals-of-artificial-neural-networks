import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def draw_curve():
    x = np.linspace(-2, 2, 100)
    y = x**4 - 4*x**2

    plt.plot(x, y, color="blue")
    plt.title("Curve incl. local minima")
    plt.xlim((-2, 2))
    plt.ylim((-4.5, 0.2))
    plt.annotate("", xy=(-(2**0.5), -4), xytext=(-1, -4.2), arrowprops=dict(facecolor="red", shrink=0.05))
    plt.plot([-(2**0.5)], [-4], 'o', color="black")
    plt.annotate("", xy=(2**0.5, -4), xytext=(1, -4.2), arrowprops=dict(facecolor="red", shrink=0.05))
    plt.plot([2**0.5], [-4], 'o', color="black")
    plt.show()

def draw_plane():    
    x, y = np.meshgrid(np.linspace(-2, 2, 25), np.linspace(-2, 2, 25))
    z = x**4 - 4*x**2 + y**4 - 2*y**2
    
    ax = plt.figure().gca(projection='3d')
    ax.plot_wireframe(x, y, z)
    ax.plot([-(2**0.5)], [1], [-5], 'o', color="black")
    ax.plot([(2**0.5)], [1], [-5], 'o', color="black")
    ax.plot([-(2**0.5)], [-1], [-5], 'o', color="black")
    ax.plot([(2**0.5)], [-1], [-5], 'o', color="black")
    plt.title("3d plane with four local minima")
    ax.set_xlabel("x")
    ax.set_xticks(range(-2, 3))
    ax.set_ylabel("y")
    ax.set_yticks(range(-2, 3))
    ax.set_zlabel("z")
    ax.set_zticks(range(-5, 9, 2))
    plt.show()

def draw_example_steps_2d(steps):
    x = np.linspace(0, 2, 50)
    y = x**4 - 4*x**2

    plt.plot(x, y, color="blue")
    plt.title("6 steps towards the local minimum")
    plt.xlim((0.25, 1.75))
    plt.ylim((-4.5, 0))
    for i in range(6):
        plt.plot([steps[i]], [steps[i]**4 - 4*steps[i]**2], "o", color = "red")
        plt.text(steps[i]-0.02, steps[i]**4 - 4*steps[i]**2-0.25, i, color = "black")
    plt.show()

def draw_example_steps_3d(steps):
    x, y = np.meshgrid(np.linspace(0.25, 1.75, 25), np.linspace(0.25, 1.6, 25))
    z = x**4 - 4*x**2 + y**4 - 2*y**2
    
    ax = plt.figure().gca(projection="3d")
    ax.plot_wireframe(x, y, z, alpha = 0.6)
    plt.title("10 steps towards the local minimum")
    
    for i in range(10):
        ax.plot([steps[i][0]], steps[i][1], [steps[i][0]**4 - 4*steps[i][0]**2 + steps[i][1]**4 - 2*steps[i][1]**2], 'o', color="red")
        ax.text(steps[i][0], steps[i][1], steps[i][0]**4 - 4*steps[i][0]**2 + steps[i][1]**4 - 2*steps[i][1]**2 + 0.25, i, color = "black")
    
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim((0.25, 1.75))
    ax.set_ylim((0.25, 1.6))
    plt.show()

def example_gradient_descent_2d():
    x  = 0.5
    lr = 0.1
    steps = []

    for i in range(100):
        steps.append(x)
        print(f"Iteration {i+1:>3}: x = {x}")
        x -= lr * (4*x**3 - 8*x)

    draw_example_steps_2d(steps)

def example_gradient_descent_3d():
    x  = 0.5
    y  = 1.5
    lr = 0.03
    steps = []

    for i in range(100):
        steps.append((x, y))
        print(f"Iteration {i+1:>3}: x = {x} / y = {y}")
        x -= lr * (4*x**3 - 8*x)
        y -= lr * (4*y**3 - 4*y)

    draw_example_steps_3d(steps)

def add_column_with_ones(x):
    if len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0], 1))
    return np.hstack((np.ones((x.shape[0], 1)).astype(int), x))

def calc_metrics(cobf, y, degree):
    text = f"{mae(cobf, y):.5}\n{rmse(cobf, y):.5}\n{r_squared(cobf, y):.5}\n{r_squared_adj(cobf, y, degree):.5}"
    return text

def gradient_descent_polynomial(degree = 1, lr = 0.001, iterations = 100):
    x = np.linspace(-2, 2, 100)
    y = x**5 - 6 * x**3 + 6 * x + 4 * np.cos(x*6) + 8

    x_extended = add_column_with_ones(x)
    
    for i in range(2, degree+1):
        x_extended = np.c_[x_extended, x**i]

    w = np.random.rand(degree+1) - 0.5    
    for _ in range(iterations):
        gradients = x_extended.T.dot(x_extended.dot(w) - y)    
        w = w - lr * gradients
   
    yhat = x_extended.dot(w)

    plt.text(1.07, 15, "MAE:\nRMSE:\nR2:\nR2-adj:              ", bbox={'facecolor': 'grey', 'alpha': 0.1, 'pad': 5})
    plt.text(1.53, 15, calc_metrics(yhat, y, degree))
    plt.ylim((-4, 20))
    plt.title(f"Gradient Descent Polynomial with degree {degree}", fontsize=9)
    plt.plot(x, y)
    plt.plot(x, yhat)
    plt.show()

def mae(yhat, y):
    return (np.absolute(yhat - y)).mean()

def rmse(yhat, y):
    diff = yhat - y
    return diff.dot(diff)**0.5

def r_squared(yhat, y):
    diff_hat = yhat - y
    diff_bar = y.mean() - y
    return 1 - (diff_hat.dot(diff_hat) / diff_bar.dot(diff_bar))

def r_squared_adj(yhat, y, degree):
    n = 100
    p = degree
    return 1 - ((1 - r_squared(yhat, y)) * (n - 1)) / (n - p - 1)

draw_curve()
draw_plane()

example_gradient_descent_2d()
example_gradient_descent_3d()

gradient_descent_polynomial(3)