import numpy as np
import matplotlib.pyplot as plt

def add_column_with_ones(x):
    if len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0], 1))
    return np.hstack((np.ones((x.shape[0], 1)).astype(int), x))

def create_2d_data_points():
    x = np.linspace(-2, 2, 100)
    y = x**5 - 6 * x**3 + 6 * x + 4 * np.cos(x*6) + 8

    return x, y

def line_of_best_fit(x, y):
    x_extended = add_column_with_ones(x)
    x_trans = x_extended.transpose()
    w = np.linalg.solve(x_trans.dot(x_extended), x_trans.dot(y))
    lobf = w[1] * x + w[0]
    return lobf

def print_poly_lobf_formula(w):
    degree = len(w)
    print(f"\nLine of best fit with degree {degree-1}:\ny = ", end = "")
    
    for i in range(degree):
        if round(w[i], 3) != 0:
            print(f"{w[i]:.3f}", end = "")
            if i >= 1:
                print("x", end = "")
            if i >= 2:
                print(f"^{i}", end = "")
            if i < degree - 1:
                print(" + ", end = "")
    print()

def poly_line_of_best_fit(x, y, degree):
    degree += 1
    x_extended = add_column_with_ones(x)
    
    for i in range(2, degree):
        x_extended = np.c_[x_extended, x**i]
        
    x_trans = x_extended.transpose()
    w = np.linalg.solve(x_trans.dot(x_extended), x_trans.dot(y))
    print_poly_lobf_formula(w)
    
    lobf = np.zeros(x.shape[0])
    for i in range(degree):
        lobf += w[i] * x**i

    return lobf

x, y = create_2d_data_points()
lobf = line_of_best_fit(x, y)

lobf = poly_line_of_best_fit(x, y, 14)
   
plt.plot(x, y)
plt.plot(x, lobf)
plt.show()