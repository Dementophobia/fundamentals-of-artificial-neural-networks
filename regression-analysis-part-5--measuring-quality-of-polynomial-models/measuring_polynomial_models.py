import numpy as np
import matplotlib.pyplot as plt

def create_2d_data_points():
    x = np.linspace(-2, 2, 100)
    y = x**5 - 6 * x**3 + 6 * x + 4 * np.cos(x*6) + 8
    return x, y

def print_poly_cobf_formula(x, y, degree):
    degree += 1
    x_extended = add_column_with_ones(x)
    
    for i in range(2, degree):
        x_extended = np.c_[x_extended, x**i]
        
    x_trans = x_extended.transpose()
    w = np.linalg.solve(x_trans.dot(x_extended), x_trans.dot(y))

    degree = len(w)
    print(f"\nCurve of best fit with degree {degree-1}:\ny = ", end = "")
    
    for i in range(degree):
        print(f"{w[i]:.3f}", end = "")
        if i >= 1:
            print("x", end = "")
        if i >= 2:
            print(f"^{i}", end = "")
        if i < degree - 1:
                print(" + ", end = "")
    print()

def add_column_with_ones(x):
    if len(x.shape) == 1:
        x = np.reshape(x, (x.shape[0], 1))
    return np.hstack((np.ones((x.shape[0], 1)).astype(int), x))

def poly_curve_of_best_fit(x, y, degree):
    degree += 1
    x_extended = add_column_with_ones(x)
    
    for i in range(2, degree):
        x_extended = np.c_[x_extended, x**i]
        
    x_trans = x_extended.transpose()
    w = np.linalg.solve(x_trans.dot(x_extended), x_trans.dot(y))
    
    return sum([w[i]*x_extended[:,i] for i in range(x_extended.shape[1])])

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

def plot_metric(x, y, method):
    values = np.empty((14,))
    
    for degree in range(1, 15):
        cobf = poly_curve_of_best_fit(x, y, degree)
        if method == "MAE":
            values[degree-1] = (mae(cobf, y))
        elif method == "RMSE":
            values[degree-1] = (rmse(cobf, y))
        elif method == "R squared":
            values[degree-1] = (r_squared(cobf, y))
        elif method == "Adjusted R squared":
            values[degree-1] = (r_squared_adj(cobf, y, degree))
    
    plt.title(f"Method: {method}")
    plt.plot(np.linspace(1, 14, 14), values)
    plt.show()

def plot_all_metrics(x, y):
    values = [np.empty((14,)) for _ in range(4)]
    label_text = ["MAE", "RMSE", "R2", "Adjusted R2"]
    
    for degree in range(1, 15):
        cobf = poly_curve_of_best_fit(x, y, degree)
        values[0][degree-1] = (mae(cobf, y))
        values[1][degree-1] = (rmse(cobf, y))
        values[2][degree-1] = (r_squared(cobf, y))
        values[3][degree-1] = (r_squared_adj(cobf, y, degree))
    
    for i in range(2):
        values[i] -= 2 * (values[i] - values[i].mean())
        values[i] /= max(values[i])
    
    for i in range(4):
        plt.plot(np.linspace(1, 14, 14), values[i], label=label_text[i])

    plt.legend()
    plt.show()    

def plot_data_points(x, y):
    plt.plot(x, y)
    plt.ylim((-4, 20))
    plt.show()

def plot_all_curves(x, y):
    curves = [poly_curve_of_best_fit(x, y, degree) for degree in range(1, 15)]
    plt.plot(x, y)
    for cobf in curves:
        plt.plot(x, cobf)
    plt.ylim((-4, 20))
    plt.show()

def plot_curve_with_metrics(x, y, degree):
    cobf = poly_curve_of_best_fit(x, y, degree)
    plt.text(1.07, 15, "MAE:\nRMSE:\nR2:\nR2-adj:              ", bbox={'facecolor': 'grey', 'alpha': 0.1, 'pad': 5})
    plt.text(1.53, 15, calc_metrics(cobf, y, degree))
    plt.text(-0.75, 20.5, f"Polynomial with degree {degree}")
    plt.plot(x, y)
    plt.plot(x, cobf)
    plt.ylim((-4, 20))
    plt.show()  

def calc_metrics(cobf, y, degree):
    text = f"{mae(cobf, y):.5}\n{rmse(cobf, y):.5}\n{r_squared(cobf, y):.5}\n{r_squared_adj(cobf, y, degree):.5}"
    return text

def print_metrics(x, y):
    for degree in range(1, 15):
        print("*********************************")
        print(f"Calculating model with degree {degree}.\n")
        cobf = poly_curve_of_best_fit(x, y, degree)

        print(f"MAE:\t{mae(cobf, y):.5}")
        print(f"RMSE:\t{rmse(cobf, y):.5}")
        print(f"R2:\t{r_squared(cobf, y):.5}")
        print(f"R2-adj:\t{r_squared_adj(cobf, y, degree):.5}\n")

x, y = create_2d_data_points()

print_metrics(x, y)

plot_data_points(x, y)
plot_all_curves(x, y)
for degree in range(1, 15):
    plot_curve_with_metrics(x, y, degree)

print_poly_cobf_formula(x, y, 6)
print_poly_cobf_formula(x, y, 7)

plot_metric(x, y, "MAE")
plot_metric(x, y, "RMSE")
plot_metric(x, y, "R squared")
plot_metric(x, y, "Adjusted R squared")

plot_all_metrics(x, y)