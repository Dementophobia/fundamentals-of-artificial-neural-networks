import numpy as np

def add_column_with_ones(x):
    return np.hstack((np.ones((x.shape[0], 1)).astype(int), x))

def generate_water_stats():
    sport = np.random.randint(200, size=100)
    temp  = np.random.randint(35,  size=100)
    
    x = np.stack((sport, temp)).transpose()
    y = sport/60 + temp/10 + 1 + np.random.randn(100)/2
    
    return x, y

def calculate_yhat(x, y):
    x_trans = x.transpose()
    w = np.linalg.solve(x_trans.dot(x), x_trans.dot(y))
    return sum([w[i]*x[:,i] for i in range(x.shape[1])])

def add_noise(x):
    noise = np.random.rand(x.shape[0],1)*10
    return np.hstack((x, noise))

def mae(yhat, y):
    return (np.absolute(yhat - y)).mean()

def rmse(yhat, y):
    diff = yhat - y
    return diff.dot(diff)**0.5

def r_squared(yhat, y):
    diff_hat = yhat - y
    diff_bar = y.mean() - y
    return 1 - (diff_hat.dot(diff_hat) / diff_bar.dot(diff_bar))

def r_squared_adj(yhat, y, x):
    n = x.shape[0]
    p = x.shape[1] - 1
    return 1 - ((1 - r_squared(yhat, y)) * (n - 1)) / (n - p - 1)
    
def print_r_squared(x, y, iterations):
    for i in range(iterations):
        yhat = calculate_yhat(x, y)
        print(f"R squared with {i} noise variables added: {r_squared(yhat, y):.5}")
        x = add_noise(x)

def print_r_squared_and_adj(x, y, iterations):
    for i in range(iterations):
        yhat = calculate_yhat(x, y)
        print(f"Noise variables: {i}\t", end="")
        print(f"R2: {r_squared(yhat, y):.5}\tR2 adj: {r_squared_adj(yhat, y, x):.5}")
        x = add_noise(x)

def print_all(x, y, iterations):
    for i in range(iterations):
        yhat = calculate_yhat(x, y)
        print(f"Noise variables: {i}\t", end="")
        print(f"R2: {r_squared(yhat, y):.5}\tR2 adj: {r_squared_adj(yhat, y, x):.5}\tMAE: {mae(yhat, y):.5}\tRMSE: {rmse(yhat, y):.5}")
        x = add_noise(x)

x, y = generate_water_stats()
x = add_column_with_ones(x)

print_r_squared(x, y, 10)
print()
print_r_squared_and_adj(x, y, 10)
print()
print_all(x, y, 10)