import numpy as np
import matplotlib.pyplot as plt


# x range [0 : 1] ecc
def function(char, x):
    if char == "A":
        return 2 * x
    elif char == "B":
        return 2 * x - 10 * x**5 + 15 * x**10
    
    
def empirical_function(char, x, sigma):
    if char == "A":
        return function("A", x) + np.random.normal(0, sigma)
    elif char == "B":
        return function("B", x) + np.random.normal(0, sigma)


def simulate_data(char, n, sigma, x_max=1):
    x = np.random.uniform(0, x_max, n)
    y = [empirical_function(char, x_i, sigma) for x_i in x]
    return np.array(x), np.array(y)
    
    
def fit_data(x, y, degree):
    return np.polyfit(x, y, degree)
    
   
def plot_data_and_fit(x, y, coeffs, char, message, title, x_max):
    plt.scatter(x, y, label='Data', color='black', s=2, alpha=1.0)
    x_plot = np.linspace(0, x_max, 500)
    plt.plot(x_plot, function(char, x_plot), label='True function', color='red', linestyle='dashed')
    colors = ["blue", "yellow", "green"]
    for i, (degree, c) in enumerate(coeffs.items()):
        color = colors[i % len(colors)]
        plt.plot(x_plot, np.polyval(c, x_plot), label=f'fit (degree {degree})', color=color)
        
    if x_max > 1:
        plt.axvline(
            x=1.0,
            color='gray',
            linestyle=':',
            label='training boundary'
            )
    
    plt.title(message)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(title + '.png')
    plt.show()
       
 
def simulation_single_func(char, n, sigma, x_max=1):
    x, y = simulate_data(char, n, sigma, x_max)
    coeffs = {}
    for degree in [1, 3, 10]:
        coeffs[degree] = fit_data(x, y, degree)
    plot_data_and_fit(x, y, coeffs, char, f'{n} pairs of training data and fit for function {char}', f'training_{char}_{n}', x_max)
    
    return coeffs
    
 
def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)
 
def simulation_single_func_with_test(char, n_train, n_test, sigma, x_max_train=1, x_max_test=1.25):
    coeffs = simulation_single_func(char, n_train, sigma, x_max_train)
    x_test, y_test = simulate_data(char, n_test, sigma, x_max_test)
    plot_data_and_fit(x_test, y_test, coeffs, char, f'{n_test} pairs of test data and fit for function {char}', f'test_{char}_{n_test}_with_train_{n_train}', x_max_test)
    #evaluate mean squared error test (for the y test values, which have the gaussian error) and mse true function for the true values of the function (without the gaussian error)
    
    mse_test = {}
    mse_true_func = {}
    mse_test_in = {}
    mse_test_out = {}
    mse_true_func_in = {}
    mse_true_func_out = {}
    
    mask_in = x_test <= x_max_train
    mask_out = x_test > x_max_train
    
    for degree in [1, 3, 10]:
        y_pred = np.polyval(coeffs[degree], x_test)
        y_true_func = function(char, x_test)
        mse_test[degree] = mean_squared_error(y_test, y_pred)
        mse_true_func[degree] = mean_squared_error(y_true_func, y_pred)
        mse_test_in[degree] = mean_squared_error(y_test[mask_in], y_pred[mask_in])
        mse_test_out[degree] = mean_squared_error(y_test[mask_out], y_pred[mask_out])
        mse_true_func_in[degree] = mean_squared_error(y_true_func[mask_in], y_pred[mask_in])
        mse_true_func_out[degree] = mean_squared_error(y_true_func[mask_out], y_pred[mask_out])
        
    mse_test_clean = {degree: float(value) for degree, value in mse_test.items()} #just to make output more readable (otherwise it will be numpy.float64)
    mse_true_func_clean = {degree: float(value) for degree, value in mse_true_func.items()}
    mse_test_in_clean = {degree: float(value) for degree, value in mse_test_in.items()}
    mse_test_out_clean = {degree: float(value) for degree, value in mse_test_out.items()}
    mse_true_func_in_clean = {degree: float(value) for degree, value in mse_true_func_in.items()}
    mse_true_func_out_clean = {degree: float(value) for degree, value in mse_true_func_out.items()}

    line1 = f'MSE_test for func {char}: {mse_test_clean}'
    line2 = f'MSE_true_func for func {char}: {mse_true_func_clean}'
    line3 = f'MSE_test_in for func {char}: {mse_test_in_clean}'
    line4 = f'MSE_test_out for func {char}: {mse_test_out_clean}'
    line5 = f'MSE_true_func_in for func {char}: {mse_true_func_in_clean}'
    line6 = f'MSE_true_func_out for func {char}: {mse_true_func_out_clean}'

    print(line1)
    print(line2)
    print(line3)
    print(line4)
    print(line5)
    print(line6)

    # Write results to file
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(f'{line1}\n')
        f.write(f'{line2}\n')
        f.write(f'{line3}\n')
        f.write(f'{line4}\n')
        f.write(f'{line5}\n')
        f.write(f'{line6}\n')

    return coeffs, mse_test, mse_true_func


RESULTS_FILE = "simulation_results.txt"

# create/overwrite the file once at the start
with open(RESULTS_FILE, "w", encoding="utf-8") as f:
    f.write("")
    
np.random.seed(12345)  # Set a random seed for reproducibility

#Let's run the simulations for both functions A and B
simulation_single_func_with_test("A", 10, 20, 0.0)
simulation_single_func_with_test("B", 10, 20, 0.0)
simulation_single_func_with_test("A", 100, 20, 1.0)
simulation_single_func_with_test("B", 100, 20, 1.0)
simulation_single_func_with_test("A", 10000, 100, 1.0)
simulation_single_func_with_test("B", 10000, 100, 1.0)
