import sys
import os

import numpy as np
import pandas as pd
import scipy.optimize
import matplotlib.pyplot as plt

# TODO: add config parser
# temp input values, as long as no configparser is implemented
input_data_filename = "example_data/PMMA_kompakt_40K.csv"
input_label_temperature = '# Temp_mean_R'
input_label_target_data = ' reacr_mean'
input_data_line_offset = 300

input_number_base_functions = 5
input_number_repetitions = 10

input_delta_T = 40 / 60

input_graphical_output_type = 'pdf'
input_graphical_output_directory = 'plots'

data = pd.read_csv(input_data_filename, header=0)

input_temperature = data[input_label_temperature].values[input_data_line_offset:]
input_target_data = data[input_label_target_data].values[input_data_line_offset:]

# number of parameters needed for each base function
_n_params = 4


# function of a single base function
def base_function(T, gauss_parameters):
    # check for correct length of parameters
    if len(gauss_parameters) != _n_params:
        print("ERROR: wrong number parameters passed to base function")
        sys.exit(1)

    # split up argument array into individual values
    # T0 : center temperature of gauss function
    # A : amplitude
    # dl : width on the left
    # dr : width on the right
    T0, A, dl, dr = gauss_parameters

    # create return array
    y = np.zeros_like(T)

    # left side (w.r.t. to x0) of the function
    y[T < T0] = A * np.exp(- (T[T < T0] - T0) ** 2 / dl ** 2)

    # right side
    y[T >= T0] = A * np.exp(- (T[T >= T0] - T0) ** 2 / dr ** 2)

    return y


# cost function for the optimisation
# p : parameter set containing the values for all involved gauss functions
def cost_function(parameters, temperature, target_data):
    if len(parameters) % _n_params != 0:
        print("ERROR: argument array has not a matching length")
        sys.exit(1)

    delta = np.copy(target_data)
    for i in range(len(parameters) // _n_params):
        delta -= base_function(temperature, parameters[_n_params * i:_n_params * (i + 1)])

    rmse = np.sum(delta ** 2) / len(temperature)

    value_penalty = 0
    penalty_factor = 10 * rmse
    for i in range(len(parameters) // _n_params):
        t0, A, dl, dr = parameters[_n_params * i:_n_params * (i + 1)]
        if A < 0:
            value_penalty += 100 * penalty_factor * np.abs(A)
        if dl < 0:
            value_penalty += penalty_factor * np.abs(dl)
        if dr < 0:
            value_penalty += penalty_factor * np.abs(dr)
        dl_dr_ratio = 2
        if (dl / dr) < dl_dr_ratio:
            value_penalty += penalty_factor * np.abs(dr / dl)
        if (dr / dl) < dl_dr_ratio:
            value_penalty += penalty_factor * np.abs(dl / dr)

    return rmse + value_penalty


# normalise data for better optimisation
T_min = np.min(input_temperature)
T_max = np.max(input_temperature)
delta_T = T_max - T_min
normed_temperature = (input_temperature - T_min) / delta_T

d_min = np.min(input_target_data)
d_max = np.max(input_target_data)
delta_d = d_max - d_min
normed_target_data = (input_target_data - d_min) / delta_d

# will hold the best parameter set
global_best_parameter_set_n = []
global_best_parameter_set = []
global_best_parameter_set_fun = []
# compute the optimal representation for all numbers of base functions
for number_base_functions in range(1, input_number_base_functions + 1):

    best_parameter_set = None
    best_parameter_set_fun = None

    # random repetitions for preventing bad initial values / local minimum
    for repetition in range(input_number_repetitions):

        parameter_set = []
        for i in range(number_base_functions):
            rnum = list(np.random.random(4))
            r_T0 = rnum[0]
            r_A = rnum[1]
            r_dl = rnum[2]
            r_dr = rnum[3]
            parameter_set += [r_T0, r_A, r_dl, r_dr]

        res = scipy.optimize.minimize(cost_function,
                                      np.array(parameter_set),
                                      method='BFGS',
                                      args=(normed_temperature, normed_target_data),
                                      options={'gtol': 1e-7, 'maxiter': 1000, 'eps': 1e-8},
                                      tol=1e-6)

        if best_parameter_set_fun is not None:
            if best_parameter_set_fun > res.fun:
                best_parameter_set_fun = res.fun
                best_parameter_set = res.x
        else:
            best_parameter_set_fun = res.fun
            best_parameter_set = res.x

    # undo normalization to parameter set
    renormalized_parameter_set = []
    for ips in range(len(best_parameter_set) // _n_params):
        n_T0, n_A, n_dl, n_dr = best_parameter_set[_n_params * ips:_n_params * (ips + 1)]

        T0 = n_T0 * delta_T + T_min
        A = n_A * delta_d + d_min
        dl = n_dl * delta_T
        dr = n_dr * delta_T

        renormalized_parameter_set.append([T0, A, dl, dr])

    print(renormalized_parameter_set)
    # append local best parameter set to global data structure
    global_best_parameter_set_n.append(number_base_functions)
    global_best_parameter_set_fun.append(best_parameter_set_fun)
    global_best_parameter_set.append(renormalized_parameter_set)


def plot_comparison(n, parameter_set, fun, temperature, target_data):
    plt.plot(temperature, target_data, 'o', label='experimental data', color='C0')

    y = np.zeros_like(temperature)

    for ips in range(len(parameter_set)):
        cy = base_function(temperature, parameter_set[ips])
        plt.plot(temperature, cy, label='reaction {}'.format(ips), color='Gray')
        y += cy

    plt.plot(temperature, y, label='global approximation', color='C1')

    plt.title('RMSE = {:8.4e}'.format(fun))
    plt.legend()
    plt.grid()

    if not os.path.isdir(input_graphical_output_directory):
        os.mkdir(input_graphical_output_directory)
    out_file_name = 'gaussian_fitting_n{:02d}.{}'.format(n, input_graphical_output_type)
    out_path = os.path.join(input_graphical_output_directory, out_file_name)
    plt.savefig(out_path)
    plt.clf()


def print_results(n, parameter_set, fun, temperature):
    A_total = 0
    Ar = []
    for ips in range(n):
        cy = base_function(temperature, parameter_set[ips])
        cA = np.sum(cy)
        Ar.append(cA)
        A_total += cA

    print("")
    print("##### Results for {} Gaussian functions".format(n))
    print("Cost function value: {:8.4e}".format(fun))
    for i in range(n):
        T0, A, dl, dr = parameter_set[i]

        # FDS user guide, equation 11.9, where Y_s(0) is a relative fraction
        R = 8.314
        AE = np.e * A * R * T0**2 / input_delta_T
        AA = np.e * A * np.exp(AE / (R*T0))


        print("## Parameter of Gaussian function number {}".format(i+1))
        print("(Temperature)   T0 = {:8.4e}".format(T0))
        print("(Amplitude)     A  = {:8.4e}".format(A))
        print("(left width)    dl = {:8.4e}".format(dl))
        print("(right width)   dr = {:8.4e}".format(dr))
        print("(Area fraction) Af = {:.2f}".format(Ar[i] / A_total))
        print("(Arrhenius E)   E  = {:8.4e}".format(AE))
        print("(Arrhenius A)   A  = {:8.4e}".format(AA))

for i in range(len(global_best_parameter_set)):
    plot_comparison(global_best_parameter_set_n[i],
                    global_best_parameter_set[i],
                    global_best_parameter_set_fun[i],
                    input_temperature,
                    input_target_data)
    print_results(global_best_parameter_set_n[i],
                  global_best_parameter_set[i],
                  global_best_parameter_set_fun[i],
                  input_temperature)
