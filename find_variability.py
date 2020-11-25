###################################################
# Find Stars Variability                          #
# Matheus J. Castro                               #
# Version 1.2.1                                   #
# Last Modification: 11/25/2020 (month/day/year)  #
###################################################

import matplotlib.pyplot as plt
import numpy as np
import os


def find_files(dir_name):
    # Find all files inside the dir_name and create a list with these files
    # with the dir_name before of each file
    files = os.listdir(dir_name)  # search for files
    if dir_name[-1] != "/":  # verify the separator character
        separator = "/"
    else:
        separator = ""

    for i in range(len(files)):
        files[i] = dir_name + separator + files[i]  # concatenate the dir_name with file names

    return files


def open_data(fl_name):
    # Open a specific file with the format needed
    data = np.genfromtxt(fl_name, delimiter=",", dtype=None, encoding="ASCII")
    file = open(fl_name).readlines()[1:17]
    param = {}
    for i in file:
        line = i[1:-1].split(",")
        try:
            param[line[0]] = [line[1], line[2]]
        except IndexError:
            param[line[0]] = line[1]

    return param, data


def format_data(data):
    # Format data to separate data for each filter
    fdata = {}
    filter_list = [data[0][3]]
    filter_verify = data[0][3]
    for i in data:  # create a list with the filters found
        if filter_verify != i[3]:
            filter_verify = i[3]
            filter_list.append(i[3])

    for i in filter_list:  # append data for each filter
        fdata[i] = [[], [], []]
        for j in data:
            if i == j[3]:
                fdata[i][0].append(j[0])
                fdata[i][1].append(j[1])
                fdata[i][2].append(j[2])

    return fdata


def plot_data(data, show=False):
    # Plot data
    plt.figure(figsize=(21, 9))
    plt.title("Change of MAG on each filter")
    plt.xlabel("Time")
    plt.ylabel("MAG")

    for i in data.keys():  # plot data for each filter
        plt.plot(data[i][0], data[i][1], ".-", label=i)
        # plt.errorbar(data_plot[i][0], data_plot[i][1], yerr=data_plot[i][2], fmt=".", label=i)

    plt.legend()
    plt.grid()
    plt.savefig("Change of MAG on each filter")
    if show:
        plt.show()
    plt.close()


def save_results(results, dir_name, fl_name):
    if not os.path.isdir("results_" + dir_name):
        os.mkdir("results_" + dir_name)
    fl_name = "results_" + fl_name

    save = "Method, Filter, Result\n"

    for method in results.keys():
        result_meth = np.array(list(results[method].items()))

        for i in range(len(result_meth)):
            save += "{}, {}, {}\n".format(method, result_meth[i][0], result_meth[i][1])

    np.savetxt(fl_name, [save], fmt="%s")


def iqr_method(data, weight=None):
    # Find the IQR value for each filter of an file opened
    if weight is None:
        weight = {}
        for i in data.keys():  # create a sequence of lists with ones (null element) if any weight is given
            weight[i] = np.ones(len(data[i][1]))

    iqr = {}
    for i in data.keys():
        data_len = len(data[i][1])
        if data_len != 1:
            # Exclude the 25% lower and highest values
            data_of_interest = np.sort(data[i][1]*weight[i])[data_len//4:-data_len//4]
        else:
            data_of_interest = data[i][1]

        data_len = len(data_of_interest)
        if data_len != 1:
            # Difference between the median of the upper and lower halves
            iqr[i] = np.median(data_of_interest[data_len//2:]) - np.median(data_of_interest[:data_len//2])
        else:
            iqr[i] = 0

    return iqr


def eta_method(data, weight=None):
    # Find the IQR value for each filter of an file opened
    if weight is None:
        weight = {}
        for i in data.keys():  # create a sequence of lists with ones (null element) if any weight is given
            weight[i] = np.ones(len(data[i][1]))

    eta = {}
    for i in data.keys():
        data_len = len(data[i][1])-1

        if data_len == 0:
            eta[i] = np.nan
            continue

        mean_square = 0
        wsum = 0
        for j in range(data_len):  # find mean square successive difference
            # Weights associated with the magnitude errors
            wmagj = 1 / (data[i][2][j + 1] ** 2 + data[i][2][j] ** 2)

            mean_square += (weight[i][j + 1] * wmagj) * (data[i][1][j + 1] - data[i][1][j]) ** 2
            # Sum of the weights
            wsum += weight[i][j + 1] * wmagj
        mean_square = (mean_square / wsum) / data_len

        var = 0
        wsum2 = 0
        for j in range(data_len+1):  # find variance manually
            # Weights associated with the magnitude error
            wmagj2 = 1. / (data[i][2][j] ** 2)
            var += wmagj2*(data[i][1][j] - np.mean(data[i][1]))**2
            wsum2 += wmagj2
        var = (var / wsum2) / data_len

        if mean_square == 0:
            eta[i] = np.nan
        else:
            eta[i] = var/mean_square

    return eta


def sigmaw_method(data, weights):
    sigmaw = {}
    for i in data.keys():
        wsum = np.sum(weights[i])
        wsum_squared = np.sum(weights[i] ** 2)
        mean = np.mean(data[i][1])

        sum_result = np.sum(weights[i] * (data[i][1] - mean) ** 2)

        if (wsum ** 2 - wsum_squared) == 0:
            sigmaw[i] = np.nan
        else:
            sigmaw[i] = np.sqrt((wsum / (wsum ** 2 - wsum_squared)) * sum_result)

    return sigmaw


def mad_method(data):
    gauss_factor = 1.4826
    mad = {}
    for i in data.keys():
        mad[i] = gauss_factor*np.median(np.abs(data[i][1] - np.median(data[i][1])))

    return mad


def excess_variance_method(data):
    var_nxs = {}
    for i in data.keys():
        overall_mean = np.mean(data[i][1])

        if overall_mean == 0:
            var_nxs[i] = np.nan
        else:
            sum_result = np.sum((data[i][1] - overall_mean) ** 2 - np.power(data[i][2], 2))
            var_nxs[i] = (1 / (len(data[i][1]) * overall_mean**2)) * sum_result

    return var_nxs


def peak_to_peak_method(data):
    nu = {}
    for i in data.keys():
        max_diff = max(np.array(data[i][1]) - np.array(data[i][2]))
        min_diff = min(np.array(data[i][1]) + np.array(data[i][2]))

        if (max_diff + min_diff) == 0:
            nu[i] = np.nan
        else:
            nu[i] = (max_diff - min_diff) / (max_diff + min_diff)

    return nu


def lag_1_method(data):
    lag = {}
    for i in data.keys():
        overall_mean = np.mean(data[i][1])

        sum_result = 0
        for j in range(len(data[i][1]) - 1):
            sum_result += (data[i][1][j] - overall_mean) * (data[i][1][j+1] - overall_mean)

        if np.sum(data[i][1] - overall_mean) == 0:
            lag[i] = np.nan
        else:
            lag[i] = sum_result / np.sum((data[i][1] - overall_mean)**2)

    return lag


def error_weights(data):
    # Create a sequence of lists using the data magnitude error
    weights = {}
    for i in data.keys():
        weights[i] = 1/np.array(data[i][2]) ** 2

    return weights


def time_weights(data, threshold=30):  # threshold defined in minutes
    # Find time weights for the data based on when the image was taken
    threshold = threshold/1440  # transform to day

    weights = {}
    for i in data.keys():
        filter_weights = [1]
        for j in range(len(data[i][0])-1):
            diff = data[i][0][j+1] - data[i][0][j]
            if diff <= threshold:
                filter_weights.append(1)
            else:
                diff = diff*24
                filter_weights.append(1/(1+diff))
        weights[i] = np.array(filter_weights)

    return weights


def main(dir_name):
    # Main function
    fls_names = find_files(dir_name)

    methods = ["IQR", "IQR Weighted", "ETA", "ETA Weighted", "Sigma Weighted", "MAD",
               "Variance NXS", "Peak to Peak", "Lag-1 Autocorrelation"]
    results = {}

    for fl_name in fls_names:
        param, data = open_data(fl_name)
        data = format_data(data)

        results[methods[0]] = iqr_method(data)
        results[methods[1]] = iqr_method(data, weight=error_weights(data))
        results[methods[2]] = eta_method(data)
        results[methods[3]] = eta_method(data, weight=time_weights(data))
        results[methods[4]] = sigmaw_method(data, error_weights(data))
        results[methods[5]] = mad_method(data)
        results[methods[6]] = excess_variance_method(data)
        results[methods[7]] = peak_to_peak_method(data)
        results[methods[8]] = lag_1_method(data)

        save_results(results, dir_name, fl_name)


if __name__ == '__main__':
    direc_name = "202006_Matheus"
    main(direc_name)
