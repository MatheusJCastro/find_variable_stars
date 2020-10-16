###################################################
# Find Stars Variability                          #
# Matheus J. Castro                               #
# Version 1.0                                     #
# Last Modification: 10/15/2020 (month/day/year)  #
###################################################

import matplotlib.pyplot as plt
import numpy as np
import sys
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

    return dir_name + separator, files


def open_fl(file):
    fl = np.loadtxt(file, delimiter=",", dtype=np.str)

    data = {}
    for i in fl:
        if i[3] not in data:
            data[i[3]] = np.asarray([list(map(float, i[:3]))])
        else:
            data[i[3]] = np.append(data[i[3]], np.asarray([list(map(float, i[:3]))]), axis=0)

    return data


def plot_3d(data, fl_name, show=False):
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlabel('Time (HDJ)')
    ax.set_zlabel('DMAG')
    ax.set_ylabel('Filter')
    ax.set_yticklabels([])

    time_values = np.array([])
    dmag_values = np.array([])
    for i in data.keys():
        time_values = np.append(time_values, data[i].T[0], axis=0)
        dmag_values = np.append(dmag_values, data[i].T[1], axis=0)

    ax.set_xlim(min(time_values), max(time_values))
    ax.set_zlim(min(dmag_values), max(dmag_values))

    count = 0
    for i in data.keys():
        ax.plot(data[i].T[0], data[i].T[1], zs=count, zdir="y", label=i, marker=".")
        count += 1

    plt.legend()
    plt.savefig("{}.jpg".format(fl_name))
    if show:
        plt.show()
    plt.close()


def create_dir(dir_name, fl_name):
    if not os.path.isdir("plots_" + dir_name):
        os.mkdir("plots_" + dir_name)
    return "plots_" + fl_name[:-4]


def find_args(args):
    show = False
    direc = "202006_Matheus/"

    help_msg = "\n\t\t\033[1;31mHelp Section\033[m\nplot_3D.py v1.0\n" \
               "Usage: python3 plot_3D.py [options] \n" \
               "\033[3mWritten by Matheus J. Castro <matheusj_castro@usp.br>\033[m\n\n" \
               "Options are:\n -h,  -help\t\t|\tShow this help;\n--h, --help\t\t|\tShow this help;\n" \
               "\t -dir [argument]|\tDefine an specific directory to read.\n" \
               "\t\t\t|\tDefault is \"202006_Mateheus/\";\n" \
               "\t -show\t\t|\tDefine to show the plots at matplotlib\n" \
               "\t\t\t|\tstructure. Default is false;\n"

    for i in range(len(args)):
        if args[i] == "-show":
            show = True
        elif args[i] == "-dir":
            direc = args[i + 1]
        elif args[i] == "-h" or args[i] == "-help" or args[i] == "--h" or args[i] == "--help":
            sys.exit(help_msg)

    return direc, show


def main(args):
    direc, show = find_args(args)
    direc, fls_names = find_files(direc)

    for fl_name in fls_names:
        data = open_fl(fl_name)
        fl_name = create_dir(direc, fl_name)
        plot_3d(data, fl_name, show=show)


if __name__ == '__main__':
    arg = sys.argv[1:]
    main(arg)
