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

    plt.rc('axes', titlesize=24)

    ax.set_title("Plot of {}\n".format(fl_name.split("/")[1]))
    ax.set_xlabel("Time (HDJ)")
    ax.set_zlabel("DMAG")
    ax.set_ylabel("Filter")
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


def plot_2d(data, fl_name, show=False):
    plt.figure(figsize=(16, 9))

    plt.title("Plot of {}\n".format(fl_name.split("/")[1]))
    plt.xlabel("Time (HDJ)")
    plt.ylabel("DMAG")

    for i in data.keys():
        plt.plot(data[i].T[0], data[i].T[1], ".", label=i)

    plt.grid()
    plt.legend()
    plt.savefig("{}.jpg".format(fl_name))
    if show:
        plt.show()
    plt.close()


def create_dir_3d(dir_name, fl_name):
    if not os.path.isdir("plots3d_" + dir_name):
        os.mkdir("plots3d_" + dir_name)
    return "plots3d_" + fl_name[:-4]


def create_dir_2d(dir_name, fl_name):
    if not os.path.isdir("plots2d_" + dir_name):
        os.mkdir("plots2d_" + dir_name)
    return "plots2d_" + fl_name[:-4]


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

        fl_name3d = create_dir_3d(direc, fl_name)
        plot_3d(data, fl_name3d, show=show)

        fl_name2d = create_dir_2d(direc, fl_name)
        plot_2d(data, fl_name2d, show=show)


if __name__ == '__main__':
    arg = sys.argv[1:]
    main(arg)
