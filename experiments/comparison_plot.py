import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    def convert(text): int(text) if text.isdigit() else text

    def alphanum_key(key): [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(l, key=alphanum_key)

if __name__ == '__main__':
    """
    This script takes as input two folders that contain JSON files, created from multiple
    runs of the SGD program, and creates comparison plots on a given quantity.
    We are assuming that the JSON files differ in one dimension as represented
    in their filename, and that both folders share the same generating procedure
    (i.e the filenames under both folders should be the same)
    Ex. usage: > comparison_plot.py -m gpu_time -o V1_vs_V2 --folderA ./results/V1/ --folderB ./results/V2/ \
     --title "V1/V2 - Features scaling" --xlabel "Number of features" --ylabel "GPU Time (ms)" --Aname "V1" --Bname "V2"
    """
    # TODO: Enable support for arbitrary number of folders/comparisons
    import argparse
    parser = argparse.ArgumentParser(description="Script for plotting comparisons between SGD implementations")
    parser.add_argument("-m", "--measure", help="The measure we are interested in plotting. Ex. \"gpu_time\" ")
    parser.add_argument("-o", "--output", help="Output directory in which plots will be generated")
    parser.add_argument("--folderA", help="First input directory to use")
    parser.add_argument("--folderB", help="Second input directory to use")
    parser.add_argument("--title", help="Title for the created plot")
    parser.add_argument("--xlabel", help="X label for the created plot")
    parser.add_argument("--ylabel", help="Y label for the created plot")
    parser.add_argument("--Aname", help="Name to be given for the process that generated folder A's contents",
                        default="CUBLAS")
    parser.add_argument("--Bname", help="Name to be given for the process that generated folder B's contents",
                        default="Plain CUDA")

    args = parser.parse_args()

    def list_files(path):
        """
        Create a list containing only the files (not directories) present under the provided directory path
        :param path: The directory under which we search for files
        :return: A list containing all the filepaths under the provided directory
        """
        from os import listdir
        from os.path import isfile, join
        return [join(path, f) for f in listdir(path) if isfile(join(path, f))]

    # Get the files from both dirs, sorted by filename
    folderA_files = sorted_nicely(list_files(args.folderA))
    folderB_files = sorted_nicely(list_files(args.folderB))

    # Get the range of the parameters in the folder
    def extract_parameter_range(file_list):
        """
        Returns a list containing the range of parameters present in the list of files/
        We are assuming that the files have a naming scheme of <some-name>-<parameter-setting>-<codepath>.json
        :param file_list: A list containing filenames with the above format
        :return: A sorted list of a range of parameters
        """
        parameter_list = []
        for filename in file_list:
            # Extract parameter, assuming <some_name>-<parameter_setting>-output.json
            # The setting will then be the second to last element after splitting on '-'
            parameter_setting = filename.split('-')[-2]
            parameter_list.append(parameter_setting)

        return parameter_list

    par_list = extract_parameter_range(folderA_files)

    # Ensure that both dirs contain the same parameter settings
    assert par_list == extract_parameter_range(folderB_files), "Folders contain different parameter settings!"

    # Initialize lists to store the measures in
    results_listA = []
    results_listB = []

    def load_json_file(filepath):
        with open(filepath, 'r') as f:
            json_string = f.read()
            experiment_vals = json.loads(json_string)
        return experiment_vals

    # We go through both file lists together, read the relevant result, add add to lists
    for fileA, fileB in zip(folderA_files, folderB_files):
        # Read each JSON experiment output file
        results_dictA = load_json_file(fileA)
        results_dictB = load_json_file(fileB)

        # Extract information
        results_listA.append(results_dictA[args.measure])
        results_listB.append(results_dictB[args.measure])

    # Plotting

    # Plotting Configuration
    dir_name = "comparison-plots"
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    folder = "./{}/".format(dir_name)
    plt_filepath = folder + args.output + ".pdf"
    frameon = False  # no background
    transparent = True
    dpi = 1200

    def create_combined_plot(x_axis_list, results_a, results_b, filepath, title, x_label, y_label):
        """
        Creates a plot containing the combined results from the two lists provided
        """
        fig = plt.figure()
        sub = fig.add_subplot(111)  # TODO: This is unnecessary
        sub.plot(x_axis_list, results_a, '-ob')
        sub.plot(x_axis_list, results_b, '-xr')
        sub.set_xlim(int(x_axis_list[0]), int(x_axis_list[-1]))  # Set the limits according to the provided x_axis_list
        sub.set_ylim(bottom=0)

        sub.set_title(title)
        sub.set_xlabel(x_label)
        sub.set_ylabel(y_label)
        sub.legend([args.Aname, args.Bname], loc='upper left')

        fig.tight_layout()
        fig.savefig(filepath, frameon=frameon,
                    transparent=transparent, dpi=dpi, bbox_indces='tight')
        plt.close(fig)

    # Create the combined plot
    create_combined_plot(
            x_axis_list=par_list,
            results_a=results_listA,
            results_b=results_listB,
            filepath=plt_filepath,
            title=args.title,
            x_label=args.xlabel,
            y_label=args.ylabel)
