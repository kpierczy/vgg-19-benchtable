# ====================================================================================================================================
# @file       charts.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 4:39:38 pm
# @project    vgg-19-testbench
# @brief      Script draws val_accuracy charts for all @a run_name trainings of every model in the @a models directory
#
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# ============================================================= Imports ============================================================ #

import os
import pickle
from matplotlib import pyplot as plt

# ============================================================== Paths ============================================================= #

# Path to the project's home directory
home = os.getenv('PROJECT_HOME')
# Path to the project's models directory
models = os.path.join(home, 'models')

# ========================================================== Configuration ========================================================= #

# Name of the output file
outname = 'val_loss.pdf'

# Metric to be plotted
metric = 'val_accuracy'

# Plot's title
title = 'Learning rates'

# Legend's location
lgd_loc = 'lower right'

# Name of the subrun that will be picked from every model in the @a model directory
run_name = 'run_1'
# Output directory
outdir = os.path.join(home, 'visualization/data')

# ============================================================== Main ============================================================== #

def main():

    # Prepare figure
    fig, ax = plt.subplots(figsize=(5,3))
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel('epoch')

    # Set log scale
    plt.yscale('log')

    # Iterate over @a run_name folders in all models 
    for model in os.listdir(models):

        # Check whether @a run_name subfolder exists
        datadir = os.path.join(os.path.join(models, model), run_name)
        if not os.path.exists(datadir):
            continue

        # Load training history
        with open(os.path.join(datadir, 'history/subrun_1.pickle'), 'rb') as h:
            history = pickle.load(h)

        # Append metric to the list
        plt.plot(history[metric])

    # Set legend
    lgnd = ax.legend(os.listdir(models), loc=lgd_loc, shadow=True)

    # Print figure
    fig.savefig(os.path.join(outdir, outname), format='pdf')
    plt.show()

# ================================================================================================================================== #

if __name__ == '__main__':
    main()
