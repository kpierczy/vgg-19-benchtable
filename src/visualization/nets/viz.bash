#!/bin/bash
# ====================================================================================================================================
# @file       viz.bash
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Sunday, 26th June 2022 4:35:19 pm
# @modified   Sunday, 26th June 2022 5:04:29 pm
# @project    vgg-19-testbench
# @details
#
#    Runs plot-neural-net framework on the python file given in the first argument to generate  neural network's 
#    visualization. The filename should be given relative to the  $PROJECT_HOME/src/visualization/nets folder. Example usage:
#
#        viz.bash pipe/pipe.py
#
# @note After running script, the framework will try to open a deleted copy of the .pdf file. It's a known behaviour.
# @note The framework also tries to remove unexisting *.vscodeLog files. It's also a know behaviour. Don't worry about it.
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# Apply patch to the PlotNeuralNet project
cp $PROJECT_HOME/src/visualization/nets/mods/tikzeng.py $PROJECT_HOME/src/visualization/nets/plot-neural-net/pycore/tikzeng.py

# Create temporary project's directory
tmp_dir=$PROJECT_HOME/src/visualization/nets/plot-neural-net/tmp
rm -rf $tmp_dir
mkdir -p $tmp_dir

# Copy source graph script into the tmp directory
dir="$(dirname $1)"
cp -r $PROJECT_HOME/src/visualization/nets/$dir/* $tmp_dir/

cd $tmp_dir

# Run the framework
file="$(basename -- $1)"
file="${file%.*}"
bash ../tikzmake.sh $file

# Move result to the initial directory
cp "$file.pdf" $PROJECT_HOME/src/visualization/nets/$dir/

# Cleanup 
rm -rf $tmp_dir
