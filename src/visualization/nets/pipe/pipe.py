# ====================================================================================================================================
# @file       pipe.py
# @author     Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @maintainer Krzysztof Pierczyk (krzysztof.pierczyk@gmail.com)
# @date       Wednesday, 6th January 2021 2:17:57 am
# @modified   Sunday, 26th June 2022 5:01:06 pm
# @project    vgg-19-testbench
# @brief      Desciption of the pipeline's graph in the PlotNeuralNet framework's language
#
# @note For author's needings the source code of the PlotNeuralNet was adjusted. The to_cor() function was modified to allow define
#     additional colours and to_Conv() was modified to allow non-default colour for the convolutional layer. Modified version of the
#     'tikzeng.py' file is stored in the 'mods/tikzeng.py'. 
#
# @copyright Krzysztof Pierczyk © 2022
# ====================================================================================================================================

# ============================================================= Imports ============================================================ #

import sys

# ============================================================= Config ============================================================= #

# The '../' must be always appended to the PATH to let plot-neural-net access additional functions
sys.path.append('../')

# ============================================================= Imports ============================================================ #

import pycore.tikzeng as pt

# ========================================================== Configuration ========================================================= #

# Distance between subsequent layers
blocks_distance = 6

# Colour definitions (requires ingerence in the source code of the framework)
cols=r"""
\def\GapColor{rgb:red,1;black,0.3}
"""

# ============================================================== Model ============================================================= #

bd = blocks_distance

# Defined architecture
arch = [
    pt.to_head( '..' ),
    pt.to_cor(cols),
    pt.to_begin(),
    # Input images
    pt.to_UnPool("img1", offset="(0.0,0,0)", width=1, height=40, depth=40),
    pt.to_UnPool("img2", offset="(0.5,0,0)", width=1, height=40, depth=40),
    pt.to_UnPool("img3", offset="(1.0,0,0)", width=1, height=40, depth=40),
    pt.to_UnPool("img4", offset="(1.5,0,0)", width=1, height=40, depth=40),
    pt.to_UnPool("img5", offset="(2.0,0,0)", width=1, height=40, depth=40),
    pt.to_UnPool("img6", offset="(2.5,0,0)", width=1, height=40, depth=40),
    # Loader
    pt.to_Conv("loader", "", "\"Loader\"", offset="({},0,0)".format(bd), to="(img6-east)", height=40, depth=40, width=16),
    pt.to_connection("img6", "loader"),
    # Split
    pt.to_Conv("training"  , "", "\"Training\"",   offset="({}, 7,0)".format(bd), to="(loader-east)", height=20, depth=20, width=16),
    pt.to_Conv("validation", "", "\"Validation\"", offset="({}, 0,0)".format(bd), to="(loader-east)", height=20, depth=20, width=16),
    pt.to_Conv("test"      , "", "\"Test\"",       offset="({},-7,0)".format(bd), to="(loader-east)", height=20, depth=20, width=16),
    pt.to_connection("loader", "training"),
    pt.to_connection("loader", "validation"),
    pt.to_connection("loader", "test"),
    # Shuffle
    pt.to_Conv("shuffling", "", "\"Shuffling\"", offset="({}, 0,0)".format(bd/2), to="(training-east)", height=20, depth=20, width=8),
    pt.to_connection("training", "shuffling"),
    # Gap
    pt.to_Conv("gap", "", "\"Gap\"", offset="({}, 0,0)".format(bd * 1.5), to="(validation-east)", height=96, depth=96, width=2, colour="\GapColor"),
    # Batch
    pt.to_Conv("training_batch_1"  , "", "",          offset="({},0,0)".format(bd * 2.5), to="(training-east)",           height=20, depth=20, width=6),
    pt.to_Conv("training_batch_2"  , "", "\"Batch\"", offset="(0.5,0,0)",                 to="(training_batch_1-east)",   height=20, depth=20, width=6),
    pt.to_Conv("training_batch_3"  , "", "",          offset="(0.5,0,0)",                 to="(training_batch_2-east)",   height=20, depth=20, width=6),
    pt.to_Conv("validation_batch_1", "", "",          offset="({},0,0)".format(bd * 2.5), to="(validation-east)",         height=20, depth=20, width=6),
    pt.to_Conv("validation_batch_2", "", "\"Batch\"", offset="(0.5,0,0)",                 to="(validation_batch_1-east)", height=20, depth=20, width=6),
    pt.to_Conv("validation_batch_3", "", "",          offset="(0.5,0,0)",                 to="(validation_batch_2-east)", height=20, depth=20, width=6),
    pt.to_Conv("test_batch_1"      , "", "",          offset="({},0,0)".format(bd * 2.5), to="(test-east)",               height=20, depth=20, width=6),
    pt.to_Conv("test_batch_2"      , "", "\"Batch\"", offset="(0.5,0,0)",                 to="(test_batch_1-east)",       height=20, depth=20, width=6),
    pt.to_Conv("test_batch_3"      , "", "",          offset="(0.5,0,0)",                 to="(test_batch_2-east)",       height=20, depth=20, width=6),
    pt.to_connection("training",   "training_batch_3"),
    pt.to_connection("validation", "validation_batch_3"),
    pt.to_connection("test",       "test_batch_3"),
    # Prefetch
    pt.to_Conv("training_prefetch"  , "", "\"Prefetch\"", offset="({},0,0)".format(bd), to="(training_batch_3-east)",   height=20, depth=20, width=16),
    pt.to_Conv("validation_prefetch", "", "\"Prefetch\"", offset="({},0,0)".format(bd), to="(validation_batch_3-east)", height=20, depth=20, width=16),
    pt.to_Conv("test_prefetch"      , "", "\"Prefetch\"", offset="({},0,0)".format(bd), to="(test_batch_3-east)",       height=20, depth=20, width=16),
    pt.to_connection("training_batch_3",  "training_prefetch"),
    pt.to_connection("validation_batch_3","validation_prefetch"),
    pt.to_connection("test_batch_3",      "test_prefetch"),
    pt.to_end()
]


# ============================================================== Main ============================================================== #

def main():
    namefile = str(sys.argv[0]).split('.')[0]
    pt.to_generate(arch, namefile + '.tex' )

# ================================================================================================================================== #

if __name__ == '__main__':
    main()