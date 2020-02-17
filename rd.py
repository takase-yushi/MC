import subprocess
import os
from joblib import Parallel, delayed


def func(config_file_name):
    cmd = ["cmake-build-debug\Encoder_Square_Affine_merge2_no_flag_coefficient.exe", config_file_name]
    subprocess.call(cmd, shell=True)


directories = ["express_way/express-way",
               "fungus/fungus",
               "cactus/cactus",
               "drone2/drone2",
               "minato/minato",
               "park_scene/park-scene",
               "station2/station2",
               "in_to_tree_1280_640/in-to-tree-1280-640"]

suffix = "-rd_Square_Affine_128"

indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]


for dir_name in directories:
    Parallel(n_jobs=9)([delayed(func)("config/" + dir_name + suffix + str(i) + ".json") for i in indices])
