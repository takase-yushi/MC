import subprocess
import os
from joblib import Parallel, delayed


def my_makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def func(config_file_name):
    cmd = ["wsl", "./cmake-build-debug/Encoder", config_file_name]
    subprocess.call(cmd, shell=True)


directories_2k = ["in_to_tree_1280_640/in-to-tree-1280-640-final-rd",
                  "fungus/fungus-final-rd",
                  "minato/minato-final-rd",
                  "cactus/cactus-final-rd",
                  "drone2/drone2-final-rd",
                  "park_scene/park-scene-final-rd",
                  "station2/station2-final-rd"]

directories_4k = [
    "express_way/express-way-final-rd"
]

indices = [x for x in range (1, 9 * 30 + 1)]

for dir_name in directories_2k:
    Parallel(n_jobs=6)([delayed(func)("config/" + dir_name + str(i) + ".json") for i in indices])

#
# for dir_name in directories_4k:
#     Parallel(n_jobs=1)([delayed(func)("config/" + dir_name + str(i) + ".json") for i in indices])
