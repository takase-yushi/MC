import subprocess
import os
from joblib import Parallel, delayed


directories_2k = ["cactus/cactus-rd",
                  "drone2/drone2-rd",
                  "fungus/fungus-rd",
                  "in_to_tree_1280_640/in-to-tree-1280-640-rd",
                  "minato/minato-rd",
                  "park_scene/park-scene-rd",
                  "station2/station2-rd"]

directories_4k = [
    "express_way/express-way-rd"
]


def func(config_file_name):
    cmd = ["wsl", "./cmake-build-debug/Encoder", config_file_name]
    subprocess.call(cmd, shell=True)


indices = [1, 2, 3, 4, 5, 6, 7, 8, 9]


for dir_name in directories_2k:
    Parallel(n_jobs=10)([delayed(func)("config/" + dir_name + str(i) + ".json") for i in indices])


for dir_name in directories_4k:
    Parallel(n_jobs=5)([delayed(func)("config/" + dir_name + str(i) + ".json") for i in indices])
