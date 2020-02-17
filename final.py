import subprocess
import os
from joblib import Parallel, delayed


def func(config_file_name):
    cmd = ["cmake-build-debug\Encoder_Square_Affine_merge2_no_flag.exe", config_file_name]
    subprocess.call(cmd, shell=True)



directories_2k = ["fungus/fungus-final-square-rd"]

directories_4k = [
    "express_way/express-way-final-square-rd"
]

indices = [x for x in range (1, 9 * 30 + 1)]

for dir_name in directories_2k:
    Parallel(n_jobs=5)([delayed(func)("config/" + dir_name + str(i) + ".json") for i in indices])

#
# for dir_name in directories_4k:
#     Parallel(n_jobs=5)([delayed(func)("config/" + dir_name + str(i) + ".json") for i in indices])
