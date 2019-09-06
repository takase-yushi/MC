//
// Created by Kamiya Keisuke on 2019-09-04.
//

#ifndef ENCODER_CONFIGUTIL_H
#define ENCODER_CONFIGUTIL_H

#include <vector>
#include "../includes/Config.h"

std::vector<Config> readTasks(std::string config_name = "config.json");
void appendConfigItem(std::string input_file_path, std::string output_file_path);

#endif //ENCODER_CONFIGUTIL_H
