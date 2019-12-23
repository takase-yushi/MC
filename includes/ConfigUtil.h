//
// Created by Kamiya Keisuke on 2019-09-04.
//

#ifndef ENCODER_CONFIGUTIL_H
#define ENCODER_CONFIGUTIL_H

#include <vector>
#include "../includes/Config.h"

std::vector<Config> readTasks(std::string config_name = "config.json");
void generateConfigItem(std::string input_file_path, std::string output_file_path);
void generateChunkedConfigItem(std::string input_file_path, std::string output_file_path, int chunk_size);
void generateConfigForTestSequence();
void generateChunkedConfigForTestSequence();
void generateChunkedRDCurveConfigForTestSequence(int ctu_size = -1);

#endif //ENCODER_CONFIGUTIL_H
