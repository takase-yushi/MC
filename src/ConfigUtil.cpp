//
// Created by Kamiya Keisuke on 2019-09-04.
//

#include "../includes/ConfigUtil.h"
#include "../includes/Config.h"
#include "../includes/env.h"
#include "../includes/Utils.h"
#include "../includes/picojson.h"
#include <vector>
#include <string>
#include <fstream>
#include <iostream>

/**
 * @fn std::vector<int> readTasks()
 * @brief config.jsonを読む
 * @param config_name コンフィグファイル名(defaultでconfig.jsonが解決される)
 * @return
 */
std::vector<Config> readTasks(std::string config_name) {
    std::vector<int> v;

    std::ifstream fs;

    fs.open(getProjectDirectory(OS) + "/" + config_name, std::ios::binary);

    if(fs.fail()){
        std::cerr << "Failed to open config.json" << std::endl;
        exit(-1);
    }

    std::string json_string((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
    fs.close();

    picojson::value val;
    const std::string err = picojson::parse(val, json_string);

    if(!err.empty()){
        std::cerr << "Failed to parse json string" << std::endl;
        exit(-1);
    }

    picojson::object& obj = val.get<picojson::object>();
    picojson::array& ary = obj["tasks"].get<picojson::array>();

    std::vector<Config> tasks;
    // iterate each-tasks
    for(auto& item : ary){
        picojson::object& task      = item.get<picojson::object>();
        bool enable_flag            = task["enable"         ].get<bool>();
        std::string img_directory   = task["img_directory"  ].get<std::string>();
        std::string gauss_ref_image = task["gauss_ref_image"].get<std::string>();
        std::string ref_image       = task["ref_image"      ].get<std::string>();
        std::string target_image    = task["target_image"   ].get<std::string>();
        std::string log_directory   = task["log_directory"].get<std::string>();
        int qp                      = static_cast<int>(task["QP"             ].get<double>());
        int ctu_width               = static_cast<int>(task["ctu_width"      ].get<double>());
        int ctu_height              = static_cast<int>(task["ctu_height"     ].get<double>());
        int division_steps          = static_cast<int>(task["division_step"  ].get<double>());

        bool lambda_enable          = task["lambda_enable"].get<bool>();
        double lambda               = static_cast<double>(task["lambda"].get<double>());
        tasks.emplace_back(enable_flag, img_directory, log_directory, gauss_ref_image, ref_image, target_image, qp, ctu_width, ctu_height, division_steps, lambda_enable, lambda);
    }

    return tasks;
}
