//
// Created by Kamiya Keisuke on 2019-09-04.
//

#include "../includes/ConfigUtil.h"
#include "../includes/Config.h"
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


void appendConfigItem(std::string input_file_path, std::string output_file_path){
    std::ifstream fs;

    fs.open(input_file_path, std::ios::binary);

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

    std::ofstream ofs;
    ofs.open(output_file_path);

    ofs << "{" << std::endl;
    ofs << "  \"tasks\":[" << std::endl;

    // iterate each-tasks
    int count = 0;
    int array_size = ary.size() - 1;
    std::vector<double> lambdas{
            0.0, 0.5, 1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            12.0, 14.0, 16.0, 18.0, 20.0,
            25.0, 30.0, 40.0, 50.0, 60.0,
            70, 80, 90, 100, 110, 120, 130,
            140, 150, 160, 170, 180, 190,
            200, 250, 300
    };

    array_size = lambdas.size() * ary.size();
    for(auto& item : ary){
        picojson::object& task      = item.get<picojson::object>();

        for(int i = 0 ; i < lambdas.size() ; i++) {
            ofs << "        {" << std::endl;
            ofs << R"(            "enable"         : )" << std::boolalpha << task["enable"].get<bool>() << "," << std::endl;
            ofs << R"(            "img_directory"  : ")" << task["img_directory"].get<std::string>() << "\"," << std::endl;
            ofs << R"(            "log_directory"  : ")" << task["log_directory"].get<std::string>() << "\"," << std::endl;
            ofs << R"(            "gauss_ref_image": ")" << task["gauss_ref_image"].get<std::string>() << "\"," << std::endl;
            ofs << R"(            "ref_image"      : ")" << task["ref_image"].get<std::string>() << "\"," << std::endl;
            ofs << R"(            "target_image"   : ")" << task["target_image"].get<std::string>() << "\"," << std::endl;
            ofs << "            \"QP\"             : " << static_cast<int>(task["QP"].get<double>()) << "," << std::endl;
            ofs << "            \"ctu_width\"      : " << static_cast<int>(task["ctu_width"].get<double>()) << "," << std::endl;
            ofs << "            \"ctu_height\"     : " << static_cast<int>(task["ctu_height"].get<double>()) << "," << std::endl;
            ofs << "            \"division_step\"  : " << static_cast<int>(task["division_step"].get<double>()) << "," << std::endl;
            ofs << "            \"lambda_enable\"  : true," << std::endl;
            ofs << "            \"lambda\"         : " << lambdas[i] << std::endl;

            if (count == array_size - 1) {
                ofs << "        }" << std::endl;
            } else {
                ofs << "        }," << std::endl;
            }
            count++;
        }
    }

    ofs << "    ]" << std::endl;
    ofs << "}" << std::endl;
}