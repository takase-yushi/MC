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
#include <iomanip>

std::vector<double> lambdas{
        1.0, 2.0, 3.0, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0, 10.0,
        12.0, 14.0, 16.0, 18.0, 20.0,
        25.0, 30.0, 40.0, 50.0, 60.0,
        70, 80, 90, 100, 110, 120, 130,
        140, 150, 160, 170, 180, 190,
        200, 250
};

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

    std::cout << getProjectDirectory(OS) + "/" + config_name << std::endl;
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
        int qp_offset               = static_cast<int>(task["QP_offset"      ].get<double>());
        int ctu_width               = static_cast<int>(task["ctu_width"      ].get<double>());
        int ctu_height              = static_cast<int>(task["ctu_height"     ].get<double>());
        int division_steps          = static_cast<int>(task["division_step"  ].get<double>());

        bool lambda_enable          = task["lambda_enable"].get<bool>();
        double lambda               = static_cast<double>(task["lambda"].get<double>());
        tasks.emplace_back(enable_flag, img_directory, log_directory, gauss_ref_image, ref_image, target_image, qp, qp_offset, ctu_width, ctu_height, division_steps, lambda_enable, lambda);
    }

    return tasks;
}


void generateConfigItem(std::string input_file_path, std::string output_file_path){
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
    int array_size = lambdas.size() * ary.size();
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
            ofs << "            \"lambda\"         : " << lambdas[i] << "," << std::endl;
            ofs << R"(            "QP_offset"      : 0)" << std::endl;

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

/**
 * @fn void generateChunkedConfigItem(std::string input_file_path, std::string out_file_path, int chunk_size)
 * @brief chunk_sizeごとに分割したコンフィグファイルを吐く
 * @param string input_file_path
 * @param string out_file_path
 * @param int chunk_size
 */
void generateChunkedConfigItem(std::string input_file_path, std::string output_file_path, int chunk_size) {
    std::ifstream fs;

    fs.open(input_file_path, std::ios::binary);

    if(fs.fail()){
        std::cerr << "Failed to open config.json" << std::endl;
        exit(-1);
    }

    std::string json_string((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
    fs.close();

    picojson::value val;
    std::string err = picojson::parse(val, json_string);

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
    int array_size = lambdas.size() * ary.size();
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
            ofs << "            \"lambda\"         : " << lambdas[i] << "," << std::endl;
            ofs << R"(            "QP_offset"      : 0)" << std::endl;

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

    ofs.close();

    fs.open(output_file_path, std::ios::binary);

    if(fs.fail()){
        std::cerr << "Failed to open config.json" << std::endl;
        exit(-1);
    }

    std::string json_string2((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
    fs.close();

    picojson::value val2;
    err = picojson::parse(val2, json_string2);

    if(!err.empty()){
        std::cerr << "Failed to parse json string" << std::endl;
        exit(-1);
    }

    obj = val2.get<picojson::object>();
    ary = obj["tasks"].get<picojson::array>();

    std::cout << ary.size() << std::endl;
    int chunked_array_size = ceil((double)ary.size() / chunk_size);

    count = 0;
    for(int chunked_array_index = 0 ; chunked_array_index < chunked_array_size ; chunked_array_index++) {
        std::ofstream ofs;

        // hoge.jsonをhoge1.jsonとかにして吐き出す
        std::string file_name = output_file_path.substr(0, output_file_path.rfind('.'));
        std::string tmp_output_file_path = file_name + std::to_string(chunked_array_index + 1) + ".json";
        ofs.open(tmp_output_file_path);

        ofs << "{" << std::endl;
        ofs << "  \"tasks\":[" << std::endl;

        array_size = (chunked_array_index == (chunked_array_size - 1) && (ary.size() % chunk_size) != 0 ? (ary.size() % chunk_size) : chunk_size);

        std::cout << array_size << std::endl;
        for(int i = 0 ; i < array_size && count < (int)ary.size(); i++){
            picojson::object& task      = ary[count].get<picojson::object>();

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
            ofs << "            \"lambda_enable\"  : " << std::boolalpha << task["lambda_enable"].get<bool>() << "," << std::endl;
            ofs << "            \"lambda\"         : " << static_cast<double>(task["lambda"].get<double>()) << "," << std::endl;
            ofs << R"(            "QP_offset"      : 0)" << std::endl;

            if (i == array_size - 1) {
                ofs << "        }" << std::endl;
            } else {
                ofs << "        }," << std::endl;
            }

            count++;
        }

        ofs << "    ]" << std::endl;
        ofs << "}" << std::endl;
        ofs.close();
    }

}

void generateChunkedRDConfigItem(std::string input_file_path, std::string output_file_path, int chunk_size) {
    std::ifstream fs;

    std::vector<double> lambdas_rd{
            10.25,12.625,15,18.5,23.375,30,38.75,50,61.25
    };

    fs.open(input_file_path, std::ios::binary);

    if(fs.fail()){
        std::cerr << "Failed to open config.json" << std::endl;
        exit(-1);
    }

    std::string json_string((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
    fs.close();

    picojson::value val;
    std::string err = picojson::parse(val, json_string);

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
    int array_size = 9;
    for(auto& item : ary){
        picojson::object& task      = item.get<picojson::object>();

        for(int i = 0 ; i < lambdas_rd.size() ; i++) {
            int qp = static_cast<int>(task["QP"].get<double>());
            if(i == 0 && qp != 22) continue;
            if(i == 1 && qp != 24) continue;
            if(i == 2 && qp != 26) continue;
            if(i == 3 && qp != 28) continue;
            if(i == 4 && qp != 30) continue;
            if(i == 5 && qp != 32) continue;
            if(i == 6 && qp != 34) continue;
            if(i == 7 && qp != 36) continue;
            if(i == 8 && qp != 37) continue;

            ofs << "        {" << std::endl;
            ofs << R"(            "enable"         : )" << std::boolalpha << task["enable"].get<bool>() << "," << std::endl;
            ofs << R"(            "img_directory"  : ")" << task["img_directory"].get<std::string>() << "\"," << std::endl;
            ofs << R"(            "log_directory"  : ")" << task["log_directory"].get<std::string>() << "\"," << std::endl;
            ofs << R"(            "gauss_ref_image": ")" << task["gauss_ref_image"].get<std::string>() << "\"," << std::endl;
            ofs << R"(            "ref_image"      : ")" << task["ref_image"].get<std::string>() << "\"," << std::endl;
            ofs << R"(            "target_image"   : ")" << task["target_image"].get<std::string>() << "\"," << std::endl;
            ofs << "            \"QP\"             : " << qp << "," << std::endl;
            ofs << "            \"ctu_width\"      : " << static_cast<int>(task["ctu_width"].get<double>()) << "," << std::endl;
            ofs << "            \"ctu_height\"     : " << static_cast<int>(task["ctu_height"].get<double>()) << "," << std::endl;
            ofs << "            \"division_step\"  : " << static_cast<int>(task["division_step"].get<double>()) << "," << std::endl;
            ofs << "            \"lambda_enable\"  : true," << std::endl;
            ofs << "            \"lambda\"         : " << lambdas_rd[i] << "," << std::endl;
            ofs << R"(            "QP_offset"      : 0)" << std::endl;

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

    ofs.close();

    fs.open(output_file_path, std::ios::binary);

    if(fs.fail()){
        std::cerr << "Failed to open config.json" << std::endl;
        exit(-1);
    }

    std::string json_string2((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
    fs.close();

    picojson::value val2;
    err = picojson::parse(val2, json_string2);

    if(!err.empty()){
        std::cerr << "Failed to parse json string" << std::endl;
        exit(-1);
    }

    obj = val2.get<picojson::object>();
    ary = obj["tasks"].get<picojson::array>();

    int chunked_array_size = ceil((double)ary.size() / chunk_size);

    count = 0;
    for(int chunked_array_index = 0 ; chunked_array_index < chunked_array_size ; chunked_array_index++) {
        std::ofstream ofs;

        // hoge.jsonをhoge1.jsonとかにして吐き出す
        std::string file_name = output_file_path.substr(0, output_file_path.rfind('.'));
        std::string tmp_output_file_path = file_name + std::to_string(chunked_array_index + 1) + ".json";
        ofs.open(tmp_output_file_path);

        ofs << "{" << std::endl;
        ofs << "  \"tasks\":[" << std::endl;

        array_size = (chunked_array_index == (chunked_array_size - 1) && (ary.size() % chunk_size) != 0 ? (ary.size() % chunk_size) : chunk_size);

        for(int i = 0 ; i < array_size && count < (int)ary.size(); i++){
            picojson::object& task      = ary[count].get<picojson::object>();

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
            ofs << "            \"lambda_enable\"  : " << std::boolalpha << task["lambda_enable"].get<bool>() << "," << std::endl;
            ofs << "            \"lambda\"         : " << static_cast<double>(task["lambda"].get<double>()) << "," << std::endl;
            ofs << R"(            "QP_offset"      : 0)" << std::endl;

            if (i == array_size - 1) {
                ofs << "        }" << std::endl;
            } else {
                ofs << "        }," << std::endl;
            }

            count++;
        }

        ofs << "    ]" << std::endl;
        ofs << "}" << std::endl;
        ofs.close();
    }

}

/**
 * @fn void generateConfigForTestSequence
 * @brief テストシーケンスに使いそうなやつのgenerateConfigItem呼び出しをまとめた．ハードコードしている
 */
void generateConfigForTestSequence(){

    std::string base_path = getProjectDirectory(OS);
    generateConfigItem(base_path + "/config/cactus/cactus.json", base_path + "/config/cactus/cactus-tmp.json");
    generateConfigItem(base_path + "/config/drone2/drone2.json", base_path + "/config/drone2/drone2-tmp.json");
    generateConfigItem(base_path + "/config/fungus/fungus.json", base_path + "/config/fungus/fungus-tmp.json");
    generateConfigItem(base_path + "/config/in_to_tree_1280_640/in-to-tree-1280-640.json", base_path + "/config/in_to_tree_1280_640/in-to-tree-1280-640-tmp.json");
    generateConfigItem(base_path + "/config/kimono1/kimono1.json", base_path + "/config/kimono1/kimono1-tmp.json");
    generateConfigItem(base_path + "/config/minato/minato.json", base_path + "/config/minato/minato-tmp.json");
    generateConfigItem(base_path + "/config/park_scene/park-scene.json", base_path + "/config/park_scene/park-scene-tmp.json");
    generateConfigItem(base_path + "/config/station2/station2.json", base_path + "/config/station2/station2-tmp.json");
    generateConfigItem(base_path + "/config/express_way/express-way.json", base_path + "/config/express-way/express-way-tmp.json");
}

/**
 * @fn void generateChunkedConfigForTestSequence
 * @brief テストシーケンスに使いそうなやつのgenerateChunkedConfigItem呼び出しをまとめた．ハードコードしている
 */
void generateChunkedConfigForTestSequence(int ctu_size){

    std::string base_path = getProjectDirectory(OS);

    std::string suffix;
    if(ctu_size != -1) {
        suffix = "-" + std::to_string(ctu_size);
    }
    generateChunkedConfigItem(base_path + "/config/cactus/cactus"                           + suffix + ".json", base_path + "/config/cactus/cactus"                           + suffix + "-tmp.json", 35);
    generateChunkedConfigItem(base_path + "/config/drone2/drone2"                           + suffix + ".json", base_path + "/config/drone2/drone2"                           + suffix + "-tmp.json", 35);
    generateChunkedConfigItem(base_path + "/config/fungus/fungus"                           + suffix + ".json", base_path + "/config/fungus/fungus"                           + suffix + "-tmp.json", 35);
    generateChunkedConfigItem(base_path + "/config/in_to_tree_1280_640/in-to-tree-1280-640" + suffix + ".json", base_path + "/config/in_to_tree_1280_640/in-to-tree-1280-640" + suffix + "-tmp.json", 35);
    generateChunkedConfigItem(base_path + "/config/kimono1/kimono1"                         + suffix + ".json", base_path + "/config/kimono1/kimono1"                         + suffix + "-tmp.json", 35);
    generateChunkedConfigItem(base_path + "/config/minato/minato"                           + suffix + ".json", base_path + "/config/minato/minato"                           + suffix + "-tmp.json", 35);
    generateChunkedConfigItem(base_path + "/config/park_scene/park-scene"                   + suffix + ".json", base_path + "/config/park_scene/park-scene"                   + suffix + "-tmp.json", 35);
    generateChunkedConfigItem(base_path + "/config/station2/station2"                       + suffix + ".json", base_path + "/config/station2/station2"                       + suffix + "-tmp.json", 35);
    generateChunkedConfigItem(base_path + "/config/express_way/express-way"                 + suffix + ".json", base_path + "/config/express_way/express-way"                 + suffix + "-tmp.json", 35);
}

/**
 * @fn void generateRDCurveConfigForTestSequence()
 * @brief RDカーブを描く用のConfigジェネレーター．並列に回せるよう，ChunkSize=1となっている
 */
void generateChunkedRDCurveConfigForTestSequence(int ctu_size){
    std::string base_path = getProjectDirectory(OS);

    std::string suffix;
    if(ctu_size != -1) {
        suffix = "-" + std::to_string(ctu_size);
    }

    generateChunkedRDConfigItem(base_path + "/config/cactus/cactus"                           + suffix + ".json", base_path + "/config/cactus/cactus"                           + suffix + "-rd.json", 1);
    generateChunkedRDConfigItem(base_path + "/config/drone2/drone2"                           + suffix + ".json", base_path + "/config/drone2/drone2"                           + suffix + "-rd.json", 1);
    generateChunkedRDConfigItem(base_path + "/config/fungus/fungus"                           + suffix + ".json", base_path + "/config/fungus/fungus"                           + suffix + "-rd.json", 1);
    generateChunkedRDConfigItem(base_path + "/config/in_to_tree_1280_640/in-to-tree-1280-640" + suffix + ".json", base_path + "/config/in_to_tree_1280_640/in-to-tree-1280-640" + suffix + "-rd.json", 1);
    generateChunkedRDConfigItem(base_path + "/config/minato/minato"                           + suffix + ".json", base_path + "/config/minato/minato"                           + suffix + "-rd.json", 1);
    generateChunkedRDConfigItem(base_path + "/config/park_scene/park-scene"                   + suffix + ".json", base_path + "/config/park_scene/park-scene"                   + suffix + "-rd.json", 1);
    generateChunkedRDConfigItem(base_path + "/config/station2/station2"                       + suffix + ".json", base_path + "/config/station2/station2"                       + suffix + "-rd.json", 1);
    generateChunkedRDConfigItem(base_path + "/config/express_way/express-way"                 + suffix + ".json", base_path + "/config/express_way/express-way"                 + suffix + "-rd.json", 1);
}

/**
 * @fn void generateRDCurveConfigForTestSequence()
 * @brief RDカーブを描く用のConfigジェネレーター．並列に回せるよう，ChunkSize=1となっている
 */
void generateChunkedRDCurveConfigForTestSequenceForPaper(int ctu_size){
    std::string base_path = getProjectDirectory(OS);

    std::string suffix;
    if(ctu_size != -1) {
        suffix = "-" + std::to_string(ctu_size);
    }

    generateChunkedRDConfigItemForPaper(base_path + "/config/cactus/cactus-final"                           + suffix + ".json", base_path + "/config/cactus/cactus-final"                           + suffix + "-rd.json", 1, "cactus"     , false);
    generateChunkedRDConfigItemForPaper(base_path + "/config/drone2/drone2-final"                           + suffix + ".json", base_path + "/config/drone2/drone2-final"                           + suffix + "-rd.json", 1, "drone2"     , false);
    generateChunkedRDConfigItemForPaper(base_path + "/config/fungus/fungus-final"                           + suffix + ".json", base_path + "/config/fungus/fungus-final"                           + suffix + "-rd.json", 1, "fungus"     , false);
    generateChunkedRDConfigItemForPaper(base_path + "/config/in_to_tree_1280_640/in-to-tree-1280-640-final" + suffix + ".json", base_path + "/config/in_to_tree_1280_640/in-to-tree-1280-640-final" + suffix + "-rd.json", 1, "in_to_tree" , false);
    generateChunkedRDConfigItemForPaper(base_path + "/config/minato/minato-final"                           + suffix + ".json", base_path + "/config/minato/minato-final"                           + suffix + "-rd.json", 1, "minato"     , false);
    generateChunkedRDConfigItemForPaper(base_path + "/config/park_scene/park-scene-final"                   + suffix + ".json", base_path + "/config/park_scene/park-scene-final"                   + suffix + "-rd.json", 1, "park_scene" , false);
    generateChunkedRDConfigItemForPaper(base_path + "/config/station2/station2-final"                       + suffix + ".json", base_path + "/config/station2/station2-final"                       + suffix + "-rd.json", 1, "station2"   , false);
//    generateChunkedRDConfigItemForPaper(base_path + "/config/express_way/express-way"                 + suffix + ".json", base_path + "/config/express_way/express-way"                 + suffix + "-rd.json", 1, "express_way", false);

}

/**
 * @fn void generateRDCurveConfigForTestSequence()
 * @brief RDカーブを描く用のConfigジェネレーター．並列に回せるよう，ChunkSize=1となっている
 */
void generateChunkedSquareRDCurveConfigForTestSequenceForPaper(int ctu_size){
    std::string base_path = getProjectDirectory(OS);

    std::string suffix;
    if(ctu_size != -1) {
        suffix = "-" + std::to_string(ctu_size);
    }

    generateChunkedRDConfigItemForPaper(base_path + "/config/cactus/cactus-final"                           + suffix + ".json", base_path + "/config/cactus/cactus-final-square"                           + suffix + "-rd.json", 1, "cactus"     , true);
    generateChunkedRDConfigItemForPaper(base_path + "/config/drone2/drone2-final"                           + suffix + ".json", base_path + "/config/drone2/drone2-final-square"                           + suffix + "-rd.json", 1, "drone2"     , true);
    generateChunkedRDConfigItemForPaper(base_path + "/config/fungus/fungus-final"                           + suffix + ".json", base_path + "/config/fungus/fungus-final-square"                           + suffix + "-rd.json", 1, "fungus"     , true);
    generateChunkedRDConfigItemForPaper(base_path + "/config/in_to_tree_1280_640/in-to-tree-1280-640-final" + suffix + ".json", base_path + "/config/in_to_tree_1280_640/in-to-tree-1280-640-final-square" + suffix + "-rd.json", 1, "in_to_tree" , true);
    generateChunkedRDConfigItemForPaper(base_path + "/config/minato/minato-final"                           + suffix + ".json", base_path + "/config/minato/minato-final-square"                           + suffix + "-rd.json", 1, "minato"     , true);
    generateChunkedRDConfigItemForPaper(base_path + "/config/park_scene/park-scene-final"                   + suffix + ".json", base_path + "/config/park_scene/park-scene-final-square"                   + suffix + "-rd.json", 1, "park_scene" , true);
    generateChunkedRDConfigItemForPaper(base_path + "/config/station2/station2-final"                       + suffix + ".json", base_path + "/config/station2/station2-final-square"                       + suffix + "-rd.json", 1, "station2"   , true);
    generateChunkedRDConfigItemForPaper(base_path + "/config/express_way/express-way"                 + suffix + ".json", base_path + "/config/express_way/express-way-square"                 + suffix + "-rd.json", 1, "express_way", true);

}

void generateChunkedRDConfigItemForPaper(std::string input_file_path, std::string output_file_path, int chunk_size, std::string sequence_name, bool isSquare){
    std::ifstream fs;

    std::vector<double> lambdas_rd{
            10.25,12.625,15,18.5,23.375,30,38.75,50,61.25
    };

    std::vector<double> lambdas_rd_for_square {
            8.125,9.625,11.5,14.5,17.75,23.625,31.125,42.5,53.75
    };

    std::vector<int> qps_rd {
        22, 24, 26, 28, 30, 32, 34, 36, 37
    };

    fs.open(input_file_path, std::ios::binary);

    if(fs.fail()){
        std::cerr << "Failed to open config.json" << std::endl;
        exit(-1);
    }

    std::string json_string((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
    fs.close();

    picojson::value val;
    std::string err = picojson::parse(val, json_string);

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
    int array_size = 9 * 30;
    for(auto& item : ary){
        picojson::object& task      = item.get<picojson::object>();

        for(int i = 0 ; i < lambdas_rd.size() ; i++) {
            for(int start_idx = 0 ; start_idx < 90 ; start_idx+=3) {
                int qp = static_cast<int>(task["QP"].get<double>());
                if(i == 0 && qp != 22) continue;
                if(i == 1 && qp != 24) continue;
                if(i == 2 && qp != 26) continue;
                if(i == 3 && qp != 28) continue;
                if(i == 4 && qp != 30) continue;
                if(i == 5 && qp != 32) continue;
                if(i == 6 && qp != 34) continue;
                if(i == 7 && qp != 36) continue;
                if(i == 8 && qp != 37) continue;

                ofs << "        {" << std::endl;
                ofs << R"(            "enable"         : )" << std::boolalpha << task["enable"].get<bool>() << "," << std::endl;
                ofs << R"(            "img_directory"  : ")" << task["img_directory"].get<std::string>() << "\"," << std::endl;
                ofs << R"(            "log_directory"  : ")" << task["log_directory"].get<std::string>() << "\"," << std::endl;
                ofs << R"(            "gauss_ref_image": ")" << task["gauss_ref_image"].get<std::string>() + std::to_string(start_idx) + "_" + std::to_string(start_idx + 3) + "_I_qp_22.png" << "\"," << std::endl;
                ofs << R"(            "ref_image"      : ")" << task["ref_image"].get<std::string>() + std::to_string(start_idx) + "_" + std::to_string(start_idx + 3) + "_I_qp_" + std::to_string(qps_rd[i]) + ".png"<< "\"," << std::endl;
                ofs << R"(            "target_image"   : ")" << task["target_image"].get<std::string>() <<  std::setfill('0') << std::right << std::setw(6) << start_idx + 3 << ".png" << "\"," << std::endl;
                ofs << "            \"QP\"             : " << qp << "," << std::endl;
                ofs << "            \"ctu_width\"      : " << static_cast<int>(task["ctu_width"].get<double>()) << "," << std::endl;
                ofs << "            \"ctu_height\"     : " << static_cast<int>(task["ctu_height"].get<double>()) << "," << std::endl;
                ofs << "            \"division_step\"  : " << static_cast<int>(task["division_step"].get<double>()) << "," << std::endl;
                ofs << "            \"lambda_enable\"  : true," << std::endl;
                ofs << "            \"lambda\"         : " << (isSquare ? lambdas_rd_for_square[i] : lambdas_rd[i]) << "," << std::endl;
                ofs << R"(            "QP_offset"      : 0)" << std::endl;

                if (count == array_size - 1) {
                    ofs << "        }" << std::endl;
                } else {
                    ofs << "        }," << std::endl;
                }
                count++;
            }
        }
    }

    ofs << "    ]" << std::endl;
    ofs << "}" << std::endl;

    ofs.close();

    fs.open(output_file_path, std::ios::binary);

    if(fs.fail()){
        std::cerr << "Failed to open config.json" << std::endl;
        exit(-1);
    }

    std::string json_string2((std::istreambuf_iterator<char>(fs)), std::istreambuf_iterator<char>());
    fs.close();

    picojson::value val2;
    err = picojson::parse(val2, json_string2);

    if(!err.empty()){
        std::cerr << "Failed to parse json string" << std::endl;
        exit(-1);
    }

    obj = val2.get<picojson::object>();
    ary = obj["tasks"].get<picojson::array>();

    int chunked_array_size = ceil((double)ary.size() / chunk_size);

    count = 0;
    for(int chunked_array_index = 0 ; chunked_array_index < chunked_array_size ; chunked_array_index++) {
        std::ofstream ofs;

        // hoge.jsonをhoge1.jsonとかにして吐き出す
        std::string file_name = output_file_path.substr(0, output_file_path.rfind('.'));
        std::string tmp_output_file_path = file_name + std::to_string(chunked_array_index + 1) + ".json";
        ofs.open(tmp_output_file_path);

        ofs << "{" << std::endl;
        ofs << "  \"tasks\":[" << std::endl;

        array_size = (chunked_array_index == (chunked_array_size - 1) && (ary.size() % chunk_size) != 0 ? (ary.size() % chunk_size) : chunk_size);

        for(int i = 0 ; i < array_size && count < (int)ary.size(); i++){
            picojson::object& task      = ary[count].get<picojson::object>();

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
            ofs << "            \"lambda_enable\"  : " << std::boolalpha << task["lambda_enable"].get<bool>() << "," << std::endl;
            ofs << "            \"lambda\"         : " << static_cast<double>(task["lambda"].get<double>()) << "," << std::endl;
            ofs << R"(            "QP_offset"      : 0)" << std::endl;

            if (i == array_size - 1) {
                ofs << "        }" << std::endl;
            } else {
                ofs << "        }," << std::endl;
            }

            count++;
        }

        ofs << "    ]" << std::endl;
        ofs << "}" << std::endl;
        ofs.close();
    }
}