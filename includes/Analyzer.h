//
// Created by Kamiya Keisuke on 2019-07-18.
//

#ifndef ENCODER_ANALYZER_H
#define ENCODER_ANALYZER_H


#include <vector>
#include <map>
#include "CodingTreeUnit.h"

class CodingTreeUnit;

class Analyzer {
public:
    void storeDistributionOfMv(std::vector<CodingTreeUnit*> ctus, std::string log_path);
    void storeMarkdownFile(double psnr, std::string log_path);
    Analyzer(const std::string &fileSuffix);

private:
    void storeDistributionOfMv(CodingTreeUnit *ctu);

    // mvdのカウンター
    std::map<int, int> mvd_counter;
    std::map<int, int> mvd_counter_x;
    std::map<int, int> mvd_counter_y;
    std::map<int, int> MV_counter;

    // greater_0フラグの頻度
    std::map<int, int> greater_0_flag_counter;
    int greater_0_flag_sum;

    // greater_1フラグの頻度
    std::map<int, int> greater_1_flag_counter;
    int greater_1_flag_sum;

    // サインフラグの頻度
    std::map<int, int> sign_flag_counter;
    int sign_flag_sum;

    // 符号量関連
    int mvd_code_sum;
    int mvd_warping_code_sum;
    int mvd_parallel_code_sum;
    int code_sum;

    // ファイルの最後につける値
    std::string file_suffix;

    int warping_patch_num;
    int parallel_patch_num;

    int merge_counter;
    int spatial_counter;

};


#endif //ENCODER_ANALYZER_H
