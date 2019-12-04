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
    void storeDistributionOfMv(std::vector<CodingTreeUnit*> ctus, std::string log_path, std::vector<int> pells, std::vector<double> residuals);
    void storeMarkdownFile(double psnr, std::string log_path);
    Analyzer(const std::string &fileSuffix);
    void storeCsvFileWithStream(std::ofstream &ofs, double psnr, double time);
    void storeMergeMvLog(std::vector<CodingTreeUnit*> ctus, std::string log_path);


private:
    void storeDistributionOfMv(CodingTreeUnit *ctu);
    void storeMergeMvLog(CodingTreeUnit* ctu, std::ofstream &ofs);
    int getEntropyCodingCode();

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

    // マージの分布
    std::map<int, int> merge_flag_counter;

    // イントラの分布
    std::map<int, int> intra_flag_counter;

    // 符号量関連
    int mvd_code_sum;
    int mvd_warping_code_sum;
    int mvd_translation_code_sum;
    int code_sum;

    // ファイルの最後につける値
    std::string file_suffix;

    int warping_patch_num;
    int merge2_counter;
    int translation_patch_num;

    int merge_counter;
    int spatial_counter;

    int intra_counter;

    int patch_num;

    float max_merge_mv_diff_x;
    float max_merge_mv_diff_y;
};


#endif //ENCODER_ANALYZER_H
