//
// Created by Kamiya Keisuke on 2019-07-18.
//

#include "../includes/Analyzer.h"
#include "../includes/Utils.h"
#include "../includes/Encode.h"
#include <cstdio>
#include <iostream>
#include <sys/stat.h>
#include <fstream>
#include <algorithm>

/**
 *
 * @param ctus
 */
void Analyzer::storeDistributionOfMv(std::vector<CodingTreeUnit *> ctus, std::string log_path) {
    greater_0_flag_sum = greater_1_flag_sum = sign_flag_sum = mvd_code_sum = warping_patch_num = translation_patch_num = 0;
    mvd_warping_code_sum = mvd_translation_code_sum = 0;
    merge_counter = spatial_counter = 0;
    code_sum = 0;
    intra_counter = 0;
    patch_num = 0;
    max_merge_mv_diff_x = 0;
    max_merge_mv_diff_y = 0;

    for(auto ctu : ctus){
        storeDistributionOfMv(ctu);
    }

    log_path = log_path + "/log" + file_suffix;
    mkdir((log_path).c_str(), 0744);

    FILE *fp = std::fopen((log_path + "/mvd_distribution" + file_suffix + ".csv").c_str(), "w");
    for(auto x : mvd_counter){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((log_path + "/mvd_distribution_x" + file_suffix + ".csv").c_str(), "w");
    for(auto x : mvd_counter_x){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((log_path + "/mvd_distribution_y" + file_suffix + ".csv").c_str(), "w");
    for(auto x : mvd_counter_y){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((log_path + "/MV_distribution" + file_suffix + ".csv").c_str(), "w");
    for(auto x : MV_counter){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((log_path + "/mvd_greater_0_flag_distribution" + file_suffix + ".csv").c_str(), "w");
    for(auto x : greater_0_flag_counter){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((log_path + "/mvd_greater_1_flag_distribution" + file_suffix + ".csv").c_str(), "w");
    for(auto x : greater_1_flag_counter){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((log_path + "/mvd_result" + file_suffix + ".txt").c_str(), "w");
    fprintf(fp, "code_sum              :%d\n", code_sum);
    fprintf(fp, "patch num             :%d\n", patch_num);
    fprintf(fp, "greater_0_flag        :%d\n", greater_0_flag_sum);
    fprintf(fp, "greater_0_flag entropy:%f\n", getEntropy({greater_0_flag_counter[0], greater_0_flag_counter[1]}));
    fprintf(fp, "greater_1_flag        :%d\n", greater_1_flag_sum);
    fprintf(fp, "greater_1_flag entropy:%f\n", getEntropy({greater_1_flag_counter[0], greater_1_flag_counter[1]}));
    fprintf(fp, "sign_flag             :%d\n", sign_flag_sum);
    fprintf(fp, "mvd_code              :%d\n", mvd_code_sum);
    fprintf(fp, "warping_code          :%d\n", mvd_warping_code_sum);
    fprintf(fp, "warping_patch         :%d\n", warping_patch_num);
    fprintf(fp, "translation_code      :%d\n", mvd_translation_code_sum);
    fprintf(fp, "translation_patch     :%d\n", translation_patch_num);
    fprintf(fp, "Spatial_patch         :%d\n", spatial_counter);
    fprintf(fp, "merge_patch           :%d\n", merge_counter);
    fprintf(fp, "merge_flag_entropy    :%f\n", getEntropy({merge_flag_counter[0], merge_flag_counter[1]}));

    if(INTRA_MODE) {
        fprintf(fp, "intra_patch           :%d\n", intra_counter);
        fprintf(fp, "intra_flag_entropy    :%f\n", getEntropy({intra_flag_counter[0], intra_flag_counter[1]}));
    }

    fclose(fp);
}


/**
 *
 * @param ctu
 */
void Analyzer::storeDistributionOfMv(CodingTreeUnit *ctu) {
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr){
        if(INTRA_MODE) code_sum += (1 + ctu->code_length + 1);
        else code_sum += (1 + ctu->code_length);
        patch_num++;

        if(ctu->method != MV_CODE_METHOD::MERGE && ctu->method != MV_CODE_METHOD::INTRA){
            if(ctu->translation_flag){
                int x_ = (int)abs(((ctu->mv1).x * 4));
                int y_ = (int)abs(((ctu->mv1).y * 4));
                MV_counter[x_]++;
                MV_counter[y_]++;
                int x = (ctu->mvds_x)[0];
                mvd_counter_x[x]++;
                int y = (ctu->mvds_y)[0];
                mvd_counter_y[y]++;

                mvd_counter[x]++;
                mvd_counter[y]++;

                greater_0_flag_counter[(int)(ctu->flags_code_sum.getXGreater0Flag()[0])]++;
                greater_0_flag_counter[(int)(ctu->flags_code_sum.getYGreater0Flag()[0])]++;

                if(ctu->flags_code_sum.getXGreater0Flag()[0]) {
                    greater_1_flag_counter[(int)(ctu->flags_code_sum.getXGreater1Flag()[0])]++;
                }
                if(ctu->flags_code_sum.getYGreater0Flag()[0]) {
                    greater_1_flag_counter[(int)(ctu->flags_code_sum.getYGreater1Flag()[0])]++;
                }

                mvd_translation_code_sum += ctu->flags_code_sum.getMvdCodeLength();
            }else{
                for(int i = 0 ; i < 3 ; i++) {
                    int x = (ctu->mvds_x)[i];
                    mvd_counter_x[x]++;
                    int y = (ctu->mvds_y)[i];
                    mvd_counter_y[y]++;
                    mvd_counter[x]++;
                    mvd_counter[y]++;

                    if(ctu->flags_code_sum.getXGreater0Flag()[i]) {
                        greater_1_flag_counter[(int)(ctu->flags_code_sum.getXGreater1Flag()[i])]++;
                    }
                    if(ctu->flags_code_sum.getYGreater0Flag()[i]) {
                        greater_1_flag_counter[(int)(ctu->flags_code_sum.getYGreater1Flag()[i])]++;
                    }

                    greater_0_flag_counter[(int)(ctu->flags_code_sum.getXGreater0Flag()[i])]++;
                    greater_0_flag_counter[(int)(ctu->flags_code_sum.getYGreater0Flag()[i])]++;
                }
                mvd_warping_code_sum += ctu->flags_code_sum.getMvdCodeLength();
            }

            greater_0_flag_sum += ctu->flags_code_sum.getGreater0FlagCodeLength();
            greater_1_flag_sum += ctu->flags_code_sum.getGreaterThanOneCodeLength();
            sign_flag_sum += ctu->flags_code_sum.getSignFlagCodeLength();
            mvd_code_sum += ctu->flags_code_sum.getMvdCodeLength();
            spatial_counter++;

            merge_flag_counter[0]++;
            intra_flag_counter[0]++;
        }else if(ctu->method == MV_CODE_METHOD::MERGE){
            merge_counter++;
            merge_flag_counter[1]++;
            intra_flag_counter[0]++;
        }else if(ctu->method == MV_CODE_METHOD::INTRA){
            intra_counter++;
            merge_flag_counter[0]++;
            intra_flag_counter[1]++;
        }

        if(ctu->translation_flag) translation_patch_num++;
        else warping_patch_num++;

        return;
    }

    if(ctu->node1 != nullptr) storeDistributionOfMv(ctu->node1);
    if(ctu->node2 != nullptr) storeDistributionOfMv(ctu->node2);
    if(ctu->node3 != nullptr) storeDistributionOfMv(ctu->node3);
    if(ctu->node4 != nullptr) storeDistributionOfMv(ctu->node4);
    code_sum += 1;
}

Analyzer::Analyzer(const std::string &fileSuffix) : file_suffix(fileSuffix) {}

/**
 * @fn void Analyzer::storeMarkdownFile(double psnr, std::string log_path)
 * @brief Markdownとして結果を書き出す
 * @param psnr PSNR
 * @param log_path ログのパス
 */
void Analyzer::storeMarkdownFile(double psnr, std::string log_path) {
    log_path = log_path + "/log" + file_suffix;

    extern int qp;
    FILE *fp = fopen((log_path + "/result.md").c_str(), "w");
    fprintf(fp, "|%d|%f|%d|%f|%d|\n", qp, getLambdaPred(qp, 1.0), code_sum, psnr, warping_patch_num + translation_patch_num);
    fclose(fp);
}

/**
 * @fn void Analyzer::storeCsvFileWithStream(std::ofstream &ofs, double psnr)
 * @breif 外部からOutputStreamを受け取って，そこにCSV形式で書き出す
 * @param ofs OutputStream
 * @param psnr PSNR値
 */
void Analyzer::storeCsvFileWithStream(std::ofstream &ofs, double psnr) {
    extern int qp;
    int tmp_code_sum = code_sum - (int)ceil(greater_0_flag_sum * (1.0-getEntropy({greater_0_flag_counter[0], greater_0_flag_counter[1]})));
    std::cout << (int)ceil(greater_0_flag_sum * getEntropy({greater_0_flag_counter[0], greater_0_flag_counter[1]}))<< std::endl;
    std::cout << "tmp_code_sum:" << tmp_code_sum << std::endl;
    tmp_code_sum = tmp_code_sum - (int)ceil(greater_1_flag_sum * (1.0 - getEntropy({greater_1_flag_counter[0], greater_1_flag_counter[1]})));

#if MERGE_MODE
    std::cout << (int)ceil(patch_num * getEntropy({merge_flag_counter[0], merge_flag_counter[1]})) << std::endl;
    std::cout << "tmp_code_sum:" << tmp_code_sum << std::endl;
    tmp_code_sum = tmp_code_sum - (int)ceil(patch_num * (1.0 - getEntropy({merge_flag_counter[0], merge_flag_counter[1]})));
#endif

#if INTRA_MODE
    std::cout << "tmp_code_sum:" << tmp_code_sum << std::endl;
    if(INTRA_MODE) tmp_code_sum = tmp_code_sum - (int)ceil(intra_counter * getEntropy({intra_flag_counter[0], intra_flag_counter[1]}));
#endif

    ofs << qp << "," << getLambdaPred(qp, 1.0) << "," << code_sum << "," << tmp_code_sum << "," << psnr << "," << patch_num << "," << spatial_counter << "," << merge_counter << "," << intra_counter << std::endl;
}

void Analyzer::storeMergeMvLog(std::vector<CodingTreeUnit*> ctus, std::string log_path) {
    std::ofstream ofs;
    ofs.open(log_path);

    for(const auto ctu : ctus) {
        storeMergeMvLog(ctu, ofs);
    }

    ofs << std::endl;
    ofs << "max_diff(x):" << max_merge_mv_diff_x << " max_diff(y):" << max_merge_mv_diff_y << std::endl;

    ofs.close();
}

void Analyzer::storeMergeMvLog(CodingTreeUnit *ctu, std::ofstream &ofs) {
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        if(ctu->method != MV_CODE_METHOD::MERGE) return;

        if(ctu->translation_flag) {
            ofs << "merge(transaltion)" << std::endl;
            ofs << "original:" << ctu->original_mv1 << " merged_mv:" << ctu->mv1 << std::endl;
            ofs << std::endl;

            max_merge_mv_diff_x = std::max(max_merge_mv_diff_x, std::fabs(ctu->original_mv1.x - ctu->mv1.x));
            max_merge_mv_diff_y = std::max(max_merge_mv_diff_y, std::fabs(ctu->original_mv1.y - ctu->mv1.y));
        } else {
            ofs << "merge(warping)" << std::endl;
            ofs << "original(1):" << ctu->original_mv1 << " merged_mv(1):" << ctu->mv1 << std::endl;
            ofs << "original(2):" << ctu->original_mv2 << " merged_mv(2):" << ctu->mv2 << std::endl;
            ofs << "original(3):" << ctu->original_mv3 << " merged_mv(3):" << ctu->mv3 << std::endl;
            ofs << std::endl;

            max_merge_mv_diff_x = std::max(max_merge_mv_diff_x, std::fabs(ctu->original_mv1.x - ctu->mv1.x));
            max_merge_mv_diff_x = std::max(max_merge_mv_diff_x, std::fabs(ctu->original_mv2.x - ctu->mv2.x));
            max_merge_mv_diff_x = std::max(max_merge_mv_diff_x, std::fabs(ctu->original_mv3.x - ctu->mv3.x));
            max_merge_mv_diff_y = std::max(max_merge_mv_diff_y, std::fabs(ctu->original_mv1.y - ctu->mv1.y));
            max_merge_mv_diff_y = std::max(max_merge_mv_diff_y, std::fabs(ctu->original_mv2.y - ctu->mv2.y));
            max_merge_mv_diff_y = std::max(max_merge_mv_diff_y, std::fabs(ctu->original_mv3.y - ctu->mv3.y));
        }
        return;
    }

    storeMergeMvLog(ctu->node1, ofs);
    storeMergeMvLog(ctu->node2, ofs);
    storeMergeMvLog(ctu->node3, ofs);
    storeMergeMvLog(ctu->node4, ofs);
}
