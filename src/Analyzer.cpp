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

/**
 *
 * @param ctus
 */
void Analyzer::storeDistributionOfMv(std::vector<CodingTreeUnit *> ctus, std::string log_path) {
    greater_0_flag_sum = greater_1_flag_sum = sign_flag_sum = mvd_code_sum = warping_patch_num = parallel_patch_num = 0;
    mvd_warping_code_sum = mvd_parallel_code_sum = 0;
    merge_counter = spatial_counter = 0;
    code_sum = 0;

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
    fprintf(fp, "greater_0_flag        :%d\n", greater_0_flag_sum);
    fprintf(fp, "greater_0_flag entropy:%f\n", getEntropy({greater_0_flag_counter[0], greater_0_flag_counter[1]}));
    fprintf(fp, "greater_1_flag        :%d\n", greater_1_flag_sum);
    fprintf(fp, "greater_1_flag entropy:%f\n", getEntropy({greater_1_flag_counter[0], greater_1_flag_counter[1]}));
    fprintf(fp, "sign_flag             :%d\n", sign_flag_sum);
    fprintf(fp, "mvd_code              :%d\n", mvd_code_sum);
    fprintf(fp, "warping_code          :%d\n", mvd_warping_code_sum);
    fprintf(fp, "warping_patch         :%d\n", warping_patch_num);
    fprintf(fp, "parallel_code         :%d\n", mvd_parallel_code_sum);
    fprintf(fp, "parallel_patch        :%d\n", parallel_patch_num);
    fprintf(fp, "Spatial_patch         :%d\n", spatial_counter);
    fprintf(fp, "merge_patch           :%d\n", merge_counter);

    fclose(fp);
}


/**
 *
 * @param ctu
 */
void Analyzer::storeDistributionOfMv(CodingTreeUnit *ctu) {
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr){
        code_sum += (1 + ctu->code_length);
        if(ctu->method != MV_CODE_METHOD::MERGE){
            if(ctu->parallel_flag){
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

                mvd_parallel_code_sum += ctu->flags_code_sum.getMvdCodeLength();
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
        }else{
            merge_counter++;
        }

        if(ctu->parallel_flag) parallel_patch_num++;
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

void Analyzer::storeMarkdownFile(double psnr, std::string log_path) {
    log_path = log_path + "/log" + file_suffix;

    extern int qp;
    FILE *fp = fopen((log_path + "/result.md").c_str(), "w");
    fprintf(fp, "|%d|%f|%d|%f|%d|\n", qp, getLambdaPred(qp, 1.0), code_sum, psnr, warping_patch_num + parallel_patch_num);
    fclose(fp);
}


void Analyzer::storeMarkdownFile(std::ofstream &ofs, double psnr) {
    extern int qp;

    ofs << qp << "," << getLambdaPred(qp, 1.0) << "," << code_sum << "," << psnr << std::endl;
}
