//
// Created by Kamiya Keisuke on 2019-07-18.
//

#include "../includes/Analyzer.h"
#include "../includes/Utils.h"
#include <cstdio>
#include <iostream>

/**
 *
 * @param ctus
 */
void Analyzer::storeDistributionOfMv(std::vector<CodingTreeUnit *> ctus, std::string log_path) {
    greater_0_flag_sum = greater_1_flag_sum = sign_flag_sum = mvd_code_sum = warping_patch_num = parallel_patch_num = 0;
    mvd_warping_code_sum = mvd_parallel_code_sum = 0;

    for(auto ctu : ctus){
        storeDistributionOfMv(ctu);
    }

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
    fprintf(fp, "greater_0_flag:%d\n", greater_0_flag_sum);
    fprintf(fp, "greater_1_flag:%d\n", greater_1_flag_sum);
    fprintf(fp, "sign_flag     :%d\n", sign_flag_sum);
    fprintf(fp, "mvd_code      :%d\n", mvd_code_sum);
    fprintf(fp, "warping_code  :%d\n", mvd_warping_code_sum);
    fprintf(fp, "warping_patch :%d\n", warping_patch_num);
    fprintf(fp, "parallel_code :%d\n", mvd_parallel_code_sum);
    fprintf(fp, "parallel_patch:%d\n", parallel_patch_num);

    fclose(fp);
}


/**
 *
 * @param ctu
 */
void Analyzer::storeDistributionOfMv(CodingTreeUnit *ctu) {
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr){
        if(ctu->method != MV_CODE_METHOD::MERGE){
            if(ctu->parallel_flag){
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
                }
                mvd_warping_code_sum += ctu->flags_code_sum.getMvdCodeLength();
            }

            greater_0_flag_sum += ctu->flags_code_sum.getGreater0FlagCodeLength();
            greater_1_flag_sum += ctu->flags_code_sum.getGreaterThanOneCodeLength();
            sign_flag_sum += ctu->flags_code_sum.getSignFlagCodeLength();
            mvd_code_sum += ctu->flags_code_sum.getMvdCodeLength();
        }

        if(ctu->parallel_flag) parallel_patch_num++;
        else warping_patch_num++;

        return;
    }

    if(ctu->node1 != nullptr) storeDistributionOfMv(ctu->node1);
    if(ctu->node2 != nullptr) storeDistributionOfMv(ctu->node2);
    if(ctu->node3 != nullptr) storeDistributionOfMv(ctu->node3);
    if(ctu->node4 != nullptr) storeDistributionOfMv(ctu->node4);
}

Analyzer::Analyzer(const std::string &fileSuffix) : file_suffix(fileSuffix) {}
