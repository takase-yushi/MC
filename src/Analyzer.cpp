//
// Created by Kamiya Keisuke on 2019-07-18.
//

#include "../includes/Analyzer.h"
#include "../includes/Utils.h"
#include <cstdio>

/**
 *
 * @param ctus
 */
void Analyzer::storeDistributionOfMv(std::vector<CodingTreeUnit *> ctus) {
    for(auto ctu : ctus){
        storeDistributionOfMv(ctu);
    }

    FILE *fp = std::fopen((getProjectDirectory(OS) + "/mvd_distribution" + file_suffix + ".csv").c_str(), "w");
    for(auto x : mvd_counter){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((getProjectDirectory(OS) + "/mvd_distribution_x" + file_suffix + ".csv").c_str(), "w");
    for(auto x : mvd_counter_x){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((getProjectDirectory(OS) + "/mvd_distribution_y" + file_suffix + ".csv").c_str(), "w");
    for(auto x : mvd_counter_y){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
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
            }else{
                for(int i = 0 ; i < 3 ; i++) {
                    int x = (ctu->mvds_x)[i];
                    mvd_counter_x[x]++;
                    int y = (ctu->mvds_y)[i];
                    mvd_counter_y[y]++;
                    mvd_counter[x]++;
                    mvd_counter[y]++;
                }
            }
        }
        return;
    }

    if(ctu->node1 != nullptr) storeDistributionOfMv(ctu->node1);
    if(ctu->node2 != nullptr) storeDistributionOfMv(ctu->node2);
    if(ctu->node3 != nullptr) storeDistributionOfMv(ctu->node3);
    if(ctu->node4 != nullptr) storeDistributionOfMv(ctu->node4);
}

Analyzer::Analyzer(const std::string &fileSuffix) : file_suffix(fileSuffix) {}

FlagsCodeSum::FlagsCodeSum(int greater0FlagCode, int greaterThanOneCode, int signFlagCode) : greater_0_flag_code(
        greater0FlagCode), greater_than_one_code(greaterThanOneCode), sign_flag_code(signFlagCode) {}

void FlagsCodeSum::countGreater0Code() {
    greater_0_flag_code++;
}

void FlagsCodeSum::countGreater1Code() {
    greater_than_one_code++;
}

void FlagsCodeSum::countSignFlagCode() {
    sign_flag_code++;
}

int FlagsCodeSum::getGreater0FlagCode() const {
    return greater_0_flag_code;
}

int FlagsCodeSum::getGreaterThanOneCode() const {
    return greater_than_one_code;
}

int FlagsCodeSum::getSignFlagCode() const {
    return sign_flag_code;
}
