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
void Analyzer::storeDistributionOfMv(std::vector<CodingTreeUnit *> ctus) {
    greater_0_flag_sum = greater_1_flag_sum = sign_flag_sum = mvd_code_sum = 0;

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

    fp = std::fopen((getProjectDirectory(OS) + "/mvd_greater_0_flag_distribution" + file_suffix + ".csv").c_str(), "w");
    for(auto x : greater_0_flag_counter){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((getProjectDirectory(OS) + "/mvd_greater_1_flag_distribution" + file_suffix + ".csv").c_str(), "w");
    for(auto x : greater_1_flag_counter){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((getProjectDirectory(OS) + "/mvd_result" + file_suffix + ".txt").c_str(), "w");
    fprintf(fp, "greater_0_flag:%d\n", greater_0_flag_sum);
    fprintf(fp, "greater_1_flag:%d\n", greater_1_flag_sum);
    fprintf(fp, "sign_flag     :%d\n", sign_flag_sum);
    fprintf(fp, "mvd_code      :%d\n", mvd_code_sum);

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

                greater_0_flag_counter[(int)(ctu->flags_code_sum->getXGreater0Flag()[0])]++;
                greater_0_flag_counter[(int)(ctu->flags_code_sum->getYGreater0Flag()[0])]++;

                if(ctu->flags_code_sum->getXGreater0Flag()[0]) {
                    greater_1_flag_counter[(int)(ctu->flags_code_sum->getXGreater1Flag()[0])]++;
                }
                if(ctu->flags_code_sum->getYGreater0Flag()[0]) {
                    greater_1_flag_counter[(int)(ctu->flags_code_sum->getYGreater1Flag()[0])]++;
                }
            }else{
                for(int i = 0 ; i < 3 ; i++) {
                    int x = (ctu->mvds_x)[i];
                    mvd_counter_x[x]++;
                    int y = (ctu->mvds_y)[i];
                    mvd_counter_y[y]++;
                    mvd_counter[x]++;
                    mvd_counter[y]++;

                    if(ctu->flags_code_sum->getXGreater0Flag()[i]) {
                        greater_1_flag_counter[(int)(ctu->flags_code_sum->getXGreater1Flag()[i])]++;
                    }
                    if(ctu->flags_code_sum->getYGreater0Flag()[i]) {
                        greater_1_flag_counter[(int)(ctu->flags_code_sum->getYGreater1Flag()[i])]++;
                    }
                }
            }

            greater_0_flag_sum += ctu->flags_code_sum->getGreater0FlagCodeLength();
            greater_1_flag_sum += ctu->flags_code_sum->getGreaterThanOneCodeLength();
            sign_flag_sum += ctu->flags_code_sum->getSignFlagCodeLength();
            mvd_code_sum += ctu->flags_code_sum->getMvdCodeLength();
        }
        return;
    }

    if(ctu->node1 != nullptr) storeDistributionOfMv(ctu->node1);
    if(ctu->node2 != nullptr) storeDistributionOfMv(ctu->node2);
    if(ctu->node3 != nullptr) storeDistributionOfMv(ctu->node3);
    if(ctu->node4 != nullptr) storeDistributionOfMv(ctu->node4);
}

Analyzer::Analyzer(const std::string &fileSuffix) : file_suffix(fileSuffix) {}

FlagsCodeSum::FlagsCodeSum(int greater0FlagCode, int greaterThanOneCode, int signFlagCode, int mvdCodeLength) : greater_0_flag_code_length(
        greater0FlagCode), greater_than_one_code_length(greaterThanOneCode), sign_flag_code_length(signFlagCode), mvd_code_length(mvdCodeLength) {}

void FlagsCodeSum::countGreater0Code() {
    greater_0_flag_code_length++;
}

void FlagsCodeSum::countGreater1Code() {
    greater_than_one_code_length++;
}

void FlagsCodeSum::countSignFlagCode() {
    sign_flag_code_length++;
}

void FlagsCodeSum::addMvdCodeLength(int len){
    mvd_code_length += len;
}

int FlagsCodeSum::getGreater0FlagCodeLength() const {
    return greater_0_flag_code_length;
}

int FlagsCodeSum::getGreaterThanOneCodeLength() const {
    return greater_than_one_code_length;
}

int FlagsCodeSum::getSignFlagCodeLength() const {
    return sign_flag_code_length;
}


int FlagsCodeSum::getMvdCodeLength() const {
    return mvd_code_length;
}

void FlagsCodeSum::setXGreater0Flag(bool xGreater0Flag) {
    x_greater_0_flag.emplace_back(xGreater0Flag);
}

void FlagsCodeSum::setYGreater0Flag(bool yGreater0Flag) {
    y_greater_0_flag.emplace_back(yGreater0Flag);
}

void FlagsCodeSum::setXGreater1Flag(bool xGreater1Flag) {
    x_greater_1_flag.emplace_back(xGreater1Flag);
}

void FlagsCodeSum::setYGreater1Flag(bool yGreater1Flag) {
    y_greater_1_flag.emplace_back(yGreater1Flag);
}

void FlagsCodeSum::setXSignFlag(bool xSignFlag) {
    x_sign_flag.emplace_back(xSignFlag);
}

void FlagsCodeSum::setYSignFlag(bool ySignFlag) {
    y_sign_flag.emplace_back(ySignFlag);
}

const std::vector<bool> &FlagsCodeSum::getXGreater0Flag() const {
    return x_greater_0_flag;
}

const std::vector<bool> &FlagsCodeSum::getYGreater0Flag() const {
    return y_greater_0_flag;
}

const std::vector<bool> &FlagsCodeSum::getXGreater1Flag() const {
    return x_greater_1_flag;
}

const std::vector<bool> &FlagsCodeSum::getYGreater1Flag() const {
    return y_greater_1_flag;
}

const std::vector<bool> &FlagsCodeSum::getXSignFlag() const {
    return x_sign_flag;
}

const std::vector<bool> &FlagsCodeSum::getYSignFlag() const {
    return y_sign_flag;
}
