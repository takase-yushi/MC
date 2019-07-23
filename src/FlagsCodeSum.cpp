//
// Created by kasph on 2019/07/21.
//

#include <vector>
#include "../includes/FlagsCodeSum.h"


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

FlagsCodeSum::FlagsCodeSum() {}
