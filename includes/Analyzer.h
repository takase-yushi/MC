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
    void storeDistributionOfMv(std::vector<CodingTreeUnit*> ctus);
    Analyzer(const std::string &fileSuffix);

private:
    void storeDistributionOfMv(CodingTreeUnit *ctu);

    // mvdのカウンター
    std::map<int, int> mvd_counter;
    std::map<int, int> mvd_counter_x;
    std::map<int, int> mvd_counter_y;

    // greater_0フラグの頻度
    std::map<int, int> greater_0_flag_counter;
    int greater_0_flag_sum;

    // greater_1フラグの頻度
    std::map<int, int> greater_1_flag_counter;
    int greater_1_flag_sum;

    // サインフラグの頻度
    std::map<int, int> sign_flag_counter;
    int sign_flag_sum;

    int mvd_code_sum;

    // ファイルの最後につける値
    std::string file_suffix;

};

class FlagsCodeSum {
    // 動きベクトル差分の絶対値が0より大きいのか？
    int greater_0_flag_code_length;

    // 動きベクトル差分の絶対値が1より大きいのか？
    int greater_than_one_code_length;

    // 正負の判定
    int sign_flag_code_length;

    int mvd_code_length;

    bool x_greater_0_flag;
    bool y_greater_0_flag;
    bool x_greater_1_flag;
    bool y_greater_1_flag;
    bool x_sign_flag;
    bool y_sign_flag;
public:
    void setXGreater0Flag(bool xGreater0Flag);

    void setYGreater0Flag(bool yGreater0Flag);

    void setXGreater1Flag(bool xGreater1Flag);

    void setYGreater1Flag(bool yGreater1Flag);

    void setXSignFlag(bool xSignFlag);

    void setYSignFlag(bool ySignFlag);

public:
    FlagsCodeSum(int greater0FlagCode, int greaterThanOneCode, int signFlagCode, int mvdCodeLength);
    void countGreater0Code();

    void countGreater1Code();

    void countSignFlagCode();

    void addMvdCodeLength(int len);

    int getGreater0FlagCodeLength() const;

    int getGreaterThanOneCodeLength() const;

    int getSignFlagCodeLength() const;

    int getMvdCodeLength() const;
};

#endif //ENCODER_ANALYZER_H
