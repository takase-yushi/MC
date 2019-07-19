//
// Created by Kamiya Keisuke on 2019-07-18.
//

#ifndef ENCODER_ANALYZER_H
#define ENCODER_ANALYZER_H


#include <vector>
#include <map>
#include "CodingTreeUnit.h"

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

    // フラグの頻度
    std::map<int, int> greater_0_flag_counter;
    std::map<int, int> greater_1_flag_counter;

    std::string file_suffix;

};


#endif //ENCODER_ANALYZER_H
