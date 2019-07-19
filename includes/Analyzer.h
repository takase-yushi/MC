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

private:
    void storeDistributionOfMv(CodingTreeUnit *ctu);

    std::map<int, int> counter;
    std::map<int, int> counter_x;
    std::map<int, int> counter_y;
    std::string file_suffix;
public:
    Analyzer(const std::string &fileSuffix);
};


#endif //ENCODER_ANALYZER_H
