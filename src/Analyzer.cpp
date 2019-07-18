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

    FILE *fp = std::fopen((getProjectDirectory(OS) + "/mvd_distribution.csv").c_str(), "w");
    for(auto x : counter){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((getProjectDirectory(OS) + "/mvd_distribution_x.csv").c_str(), "w");
    for(auto x : counter_x){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((getProjectDirectory(OS) + "/mvd_distribution_y.csv").c_str(), "w");
    for(auto x : counter_y){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);
}


/**
 *
 * @param ctu
 */
void Analyzer::storeDistributionOfMv(CodingTreeUnit *ctu) {
    if(ctu->leftNode == nullptr && ctu->rightNode == nullptr && (ctu->mvds).size() > 0){
        if(ctu->parallel_flag){
            int x = ctu->mvds[0].x;
            int y = ctu->mvds[0].y;
            counter[x]++;
            counter[y]++;
        }else{
            int x = ctu->mvds[0].x;
            int y = ctu->mvds[0].y;
            counter[x]++;
            counter[y]++;
            x = ctu->mvds[1].x;
            y = ctu->mvds[1].y;
            counter[x]++;
            counter[y]++;
            x = ctu->mvds[2].x;
            y = ctu->mvds[2].y;
            counter[x]++;
            counter[y]++;
        }
        return;
    }

    if(ctu->leftNode != nullptr) storeDistributionOfMv(ctu->leftNode);
    if(ctu->rightNode != nullptr) storeDistributionOfMv(ctu->rightNode);
}

Analyzer::Analyzer(const std::string &fileSuffix) : file_suffix(fileSuffix) {}
