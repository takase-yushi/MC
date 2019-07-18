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
    if(ctu->leftNode == nullptr && ctu->rightNode == nullptr){
        if(ctu->parallel_flag){
            int x = (ctu->mvds_x)[0];
            counter_x[x]++;
            int y = (ctu->mvds_y)[0];
            counter_y[y]++;

            counter[x]++;
            counter[y]++;
        }else{
            for(int i = 0 ; i < 3 ; i++) {
                int x = (ctu->mvds_x)[i];
                counter_x[x]++;
                int y = (ctu->mvds_y)[i];
                counter_y[y]++;
                counter[x]++;
                counter[y]++;
            }
        }
        return;
    }

    if(ctu->leftNode != nullptr) storeDistributionOfMv(ctu->leftNode);
    if(ctu->rightNode != nullptr) storeDistributionOfMv(ctu->rightNode);
}

Analyzer::Analyzer(const std::string &fileSuffix) : file_suffix(fileSuffix) {}
