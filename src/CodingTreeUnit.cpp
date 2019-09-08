//
// Created by kasph on 2019/05/05.
//

#include <iostream>
#include "../includes/CodingTreeUnit.h"

CodingTreeUnit::~CodingTreeUnit() {
    delete node1;
    delete node2;
    delete node3;
    delete node4;
    std::vector<int>().swap(mvds_x);
    std::vector<int>().swap(mvds_y);
    std::vector<cv::Point2f>().swap(mvds);
    std::vector<int>().swap(original_mvds_x);
    std::vector<int>().swap(original_mvds_y);
    std::vector<bool>().swap(x_greater_0_flag);
    std::vector<bool>().swap(y_greater_0_flag);
    std::vector<bool>().swap(x_greater_1_flag);
    std::vector<bool>().swap(y_greater_1_flag);
    std::vector<bool>().swap(x_sign_flag);
    std::vector<bool>().swap(y_sign_flag);
}
