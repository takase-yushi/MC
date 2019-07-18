//
// Created by kasph on 2019/05/05.
//

#ifndef ENCODER_CODINGTREEUNIT_H
#define ENCODER_CODINGTREEUNIT_H


#include <opencv2/core/types.hpp>

class CodingTreeUnit {
public:
    virtual ~CodingTreeUnit();

    int code_length;
    bool split_cu_flag;
    bool parallel_flag;
    CodingTreeUnit *leftNode;
    CodingTreeUnit *rightNode;
    CodingTreeUnit *parentNode;
    int triangle_index;
    cv::Point2f mv1, mv2, mv3;
    std::vector<int> mvds_x, mvds_y;
    std::vector<int> original_mvds_x, original_mvds_y;
    double error_bm, error_newton;
    cv::Point2f collocated_mv;
};


#endif //ENCODER_CODINGTREEUNIT_H
