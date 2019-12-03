//
// Created by kasph on 2019/05/05.
//

#ifndef ENCODER_CODINGTREEUNIT_H
#define ENCODER_CODINGTREEUNIT_H


#include <opencv2/core/types.hpp>
#include "FlagsCodeSum.h"


enum MV_CODE_METHOD {
    SPATIAL,
    Collocated,
    MERGE,
    MERGE_Collocated,
    INTRA,
    MERGE2
};


class CodingTreeUnit {
public:

    virtual ~CodingTreeUnit();

    int code_length;
    bool split_cu_flag;
    bool translation_flag;
    CodingTreeUnit *node1, *node2, *node3, *node4;
    CodingTreeUnit *parentNode;
    int triangle_index;
    int square_index;
    cv::Point2f mv1, mv2, mv3;
    std::vector<int> mvds_x, mvds_y;
    std::vector<cv::Point2f> mvds;
    int ref_triangle_idx;
    int ref_square_idx;
    std::vector<int> original_mvds_x, original_mvds_y;
    double error_bm, error_newton;
    cv::Point2f collocated_mv;
    MV_CODE_METHOD method;
    FlagsCodeSum flags_code_sum;
    std::vector<bool> x_greater_0_flag, y_greater_0_flag;
    std::vector<bool> x_greater_1_flag, y_greater_1_flag;
    std::vector<bool> x_sign_flag, y_sign_flag;

    cv::Point2f merge_triangle_ref_vector;
    cv::Point2f original_mv1, original_mv2, original_mv3;

    bool share_flag[3];
};


#endif //ENCODER_CODINGTREEUNIT_H
