//
// Created by kasph on 2019/05/05.
//

#ifndef ENCODER_CODINGTREEUNIT_H
#define ENCODER_CODINGTREEUNIT_H


#include <opencv2/core/types.hpp>

class CodingTreeUnit {
public:
    virtual ~CodingTreeUnit();

    bool split_cu_flag1;
    bool split_cu_flag2;
    CodingTreeUnit *leftNode;
    CodingTreeUnit *rightNode;
    CodingTreeUnit *parentNode;
    int depth;
    int position; // left or right
    int triangle_index;
    cv::Point2f mv1, mv2, mv3;
    cv::Point2f collocated_mv;
};


#endif //ENCODER_CODINGTREEUNIT_H