//
// Created by kasph on 2019/05/15.
//

#ifndef ENCODER_COLLOCATEDMVTREE_H
#define ENCODER_COLLOCATEDMVTREE_H


#include <opencv2/core/types.hpp>

class CollocatedMvTree {
public:

    virtual ~CollocatedMvTree();
    cv::Point2i mv_decimal;
    cv::Point2i mv_integer;
    CollocatedMvTree* leftNode;
    CollocatedMvTree* rightNode;
};


#endif //ENCODER_COLLOCATEDMVTREE_H
