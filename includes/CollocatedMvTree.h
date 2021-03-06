//
// Created by kasph on 2019/05/15.
//

#ifndef ENCODER_COLLOCATEDMVTREE_H
#define ENCODER_COLLOCATEDMVTREE_H


#include <opencv2/core/types.hpp>

class CollocatedMvTree {
public:

    virtual ~CollocatedMvTree();

    cv::Point2f mv1;
    cv::Point2f mv2;
    cv::Point2f mv3;
    CollocatedMvTree *node1, *node2, *node3, *node4;
};


#endif //ENCODER_COLLOCATEDMVTREE_H
