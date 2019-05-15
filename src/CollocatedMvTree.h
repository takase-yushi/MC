//
// Created by kasph on 2019/05/15.
//

#ifndef ENCODER_COLLOCATEDMVTREE_H
#define ENCODER_COLLOCATEDMVTREE_H


#include <opencv2/core/types.hpp>

class CollocatedMvTree {
public:

    CollocatedMvTree* leftNode;

    virtual ~CollocatedMvTree();

    cv::Point2f mv;
    CollocatedMvTree* rightNode;
};


#endif //ENCODER_COLLOCATEDMVTREE_H
