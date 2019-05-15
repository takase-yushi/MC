//
// Created by kasph on 2019/05/15.
//

#ifndef ENCODER_COLLOCATEDMVTREE_H
#define ENCODER_COLLOCATEDMVTREE_H


#include <opencv2/core/types.hpp>

class CollocatedMvTree {
    CollocatedMvTree* leftNode;
    CollocatedMvTree* rightNode;
public:
    virtual ~CollocatedMvTree();

private:
    cv::Point2f mv;
};


#endif //ENCODER_COLLOCATEDMVTREE_H
