//
// Created by kasph on 2019/08/05.
//

#ifndef ENCODER_DECODER_H
#define ENCODER_DECODER_H


#include <opencv2/core/types.hpp>
#include <set>
#include "Utils.h"
#include "CollocatedMvTree.h"

class Decoder {
public:
    void initTriangle(int _block_size_x, int _block_size_y, int _divide_steps, int _qp, int divide_flag);
    Decoder(const cv::Mat &ref_image, const cv::Mat &targetImage);

private:
    int block_size_x, block_size_y;
    int qp;
    int block_num_x;
    int block_num_y;
    int divide_steps;
    int coded_picture_num;
    int image_width, image_height;
    cv::Mat ref_image, target_image;
    cv::Mat hevc_ref_image;

private:

    std::vector<cv::Point2f> corners;
    std::vector<std::set<int>> neighbor_vtx;
    std::vector<std::set<int>> covered_triangle;
    std::vector<std::set<int>> same_corner_list;
    std::vector<std::pair<Triangle, int> > triangles;
    std::vector<std::vector<int>> corner_flag;
    std::vector<bool> delete_flag;
    std::vector<bool> isCodedTriangle;
    std::vector<std::vector<CollocatedMvTree*>> previousMvList;

    cv::Mat hevc_expansion_ref;
};


#endif //ENCODER_DECODER_H
