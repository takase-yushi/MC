//
// Created by kasph on 2019/08/05.
//

#ifndef ENCODER_DECODER_H
#define ENCODER_DECODER_H


#include <opencv2/core/types.hpp>
#include <set>
#include "Utils.h"
#include "CollocatedMvTree.h"
#include "GaussResult.h"

class Decoder {
public:
    void initTriangle(int _block_size_x, int _block_size_y, int _divide_steps, int _qp, int divide_flag);
    Decoder(const cv::Mat &ref_image, const cv::Mat &targetImage);
    void reconstructionTriangle(std::vector<CodingTreeUnit*> ctus);
    cv::Mat getReconstructionTriangleImage();

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
    std::vector<CodingTreeUnit *> decode_ctus;

private:

    std::vector<cv::Point2f> corners;
    std::vector<std::set<int>> neighbor_vtx;
    std::vector<std::set<int>> covered_triangle;
    std::vector<std::set<int>> same_corner_list;
    std::vector<std::pair<Triangle, int> > triangles;
    std::vector<std::vector<int>> corner_flag;
    std::vector<bool> delete_flag;
    std::vector<bool> isCodedTriangle;
    std::vector<std::vector<CollocatedMvTree *>> previousMvList;
    std::vector<GaussResult> triangle_info;

    cv::Mat hevc_expansion_ref;

    void addNeighborVertex(int p1_idx, int p2_idx, int p3_idx);
    void addCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_no);
    void reconstructionTriangle(CodingTreeUnit *ctu, CodingTreeUnit *decode_ctu, Point3Vec triangle, int type);
    int getCornerIndex(cv::Point2f p);
    int insertTriangle(int p1_idx, int p2_idx, int p3_idx, int type);
    void getModeImage(CodingTreeUnit *ctu, cv::Mat &out, const std::vector<std::vector<int>> &area_flag);
};

#endif //ENCODER_DECODER_H
