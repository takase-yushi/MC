//
// Created by kasph on 2019/05/23.
//

#ifndef ENCODER_IMAGEUTIL_H
#define ENCODER_IMAGEUTIL_H

#include "Utils.h"
#include "CodingTreeUnit.h"

cv::Mat getResidualImage(const cv::Mat &target_image, const cv::Mat &predict_image, int k = 1);
double getTriangleResidual(const cv::Mat ref_image, const cv::Mat &target_image, Point3Vec &triangle, std::vector<cv::Point2f> mv, const std::vector<cv::Point2f> &in_triangle_pixels);
std::vector<std::vector<cv::Mat>> getRefImages(const cv::Mat ref_image, const cv::Mat gauss_ref_image);
std::vector<std::vector<cv::Mat>> getTargetImages(const cv::Mat target_image);
EXPAND_ARRAY_TYPE getExpandImages(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, int expand);
void freeExpandImages(EXPAND_ARRAY_TYPE expand_images, int expand, int filter_num, int step_num, int rows, int cols);
cv::Mat getReconstructionDivisionImage(cv::Mat image, std::vector<CodingTreeUnit *> ctu, int block_size_x, int block_size_y);
void freeImage(unsigned char **image, cv::Size image_size, int expansion_size);
unsigned char ** getExpansionImage(cv::Mat image, int k, int expansion_size);
cv::Mat getExpansionMatImage(cv::Mat &image, int k, int expansion_size);

#endif //ENCODER_IMAGEUTIL_H
