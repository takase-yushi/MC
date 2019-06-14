//
// Created by kasph on 2019/05/23.
//

#ifndef ENCODER_IMAGEUTIL_H
#define ENCODER_IMAGEUTIL_H

#include "Utils.h"

cv::Mat getResidualImage(const cv::Mat &target_image, const cv::Mat &predict_image, int k = 1);
double getTriangleResidual(const cv::Mat ref_image, const cv::Mat &target_image, Point3Vec &triangle, std::vector<cv::Point2f> mv);
std::vector<std::vector<cv::Mat>> getRefImages(const cv::Mat ref_image, const cv::Mat gauss_ref_image);

#endif //ENCODER_IMAGEUTIL_H
