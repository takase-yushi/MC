//
// Created by kasph on 2019/05/23.
//

#ifndef ENCODER_IMAGEUTIL_H
#define ENCODER_IMAGEUTIL_H

#include "Utils.h"

cv::Mat getResidualImage(const cv::Mat &target_image, const cv::Mat &predict_image);
double getTriangleResidual(const cv::Mat ref_image, const cv::Mat &target_image, Point3Vec &triangle, std::vector<cv::Point2f> mv);

double getSquaredError(cv::Mat& ref_image, cv::Mat& target_image, Point3Vec& triangle, std::vector<cv::Point2f>& mv);

#endif //ENCODER_IMAGEUTIL_H
