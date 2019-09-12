/**
 * @file ME.h
 * @brief ME.cpp用のヘッダファイル
 * @author Keisuke KAMIYA
 */

#ifndef ENCODER_ME_H
#define ENCODER_ME_H

#include <opencv2/core/mat.hpp>
#include <vector>
#include <queue>
#include "Utils.h"

std::tuple<std::vector<cv::Point2f>, double> blockMatching(Point3Vec tr, const cv::Mat& current, cv::Mat expansion_image);
std::tuple<std::vector<cv::Point2f>, std::vector<double>> blockMatching(Point4Vec square, const cv::Mat& target_image, cv::Mat expansion_ref_image);
std::tuple<std::vector<cv::Point2f>, std::vector<double>> blockMatching(Point3Vec triangle, const cv::Mat& target_image, cv::Mat expansion_ref_image, std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, cv::Point2f fullpell_initial_vector = cv::Point2f(-10000, -10000));

double getPredictedImage(unsigned char **expand_ref, cv::Mat& target_image, cv::Mat& output_image, Point3Vec& triangle, std::vector<cv::Point2f>& mv, int offset, std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, cv::Rect block_size, unsigned char **ref_hevc = nullptr);
double getPredictedImage(unsigned char **expand_ref, cv::Mat& target_image, cv::Mat& output_image, Point4Vec& square, cv::Point2f& mv, unsigned char **ref_hevc);

std::tuple<std::vector<cv::Point2f>, cv::Point2f, double, double, int> GaussNewton(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, std::vector<std::vector<std::vector<unsigned char **>>> expand_image, Point3Vec target_corners, const std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, int block_size_x, int block_size_y, cv::Point2f init_vector, unsigned char **ref_hevc = nullptr);

double bicubic_weight(double x);
int bicubic_interpolation(unsigned char **img, double x, double y);

std::vector<cv::Point2f> getPredictedWarpingMv(std::vector<cv::Point2f>& ref_triangle_coordinate, std::vector<cv::Point2f>& ref_mvs, std::vector<cv::Point2f>& target_triangle_coordinate);

#endif //ENCODER_ME_H
