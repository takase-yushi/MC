//
// Created by kasph on 2019/05/23.
//

#ifndef ENCODER_IMAGEUTIL_H
#define ENCODER_IMAGEUTIL_H

#include "Utils.h"
#include "CodingTreeUnit.h"

typedef std::vector<std::vector<std::vector<unsigned char *>>> EXPAND_ARRAY_TYPE;

enum IP_MODE {
    BILINEAR,
    BICUBIC,
    HEVC
};

cv::Mat getResidualImage(const cv::Mat &target_image, const cv::Mat &predict_image, int k = 1);
double getTriangleResidual(const cv::Mat ref_image, const cv::Mat &target_image, Point3Vec &triangle, std::vector<cv::Point2f> mv, const std::vector<cv::Point2f> &in_triangle_pixels);
std::vector<std::vector<cv::Mat>> getRefImages(const cv::Mat& ref_image, const cv::Mat& gauss_ref_image);
std::vector<std::vector<cv::Mat>> getTargetImages(const cv::Mat target_image);
EXPAND_ARRAY_TYPE getExpandImages(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, int expand);
void freeExpandImages(EXPAND_ARRAY_TYPE expand_images, int expand, int filter_num, int step_num, int rows, int cols);
void freeHEVCExpandImage(EXPAND_ARRAY_TYPE expand_images, int expand, int filter_num, int step_num, int rows, int cols);
cv::Mat getReconstructionDivisionImage(cv::Mat image, std::vector<CodingTreeUnit *> ctu, int block_size_x, int block_size_y);
void freeImage(unsigned char **image, cv::Size image_size, int expansion_size);
unsigned char ** getExpansionImage(cv::Mat image, int k, int expansion_size, IP_MODE = IP_MODE::BICUBIC);
cv::Mat getExpansionMatImage(cv::Mat &image, int k, int expansion_size, IP_MODE mode = IP_MODE::BICUBIC);
bool isMyTriangle(const CodingTreeUnit* ctu, int triangle_index);
std::vector<cv::Point2f> getPixelsInTriangle(const Point3Vec& triangle, const std::vector<std::vector<int>>& area_flag, int triangle_index, CodingTreeUnit* ctu, int block_size_x, int block_size_y);
double w(double x);
double img_ip(unsigned char **img, cv::Rect rect, double x, double y, int mode);
double img_ip(unsigned char *img, cv::Rect rect, double x, double y, int offset = SEARCH_RANGE, int k = 4);

unsigned char* getExpansionHEVCImage(cv::Mat image, int k, int expansion_size);
cv::Mat getExpansionMatHEVCImage(cv::Mat image, int k, int expansion_size);
double getTriangleResidual(unsigned char *ref_image, const cv::Mat &target_image, Point3Vec &triangle, std::vector<cv::Point2f> mv, const std::vector<cv::Point2f> &in_triangle_pixels, cv::Rect rect);
double getTriangleSSE(unsigned char *ref_image, unsigned char *target_image, Point3Vec &triangle, std::vector<cv::Point2f> mv, const std::vector<cv::Point2f> &in_triangle_pixels, cv::Rect rect, bool useRound = false);
cv::Mat getAppliedLPFImage(const cv::Mat &image);

void store1DArrayImage(std::string output_path, unsigned char *array, int width, int height, int offset, int k);
std::vector<cv::Point2f> getPixelsInSquare(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4);

#endif //ENCODER_IMAGEUTIL_H
