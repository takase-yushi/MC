//
// Created by takahiro on 2019/05/31.
//

#ifndef ENCODER_MAIN_H
#define ENCODER_MAIN_H

#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include <queue>
#include <fstream>
#include <ctime>
#include <utility>
#include <numeric>
#include "../includes/Encode.h"
#include "../includes/config.h"
#include "../includes/ME.hpp"
#include "../includes/DelaunayTriangulation.hpp"
#include "../includes/Vector.hpp"
#include "../includes/psnr.h"
#include "../includes/Golomb.hpp"
#include "../includes/TriangleDivision.h"
#include "../includes/Reconstruction.h"
#include "../includes/ImageUtil.h"
#include "../includes/runAdaptive.h"

struct PredictedImageResult {
    cv::Mat out, mv_image;
    int freq_block, freq_warp, block_matching_pixel_nums, warping_pixel_nums, x_bits, y_bits;
    double block_matching_pixel_errors, warping_pixel_errors;

    PredictedImageResult(cv::Mat out,
                         cv::Mat mv_image,
                         int freq_block,
                         int freq_warp,
                         int block_matching_pixel_nums,
                         int warping_pixel_nums,
                         int xbits,
                         int ybits,
                         double block_matching_pixel_errors,
                         double warping_pixel_errors
    ) :
            out(std::move(out)),
            mv_image(std::move(mv_image)),
            freq_block(freq_block),
            freq_warp(freq_warp),
            block_matching_pixel_nums(block_matching_pixel_nums),
            warping_pixel_nums(warping_pixel_nums),
            x_bits(xbits),
            y_bits(ybits),
            block_matching_pixel_errors(block_matching_pixel_errors),
            warping_pixel_errors(warping_pixel_errors){}

    double getBlockMatchingFrequency() {
        return (double) freq_block / (freq_block + freq_warp) * 100;
    }

    double getWarpingFrequency() {
        return (double) freq_warp / (freq_block + freq_warp) * 100;
    }

    double getBlockMatchingPatchPSNR() {
        return 10.0 * log10(255.0 * 255.0 / (block_matching_pixel_errors / (3.0 * block_matching_pixel_nums)));
    }

    double getWarpingPatchPSNR() {
        return 10.0 * log10(255.0 * 255.0 / (warping_pixel_errors / (3.0 * warping_pixel_nums)));
    }

    int getXbits() {
        return x_bits;
    }

    int getYbits() {
        return y_bits;
    }

};

std::vector<cv::Point2f> cornersQuantization(std::vector<cv::Point2f> &corners, const cv::Mat &target);

PredictedImageResult
getPredictedImage(cv::Mat &ref,cv::Mat &target,cv::Mat &intra, std::vector<Triangle> &triangles, const std::vector<cv::Point2f> &ref_corners,
                  std::vector<cv::Point2f> &corners, DelaunayTriangulation md,std::vector<cv::Point2f> &add_corners,int &add_count,const cv::Mat& residual_ref,int &tmp_mv_x,int &tmp_mv_y,bool add_flag);

std::vector<cv::Point2f> uniqCoordinate(const std::vector<cv::Point2f> &corners);

void storeFrequency(const std::string &file_path, const std::vector<int> freq, int mid);

cv::Point2f getDifferenceVector(const Triangle &triangle, const std::vector<cv::Point2f> &corners,
                                const std::vector<cv::Point2f> &corners_mv, const cv::Point2f &mv);

cv::Mat getReconstructionDivisionImage(cv::Mat image, std::vector<CodingTreeUnit *> ctu);

int addSideCorners(cv::Mat img, std::vector<cv::Point2f> &corners);

void run();

// 問題は差分ベクトルどうするの…？って

#define HARRIS false
#define THRESHOLD true
#define LAMBDA 0.2
#define INTER_DIV true // 頂点追加するかしないか

#define DIVIDE_MODE LEFT_DIVIDE

#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"


#endif //ENCODER_MAIN_H
