//
// Created by kasph on 2019/08/06.
//

#ifndef ENCODER_GAUSSRESULT_H
#define ENCODER_GAUSSRESULT_H

#include <vector>
#include <opencv2/core/types.hpp>
#include "CodingTreeUnit.h"

class GaussResult{
public:
    GaussResult();

    GaussResult(const std::vector<cv::Point2f> &mvWarping, const cv::Point2f &mvParallel, double residual,
                int triangleSize, bool translationFlag, double residual_bm, double residual_newton);
    std::vector<cv::Point2f> mv_warping, original_mv_warping;
    cv::Point2f mv_translation, original_mv_translation;
    double residual_warping, residual_translation;
    int triangle_size;
    bool translation_flag;
    double residual_bm;
    double residual_newton;
    MV_CODE_METHOD method_warping, method_translation;
};

#endif //ENCODER_GAUSSRESULT_H
