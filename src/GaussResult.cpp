//
// Created by kasph on 2019/08/06.
//

#include "../includes/GaussResult.h"

GaussResult::GaussResult(const std::vector<cv::Point2f> &mvWarping, const cv::Point2f &mvParallel,
                         double residualWarping, double residualTranslation, int triangleSize, bool translationFlag, double residualBm, double residualNewton) : mv_warping(
        mvWarping), mv_translation(mvParallel), residual_warping(residualWarping), residual_translation(residualTranslation), triangle_size(triangleSize), translation_flag(translationFlag), residual_bm(residualBm), residual_newton(residualNewton) {}

GaussResult::GaussResult() {}