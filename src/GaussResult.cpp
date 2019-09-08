//
// Created by kasph on 2019/08/06.
//

#include "../includes/GaussResult.h"

GaussResult::GaussResult(const std::vector<cv::Point2f> &mvWarping, const cv::Point2f &mvParallel,
                         double residual, int triangleSize, bool parallelFlag, double residualBm, double residualNewton) : mv_warping(
        mvWarping), mv_parallel(mvParallel), residual(residual), triangle_size(triangleSize), parallel_flag(parallelFlag), residual_bm(residualBm), residual_newton(residualNewton) {}

GaussResult::GaussResult() {}