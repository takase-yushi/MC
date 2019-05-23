//
// Created by kasph on 2019/05/23.
//

#include <opencv2/core/mat.hpp>
#include <cassert>
#include "ImageUtil.h"
#include "../includes/Utils.h"

/**
 * @fn get_residual(const cv::Mat &target_image, const cv::Mat &predict_image)
 * @brief target_imageとpredict_imageの差分を計算し、差分画像として返す
 * @param target_image 対象画像
 * @param predict_image 予測画像
 * @return 対象画像と予測画像の差分画像
 */
cv::Mat getResidualImage(const cv::Mat &target_image, const cv::Mat &predict_image){
    assert(target_image.cols == predict_image.cols && target_image.rows == predict_image.rows);

    cv::Mat residual_image = cv::Mat::zeros(target_image.rows, target_image.cols, CV_8UC3);

    for(int row = 0 ; row < target_image.rows ; row++){
        for(int col = 0 ; col < target_image.cols ; col++){
            R(residual_image, row, col) = abs(R(target_image, row, col) - R(predict_image, row, col));
            G(residual_image, row, col) = abs(G(target_image, row, col) - G(predict_image, row, col));
            B(residual_image, row, col) = abs(B(target_image, row, col) - B(predict_image, row, col));
        }
    }

    return residual_image;
}