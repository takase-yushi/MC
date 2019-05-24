//
// Created by kasph on 2019/05/23.
//

#include <opencv2/core/mat.hpp>
#include <cassert>
#include <iostream>
#include "../includes/ImageUtil.h"
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
            R(residual_image, col, row) = abs(R(target_image, col, row) - R(predict_image, col, row));
            G(residual_image, col, row) = abs(G(target_image, col, row) - G(predict_image, col, row));
            B(residual_image, col, row) = abs(B(target_image, col, row) - B(predict_image, col, row));
        }
    }

    return residual_image;
}
