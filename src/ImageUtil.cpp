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

double getSquaredError(cv::Mat& ref_image, cv::Mat& target_image, Point3Vec& triangle, std::vector<cv::Point2f>& mv) {
    cv::Point2f pp0, pp1, pp2;

    pp0.x = triangle.p1.x + mv[0].x;
    pp0.y = triangle.p1.y + mv[0].y;
    pp1.x = triangle.p2.x + mv[1].x;
    pp1.y = triangle.p2.y + mv[1].y;
    pp2.x = triangle.p3.x + mv[2].x;
    pp2.y = triangle.p3.y + mv[2].y;

    double quantize_step = 4.0;

    double sx = std::min({(int) triangle.p1.x, (int) triangle.p2.x, (int) triangle.p3.x});
    double lx = std::max({(int) triangle.p1.x, (int) triangle.p2.x, (int) triangle.p3.x});
    double sy = std::min({(int) triangle.p1.y, (int) triangle.p2.y, (int) triangle.p3.y});
    double ly = std::max({(int) triangle.p1.y, (int) triangle.p2.y, (int) triangle.p3.y});

    std::vector<cv::Point2f> in_triangle_pixels;
    cv::Point2f xp;
    for (int j = (int) (round(sy) - 1); j <= round(ly) + 1; j++) {
        for (int i = (int) (round(sx) - 1); i <= round(lx) + 1; i++) {
            xp.x = (float) i;
            xp.y = (float) j;
            if (isInTriangle(triangle, xp) == 1) {
                in_triangle_pixels.emplace_back(xp);//三角形の内部のピクセルを格納
            }
        }
    }
    cv::Point2f X,a,b,a_later,b_later,X_later;
    double alpha,beta,det;

    double squared_error = 0.0;

    a = triangle.p3 - triangle.p1;
    b = triangle.p2 - triangle.p1;
    det = a.x * b.y - a.y * b.x;

    for(const auto& pixel : in_triangle_pixels) {
        X.x = pixel.x - triangle.p1.x;
        X.y = pixel.y - triangle.p1.y;
        alpha = (X.x * b.y - X.y * b.x) / det;
        beta = (a.x * X.y - a.y * X.x) / det;
        X.x += triangle.p1.x;
        X.y += triangle.p1.y;

        a_later = pp2 - pp0;
        b_later = pp1 - pp0;
        X_later = alpha * a_later + beta * b_later + pp0;

        if (X_later.x >= ref_image.cols - 1) {
            X_later.x = ref_image.cols - 1.001;
        }
        if (X_later.y >= ref_image.rows - 1) {
            X_later.y = ref_image.rows - 1.001;
        }

        if (X_later.x < 0) {
            X_later.x = 0;
        }
        if (X_later.y < 0) {
            X_later.y = 0;
        }
        int x0 = floor(X_later.x);
        double d_x = X_later.x - x0;
        int y0 = floor(X_later.y);
        double d_y = X_later.y - y0;

        int y = (int) floor((M(ref_image, (int) x0    , (int) y0    ) * (1 - d_x) * (1 - d_y)  +
                             M(ref_image, (int) x0 + 1, (int) y0    ) * (    d_x) * (1 - d_y)  +
                             M(ref_image, (int) x0    , (int) y0 + 1) * (1 - d_x) * (    d_y)  +
                             M(ref_image, (int) x0 + 1, (int) y0 + 1) * (    d_x) * (    d_y)) + 0.5);


        squared_error += pow((M(target_image, (int)pixel.x, (int)pixel.y) - (0.299 * y + 0.587 * y + 0.114 * y)), 2);
    }

    return squared_error;
}