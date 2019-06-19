//
// Created by kasph on 2019/05/23.
//

#include <opencv2/core/mat.hpp>
#include <cassert>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "../includes/ImageUtil.h"
#include "../includes/Utils.h"
#include "../includes/CodingTreeUnit.h"
#include "../includes/Reconstruction.h"
#include "../includes/TriangleDivision.h"

/**
 * @fn get_residual(const cv::Mat &target_image, const cv::Mat &predict_image)
 * @brief target_imageとpredict_imageの差分を計算し、差分画像として返す
 * @param target_image 対象画像
 * @param predict_image 予測画像
 * @return 対象画像と予測画像の差分画像
 */
cv::Mat getResidualImage(const cv::Mat &target_image, const cv::Mat &predict_image, int k){
    assert(target_image.cols == predict_image.cols && target_image.rows == predict_image.rows);

    cv::Mat residual_image = cv::Mat::zeros(target_image.rows, target_image.cols, CV_8UC3);

    for(int row = 0 ; row < target_image.rows ; row++){
        for(int col = 0 ; col < target_image.cols ; col++){
            R(residual_image, col, row) = k * abs(R(target_image, col, row) - R(predict_image, col, row));
            G(residual_image, col, row) = k * abs(G(target_image, col, row) - G(predict_image, col, row));
            B(residual_image, col, row) = k * abs(B(target_image, col, row) - B(predict_image, col, row));
        }
    }

    return residual_image;
}

/***
 * @fn double getTriangleResidual(const cv::Mat &target_image, const cv::Mat ref_image, Point3Vec vec)
 * @brief ある三角形を変形した際の残差を返す
 * @param ref_image 参照画像
 * @param target_image 対象画像
 * @param triangle 三角パッチの座標
 * @param vec 動きベクトル
 * @return 残差
 */
double getTriangleResidual(const cv::Mat ref_image, const cv::Mat &target_image, Point3Vec &triangle, std::vector<cv::Point2f> mv, const std::vector<cv::Point2f> &in_triangle_pixels){
    double residual = 0.0;

    cv::Point2f pixels_in_triangle;

    cv::Point2f pp0, pp1, pp2;

    pp0.x = triangle.p1.x + mv[0].x;
    pp0.y = triangle.p1.y + mv[0].y;
    pp1.x = triangle.p2.x + mv[1].x;
    pp1.y = triangle.p2.y + mv[1].y;
    pp2.x = triangle.p3.x + mv[2].x;
    pp2.y = triangle.p3.y + mv[2].y;

    double quantize_step = 4.0;

    // TODO: 置き換え
    double sx = std::min({(int) triangle.p1.x, (int) triangle.p2.x, (int) triangle.p3.x});
    double lx = std::max({(int) triangle.p1.x, (int) triangle.p2.x, (int) triangle.p3.x});
    double sy = std::min({(int) triangle.p1.y, (int) triangle.p2.y, (int) triangle.p3.y});
    double ly = std::max({(int) triangle.p1.y, (int) triangle.p2.y, (int) triangle.p3.y});

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

        if (X_later.x >= ref_image.cols - 1) X_later.x = ref_image.cols - 1.001;

        if (X_later.y >= ref_image.rows - 1) X_later.y = ref_image.rows - 1.001;

        if (X_later.x < 0) X_later.x = 0;

        if (X_later.y < 0) X_later.y = 0;

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

/**
 * @fn std::vector<std::vector<cv::Mat>> getRefImages(const cv::Mat ref_image, const cv::Mat gauss_ref_image)
 * @brief 参照画像の集まりを返す
 * @param ref_image 参照画像
 * @param gauss_ref_image ガウスニュートン法で使う参照画像
 * @return あつまれ～！
 */
std::vector<std::vector<cv::Mat>> getRefImages(const cv::Mat ref_image, const cv::Mat gauss_ref_image){
    std::vector<std::vector<cv::Mat>> ref_images;

    // 参照画像のフィルタ処理（１）
    std::vector<cv::Mat> ref1_levels;
    cv::Mat ref_level_1, ref_level_2, ref_level_3, ref_level_4;
    ref_level_1 = gauss_ref_image;
    ref_level_2 = half(ref_level_1, 2);
    ref_level_3 = half(ref_level_2, 2);
    ref_level_4 = half(ref_level_3, 2);
    ref1_levels.emplace_back(ref_level_4);
    ref1_levels.emplace_back(ref_level_3);
    ref1_levels.emplace_back(ref_level_2);
    ref1_levels.emplace_back(ref_image);

    // 参照画像のフィルタ処理（２）
    std::vector<cv::Mat> ref2_levels;
    cv::Mat ref2_level_1 = gauss_ref_image;
    cv::Mat ref2_level_2 = half(ref2_level_1, 2);
    cv::Mat ref2_level_3 = half(ref2_level_2, 1);
    cv::Mat ref2_level_4 = half(ref2_level_3, 1);
    ref2_levels.emplace_back(ref2_level_4);
    ref2_levels.emplace_back(ref2_level_3);
    ref2_levels.emplace_back(ref2_level_2);
    ref2_levels.emplace_back(ref_image);

    ref_images.emplace_back(ref1_levels);
    ref_images.emplace_back(ref2_levels);

    return ref_images;
}

/**
 * @fn std::vector<std::vector<cv::Mat>> getTargetImages(const cv::Mat target_image)
 * @brief ガウス・ニュートン法で使う対象画像の集まりを返す
 * @param target_image 対象画像
 * @return vectorのvector(死)
 */
std::vector<std::vector<cv::Mat>> getTargetImages(const cv::Mat target_image){
    std::vector<std::vector<cv::Mat>> target_images;

    // 対象画像のフィルタ処理（１）
    std::vector<cv::Mat> target1_levels;
    cv::Mat target_level_1, target_level_2, target_level_3, target_level_4;
    target_level_1 = target_image;
    target_level_2 = half(target_level_1, 2);
    target_level_3 = half(target_level_2, 2);
    target_level_4 = half(target_level_3, 2);
    target1_levels.emplace_back(target_level_4);
    target1_levels.emplace_back(target_level_3);
    target1_levels.emplace_back(target_level_2);
    target1_levels.emplace_back(target_level_1);


    // 対象画像のフィルタ処理（２）
    std::vector<cv::Mat> target2_levels;
    cv::Mat target2_level_1 = target_image;
    cv::Mat target2_level_2 = half(target2_level_1, 2);
    cv::Mat target2_level_3 = half(target2_level_2, 1);
    cv::Mat target2_level_4 = half(target2_level_3, 1);
    target2_levels.emplace_back(target2_level_4);
    target2_levels.emplace_back(target2_level_3);
    target2_levels.emplace_back(target2_level_2);
    target2_levels.emplace_back(target2_level_1);

    target_images.emplace_back(target1_levels);
    target_images.emplace_back(target2_levels);

    return target_images;
}

/**
 * @fn EXPAND_ARRAY_TYPE getExpandImages(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, int expand)
 * @brief 拡大した画像をいい感じに格納して返す
 * @param ref_images 参照画像の集合
 * @param target_images 対象画像の集合
 * @param expand 拡大する範囲
 * @return しゅうごう～
 */
EXPAND_ARRAY_TYPE getExpandImages(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, int expand){
    EXPAND_ARRAY_TYPE expand_images;
    expand_images.resize(ref_images.size());
    for (int filter_num = 0; filter_num < static_cast<int>(ref_images.size()); filter_num++) {
        expand_images[filter_num].resize(ref_images[filter_num].size());
        for (int step = 0; step < static_cast<int>(ref_images[filter_num].size()); step++) {
            expand_images[filter_num][step].resize(4);
        }
    }

    for(int filter = 0 ; filter < static_cast<int>(ref_images.size()) ; filter++){
        for(int step = 0 ; step < static_cast<int>(ref_images[filter].size()) ; step++){
            cv::Mat current_target_image = mv_filter(target_images[filter][step], 2);
            cv::Mat current_ref_image = mv_filter(ref_images[filter][step], 2);

            auto **current_target_expand = (unsigned char **) std::malloc(
                    sizeof(unsigned char *) * (current_target_image.cols + expand * 2));
            current_target_expand += expand;
            auto **current_target_org_expand = (unsigned char **) std::malloc(
                    sizeof(unsigned char *) * (current_target_image.cols + expand * 2));
            current_target_org_expand += expand;

            for (int j = -expand; j < current_target_image.cols + expand; j++) {
                current_target_expand[j] = (unsigned char *) std::malloc(
                        sizeof(unsigned char) * (current_target_image.rows + expand * 2));
                current_target_expand[j] += expand;

                current_target_org_expand[j] = (unsigned char *) std::malloc(
                        sizeof(unsigned char) * (current_target_image.rows + expand * 2));
                current_target_org_expand[j] += expand;
            }

            auto **current_ref_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (current_target_image.cols + expand * 2));
            current_ref_expand += expand;
            auto **current_ref_org_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (current_target_image.cols + expand * 2));
            current_ref_org_expand += expand;

            for (int j = -expand; j < current_ref_image.cols + expand; j++) {
                if ((current_ref_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (current_target_image.rows + expand * 2))) == nullptr) {
                }
                current_ref_expand[j] += expand;

                (current_ref_org_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (current_target_image.rows + expand * 2)));
                current_ref_org_expand[j] += expand;
            }

            for (int j = -expand; j < current_target_image.rows + expand; j++) {
                for (int i = -expand; i < current_target_image.cols + expand; i++) {
                    if (j >= 0 && j < current_target_image.rows && i >= 0 && i < current_target_image.cols) {
                        current_target_expand[i][j] = M(current_target_image, i, j);
                        current_ref_expand[i][j] = M(current_ref_image, i, j);

                        current_target_org_expand[i][j] = M(target_images[filter][step], i, j);
                        current_ref_org_expand[i][j] = M(ref_images[filter][step], i, j);
                    } else {
                        current_target_expand[i][j] = 0;
                        current_ref_expand[i][j] = 0;
                        current_target_org_expand[i][j] = 0;
                        current_ref_org_expand[i][j] = 0;
                    }
                }
            }
            int spread = 18;// 双3次補間を行うために、画像の周り(16+2)=18ピクセルだけ折り返し
            for (int j = 0; j < current_target_image.rows; j++) {
                for (int i = 1; i <= spread; i++) {
                    current_target_expand[-i][j] = current_target_expand[0][j];
                    current_target_expand[current_target_image.cols - 1 + i][j] = current_target_expand[
                            current_target_image.cols - 1][j];
                    current_ref_expand[-i][j] = current_ref_expand[0][j];
                    current_ref_expand[current_target_image.cols - 1 + i][j] = current_ref_expand[
                            current_target_image.cols - 1][j];
                    current_target_org_expand[-i][j] = current_target_org_expand[0][j];
                    current_target_org_expand[current_target_image.cols - 1 + i][j] = current_target_org_expand[
                            current_target_image.cols - 1][j];
                    current_ref_org_expand[-i][j] = current_ref_org_expand[0][j];
                    current_ref_org_expand[current_target_image.cols - 1 + i][j] = current_ref_org_expand[
                            current_target_image.cols - 1][j];
                }
            }
            for (int i = -spread; i < current_target_image.cols + spread; i++) {
                for (int j = 1; j <= spread; j++) {
                    current_target_expand[i][-j] = current_target_expand[i][0];
                    current_target_expand[i][current_target_image.rows - 1 + j] = current_target_expand[i][
                            current_target_image.rows - 1];
                    current_ref_expand[i][-j] = current_ref_expand[i][0];
                    current_ref_expand[i][current_target_image.rows - 1 + j] = current_ref_expand[i][
                            current_target_image.rows - 1];

                    current_target_org_expand[i][-j] = current_target_org_expand[i][0];
                    current_target_org_expand[i][current_target_image.rows - 1 + j] = current_target_org_expand[i][
                            current_target_image.rows - 1];
                    current_ref_org_expand[i][-j] = current_ref_org_expand[i][0];
                    current_ref_org_expand[i][current_target_image.rows - 1 + j] = current_ref_org_expand[i][
                            current_target_image.rows - 1];
                }
            }

            expand_images[filter][step][0] = current_ref_expand;
            expand_images[filter][step][1] = current_ref_org_expand;
            expand_images[filter][step][2] = current_target_expand;
            expand_images[filter][step][3] = current_target_org_expand;
        }
    }

    return expand_images;
}

/**
 * @fn void freeExpandImages(EXPAND_ARRAY_TYPE expand_images, int expand, int filter_num, int step_num, int rows, int cols)
 * @breif 拡張した画像(mallocで取得)を開放、通称「拡張フリー」をする
 * @param expand_images freeeしたい画像
 * @param expand 拡大した画素数
 * @param filter_num フィルターの個数
 * @param step_num ステップ数
 * @param rows 原画像の横幅
 * @param cols 原画像の縦幅
 */
void freeExpandImages(EXPAND_ARRAY_TYPE expand_images, int expand, int filter_num, int step_num, int rows, int cols){
    for(int filter = 0 ; filter < filter_num ; filter++){
        for(int step = 0 ; step < step_num ; step++){
            int scaled_col = cols / std::pow(2, 3 - step);
            auto current_ref_expand = expand_images[filter][step][0];
            auto current_ref_org_expand = expand_images[filter][step][1];
            auto current_target_expand = expand_images[filter][step][2];
            auto current_target_org_expand = expand_images[filter][step][3];

            for (int d = -expand; d < scaled_col + expand; d++) {
                current_target_expand[d] -= expand;
                current_ref_expand[d] -= expand;
                free(current_ref_expand[d]);
                free(current_target_expand[d]);

                current_target_org_expand[d] -= expand;
                current_ref_org_expand[d] -= expand;
                free(current_ref_org_expand[d]);
                free(current_target_org_expand[d]);
            }

            current_target_expand -= expand;
            current_ref_expand -= expand;
            free(current_target_expand);
            free(current_ref_expand);

            current_target_org_expand -= expand;
            current_ref_org_expand -= expand;
            free(current_target_org_expand);
            free(current_ref_org_expand);
        }
    }

}

/**
 * @fn cv::Mat getReconstructionDivisionImage(cv::Mat image, std::vector<CodingTreeUnit *> ctu)
 * @brief CodingTreeをもらって、三角形を書いた画像を返す
 * @param image 下地
 * @param ctu CodingTree
 * @return 画像
 */
cv::Mat getReconstructionDivisionImage(cv::Mat image, std::vector<CodingTreeUnit *> ctu, int block_size_x, int block_size_y) {
    Reconstruction rec(image);
    rec.init(block_size_x, block_size_y, LEFT_DIVIDE);
    puts("");
    rec.reconstructionTriangle(ctu);
    std::vector<Point3Vec> hoge = rec.getTriangleCoordinateList();

    cv::Mat reconstructedImage = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp");
    for(const auto foo : hoge) {
        drawTriangle(reconstructedImage, foo.p1, foo.p2, foo.p3, cv::Scalar(255, 255, 255));
    }
//    cv::imwrite(getProjectDirectory(OS) + "/img/minato/reconstruction_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png", reconstructedImage);

    return reconstructedImage;
}