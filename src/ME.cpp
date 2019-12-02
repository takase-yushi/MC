/**
 * @file ME.cpp
 * @brief ME(Motion Estimation)のためのBlockMatchingとWarpingを定義したファイル
 * @author Keisuke KAMIYA
 */

#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include <queue>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <memory>
#include "../includes/Utils.h"
#include "../includes/ImageUtil.h"
#include "../includes/MELog.h"
#include <algorithm>

#define HEIGHT 1024
#define WIDTH 1920

/**
 * @fn void block_matching(cv::Mat &prev, cv::Mat &current, double &error, cv::Point2f &mv, Point3Vec tr, cv::Mat expansion_image)
 * @brief ブロックマッチングを行い, 動きベクトルを求める
 * @param[in]  prev             参照画像
 * @param[in]  current          対象画像
 * @param[out] error            誤差
 * @param[out] mv               ブロックマッチングの平行移動ベクトル
 * @param[in]  triangle               三角形を表す3点の座標
 * @param[in]  expansion_image  2倍に拡大した(補間した)画像
 * @details
 *
 */
std::tuple<std::vector<cv::Point2f>, double> blockMatching(Point3Vec tr, const cv::Mat& current, cv::Mat expansion_image) {
    double sx, sy, lx, ly;
    cv::Point2f tp1, tp2, tp3;

    tp1 = tr.p1;
    tp2 = tr.p2;
    tp3 = tr.p3;

    tp1.x = (tp1.x + 1) * 4 - 1;
    tp1.y = (tp1.y + 1) * 4 - 1;
    tp2.x = (tp2.x + 1) * 4 - 1;
    tp2.y = (tp2.y + 1) * 4 - 1;
    tp3.x = (tp3.x + 1) * 4 - 1;
    tp3.y = (tp3.y + 1) * 4 - 1;

    sx = std::min({tp1.x, tp2.x, tp3.x});
    sy = std::min({tp1.y, tp2.y, tp3.y});
    lx = std::max({tp1.x, tp2.x, tp3.x});
    ly = std::max({tp1.y, tp2.y, tp3.y});

    cv::Point2f mv_tmp(0.0, 0.0); //三角パッチの動きベクトル
    int SX = SEARCH_RANGE; // ブロックマッチングの探索範囲(X)
    int SY = SEARCH_RANGE; // ブロックマッチングの探索範囲(Y)

    double e, error_min;
    int e_count;

    error_min = 1 << 20;
    cv::Point2d xp(0.0, 0.0);
    cv::Point2f mv_min;
    int spread_quarter = SEARCH_RANGE * 4;
    int s = 4;                   //4 : Full-pel, 2 : Half-pel, 1 : Quarter-pel

    for(int j = -SY * 4 ; j <= SY * 4 ; j += s) {            //j : y方向のMV
        for(int i = -SX * 4 ; i <= SX * 4 ; i += s) {        //i : x方向のMV
            //探索範囲が画像上かどうか判定
            if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_image.cols - spread_quarter
               && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_image.rows - spread_quarter) {
                e = 0.0;
                e_count = 0;
                for (int y = (int) (round(sy) / 4 - 1); y <= (int) (round(ly) / 4 + 1); y++) {
                    for (int x = (int) (round(sx) / 4 - 1); x <= (int) (round(lx) / 4 + 1); x++) {
                        xp.x = (double)x;
                        xp.y = (double)y;
                        //xpが三角形の中かどうか判定
                        if(isInTriangle(tr, xp)){
                            e += fabs(R(expansion_image, i + 4 * x + spread_quarter, j + 4 * y + spread_quarter) - M(current, x, y));
                            e_count++;
                        }
                    }
                }
                if(error_min > e && e_count > 0){
                    error_min = e;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }

    std::vector<cv::Point2f> mvs;
    mvs.emplace_back(mv_min.x, mv_min.y);

    mv_tmp.x = mv_min.x * 4;
    mv_tmp.y = mv_min.y * 4;

    s = 2;
    error_min = 1 << 20;
    for(int j = - s + mv_tmp.y ; j <= s + mv_tmp.y ; j += s){            //j : y方向のMV
        for(int i = - s + mv_tmp.x ; i <= s + mv_tmp.x ; i += s){        //i : x方向のMV
            if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_image.cols - spread_quarter
               && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_image.rows - spread_quarter) {
                e = 0.0;
                e_count = 0;
                for (int y = (int) (round(sy) / 4 - 1); y <= (int) (round(ly) / 4 + 1); y++) {
                    for (int x = (int) (round(sx) / 4 - 1); x <= (int) (round(lx) / 4 + 1); x++) {
                        xp.x = (double)x;
                        xp.y = (double)y;
                        //xpが三角形の中かどうか判定
                        if(isInTriangle(tr, xp)){
                            e += fabs(R(expansion_image, i + 4 * x + spread_quarter, j + 4 * y + spread_quarter) - M(current, x, y));
                            e_count++;
                        }
                    }
                }
                if(error_min > e && e_count > 0){
                    error_min = e;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }

    mvs.emplace_back(mv_min.x, mv_min.y);
    mv_tmp.x = mv_min.x * 4;
    mv_tmp.y = mv_min.y * 4;

    s = 1;
    error_min = 1 << 20;

    for(int j = - s + mv_tmp.y ; j <= s + mv_tmp.y ; j += s){            //j : y方向のMV
        for(int i = - s + mv_tmp.x ; i <= s + mv_tmp.x ; i += s){        //i : x方向のMV
            if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_image.cols - spread_quarter
               && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_image.rows - spread_quarter) {
                e = 0.0;
                e_count = 0;
                for (int y = (int) (round(sy) / 4 - 1); y <= (int) (round(ly) / 4 + 1); y++) {
                    for (int x = (int) (round(sx) / 4 - 1); x <= (int) (round(lx) / 4 + 1); x++) {
                        xp.x = (double)x;
                        xp.y = (double)y;
                        //xpが三角形の中かどうか判定
                        if(isInTriangle(tr, xp)){
                            e += fabs(R(expansion_image, i + 4 * x + spread_quarter, j + 4 * y + spread_quarter) - M(current, x, y));
                            e_count++;
                        }
                    }
                }
                if(error_min > e && e_count > 0){
                    error_min = e;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }

    double error = error_min;
    mvs.emplace_back(mv_min.x, mv_min.y);

    return std::make_tuple(mvs, error);
}

/**
 * @fn std::tuple<std::vector<cv::Point2f>, double> blockMatching(Point3Vec tr, const cv::Mat& target_image, cv::Mat expansion_image, std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu)
 * @brief ブロックマッチング（三段探索）をして，動きベクトルを求める
 * @param triangle 三角パッチ
 * @param target_image 対象画像のMat
 * @param expansion_ref_image 拡大した参照画像
 * @param area_flag getPixelsInTriangleで使用するマスク
 * @param triangle_index 三角パッチの番号△
 * @param ctu CodingTree
 * @return 動きベクトルのvector(vec[0]: full-pell vec[1]: half-pell vec[2]: quarter-pell)と
 */
std::tuple<std::vector<cv::Point2f>, std::vector<double>> blockMatching(Point3Vec triangle, const cv::Mat& target_image, cv::Mat expansion_ref_image, std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, cv::Point2f fullpell_initial_vector) {
    double sx, sy, lx, ly;
    cv::Point2f tp1, tp2, tp3;

    tp1 = triangle.p1;
    tp2 = triangle.p2;
    tp3 = triangle.p3;

    sx = std::min({tp1.x, tp2.x, tp3.x});
    sy = std::min({tp1.y, tp2.y, tp3.y});
    lx = std::max({tp1.x, tp2.x, tp3.x});
    ly = std::max({tp1.y, tp2.y, tp3.y});

    int width = lx - sx + 1;
    int height = ly - sy + 1;

    sx = sx * 4;
    sy = sy * 4;
    lx = sx + width * 4 - 1;
    ly = sy + height * 4 - 1;

    cv::Point2f mv_tmp(0.0, 0.0); //三角パッチの動きベクトル
    int SX = 32; // ブロックマッチングの探索範囲(X)
    int SY = 32; // ブロックマッチングの探索範囲(Y)

    double e = 1e9, error_min = 1e9;
    int e_count;
    cv::Point2f mv_min;
    int spread_quarter = SEARCH_RANGE * 4;
    int s = 4;                   //4 : Full-pel, 2 : Half-pel, 1 : Quarter-pel
    extern int block_size_x;
    extern int block_size_y;
    std::vector<cv::Point2f> pixels = getPixelsInTriangle(triangle, area_flag, triangle_index, ctu, block_size_x, block_size_y);

    if(fullpell_initial_vector.x == -10000 && fullpell_initial_vector.y == -10000){
        for(int j = -SY * 4 ; j <= SY * 4 ; j += s) {            //j : y方向のMV
            for(int i = -SX * 4 ; i <= SX * 4 ; i += s) {        //i : x方向のMV
                //探索範囲が画像上かどうか判定
                if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
                   && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                    e = 0.0;
                    e_count = 0;
                    for(auto &pixel : pixels) {
                        int ref_x = std::max((int)(4 * pixel.x), 0);
                        ref_x = (i + ref_x + spread_quarter);
                        int ref_y = std::max((int)((4 * pixel.y)), 0);
                        ref_y = (j + ref_y + spread_quarter);
                        e += fabs(R(expansion_ref_image, ref_x, ref_y) - R(target_image, (int)pixel.x, (int)pixel.y));
                        e_count++;
                    }
                }
                if(error_min > e){
                    error_min = e;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }else{
        mv_min.x = (fullpell_initial_vector.x > 0 ? (int)(fullpell_initial_vector.x + 0.5) : (int) (fullpell_initial_vector.x - 0.5));
        mv_min.y = (fullpell_initial_vector.y > 0 ? (int)(fullpell_initial_vector.y + 0.5) : (int) (fullpell_initial_vector.y - 0.5));
    }

    std::vector<cv::Point2f> mvs;
    std::vector<double> errors;
    mvs.emplace_back(mv_min.x, mv_min.y);
    errors.emplace_back(error_min);

    mv_tmp.x = mv_min.x * 4;
    mv_tmp.y = mv_min.y * 4;

    s = 2;
    error_min = 1 << 20;
    for(int j = -2 *  s + mv_tmp.y ; j <= 2 * s + mv_tmp.y ; j += s){            //j : y方向のMV
        for(int i = - 2 * s + mv_tmp.x ; i <= 2 * s + mv_tmp.x ; i += s){        //i : x方向のMV
            if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
               && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                e = 0.0;
                e_count = 0;
                for(auto &pixel : pixels) {
                    int ref_x = std::max((int)(4 * pixel.x), 0);
                    ref_x = (i + ref_x + spread_quarter);
                    int ref_y = std::max((int)((4 * pixel.y)), 0);
                    ref_y = (j + ref_y + spread_quarter);
                    e += fabs(R(expansion_ref_image, ref_x, ref_y) - R(target_image, (int)pixel.x, (int)pixel.y));
                    e_count++;
                }
                if(error_min > e && e_count > 0){
                    error_min = e;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }
    mvs.emplace_back(mv_min.x, mv_min.y);
    errors.emplace_back(error_min);

    mv_tmp.x = mv_min.x * 4;
    mv_tmp.y = mv_min.y * 4;

    s = 1;
    error_min = 1 << 20;

    for(int j = - 2 * s + mv_tmp.y ; j <= 2 * s + mv_tmp.y ; j += s){            //j : y方向のMV
        for(int i = - 2 * s + mv_tmp.x ; i <= 2 * s + mv_tmp.x ; i += s){        //i : x方向のMV
            if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
               && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                e = 0.0;
                for(auto &pixel : pixels) {
                    int ref_x = std::max((int)(4 * pixel.x), 0);
                    ref_x = (i + ref_x + spread_quarter);
                    int ref_y = std::max((int)((4 * pixel.y)), 0);
                    ref_y = (j + ref_y + spread_quarter);
                    e += fabs(R(expansion_ref_image, ref_x, ref_y) - R(target_image, (int)pixel.x, (int)pixel.y));
                }

                if(error_min > e){
                    error_min = e;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }
    double error = error_min;
    mvs.emplace_back(mv_min.x, mv_min.y);
    errors.emplace_back(error);

    return std::make_tuple(mvs, errors);
}

/**
 * @fn double bicubic_weight(double x)
 * @brief bicubicフィルタの重みを計算する
 * @param x 重み関数に渡すアレ
 * @return 重み
 */
double bicubic_weight(double x){
    double abs_x = fabs(x);

    if (abs_x <= 1.0) {
        return abs_x * abs_x * abs_x - 2 * abs_x * abs_x + 1;
    } else if (abs_x <= 2.0) {
        return - abs_x * abs_x * abs_x + 5 * abs_x * abs_x - 8 * abs_x + 4;
    } else {
        return 0.0;
    }
}

/**
 * @fn int bicubic_interpolation(unsigned char **img, double x, double y)
 * @brief 与えられた座標を双三次補間を行い画素値を返す
 * @param img 折返しなどで拡大した画像
 * @param x 小数精度のx座標
 * @param y 小数精度のy座標
 * @return 補間した値
 */
int bicubic_interpolation(unsigned char **img, double x, double y){
    int x0 = floor(x);
    double d_x = x - x0;
    int y0 = floor(y);
    double d_y = y - y0;

    double val = 0.0;
    for(int ny = -1 ; ny <= 2 ; ny++) {
        for(int nx = -1 ; nx <= 2 ; nx++) {
            val += img[x0 + nx][y0 + ny] * bicubic_weight(nx - d_x) * bicubic_weight(ny - d_y);
        }
    }

    if(val >= 255.5) return 255;
    else if(val < -0.5) return 0;
    else return (int)(val + 0.5);
}

/**
 * @fn void getPredictedImage(cv::Mat& ref_image, cv::Mat& target_image, cv::Mat& output_image, std::vector<cv::Point2f>& mv)
 * @brief 動きベクトルをもらって、out_imageに画像を書き込む
 * @param[in] ref_image
 * @param[in] target_image
 * @param[out] output_image
 * @param[in] triangle
 * @param[in] mv
 * @param[in] translation_flag
 * @return 2乗誤差
 */
double getPredictedImage(unsigned char **expand_ref, cv::Mat& target_image, cv::Mat& output_image, Point3Vec& triangle, std::vector<cv::Point2f>& mv, int offset, std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, cv::Rect block_size, unsigned char *ref_hevc) {
    cv::Point2f pp0, pp1, pp2;

    pp0.x = triangle.p1.x + mv[0].x;
    pp0.y = triangle.p1.y + mv[0].y;
    pp1.x = triangle.p2.x + mv[1].x;
    pp1.y = triangle.p2.y + mv[1].y;
    pp2.x = triangle.p3.x + mv[2].x;
    pp2.y = triangle.p3.y + mv[2].y;

    std::vector<cv::Point2f> in_triangle_pixels = getPixelsInTriangle(triangle, area_flag, triangle_index, ctu, block_size.width, block_size.height);

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

        int y;
        if(ref_hevc != nullptr){
            y = img_ip(ref_hevc, cv::Rect(-SEARCH_RANGE * 4, -SEARCH_RANGE * 4, 4 * (target_image.cols + 2 * SEARCH_RANGE), 4 * (target_image.rows + 2 * SEARCH_RANGE)), 4 * X_later.x, 4 * X_later.y);
        }else{
            // y = bicubic_interpolation(expand_ref, X_later.x, X_later.y);
        }

        R(output_image, (int)pixel.x, (int)pixel.y) = y;
        G(output_image, (int)pixel.x, (int)pixel.y) = y;
        B(output_image, (int)pixel.x, (int)pixel.y) = y;

        squared_error += pow((M(target_image, (int)pixel.x, (int)pixel.y) - (0.299 * y + 0.587 * y + 0.114 * y)), 2);
    }

    return squared_error;
}

/**
 * @fn std::pair<std::vector<cv::Point2f>, cv::Point2f> GaussNewton(cv::Mat ref_image, cv::Mat target_mage, cv::Mat gauss_ref_image, Point3Vec target_corners)
 * @brief ガウス・ニュートン法を行い、動きベクトル・予測残差・面積を返す
 * @param ref_images
 * @param target_images
 * @param expand_image
 * @param target_corners
 * @param area_flag
 * @param triangle_index
 * @param ctu
 * @param block_size_x
 * @param block_size_y
 * @return ワーピングの動きベクトル・平行移動の動きベクトル・予測残差・面積・平行移動のフラグのtuple
 */
std::tuple<std::vector<cv::Point2f>, cv::Point2f, double, double, int> GaussNewton(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, std::vector<std::vector<std::vector<unsigned char *>>> expand_image, Point3Vec target_corners, const std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, int block_size_x, int block_size_y, cv::Point2f init_vector, unsigned char *ref_hevc){
    // 画像の初期化 vector[filter][picture_number]

    /**
     * Translation用の動きベクトル推定
     * - 変動量(u,v)を求める
     *
     * ## 方程式の命名
     *
     * -                - -          -     -   -
     * |                | |          |     |   |
     * |                | |          |     |   |
     * | gg_translation | | delta_uv |  =  | B |
     * |                | |          |     |   |
     * |                | |          |     |   |
     * -                - -          -     -   -
     *
     */
    const int warping_matrix_dim     = 6; // 方程式の次元
    const int translation_matrix_dim = 2;
    cv::Mat gg_warping               = cv::Mat::zeros(warping_matrix_dim, warping_matrix_dim, CV_64F);         // 式(45)の左辺6×6行列
    cv::Mat gg_translation           = cv::Mat::zeros(translation_matrix_dim, translation_matrix_dim, CV_64F); // 式(52)の左辺2×2行列
    cv::Mat B_warping                = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F);                          // 式(45)の右辺
    cv::Mat B_translation            = cv::Mat::zeros(translation_matrix_dim, 1, CV_64F);                      // 式(52)の右辺
    cv::Mat delta_uv_warping         = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F);                          // 式(45)の左辺 delta
    cv::Mat delta_uv_translation     = cv::Mat::zeros(translation_matrix_dim, 1, CV_64F);                      // 式(52)の右辺 delta

    double min_error_warping = 1E6, min_error_translation = 1E6;
    double max_PSNR_warping = -1, max_PSNR_translation = -1;

    cv::Point2f p0, p1, p2;
    std::vector<cv::Point2f> max_v_warping;
    cv::Point2f max_v_translation;

    std::vector<std::pair<std::vector<cv::Point2f>,double>> v_stack_warping;
    std::vector<std::pair<cv::Point2f,double>> v_stack_translation;
    std::vector<cv::Point2f> pixels_in_triangle;

    cv::Point2f initial_vector(0.0, 0.0);

    p0 = target_corners.p1;
    p1 = target_corners.p2;
    p2 = target_corners.p3;

    Point3Vec current_triangle_coordinates(p0, p1, p2);

    pixels_in_triangle = getPixelsInTriangle(current_triangle_coordinates, area_flag, triangle_index, ctu, block_size_x, block_size_y);

    double sx = std::min({(int) p0.x, (int) p1.x, (int) p2.x});
    double lx = std::max({(int) p0.x, (int) p1.x, (int) p2.x});
    double sy = std::min({(int) p0.y, (int) p1.y, (int) p2.y});
    double ly = std::max({(int) p0.y, (int) p1.y, (int) p2.y});

    int bm_x_offset = 32;
    int bm_y_offset = 32;
    double error_bm_min = 1e9;

    int image_width  = ref_images[0][3].cols;
    int image_height = ref_images[0][3].rows;

    if(init_vector.x == -1000 && init_vector.y == -1000) {
        for (int by = -bm_y_offset; by < bm_y_offset; by++) {
            for (int bx = -bm_x_offset; bx < bm_x_offset; bx++) {
                if (sx + bx < -SEARCH_RANGE || ref_images[0][3].cols + SEARCH_RANGE <= (lx + bx) || sy + by < -SEARCH_RANGE ||
                    ref_images[0][3].rows + SEARCH_RANGE <= (ly + by))
                    continue;
                double error_tmp = 0.0;
                for (const auto &pixel : pixels_in_triangle) {
#if GAUSS_NEWTON_HEVC_IMAGE

                    unsigned char pel1 = F(expand_image[0][3][1], 4 * (int) (pixel.x + bx), 4 * (int) (pixel.y + by), 4 * SEARCH_RANGE, 4 * (ref_images[0][3].cols));
                    unsigned char pel2 = F(expand_image[0][3][3], 4 * (int) (pixel.x)     , 4 * (int) (pixel.y)     , 4 * SEARCH_RANGE, 4 * (ref_images[0][3].cols));
                    error_tmp += (pel1 - pel2) * (pel1 - pel2);
#else
                    error_tmp += abs(expand_image[0][3][1][(int) (pixel.x + bx)][(int) (pixel.y + by)] -
                                                     expand_image[0][3][3][(int) (pixel.x)][(int) (pixel.y)]);
#endif
                }
                if (error_bm_min > error_tmp) {
                    error_bm_min = error_tmp;
                    initial_vector.x = bx;
                    initial_vector.y = by;
                }
            }
        }
    }else{
        initial_vector.x = init_vector.x;
        initial_vector.y = init_vector.y;
    }

#if STORE_NEWTON_LOG
    extern std::vector<MELog> ME_log_translation_0;
    extern std::vector<MELog> ME_log_translation_1;
    extern std::vector<MELog> ME_log_warping_0;
    extern std::vector<MELog> ME_log_warping_1;

    ME_log_translation_0.emplace_back();
    ME_log_translation_1.emplace_back();
#endif


    for(int filter_num = 0 ; filter_num < static_cast<int>(ref_images.size()) ; filter_num++){
        cv::Point2f tmp_mv_translation(initial_vector.x, initial_vector.y);
        bool translation_update_flag = true;

#if STORE_NEWTON_LOG
        MELog& current_me_log = (filter_num == 0 ? ME_log_translation_0.back() : ME_log_translation_1.back());
#endif
        // Marquardtの係数
        double alpha_marquardt = 0.5;

        v_stack_translation.emplace_back(tmp_mv_translation, error_bm_min);

        for(int step = 3 ; step < static_cast<int>(ref_images[filter_num].size()) ; step++){
            double SSE_translation = 0.0;

            double scale = pow(2, 3 - step);
            cv::Mat current_ref_image = ref_images[filter_num][step];
            cv::Mat current_target_image = target_images[filter_num][step];

            unsigned char *current_target_expand, *current_target_org_expand; //画像の周りに500ピクセルだけ黒の領域を設ける(念のため)
            unsigned char *current_ref_expand, *current_ref_org_expand;    //f_expandと同様

            current_ref_expand        = expand_image[filter_num][step][0];
            current_ref_org_expand    = expand_image[filter_num][step][1];
            current_target_expand     = expand_image[filter_num][step][2];
            current_target_org_expand = expand_image[filter_num][step][3];

            int spread = SEARCH_RANGE; // 探索範囲は16までなので16に戻す

            int scaled_spread = spread / scale;
            p0 = target_corners.p1 / scale;
            p1 = target_corners.p2 / scale;
            p2 = target_corners.p3 / scale;

            // 端の頂点の調整
            if (target_corners.p1.x == target_images[0][3].cols - 1) p0.x = target_images[0][step].cols - 1;
            if (target_corners.p1.y == target_images[0][3].rows - 1) p0.y = target_images[0][step].rows - 1;
            if (target_corners.p2.x == target_images[0][3].cols - 1) p1.x = target_images[0][step].cols - 1;
            if (target_corners.p2.y == target_images[0][3].rows - 1) p1.y = target_images[0][step].rows - 1;
            if (target_corners.p3.x == target_images[0][3].cols - 1) p2.x = target_images[0][step].cols - 1;
            if (target_corners.p3.y == target_images[0][3].rows - 1) p2.y = target_images[0][step].rows - 1;

            if(fabs((p2 - p0).x * (p1 - p0).y - (p2 - p0).y * (p1 - p0).x) <= 0) break;

            current_triangle_coordinates.p1 = p0;
            current_triangle_coordinates.p2 = p1;
            current_triangle_coordinates.p3 = p2;
            pixels_in_triangle = getPixelsInTriangle(current_triangle_coordinates, area_flag, triangle_index, ctu, block_size_x, block_size_y);

            std::vector<cv::Point2f> scaled_coordinates{p0, p1, p2};

            if(step != 0) {
                // 画面外にはみ出してる場合、２倍からだんだん小さく縮小していく

                // 平行移動
                double magnification = 1.0;
                while ( (p0.x + tmp_mv_translation.x * magnification < -scaled_spread && p0.x + tmp_mv_translation.x * magnification > current_target_image.cols - 1 + scaled_spread) &&
                        (p1.x + tmp_mv_translation.x * magnification < -scaled_spread && p1.x + tmp_mv_translation.x * magnification > current_target_image.cols - 1 + scaled_spread) &&
                        (p2.x + tmp_mv_translation.x * magnification < -scaled_spread && p2.x + tmp_mv_translation.x * magnification > current_target_image.cols - 1 + scaled_spread) &&
                        (p0.y + tmp_mv_translation.y * magnification < -scaled_spread && p0.y + tmp_mv_translation.y * magnification > current_target_image.rows - 1 + scaled_spread) &&
                        (p1.y + tmp_mv_translation.y * magnification < -scaled_spread && p1.y + tmp_mv_translation.y * magnification > current_target_image.rows - 1 + scaled_spread) &&
                        (p2.y + tmp_mv_translation.y * magnification < -scaled_spread && p2.y + tmp_mv_translation.y * magnification > current_target_image.rows - 1 + scaled_spread) ) {
                    if(magnification <= 1)break;
                    magnification -= 0.1;
                }
                tmp_mv_translation *= magnification;
            }
            v_stack_translation.clear();
            v_stack_translation.emplace_back(tmp_mv_translation, error_bm_min);

            double prev_SSE_translation = error_bm_min;
            cv::Point2f prev_mv_translation = tmp_mv_translation;

            int iterate_counter = 0;

#if STORE_NEWTON_LOG
            current_me_log.mv_newton_translation.emplace_back(tmp_mv_translation);
            current_me_log.residual.emplace_back(error_bm_min);
            current_me_log.coordinate_after_move1.emplace_back();
            current_me_log.coordinate_after_move2.emplace_back();
            current_me_log.coordinate_after_move3.emplace_back();
#endif

            while(true){
                // 移動後の座標を格納する
                cv::Point2f a = p2 - p0;
                cv::Point2f b = p1 - p0;
                double det = a.x * b.y - a.y * b.x;
                // tmp_mv_warping, tmp_mv_translationは現在の動きベクトル
                // 初回は初期値が入ってる

                gg_translation = cv::Mat::zeros(translation_matrix_dim, translation_matrix_dim, CV_64F);
                B_translation = cv::Mat::zeros(translation_matrix_dim, 1, CV_64F);
                delta_uv_translation = cv::Mat::zeros(translation_matrix_dim, 1, CV_64F);

                double delta_g_translation[translation_matrix_dim] = {0};

                cv::Point2f X;
                SSE_translation = 0.0;

                double E_delta_x = 0.0;
                double E_delta_y = 0.0;

                for(const auto& pixel : pixels_in_triangle) {
                    X.x = pixel.x - p0.x;
                    X.y = pixel.y - p0.y;

                    double alpha = (X.x * b.y - X.y * b.x) / det;
                    double beta = (a.x * X.y - a.y * X.x)/ det;;

                    // 参照フレームの前進差分（平行移動）


                    // 移動後の頂点を計算し格納
                    std::vector<cv::Point2f> triangle_later_translation(3);
                    triangle_later_translation[0] = p0 + tmp_mv_translation;
                    triangle_later_translation[1] = p1 + tmp_mv_translation;
                    triangle_later_translation[2] = p2 + tmp_mv_translation;

                    cv::Point2f a_later_translation = triangle_later_translation[2] - triangle_later_translation[0];
                    cv::Point2f b_later_translation = triangle_later_translation[1] - triangle_later_translation[0];
                    cv::Point2f X_later_translation = alpha * a_later_translation + beta * b_later_translation + triangle_later_translation[0];

                    if(X_later_translation.x >= (current_ref_image.cols - 1 + scaled_spread)) X_later_translation.x = current_ref_image.cols - 1 + scaled_spread;
                    if(X_later_translation.y >= (current_ref_image.rows - 1 + scaled_spread)) X_later_translation.y = current_ref_image.rows - 1 + scaled_spread;
                    if(X_later_translation.x < -scaled_spread) X_later_translation.x = -scaled_spread;
                    if(X_later_translation.y < -scaled_spread) X_later_translation.y = -scaled_spread;

#if GAUSS_NEWTON_HEVC_IMAGE

                    /**
                     * 微分を行う
                     *
                     * - 4倍画像上でやる->1/4しか進んでいないので，最終的な微分の結果は4倍する
                     *
                     *  (x_int, y_int)     (x_int + 1, y_int)
                     *               o x x x x o
                     *               x x x x x x
                     *               x x x x x x
                     *               x x x x x x
                     *               x x x x x x
                     *               o x x x x o
                     *  (x_int, y_int+1)   (x_int + 1, y_int + 1)
                     *
                     */

                    int x_int = (int)floor(X_later_translation.x);
                    int y_int = (int)floor(X_later_translation.y);
                    double dx = X_later_translation.x - x_int;
                    double dy = X_later_translation.y - y_int;

                    double x1_slope = F(current_ref_expand, 4 * (x_int) + 1, 4 * y_int    , 4 * SEARCH_RANGE, 4 * image_width) - F(current_ref_expand, 4 * x_int, 4 * y_int    , 4 * SEARCH_RANGE, 4 * image_width);
                    double x2_slope = F(current_ref_expand, 4 * (x_int) + 1, 4 * y_int + 1, 4 * SEARCH_RANGE, 4 * image_width) - F(current_ref_expand, 4 * x_int, 4 * y_int + 1, 4 * SEARCH_RANGE, 4 * image_width);
                    double g_x_translation = 4 * (x1_slope * (1 - dy) + x2_slope * dy);

                    double y1_slope = F(current_ref_expand, 4 * (x_int)    , 4 * y_int + 1, 4 * SEARCH_RANGE, 4 * image_width) - F(current_ref_expand, 4 * x_int    , 4 * y_int, 4 * SEARCH_RANGE, 4 * image_width);
                    double y2_slope = F(current_ref_expand, 4 * (x_int) + 1, 4 * y_int + 1, 4 * SEARCH_RANGE, 4 * image_width) - F(current_ref_expand, 4 * x_int + 1, 4 * y_int, 4 * SEARCH_RANGE, 4 * image_width);

                    double g_y_translation = 4 * (y1_slope * (1 - dx) +  y2_slope * dx);

#else
                    g_x   = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x  + 1 , X_later_warping.y    , 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x  - 1, X_later_warping.y     , 1)) / 2.0;  // (current_ref_expand[x_warping_tmp + 4 ][y_warping_tmp     ] - current_ref_expand[x_warping_tmp - 4 ][y_warping_tmp     ]) / 2.0;
                            g_y   = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x     , X_later_warping.y  + 1, 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x     , X_later_warping.y  - 1, 1)) / 2.0;  // (current_ref_expand[x_warping_tmp     ][y_warping_tmp + 4 ] - current_ref_expand[x_warping_tmp     ][y_warping_tmp - 4 ]) / 2.0;
                            g_x_translation = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x + 1, X_later_translation.y    , 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x - 1, X_later_translation.y    , 1)) / 2.0;  // (current_ref_expand[x_translation_tmp + 4][y_translation_tmp    ] - current_ref_expand[x_translation_tmp - 4][y_translation_tmp    ]) / 2.0;
                            g_y_translation = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x    , X_later_translation.y + 1, 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x    , X_later_translation.y - 1, 1)) / 2.0;  // (current_ref_expand[x_translation_tmp    ][y_translation_tmp + 4] - current_ref_expand[x_translation_tmp    ][y_translation_tmp - 4]) / 2.0;
#endif

                    delta_g_translation[0] = g_x_translation;
                    delta_g_translation[1] = g_y_translation;

#if GAUSS_NEWTON_HEVC_IMAGE
                    double f              = img_ip(current_target_expand    , cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *               pixel.x, 4 *               pixel.y);
                    double f_org          = img_ip(current_target_org_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *               pixel.x, 4 *               pixel.y);
                    double g_translation  = img_ip(current_ref_expand       , cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * X_later_translation.x, 4 * X_later_translation.y);
#else
                    f              = img_ip(current_target_expand    , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),                X.x,                X.y, 2);
                    f_org          = img_ip(current_target_org_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),                X.x,                X.y, 2);
                    g_warping      = img_ip(current_ref_expand       , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),  X_later_warping.x,  X_later_warping.y, 2);
                    g_translation     = img_ip(current_ref_expand       , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x, X_later_translation.y, 2);
#endif
                    double g_org_translation;

                    if(ref_hevc != nullptr) {
                        g_org_translation = img_ip(ref_hevc, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * X_later_translation.x, 4 * X_later_translation.y);
                    }else {
#if GAUSS_NEWTON_HEVC_IMAGE
                        g_org_translation = img_ip(current_ref_org_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * X_later_translation.x, 4 * X_later_translation.y);
#else
                        g_org_warping  = img_ip(current_ref_org_expand, cv::Rect(-spread, -spread, current_target_image.cols + 2 * spread, current_target_image.rows + 2 * spread),  tmp_X_later_warping.x, tmp_X_later_warping.y, 2);
                            g_org_translation = img_ip(current_ref_org_expand, cv::Rect(-spread, -spread, current_target_image.cols + 2 * spread, current_target_image.rows + 2 * spread), tmp_X_later_translation.x, tmp_X_later_translation.y, 2);
#endif
                    }

                    E_delta_x += -2 * (f - g_translation) * delta_g_translation[0];
                    E_delta_y += -2 * (f - g_translation) * delta_g_translation[1];

                    for (int row = 0; row < 2; row++) {
                        for (int col = 0; col < 2; col++) {
                            gg_translation.at<double>(row, col) += delta_g_translation[row] * delta_g_translation[col];
                        }
                        B_translation.at<double>(row, 0) += (f - g_translation) * delta_g_translation[row];
                    }
                }

                gg_translation.at<double>(0, 0) *= (1 + alpha_marquardt);
                gg_translation.at<double>(1, 1) *= (1 + alpha_marquardt);

                cv::solve(gg_translation, B_translation, delta_uv_translation);

                // 更新量がしきい値以上であれば打ち切る
                double delta_u = delta_uv_translation.at<double>(0, 0);
                double delta_v = delta_uv_translation.at<double>(1, 0);
                if(fabs(delta_u) >= DELTA_UV_THRESHOLD || fabs(delta_v) >= DELTA_UV_THRESHOLD){
                    break;
                }

                if(translation_update_flag && prev_SSE_translation > SSE_translation) {
                    for (int k = 0; k < 2; k++) {
                        if (k % 2 == 0) {
                            double translated_x = tmp_mv_translation.x + delta_u;
                            if ((-scaled_spread <= scaled_coordinates[0].x + translated_x) &&
                                (target_images[0][step].cols - 1 + scaled_spread >= scaled_coordinates[0].x + translated_x) &&
                                (-scaled_spread <= scaled_coordinates[1].x + translated_x) &&
                                (target_images[0][step].cols - 1 + scaled_spread >= scaled_coordinates[1].x + translated_x) &&
                                (-scaled_spread <= scaled_coordinates[2].x + translated_x) &&
                                (target_images[0][step].cols - 1 + scaled_spread >= scaled_coordinates[2].x + translated_x)) {
                                tmp_mv_translation.x = translated_x;
                            }
                        } else {
                            double translated_y = tmp_mv_translation.y + delta_v;
                            if ((-scaled_spread <= scaled_coordinates[0].y + translated_y) &&
                                (target_images[0][step].rows - 1 + scaled_spread >= scaled_coordinates[0].y + translated_y) &&
                                (-scaled_spread <=scaled_coordinates[1].y + translated_y) &&
                                (target_images[0][step].rows - 1 + scaled_spread >= scaled_coordinates[1].y + translated_y) &&
                                (-scaled_spread <=scaled_coordinates[2].y + translated_y) &&
                                (target_images[0][step].rows - 1 + scaled_spread >= scaled_coordinates[2].y + translated_y)) {
                                tmp_mv_translation.y = translated_y;
                            }
                        }
                    }
                }

                // 移動後のSSEを求める
                std::vector<cv::Point2f> translation_mvs{tmp_mv_translation, tmp_mv_translation, tmp_mv_translation};
                SSE_translation = getTriangleSSE(ref_hevc, current_target_org_expand, target_corners, translation_mvs, pixels_in_triangle, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)));


                double eps = 1e-3;

                iterate_counter++;

#if STORE_NEWTON_LOG
                current_me_log.residual.emplace_back(SSE_translation);
                current_me_log.mv_newton_translation.emplace_back(tmp_mv_translation);
                current_me_log.coordinate_after_move1.emplace_back(tmp_mv_translation + p0);
                current_me_log.coordinate_after_move2.emplace_back(tmp_mv_translation + p1);
                current_me_log.coordinate_after_move3.emplace_back(tmp_mv_translation + p2);
#endif

                if(prev_SSE_translation > SSE_translation){
                    alpha_marquardt *= 0.2;
                    prev_SSE_translation = SSE_translation;
                    prev_mv_translation = tmp_mv_translation;
                    v_stack_translation.emplace_back(tmp_mv_translation, SSE_translation);
                }else{
                    alpha_marquardt *= 10;
                    tmp_mv_translation = prev_mv_translation;
                }

                if ((fabs(prev_SSE_translation - SSE_translation) / SSE_translation) < eps) {
                    translation_update_flag = false;
                }

                if(iterate_counter > 30 || !(translation_update_flag)){
                    break;
                }

                SSE_translation = 0.0;
            }

#if STORE_NEWTON_LOG
            extern std::vector<std::vector<double>> freq_newton_translation;
            freq_newton_translation[filter_num][std::min(iterate_counter, 20)]++;

            current_me_log.percentage = (fabs(current_me_log.residual.back() - current_me_log.residual.front()) / current_me_log.residual.front() * 100);
#endif

//            if(iterate_counter < 20 && !slow_newton_translation[filter_num].empty()){
//                slow_newton_translation[filter_num].erase(slow_newton_translation[filter_num].begin() + slow_newton_translation[filter_num].size() - 1);
//                mv_newton_translation[filter_num].erase(mv_newton_translation[filter_num].begin() + mv_newton_translation[filter_num].size() - 1);
//                coordinate_newton_translation1[filter_num].erase(coordinate_newton_translation1[filter_num].begin() + coordinate_newton_translation1[filter_num].size() - 1);
//                coordinate_newton_translation2[filter_num].erase(coordinate_newton_translation2[filter_num].begin() + coordinate_newton_translation2[filter_num].size() - 1);
//                coordinate_newton_translation3[filter_num].erase(coordinate_newton_translation3[filter_num].begin() + coordinate_newton_translation3[filter_num].size() - 1);
//                p0_newton_translation[filter_num].erase(p0_newton_translation[filter_num].begin() + p0_newton_translation[filter_num].size() - 1);
//                p1_newton_translation[filter_num].erase(p1_newton_translation[filter_num].begin() + p1_newton_translation[filter_num].size() - 1);
//                p2_newton_translation[filter_num].erase(p2_newton_translation[filter_num].begin() + p2_newton_translation[filter_num].size() - 1);
//
//            }else if((slow_newton_translation[filter_num][slow_newton_translation[filter_num].size() - 1][slow_newton_translation[filter_num][slow_newton_translation[filter_num].size() - 1].size() - 1]
//                      - slow_newton_translation[filter_num][slow_newton_translation[filter_num].size() - 1][0]) < 0){
//                slow_newton_translation[filter_num].erase(slow_newton_translation[filter_num].begin() + slow_newton_translation[filter_num].size() - 1);
//                mv_newton_translation[filter_num].erase(mv_newton_translation[filter_num].begin() + mv_newton_translation[filter_num].size() - 1);
//                coordinate_newton_translation1[filter_num].erase(coordinate_newton_translation1[filter_num].begin() + coordinate_newton_translation1[filter_num].size() - 1);
//                coordinate_newton_translation2[filter_num].erase(coordinate_newton_translation2[filter_num].begin() + coordinate_newton_translation2[filter_num].size() - 1);
//                coordinate_newton_translation3[filter_num].erase(coordinate_newton_translation3[filter_num].begin() + coordinate_newton_translation3[filter_num].size() - 1);
//                p0_newton_translation[filter_num].erase(p0_newton_translation[filter_num].begin() + p0_newton_translation[filter_num].size() - 1);
//                p1_newton_translation[filter_num].erase(p1_newton_translation[filter_num].begin() + p1_newton_translation[filter_num].size() - 1);
//                p2_newton_translation[filter_num].erase(p2_newton_translation[filter_num].begin() + p2_newton_translation[filter_num].size() - 1);
//            }

            std::sort(v_stack_translation.begin(), v_stack_translation.end(), [](std::pair<cv::Point2f,double> a, std::pair<cv::Point2f,double> b){
                return a.second < b.second;
            });

            tmp_mv_translation = v_stack_translation[0].first;
            double Error_translation = v_stack_translation[0].second;
            double PSNR_translation = 10 * log10((255 * 255) / (Error_translation / (double)pixels_in_triangle.size()));

            if(step == 3) {//一番下の階層で
                if(PSNR_translation >= max_PSNR_translation){//2種類のボケ方で良い方を採用
                    max_PSNR_translation = PSNR_translation;
                    min_error_translation = Error_translation;
                    max_v_translation = roundVecQuarter(tmp_mv_translation);
                }
            }
        }
    }

    /**
     *
     * ワーピングの推定
     *
     * 3本の動きベクトルを推定してワーピングする
     *
     */
#if STORE_NEWTON_LOG
    ME_log_warping_0.emplace_back();
    ME_log_warping_1.emplace_back();
#endif

    for(int filter_num = 0 ; filter_num < static_cast<int>(ref_images.size()) ; filter_num++){
        std::vector<cv::Point2f> tmp_mv_warping(3, cv::Point2f(initial_vector.x, initial_vector.y));
        bool warping_update_flag = true;

#if STORE_NEWTON_LOG
        MELog& current_me_log = (filter_num == 0 ? ME_log_warping_0.back() : ME_log_warping_1.back());
#endif

        // Marquardtの係数
        double alpha_marquardt = 0.5;

        v_stack_warping.emplace_back(tmp_mv_warping, error_bm_min);

        for(int step = 3 ; step < static_cast<int>(ref_images[filter_num].size()) ; step++){
            double SSE_warping = 0.0;
            double scale = pow(2, 3 - step);
            cv::Mat current_ref_image = ref_images[filter_num][step];
            cv::Mat current_target_image = target_images[filter_num][step];

            unsigned char *current_target_expand, *current_target_org_expand; //画像の周りに500ピクセルだけ黒の領域を設ける(念のため)
            unsigned char *current_ref_expand, *current_ref_org_expand;    //f_expandと同様

            current_ref_expand        = expand_image[filter_num][step][0];
            current_ref_org_expand    = expand_image[filter_num][step][1];
            current_target_expand     = expand_image[filter_num][step][2];
            current_target_org_expand = expand_image[filter_num][step][3];

            int spread = SEARCH_RANGE; // 探索範囲は16までなので16に戻す

            int scaled_spread = spread / scale;
            p0 = target_corners.p1 / scale;
            p1 = target_corners.p2 / scale;
            p2 = target_corners.p3 / scale;

            // 端の頂点の調整
            if (target_corners.p1.x == target_images[0][3].cols - 1) p0.x = target_images[0][step].cols - 1;
            if (target_corners.p1.y == target_images[0][3].rows - 1) p0.y = target_images[0][step].rows - 1;
            if (target_corners.p2.x == target_images[0][3].cols - 1) p1.x = target_images[0][step].cols - 1;
            if (target_corners.p2.y == target_images[0][3].rows - 1) p1.y = target_images[0][step].rows - 1;
            if (target_corners.p3.x == target_images[0][3].cols - 1) p2.x = target_images[0][step].cols - 1;
            if (target_corners.p3.y == target_images[0][3].rows - 1) p2.y = target_images[0][step].rows - 1;

            if(fabs((p2 - p0).x * (p1 - p0).y - (p2 - p0).y * (p1 - p0).x) <= 0) break;

            current_triangle_coordinates.p1 = p0;
            current_triangle_coordinates.p2 = p1;
            current_triangle_coordinates.p3 = p2;
            pixels_in_triangle = getPixelsInTriangle(current_triangle_coordinates, area_flag, triangle_index, ctu, block_size_x, block_size_y);

            std::vector<cv::Point2f> scaled_coordinates{p0, p1, p2};

            if(step != 0) {
                // 画面外にはみ出してる場合、２倍からだんだん小さく縮小していく

                // ワーピング
                double magnification = 1.0;
                while ( (p0.x + tmp_mv_warping[0].x * magnification < -scaled_spread && p0.x + tmp_mv_warping[0].x * magnification > current_target_image.cols - 1 + scaled_spread) &&
                        (p1.x + tmp_mv_warping[1].x * magnification < -scaled_spread && p1.x + tmp_mv_warping[1].x * magnification > current_target_image.cols - 1 + scaled_spread) &&
                        (p2.x + tmp_mv_warping[2].x * magnification < -scaled_spread && p2.x + tmp_mv_warping[2].x * magnification > current_target_image.cols - 1 + scaled_spread) &&
                        (p0.y + tmp_mv_warping[0].y * magnification < -scaled_spread && p0.y + tmp_mv_warping[0].y * magnification > current_target_image.rows - 1 + scaled_spread) &&
                        (p1.y + tmp_mv_warping[1].y * magnification < -scaled_spread && p1.y + tmp_mv_warping[1].y * magnification > current_target_image.rows - 1 + scaled_spread) &&
                        (p2.y + tmp_mv_warping[2].y * magnification < -scaled_spread && p2.y + tmp_mv_warping[2].y * magnification > current_target_image.rows - 1 + scaled_spread) ) {
                    if(magnification <= 1)break;
                    magnification -= 0.1;
                }
                for (int s = 0; s < 3; s++) tmp_mv_warping[s] *= magnification;

            }
            v_stack_warping.clear();

            v_stack_warping.emplace_back(tmp_mv_warping, error_bm_min);

            double prev_SSE_warping = error_bm_min;
            std::vector<cv::Point2f> prev_mv_warping{initial_vector, initial_vector, initial_vector};

#if STORE_NEWTON_LOG
            current_me_log.mv_newton_warping.emplace_back(tmp_mv_warping);
            current_me_log.residual.emplace_back(error_bm_min);
            current_me_log.coordinate_after_move1.emplace_back();
            current_me_log.coordinate_after_move2.emplace_back();
            current_me_log.coordinate_after_move3.emplace_back();
#endif

            int iterate_counter = 0;
            while(true){
                // 移動後の座標を格納する
                std::vector<cv::Point2f> ref_coordinates_warping;
                SSE_warping = 0.0;

                ref_coordinates_warping.emplace_back(p0);
                ref_coordinates_warping.emplace_back(p1);
                ref_coordinates_warping.emplace_back(p2);

                cv::Point2f a = p2 - p0;
                cv::Point2f b = p1 - p0;
                double det = a.x * b.y - a.y * b.x;
                // tmp_mv_warping, tmp_mv_translationは現在の動きベクトル
                // 初回は初期値が入ってる
                cv::Point2f c = tmp_mv_warping[2] - tmp_mv_warping[0];
                cv::Point2f d = tmp_mv_warping[1] - tmp_mv_warping[0];

                double S[6];
                S[0] = -0.5*(a.y + c.y - b.y - d.y);
                S[1] = -0.5*(b.x + d.x - a.x - c.x);
                S[2] = 0.5*(a.y + c.y);
                S[3] = -0.5*(a.x + c.x);
                S[4] = -0.5*(b.y + d.y);
                S[5] = 0.5*(b.x + d.x);

                gg_warping = cv::Mat::zeros(warping_matrix_dim, warping_matrix_dim, CV_64F);
                B_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F);
                delta_uv_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F);

                double delta_g_warping[warping_matrix_dim] = {0};

                cv::Point2f X;
                double RMSE_warping_filter = 0;
                for(auto pixel : pixels_in_triangle) {
                    X.x = pixel.x - p0.x;
                    X.y = pixel.y - p0.y;

                    double alpha = (X.x * b.y - X.y * b.x) / det;
                    double beta = (a.x * X.y - a.y * X.x)/ det;
                    X.x += p0.x;
                    X.y += p0.y;

                    // 移動後の頂点を計算し格納
                    std::vector<cv::Point2f> triangle_later_warping(3);
                    triangle_later_warping[0] = p0 + tmp_mv_warping[0];
                    triangle_later_warping[1] = p1 + tmp_mv_warping[1];
                    triangle_later_warping[2] = p2 + tmp_mv_warping[2];

                    cv::Point2f a_later_warping  =  triangle_later_warping[2] -  triangle_later_warping[0];
                    cv::Point2f b_later_warping  =  triangle_later_warping[1] -  triangle_later_warping[0];
                    cv::Point2f  X_later_warping  = alpha *  a_later_warping + beta *  b_later_warping +  triangle_later_warping[0];

                    if(X_later_warping.x >= current_ref_image.cols - 1 + scaled_spread) X_later_warping.x = current_ref_image.cols - 1.00 + scaled_spread;
                    if(X_later_warping.y >= current_ref_image.rows - 1 + scaled_spread) X_later_warping.y = current_ref_image.rows - 1.00 + scaled_spread;
                    if(X_later_warping.x < -scaled_spread) X_later_warping.x = -scaled_spread;
                    if(X_later_warping.y < -scaled_spread) X_later_warping.y = -scaled_spread;


                    #if GAUSS_NEWTON_HEVC_IMAGE
                    //
                    // (x_int, y_int)     (x_int + 1, y_int)
                    //             o x x x x o
                    //             x x x x x x
                    //             x x x x x x
                    //             x x x x x x
                    //             x x x x x x
                    //             o x x x x o
                    // (x_int, y_int+1)   (x_int + 1, y_int + 1)
                    //

                    int x_int = (int)floor(X_later_warping.x);
                    int y_int = (int)floor(X_later_warping.y);
                    double dx = X_later_warping.x - x_int;
                    double dy = X_later_warping.y - y_int;


                    double x1_slope = F(current_ref_expand, 4 * (x_int) + 1, 4 * y_int    , 4 * SEARCH_RANGE, 4 * image_width) - F(current_ref_expand, 4 * x_int, 4 * y_int    , 4 * SEARCH_RANGE, 4 * image_width);
                    double x2_slope = F(current_ref_expand, 4 * (x_int) + 1, 4 * y_int + 1, 4 * SEARCH_RANGE, 4 * image_width) - F(current_ref_expand, 4 * x_int, 4 * y_int + 1, 4 * SEARCH_RANGE, 4 * image_width);
                    double g_x = 4 * (x1_slope * (1 - dy) + x2_slope * dy);


                    double y1_slope = F(current_ref_expand, 4 * (x_int)    , 4 * y_int + 1, 4 * SEARCH_RANGE, 4 * image_width) - F(current_ref_expand, 4 * x_int    , 4 * y_int, 4 * SEARCH_RANGE, 4 * image_width);
                    double y2_slope = F(current_ref_expand, 4 * (x_int) + 1, 4 * y_int + 1, 4 * SEARCH_RANGE, 4 * image_width) - F(current_ref_expand, 4 * x_int + 1, 4 * y_int, 4 * SEARCH_RANGE, 4 * image_width);

                    double g_y = 4 * (y1_slope * (1 - dx) +  y2_slope * dx);
                    #else
                        g_x   = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x  + 1 , X_later_warping.y    , 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x  - 1, X_later_warping.y     , 1)) / 2.0;  // (current_ref_expand[x_warping_tmp + 4 ][y_warping_tmp     ] - current_ref_expand[x_warping_tmp - 4 ][y_warping_tmp     ]) / 2.0;
                            g_y   = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x     , X_later_warping.y  + 1, 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x     , X_later_warping.y  - 1, 1)) / 2.0;  // (current_ref_expand[x_warping_tmp     ][y_warping_tmp + 4 ] - current_ref_expand[x_warping_tmp     ][y_warping_tmp - 4 ]) / 2.0;
                            g_x_translation = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x + 1, X_later_translation.y    , 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x - 1, X_later_translation.y    , 1)) / 2.0;  // (current_ref_expand[x_translation_tmp + 4][y_translation_tmp    ] - current_ref_expand[x_translation_tmp - 4][y_translation_tmp    ]) / 2.0;
                            g_y_translation = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x    , X_later_translation.y + 1, 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x    , X_later_translation.y - 1, 1)) / 2.0;  // (current_ref_expand[x_translation_tmp    ][y_translation_tmp + 4] - current_ref_expand[x_translation_tmp    ][y_translation_tmp - 4]) / 2.0;
                    #endif


                    for(int i = 0 ; i < 6 ; i++) {
                        // 頂点を動かしたときのパッチ内の変動量x軸y軸独立に計算(delta_gを求めるために必要)
                        double delta_x, delta_y;
                        switch (i) {//頂点ごとにxy軸独立に偏微分
                            case 0: // u0
                                delta_x = 1 - alpha - beta;
                                delta_y = 0;
                                break;
                            case 1: // v0
                                delta_x = 0;
                                delta_y = 1 - alpha - beta;
                                break;
                            case 2: // u1
                                delta_x = beta;
                                delta_y = 0;
                                break;
                            case 3: // v1
                                delta_x = 0;
                                delta_y = beta;
                                break;
                            case 4: // u2
                                delta_x = alpha;
                                delta_y = 0;
                                break;
                            case 5: // v2
                                delta_x = 0;
                                delta_y = alpha;
                                break;
                            default:
                                break;
                        }

                        // 式(28)～(33)
                        delta_g_warping[i] = g_x * delta_x + g_y * delta_y;
                    }

                    double f;
                    double f_org;
                    double g_warping;

#if GAUSS_NEWTON_HEVC_IMAGE
                    f              = img_ip(current_target_expand    , cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *                X.x, 4 *                X.y);
                    f_org          = img_ip(current_target_org_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *                X.x, 4 *                X.y);
                    g_warping      = img_ip(current_ref_expand       , cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *  X_later_warping.x, 4 *  X_later_warping.y);
#else
                    f              = img_ip(current_target_expand    , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),                X.x,                X.y, 2);
                    f_org          = img_ip(current_target_org_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),                X.x,                X.y, 2);
                    g_warping      = img_ip(current_ref_expand       , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),  X_later_warping.x,  X_later_warping.y, 2);
                    g_translation     = img_ip(current_ref_expand       , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x, X_later_translation.y, 2);
#endif
                    double g_org_warping;

                    if(ref_hevc != nullptr) {
                        g_org_warping  = img_ip(ref_hevc, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * X_later_warping.x, 4 * X_later_warping.y);
                    }else {
#if GAUSS_NEWTON_HEVC_IMAGE
                        g_org_warping  = img_ip(current_ref_org_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * X_later_warping.x, 4 * X_later_warping.y);
#else
                        g_org_warping  = img_ip(current_ref_org_expand, cv::Rect(-spread, -spread, current_target_image.cols + 2 * spread, current_target_image.rows + 2 * spread),  tmp_X_later_warping.x, tmp_X_later_warping.y, 2);
                            g_org_translation = img_ip(current_ref_org_expand, cv::Rect(-spread, -spread, current_target_image.cols + 2 * spread, current_target_image.rows + 2 * spread), tmp_X_later_translation.x, tmp_X_later_translation.y, 2);
#endif
                    }


                    for (int row = 0; row < warping_matrix_dim; row++) {
                        for (int col = 0; col < warping_matrix_dim; col++) {
                            gg_warping.at<double>(row, col) += delta_g_warping[row] * delta_g_warping[col];//A_0の行列を生成(左辺の6x6の行列に相当)
                        }
                        B_warping.at<double>(row, 0) += (f - g_warping) * delta_g_warping[row];//bの行列を生成(右辺の6x1のベクトルに相当)
                    }

                }


                for(int k = 0 ; k < warping_matrix_dim ; k++){
                    gg_warping.at<double>(k, k) *= (1 + alpha_marquardt);
                }

                cv::solve(gg_warping, B_warping, delta_uv_warping); //6x6の連立方程式を解いてdelta_uvに格納

                // delta_uvの値がしきい値を超えたら更新を終了する
                bool delta_uv_threshold_flag = false;
                for(int row = 0 ; row < warping_matrix_dim ; row++ ){
                    if(fabs(delta_uv_warping.at<double>(row, 0)) >= DELTA_UV_THRESHOLD){
                        delta_uv_threshold_flag = true;
                        break;
                    }
                }
                if(delta_uv_threshold_flag) break;

                if(warping_update_flag && prev_SSE_warping > SSE_warping) {
                    for (int k = 0; k < 6; k++) {
                        if (k % 2 == 0) {
                            if ((-scaled_spread <=
                                 scaled_coordinates[(int) (k / 2)].x + tmp_mv_warping[(int) (k / 2)].x +
                                 delta_uv_warping.at<double>(k, 0)) &&
                                (target_images[0][step].cols - 1 + scaled_spread >=
                                 scaled_coordinates[(int) (k / 2)].x + tmp_mv_warping[(int) (k / 2)].x +
                                 delta_uv_warping.at<double>(k, 0))) {
                                tmp_mv_warping[(int) (k / 2)].x = tmp_mv_warping[(int) (k / 2)].x +
                                                                  delta_uv_warping.at<double>(k,
                                                                                              0);//動きベクトルを更新(画像の外に出ないように)
                            }
                        } else {
                            if ((-scaled_spread <=
                                 scaled_coordinates[(int) (k / 2)].y + tmp_mv_warping[(int) (k / 2)].y +
                                 delta_uv_warping.at<double>(k, 0)) &&
                                (target_images[0][step].rows - 1 + scaled_spread >=
                                 scaled_coordinates[(int) (k / 2)].y + tmp_mv_warping[(int) (k / 2)].y +
                                 delta_uv_warping.at<double>(k, 0))) {
                                tmp_mv_warping[(int) (k / 2)].y =
                                        tmp_mv_warping[(int) (k / 2)].y + delta_uv_warping.at<double>(k, 0);
                            }
                        }
                    }
                }

                SSE_warping = getTriangleSSE(ref_hevc, current_target_org_expand, target_corners, tmp_mv_warping, pixels_in_triangle, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)));

                iterate_counter++;
                double eps = 1e-3;

#if STORE_NEWTON_LOG
                current_me_log.mv_newton_warping.emplace_back(tmp_mv_warping);
                current_me_log.residual.emplace_back(SSE_warping);
                current_me_log.coordinate_after_move1.emplace_back(tmp_mv_warping[0] + p0);
                current_me_log.coordinate_after_move2.emplace_back(tmp_mv_warping[1] + p1);
                current_me_log.coordinate_after_move3.emplace_back(tmp_mv_warping[2] + p2);
#endif

                if(prev_SSE_warping > SSE_warping){
                    alpha_marquardt *= 0.2;
                    prev_SSE_warping = SSE_warping;
                    prev_mv_warping = tmp_mv_warping;
                    v_stack_warping.emplace_back(tmp_mv_warping, SSE_warping);
                }else{
                    alpha_marquardt *= 10;
                    tmp_mv_warping = prev_mv_warping;
                }

                if ((fabs(prev_SSE_warping - SSE_warping) / SSE_warping < eps)) {
                    warping_update_flag = false;
                }

                if(iterate_counter > 20 || !(warping_update_flag)){
                    break;
                }

            }

#if STORE_NEWTON_LOG
            extern std::vector<std::vector<double>> freq_newton_warping;
            freq_newton_warping[filter_num][std::min(iterate_counter, 20)]++;

            current_me_log.percentage = (fabs(current_me_log.residual.back() - current_me_log.residual.front()) / current_me_log.residual.front() * 100);
#endif

            std::sort(v_stack_warping.begin(), v_stack_warping.end(), [](std::pair<std::vector<cv::Point2f>,double> a, std::pair<std::vector<cv::Point2f>,double> b){
                return a.second < b.second;
            });

            tmp_mv_warping = v_stack_warping[0].first;//一番良い動きベクトルを採用
            double Error_warping = v_stack_warping[0].second;
            double PSNR_warping = 10 * log10((255 * 255) / (Error_warping / (double)pixels_in_triangle.size()));

            if(step == 3) {//一番下の階層で
                if (PSNR_warping >= max_PSNR_warping) {
                    max_PSNR_warping = PSNR_warping;
                    min_error_warping = Error_warping;
                    max_v_warping = tmp_mv_warping;
                }
            }
        }
    }

    // 量子化
    double quantize_offset = 0.125;
    if(max_v_translation.x < 0) {
        max_v_translation.x = ((int)((max_v_translation.x - quantize_offset) * 4) / 4.0);
    }else{
        max_v_translation.x = ((int)((max_v_translation.x + quantize_offset) * 4) / 4.0);
    }

    if(max_v_translation.y < 0) {
        max_v_translation.y = ((int) ((max_v_translation.y - quantize_offset) * 4) / 4.0);
    }else{
        max_v_translation.y = ((int) ((max_v_translation.y + quantize_offset) * 4) / 4.0);
    }

    for(int i = 0 ; i < 3 ; i++){
        if(max_v_warping[i].x < 0) max_v_warping[i].x -= quantize_offset;
        else max_v_warping[i].x += quantize_offset;

        if(max_v_warping[i].y < 0) max_v_warping[i].y -= quantize_offset;
        else max_v_warping[i].y += quantize_offset;

        max_v_warping[i].x = ((int)((max_v_warping[i].x) * 4) / 4.0);
        max_v_warping[i].y = ((int)((max_v_warping[i].y) * 4) / 4.0);
    }

    return std::make_tuple(std::vector<cv::Point2f>{max_v_warping[0], max_v_warping[1], max_v_warping[2]}, max_v_translation, min_error_warping, min_error_translation, pixels_in_triangle.size());
}

/**
 * @fn
 * @brief
 * @param ref_triangle_coordinate 参照パッチの各点の座標
 * @param ref_mvs 参照パッチの各点の動きベクトル
 * @param target_triangle_coordinate 符号化対象パッチの頂点の座標
 * @return 予測した動きベクトル
 */
std::vector<cv::Point2f> getPredictedWarpingMv(std::vector<cv::Point2f>& ref_triangle_coordinate, std::vector<cv::Point2f>& ref_mvs, std::vector<cv::Point2f>& target_triangle_coordinate){
    cv::Point2f p0,p1,p2;
    p0 = ref_triangle_coordinate[0];
    p1 = ref_triangle_coordinate[1];
    p2 = ref_triangle_coordinate[2];
    cv::Point2f a = p1 - p0;
    cv::Point2f b = p2 - p0;

    std::vector<cv::Point2f> v;

    double det = a.x * b.y - a.y * b.x;

    for (int i = 0; i < static_cast<int>(target_triangle_coordinate.size()); i++) {
        cv::Point2f target = target_triangle_coordinate[i] - ref_triangle_coordinate[0];
        double alpha = (target.x * b.y - target.y * b.x) / det;
        double beta = (a.x * target.y - a.y * target.x) / det;
        cv::Point2f X_ = alpha * (p1 + ref_mvs[1] - (p0 + ref_mvs[0])) + beta * (p2 + ref_mvs[2] - (p0 + ref_mvs[0])) + p0 + ref_mvs[0];
        v.emplace_back(roundVecQuarter(X_ - target_triangle_coordinate[i]));
    }

    return v;
}
