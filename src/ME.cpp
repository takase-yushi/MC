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
#include "../includes/DelaunayTriangulation.hpp"
#include "../includes/Utils.h"
#include "../includes/ImageUtil.h"
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
    int SX = 16; // ブロックマッチングの探索範囲(X)
    int SY = 16; // ブロックマッチングの探索範囲(Y)

    double e, error_min;
    int e_count;

    error_min = 1 << 20;
    cv::Point2d xp(0.0, 0.0);
    cv::Point2f mv_min;
    int spread_quarter = 64;
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
    int spread_quarter = 64;
    int s = 4;                   //4 : Full-pel, 2 : Half-pel, 1 : Quarter-pel
    std::vector<cv::Point2f> pixels = getPixelsInTriangle(triangle, area_flag, triangle_index, ctu, 128, 128);

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
                        e += fabs(M(expansion_ref_image, ref_x, ref_y) - M(target_image, (int)pixel.x, (int)pixel.y));
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
                    e += fabs(M(expansion_ref_image, ref_x, ref_y) - M(target_image, (int)pixel.x, (int)pixel.y));
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
                    e += fabs(M(expansion_ref_image, ref_x, ref_y) - M(target_image, (int)pixel.x, (int)pixel.y));
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
 * @fn std::vector<cv::Point2f> warping(cv::Mat &prev_gray, cv::Mat &current_gray, cv::Mat &prev_color, cv::Mat &current_color, double &error_warp, Point3Vec target_corners, Point3Vec ref_corners)
 * @brief ワーピングを行い、動きベクトルと誤差を返す
 * @param[in]  prev_gray      参照画像のグレースケール画像
 * @param[in]  current_gray   対象画像のグレースケール画像
 * @param[in]  prev_color     参照画像のカラー画像
 * @param[in]  current_color  対象画像のカラー画像
 * @param[out] error_warp     ワーピングの誤差
 * @param[in]  target_corners        三角点を構成する3つの頂点
 * @return std::vector<cv::Point2f> 三角形3点の移動先座標prev_cornersを返す. corners[i]の動きベクトルはprev_corners[i]に格納される.
 */
std::vector<cv::Point2f> warping(const cv::Mat& prev_color, const cv::Mat& current_color,
                                 double &error_warp, Point3Vec target_corners, Point3Vec& ref_corners) {
    std::vector<cv::Point2f> prev_corners;
    prev_corners.emplace_back(ref_corners.p1); // こいつはハーフペルになっている
    prev_corners.emplace_back(ref_corners.p2);
    prev_corners.emplace_back(ref_corners.p3);

    cv::Mat expansion_ref = bilinearInterpolation(prev_color.clone());

    // 示す先が範囲外の場合
    for (int i = 0; i < 3; i++) {
        if (prev_corners[i].x < 0 || expansion_ref.cols <= prev_corners[i].x || prev_corners[i].y < 0 || expansion_ref.rows <= prev_corners[i].y) {
            error_warp = 10000;
            return prev_corners;
        }
    }

    cv::Point2f xp(0.0, 0.0), va(0.0, 0.0), vb(0.0, 0.0), ta, tb, tc;

    ta = target_corners.p1;
    tb = target_corners.p2;
    tc = target_corners.p3;

    double mmx, mmy;
    double ii, jj;
    unsigned char rr_prev, gg_prev, bb_prev;
    double error_tmp = 0.0;
    int error_count = 0;

    double sx = std::min({(int)target_corners.p1.x, (int)target_corners.p2.x, (int)target_corners.p3.x});
    double lx = std::max({(int)target_corners.p1.x, (int)target_corners.p2.x, (int)target_corners.p3.x});
    double sy = std::min({(int)target_corners.p1.y, (int)target_corners.p2.y, (int)target_corners.p3.y});
    double ly = std::max({(int)target_corners.p1.y, (int)target_corners.p2.y, (int)target_corners.p3.y});

    for (int j = (int) (round(sy) - 1); j <= round(ly) + 1; j++) {
        for (int i = (int) (round(sx) - 1); i <= round(lx) + 1; i++) {
            xp.x = (float)i; xp.y = (float)j;

            // (i, j)がvecで構成される三角形の内部であるとき
            if (isInTriangle(target_corners, xp) == 1) {
                // cからaへのベクトル
                va.x = ta.x - tc.x;
                va.y = ta.y - tc.y;
                // cからbへのベクトル
                vb.x = tb.x - tc.x;
                vb.y = tb.y - tc.y;

                // Pっぽい？
                double p = (((double) i - tc.x) * va.y) - (((double) j - tc.y) * va.x);
                p /= ((va.y * vb.x) - (va.x * vb.y));

                // lを求める
                double l = ((vb.x - va.x) * ((double) j - tc.y)) - ((vb.y - va.y) * ((double) i - tc.x));
                l /= ((va.y * vb.x) - (va.x * vb.y));

                // kを求める
                double k = (l != 0 ? p / l : 0.0);

                mmx = (prev_corners[1].x - prev_corners[0].x) * k + prev_corners[0].x;
                mmy = (prev_corners[1].y - prev_corners[0].y) * k + prev_corners[0].y;

                ii = (mmx - prev_corners[2].x) * l + prev_corners[2].x;
                jj = (mmy - prev_corners[2].y) * l + prev_corners[2].y;

                // 補間
                interpolation(expansion_ref, ii, jj, rr_prev, gg_prev, bb_prev);

                error_tmp += fabs(M(current_color, i, j) - (0.299 * (double) rr_prev + 0.587 * (double) gg_prev + 0.114 * (double) bb_prev));
                error_count++;
            }
        }
    }

    error_tmp /= (double) error_count;

    error_warp = error_tmp;

    return prev_corners;
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
 * @param[in] parallel_flag
 * @return 2乗誤差
 */
double getPredictedImage(unsigned char **expand_ref, cv::Mat& target_image, cv::Mat& output_image, Point3Vec& triangle, std::vector<cv::Point2f>& mv, int offset, std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, cv::Rect block_size, unsigned char **ref_hevc) {
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
            y = img_ip(ref_hevc, cv::Rect(-64, -64, 4 * (target_image.cols + 2 * 16), 4 * (target_image.rows + 2 * 16)), 4 * X_later.x, 4 * X_later.y, 1);
        }else{
            std::cout << X_later.x << " " << X_later.y << std::endl;
            y = bicubic_interpolation(expand_ref, X_later.x, X_later.y);
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
std::tuple<std::vector<cv::Point2f>, cv::Point2f, double, int, bool> GaussNewton(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, std::vector<std::vector<std::vector<unsigned char **>>> expand_image, Point3Vec target_corners, const std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, int block_size_x, int block_size_y, cv::Point2f init_vector, unsigned char **ref_hevc){
    // 画像の初期化 vector[filter][picture_number]

    const int warping_matrix_dim = 6; // 方程式の次元
    const int parallel_matrix_dim = 2;
    cv::Mat gg_warping = cv::Mat::zeros(warping_matrix_dim, warping_matrix_dim, CV_64F); // 式(45)の左辺6×6行列
    cv::Mat gg_parallel = cv::Mat::zeros(parallel_matrix_dim, parallel_matrix_dim, CV_64F); // 式(52)の左辺2×2行列
    cv::Mat B_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F); // 式(45)の右辺
    cv::Mat B_parallel = cv::Mat::zeros(parallel_matrix_dim, 1, CV_64F); // 式(52)の右辺
    cv::Mat delta_uv_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F); // 式(45)の左辺 delta
    cv::Mat delta_uv_parallel = cv::Mat::zeros(parallel_matrix_dim, 1, CV_64F); // 式(52)の右辺 delta

    double MSE_warping, MSE_parallel;
    double min_error_warping = 1E6, min_error_parallel = 1E6;
    double max_PSNR_warping = -1, max_PSNR_parallel = -1;

    cv::Point2f p0, p1, p2;
    std::vector<cv::Point2f> max_v_warping;
    cv::Point2f max_v_parallel;

    std::vector<std::pair<std::vector<cv::Point2f>,double>> v_stack_warping;
    std::vector<std::pair<cv::Point2f,double>> v_stack_parallel;
    std::vector<cv::Point2f> pixels_in_triangle;

    bool parallel_flag = true;

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
    double error_min = 1e9;

    if(init_vector.x == -1000 && init_vector.y == -1000) {
        for (int by = -bm_y_offset; by < bm_y_offset; by++) {
            for (int bx = -bm_x_offset; bx < bm_x_offset; bx++) {
                if (sx + bx < -16 || ref_images[0][3].cols + 16 <= (lx + bx) || sy + by < -16 ||
                    ref_images[0][3].rows + 16 <= (ly + by))
                    continue;
                double error_tmp = 0.0;
                for (const auto &pixel : pixels_in_triangle) {
                    error_tmp += abs(expand_image[0][3][1][(int) (pixel.x + bx)][(int) (pixel.y + by)] -
                                     expand_image[0][3][3][(int) (pixel.x)][(int) (pixel.y)]);
                }
                if (error_min > error_tmp) {
                    error_min = error_tmp;
                    initial_vector.x = bx;
                    initial_vector.y = by;
                }
            }
        }
    }else{
        initial_vector.x = init_vector.x;
        initial_vector.y = init_vector.y;
    }

    initial_vector /= 2.0;
    for(int filter_num = 0 ; filter_num < static_cast<int>(ref_images.size()) ; filter_num++){
        std::vector<cv::Point2f> tmp_mv_warping(3, cv::Point2f(initial_vector.x, initial_vector.y));
        cv::Point2f tmp_mv_parallel(initial_vector.x, initial_vector.y);

        for(int step = 3 ; step < static_cast<int>(ref_images[filter_num].size()) ; step++){

            double scale = pow(2, 3 - step);
            cv::Mat current_ref_image = ref_images[filter_num][step];
            cv::Mat current_target_image = target_images[filter_num][step];

            unsigned char **current_target_expand, **current_target_org_expand; //画像の周りに500ピクセルだけ黒の領域を設ける(念のため)
            unsigned char **current_ref_expand, **current_ref_org_expand;    //f_expandと同様

            current_ref_expand        = expand_image[filter_num][step][0];
            current_ref_org_expand    = expand_image[filter_num][step][1];
            current_target_expand     = expand_image[filter_num][step][2];
            current_target_org_expand = expand_image[filter_num][step][3];

            int spread = 16; // 探索範囲は16までなので16に戻す

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
                double magnification = 2.0;
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

                // 平行移動
                magnification = 2.0;
                while ( (p0.x + tmp_mv_parallel.x * magnification < -scaled_spread && p0.x + tmp_mv_parallel.x * magnification > current_target_image.cols - 1 + scaled_spread) &&
                        (p1.x + tmp_mv_parallel.x * magnification < -scaled_spread && p1.x + tmp_mv_parallel.x * magnification > current_target_image.cols - 1 + scaled_spread) &&
                        (p2.x + tmp_mv_parallel.x * magnification < -scaled_spread && p2.x + tmp_mv_parallel.x * magnification > current_target_image.cols - 1 + scaled_spread) &&
                        (p0.y + tmp_mv_parallel.y * magnification < -scaled_spread && p0.y + tmp_mv_parallel.y * magnification > current_target_image.rows - 1 + scaled_spread) &&
                        (p1.y + tmp_mv_parallel.y * magnification < -scaled_spread && p1.y + tmp_mv_parallel.y * magnification > current_target_image.rows - 1 + scaled_spread) &&
                        (p2.y + tmp_mv_parallel.y * magnification < -scaled_spread && p2.y + tmp_mv_parallel.y * magnification > current_target_image.rows - 1 + scaled_spread) ) {
                    if(magnification <= 1)break;
                    magnification -= 0.1;
                }
                tmp_mv_parallel *= magnification;
            }
            v_stack_parallel.clear();
            v_stack_warping.clear();

            double prev_error_warping = 1e6, prev_error_parallel = 1e6;
            cv::Point2f prev_mv_parallel;
            std::vector<cv::Point2f> prev_mv_warping;
            bool warping_update_flag = true, parallel_update_flag = true;

            int iterate_counter = 0;
            while(true){
                // 移動後の座標を格納する
                std::vector<cv::Point2f> ref_coordinates_warping;
                std::vector<cv::Point2f> ref_coordinates_parallel;

                ref_coordinates_warping.emplace_back(p0);
                ref_coordinates_warping.emplace_back(p1);
                ref_coordinates_warping.emplace_back(p2);

                ref_coordinates_parallel.emplace_back(p0);
                ref_coordinates_parallel.emplace_back(p1);
                ref_coordinates_parallel.emplace_back(p2);

                cv::Point2f a = p2 - p0;
                cv::Point2f b = p1 - p0;
                double det = a.x * b.y - a.y * b.x;
                // tmp_mv_warping, tmp_mv_parallelは現在の動きベクトル
                // 初回は初期値が入ってる
                cv::Point2f c = tmp_mv_warping[2] - tmp_mv_warping[0];
                cv::Point2f d = tmp_mv_warping[1] - tmp_mv_warping[0];

                double area_before_move = 0.5 * fabs(det); // 移動前の面積
                double area_after_move = 0.5 * fabs((b.x + d.x)*(a.y + c.y) - (a.x + c.x)*(b.y + d.y)); // 移動後の面積

                double S[6];
                S[0] = -0.5*(a.y + c.y - b.y - d.y);
                S[1] = -0.5*(b.x + d.x - a.x - c.x);
                S[2] = 0.5*(a.y + c.y);
                S[3] = -0.5*(a.x + c.x);
                S[4] = -0.5*(b.y + d.y);
                S[5] = 0.5*(b.x + d.x);

                MSE_parallel = MSE_warping = 0.0;
                gg_warping = cv::Mat::zeros(warping_matrix_dim, warping_matrix_dim, CV_64F);
                B_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F);
                delta_uv_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F);

                gg_parallel = cv::Mat::zeros(parallel_matrix_dim, parallel_matrix_dim, CV_64F);
                B_parallel = cv::Mat::zeros(parallel_matrix_dim, 1, CV_64F);
                delta_uv_parallel = cv::Mat::zeros(parallel_matrix_dim, 1, CV_64F);

                double delta_g_warping[warping_matrix_dim] = {0};
                double delta_g_parallel[parallel_matrix_dim] = {0};

                cv::Point2f X;
                double RMSE_warping_filter = 0;
                double RMSE_parallel_filter = 0;
                for(auto pixel : pixels_in_triangle) {
                    X.x = pixel.x - p0.x;
                    X.y = pixel.y - p0.y;

                    double alpha = (X.x * b.y - X.y * b.x) / det;
                    double beta = (a.x * X.y - a.y * X.x)/ det;
                    X.x += p0.x;
                    X.y += p0.y;

                    int x_integer = (int)floor(X.x);
                    int y_integer = (int)floor(X.y);
                    int x_decimal = X.x - x_integer;
                    int y_decimal = X.y - y_integer;

                    // 参照フレームの前進差分（平行移動）
                    double g_x_parallel;
                    double g_y_parallel;
                    cv::Point2f X_later_parallel, X_later_warping;

                    for(int i = 0 ; i < 6 ; i++) {
                        // 移動後の頂点を計算し格納
                        ref_coordinates_warping[0] = p0 + tmp_mv_warping[0];
                        ref_coordinates_warping[1] = p1 + tmp_mv_warping[1];
                        ref_coordinates_warping[2] = p2 + tmp_mv_warping[2];
                        ref_coordinates_parallel[0] = p0 + tmp_mv_parallel;
                        ref_coordinates_parallel[1] = p1 + tmp_mv_parallel;
                        ref_coordinates_parallel[2] = p2 + tmp_mv_parallel;

                        std::vector<cv::Point2f> triangle_later_warping;
                        std::vector<cv::Point2f> triangle_later_parallel;
                        triangle_later_warping.emplace_back(ref_coordinates_warping[0]);
                        triangle_later_warping.emplace_back(ref_coordinates_warping[1]);
                        triangle_later_warping.emplace_back(ref_coordinates_warping[2]);
                        triangle_later_parallel.emplace_back(ref_coordinates_parallel[0]);
                        triangle_later_parallel.emplace_back(ref_coordinates_parallel[1]);
                        triangle_later_parallel.emplace_back(ref_coordinates_parallel[2]);

                        cv::Point2f a_later_warping, a_later_parallel;
                        cv::Point2f b_later_warping, b_later_parallel;

                        a_later_warping  =  triangle_later_warping[2] -  triangle_later_warping[0];
                        a_later_parallel = triangle_later_parallel[2] - triangle_later_parallel[0];
                        b_later_warping  =  triangle_later_warping[1] -  triangle_later_warping[0];
                        b_later_parallel = triangle_later_parallel[1] - triangle_later_parallel[0];
                        X_later_warping  = alpha *  a_later_warping + beta *  b_later_warping +  triangle_later_warping[0];
                        X_later_parallel = alpha * a_later_parallel + beta * b_later_parallel + triangle_later_parallel[0];

                        if(X_later_warping.x >= current_ref_image.cols - 1 + scaled_spread) X_later_warping.x = current_ref_image.cols - 1.00 + scaled_spread;
                        if(X_later_warping.y >= current_ref_image.rows - 1 + scaled_spread) X_later_warping.y = current_ref_image.rows - 1.00 + scaled_spread;
                        if(X_later_warping.x < -scaled_spread) X_later_warping.x = -scaled_spread;
                        if(X_later_warping.y < -scaled_spread) X_later_warping.y = -scaled_spread;

                        if(X_later_parallel.x >= (current_ref_image.cols - 1 + scaled_spread)) X_later_parallel.x = current_ref_image.cols - 1 + scaled_spread;
                        if(X_later_parallel.y >= (current_ref_image.rows - 1 + scaled_spread)) X_later_parallel.y = current_ref_image.rows - 1 + scaled_spread;
                        if(X_later_parallel.x < -scaled_spread) X_later_parallel.x = -scaled_spread;
                        if(X_later_parallel.y < -scaled_spread) X_later_parallel.y = -scaled_spread;

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

                        // 参照フレームの中心差分
                        spread+=1;
                        double g_x   = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x  + 1 , X_later_warping.y    , 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x  - 1, X_later_warping.y     , 1)) / 2.0;  // (current_ref_expand[x_warping_tmp + 4 ][y_warping_tmp     ] - current_ref_expand[x_warping_tmp - 4 ][y_warping_tmp     ]) / 2.0;
                        double g_y   = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x     , X_later_warping.y  + 1, 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x     , X_later_warping.y  - 1, 1)) / 2.0;  // (current_ref_expand[x_warping_tmp     ][y_warping_tmp + 4 ] - current_ref_expand[x_warping_tmp     ][y_warping_tmp - 4 ]) / 2.0;
                        g_x_parallel = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_parallel.x + 1, X_later_parallel.y    , 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_parallel.x - 1, X_later_parallel.y    , 1)) / 2.0;  // (current_ref_expand[x_parallel_tmp + 4][y_parallel_tmp    ] - current_ref_expand[x_parallel_tmp - 4][y_parallel_tmp    ]) / 2.0;
                        g_y_parallel = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_parallel.x    , X_later_parallel.y + 1, 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_parallel.x    , X_later_parallel.y - 1, 1)) / 2.0;  // (current_ref_expand[x_parallel_tmp    ][y_parallel_tmp + 4] - current_ref_expand[x_parallel_tmp    ][y_parallel_tmp - 4]) / 2.0;
                        spread-=1;
                        
                        // 式(28)～(33)
                        delta_g_warping[i] = g_x * delta_x + g_y * delta_y;
                    }
                    delta_g_parallel[0] = g_x_parallel;
                    delta_g_parallel[1] = g_y_parallel;

                    double f              = img_ip(current_target_expand    , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),                X.x,                X.y, 2);
                    double f_org          = img_ip(current_target_org_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),                X.x,                X.y, 2);
                    double g_warping      = img_ip(current_ref_expand       , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),  X_later_warping.x,  X_later_warping.y, 2);
                    double g_parallel     = img_ip(current_ref_expand       , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_parallel.x, X_later_parallel.y, 2);
                    double g_org_warping;
                    double g_org_parallel;

                    RMSE_warping_filter += fabs(f - g_warping);
                    RMSE_parallel_filter += fabs(f - g_parallel);

                    cv::Point2f tmp_X_later_warping, tmp_X_later_parallel;
                    tmp_X_later_warping.x = X_later_warping.x;
                    tmp_X_later_warping.y = X_later_warping.y;
                    tmp_X_later_parallel.x = X_later_parallel.x;
                    tmp_X_later_parallel.y = X_later_parallel.y;

                    tmp_X_later_warping = roundVecQuarter(tmp_X_later_warping);
                    tmp_X_later_parallel = roundVecQuarter(tmp_X_later_parallel);

                    if(ref_hevc != nullptr) {
                        g_org_warping  = img_ip(ref_hevc, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * tmp_X_later_warping.x,  4 * tmp_X_later_warping.y, 1);
                        g_org_parallel = img_ip(ref_hevc, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * tmp_X_later_parallel.x, 4 * tmp_X_later_parallel.y, 1);
                    }else {
                        g_org_warping  = img_ip(current_ref_org_expand, cv::Rect(-spread, -spread, current_target_image.cols + 2 * spread, current_target_image.rows + 2 * spread),  tmp_X_later_warping.x, tmp_X_later_warping.y, 2);
                        g_org_parallel = img_ip(current_ref_org_expand, cv::Rect(-spread, -spread, current_target_image.cols + 2 * spread, current_target_image.rows + 2 * spread), tmp_X_later_parallel.x, tmp_X_later_parallel.y, 2);
                    }

                    if(iterate_counter > 4){
                        f = f_org;
                        g_warping = g_org_warping;
                        g_parallel = g_org_parallel;
                    }

                    for (int row = 0; row < warping_matrix_dim; row++) {
                        for (int col = 0; col < warping_matrix_dim; col++) {
                            gg_warping.at<double>(row, col) += delta_g_warping[row] * delta_g_warping[col];//A_0の行列を生成(左辺の6x6の行列に相当)
                        }
                        B_warping.at<double>(row, 0) += (f - g_warping) * delta_g_warping[row];//bの行列を生成(右辺の6x1のベクトルに相当)
                    }
                    for (int row = 0; row < 2; row++) {
                        for (int col = 0; col < 2; col++) {
                            gg_parallel.at<double>(row, col) += delta_g_parallel[row] * delta_g_parallel[col];
                        }
                        B_parallel.at<double>(row, 0) += (f - g_parallel) * delta_g_parallel[row];
                    }


                    MSE_warping += fabs(f_org - g_org_warping);   // * (f_org - g_org_warping);
                    MSE_parallel += fabs(f_org - g_org_parallel); // * (f_org - g_org_parallel);
                }

                double mu = 10;
                for(int row = 0 ; row < warping_matrix_dim ; row++){
                    for(int col = 0 ; col < warping_matrix_dim ; col++) {
                        gg_warping.at<double>(row, col) += mu * ((area_after_move * area_after_move * area_after_move - area_before_move * area_before_move * area_after_move) / (area_before_move * area_before_move * area_before_move * area_before_move) * S[row]);
                    }
                }

                double Error_warping = MSE_warping;
                double Error_parallel = MSE_parallel;



                if(prev_error_warping < MSE_warping && warping_update_flag){
                    warping_update_flag  = false;
                    tmp_mv_warping = prev_mv_warping;
                }

                if(warping_update_flag) {
                    cv::solve(gg_warping, B_warping, delta_uv_warping); //6x6の連立方程式を解いてdelta_uvに格納
                    v_stack_warping.emplace_back(tmp_mv_warping, Error_warping);

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

                if(prev_error_parallel < MSE_parallel && parallel_update_flag){
                    parallel_update_flag = false;
                    tmp_mv_parallel = prev_mv_parallel;
                }
//                std::cout << iterate_counter + 1 << " " << MSE_parallel << " " << RMSE_parallel_filter << " " << tmp_mv_parallel << " " << parallel_update_flag << std::endl;

                if(parallel_update_flag) {
                    cv::solve(gg_parallel, B_parallel, delta_uv_parallel);
                    v_stack_parallel.emplace_back(tmp_mv_parallel, Error_parallel);

                    for (int k = 0; k < 2; k++) {
                        if (k % 2 == 0) {
                            if ((-scaled_spread <=
                                 scaled_coordinates[0].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) &&
                                (target_images[0][step].cols - 1 + scaled_spread >=
                                 scaled_coordinates[0].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) &&
                                (-scaled_spread <=
                                 scaled_coordinates[1].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) &&
                                (target_images[0][step].cols - 1 + scaled_spread >=
                                 scaled_coordinates[1].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) &&
                                (-scaled_spread <=
                                 scaled_coordinates[2].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) &&
                                (target_images[0][step].cols - 1 + scaled_spread >=
                                 scaled_coordinates[2].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0))) {
                                tmp_mv_parallel.x = tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0);
                            }
                        } else {
                            if ((-scaled_spread <=
                                 scaled_coordinates[0].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                                (target_images[0][step].rows - 1 + scaled_spread >=
                                 scaled_coordinates[0].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                                (-scaled_spread <=
                                 scaled_coordinates[1].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                                (target_images[0][step].rows - 1 + scaled_spread >=
                                 scaled_coordinates[1].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                                (-scaled_spread <=
                                 scaled_coordinates[2].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                                (target_images[0][step].rows - 1 + scaled_spread >=
                                 scaled_coordinates[2].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0))) {
                                tmp_mv_parallel.y = tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0);
                            }
                        }
                    }
                }

                double eps = 1e-3;
//                std::cout << fabs(prev_error_warping - MSE_warping) << " " << MSE_warping << " " <<(fabs(prev_error_warping - MSE_warping) / MSE_warping) << std::endl;
                if(((fabs(prev_error_parallel - MSE_parallel) / MSE_parallel) < eps && (fabs(prev_error_warping - MSE_warping) / MSE_warping < eps)) || (!parallel_update_flag && !warping_update_flag) || iterate_counter > 20){
                    break;
                }

                prev_error_parallel = MSE_parallel;
                prev_error_warping = MSE_warping;
                prev_mv_parallel = tmp_mv_parallel;
                prev_mv_warping = tmp_mv_warping;
                iterate_counter++;
            }

//            std::sort(v_stack_warping.begin(), v_stack_warping.end(), [](std::pair<std::vector<cv::Point2f>,double> a, std::pair<std::vector<cv::Point2f>,double> b){
//              return a.second < b.second;
//            });
//
//            std::sort(v_stack_parallel.begin(), v_stack_parallel.end(), [](std::pair<cv::Point2f,double> a, std::pair<cv::Point2f,double> b){
//              return a.second < b.second;
//            });

            std::reverse(v_stack_parallel.begin(), v_stack_parallel.end());
            std::reverse(v_stack_warping.begin(), v_stack_warping.end());

            tmp_mv_warping = v_stack_warping[0].first;//一番良い動きベクトルを採用
            double Error_warping = v_stack_warping[0].second;
            tmp_mv_parallel = v_stack_parallel[0].first;
            double Error_parallel = v_stack_parallel[0].second;
            MSE_warping = Error_warping / (double)pixels_in_triangle.size();
            MSE_parallel = Error_parallel / (double)pixels_in_triangle.size();
            double PSNR_warping = 10 * log10((255 * 255) / MSE_warping);
            double PSNR_parallel = 10 * log10((255 * 255) / MSE_parallel);

            if(step == 3) {//一番下の階層で
                if(PSNR_parallel >= max_PSNR_parallel){//2種類のボケ方で良い方を採用
                    max_PSNR_parallel = PSNR_parallel;
                    min_error_parallel = Error_parallel;
                    max_v_parallel = tmp_mv_parallel;
                }
                if (PSNR_warping >= max_PSNR_warping) {
                    max_PSNR_warping = PSNR_warping;
                    min_error_warping = Error_warping;
                    max_v_warping = tmp_mv_warping;
                }

                if (fabs(max_PSNR_warping - max_PSNR_parallel) <= 0.5 || max_PSNR_parallel > max_PSNR_warping) {//ワーピングと平行移動でRDのようなことをする
                    parallel_flag = true;//平行移動を採用
                } else{
                    parallel_flag = false;//ワーピングを採用
                }
            }
        }
    }

    // 量子化
    double quantize_offset = 0.125;
    if(max_v_parallel.x < 0) {
        max_v_parallel.x = ((int)((max_v_parallel.x - quantize_offset) * 4) / 4.0);
    }else{
        max_v_parallel.x = ((int)((max_v_parallel.x + quantize_offset) * 4) / 4.0);
    }

    if(max_v_parallel.y < 0) {
        max_v_parallel.y = ((int) ((max_v_parallel.y - quantize_offset) * 4) / 4.0);
    }else{
        max_v_parallel.y = ((int) ((max_v_parallel.y + quantize_offset) * 4) / 4.0);
    }

    for(int i = 0 ; i < 3 ; i++){
        if(max_v_warping[i].x < 0) max_v_warping[i].x -= quantize_offset;
        else max_v_warping[i].x += quantize_offset;

        if(max_v_warping[i].y < 0) max_v_warping[i].y -= quantize_offset;
        else max_v_warping[i].y += quantize_offset;

        max_v_warping[i].x = ((int)((max_v_warping[i].x) * 4) / 4.0);
        max_v_warping[i].y = ((int)((max_v_warping[i].y) * 4) / 4.0);
    }

    double error;
    if(parallel_flag) {
        error = min_error_parallel;
    }else{
        error = min_error_warping;
    }

    return std::make_tuple(std::vector<cv::Point2f>{max_v_warping[0], max_v_warping[1], max_v_warping[2]}, max_v_parallel, error, pixels_in_triangle.size(), parallel_flag);
}

/**
 * @fn std::tuple<std::vector<cv::Point2f>, cv::Point2f, double, int, bool> Marquardt(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, std::vector<std::vector<std::vector<unsigned char **>>> expand_image, Point3Vec target_corners, const std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, int block_size_x, int block_size_y)
 * @brief[in] レーベンバーグ・マーカート法を行い、動きベクトル・予測残差・面積を返す
 * @param[in] ref_images
 * @param[in] target_images
 * @param[in] expand_image
 * @param[in] target_corners
 * @param[in] area_flag
 * @param[in] triangle_index
 * @param[in] ctu
 * @param[in] block_size_x
 * @param[in] block_size_y
 * @return ワーピングの動きベクトル・平行移動の動きベクトル・予測残差・面積・平行移動のフラグのtuple
 */
std::tuple<std::vector<cv::Point2f>, cv::Point2f, double, int, bool> Marquardt(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, std::vector<std::vector<std::vector<unsigned char **>>> expand_image, Point3Vec target_corners, const std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, int block_size_x, int block_size_y){
    // 画像の初期化 vector[filter][picture_number]

    const int warping_matrix_dim = 6; // 方程式の次元
    const int parallel_matrix_dim = 2;
    cv::Mat gg_warping = cv::Mat::zeros(warping_matrix_dim, warping_matrix_dim, CV_64F); // 式(45)の左辺6×6行列
    cv::Mat gg_parallel = cv::Mat::zeros(parallel_matrix_dim, parallel_matrix_dim, CV_64F); // 式(52)の左辺2×2行列
    cv::Mat B_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F); // 式(45)の右辺
    cv::Mat B_parallel = cv::Mat::zeros(parallel_matrix_dim, 1, CV_64F); // 式(52)の右辺

    /* マーカート法で使う一つ前の行列 */
    cv::Mat gg_warping_prev = cv::Mat::zeros(warping_matrix_dim, warping_matrix_dim, CV_64F); // 式(45)の左辺6×6行列
    cv::Mat gg_parallel_prev = cv::Mat::zeros(parallel_matrix_dim, parallel_matrix_dim, CV_64F); // 式(52)の左辺2×2行列
    cv::Mat B_warping_prev = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F); // 式(45)の右辺
    cv::Mat B_parallel_prev = cv::Mat::zeros(parallel_matrix_dim, 1, CV_64F); // 式(52)の右辺


    cv::Mat delta_uv_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F); // 式(45)の左辺 delta
    cv::Mat delta_uv_parallel = cv::Mat::zeros(parallel_matrix_dim, 1, CV_64F); // 式(52)の右辺 delta

    double MSE_warping, MSE_parallel;
    double min_error_warping = 1E6, min_error_parallel = 1E6;
    double max_PSNR_warping = -1, max_PSNR_parallel = -1;

    cv::Point2f p0, p1, p2;
    std::vector<cv::Point2f> max_v_warping;
    cv::Point2f max_v_parallel;

    std::vector<std::pair<std::vector<cv::Point2f>,double>> v_stack_warping;
    std::vector<std::pair<cv::Point2f,double>> v_stack_parallel;
    std::vector<cv::Point2f> pixels_in_triangle;

    bool parallel_flag = true;

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

    int bm_x_offset = 10;
    int bm_y_offset = 10;
    double error_min = 1e9;

    for(int by = -bm_y_offset ; by < bm_y_offset ; by++){
        for(int bx = -bm_x_offset ; bx < bm_x_offset ; bx++){
            if(sx + bx < -16 || ref_images[0][3].cols + 16 <= (lx + bx) || sy + by < -16 || ref_images[0][3].rows + 16 <=(ly + by)) continue;
            double error_tmp = 0.0;
            for(const auto& pixel : pixels_in_triangle) {
                error_tmp += abs(expand_image[0][3][1][(int)(pixel.x + bx)][(int)(pixel.y + by)] - expand_image[0][3][3][(int)(pixel.x)][(int)(pixel.y)]);
            }
            if(error_min > error_tmp) {
                error_min = error_tmp;
                initial_vector.x = bx;
                initial_vector.y = by;
            }
        }
    }

    initial_vector /= 4.0;
    for(int filter_num = 0 ; filter_num < static_cast<int>(ref_images.size()) ; filter_num++){
        std::vector<cv::Point2f> tmp_mv_warping(3, cv::Point2f(initial_vector.x, initial_vector.y));
        cv::Point2f tmp_mv_parallel(initial_vector.x, initial_vector.y);

        for(int step = 2 ; step < static_cast<int>(ref_images[filter_num].size()) ; step++){

            double scale = pow(2, 3 - step);
            cv::Mat current_ref_image = mv_filter(ref_images[filter_num][step],2);
            cv::Mat current_target_image = mv_filter(target_images[filter_num][step],2);

            unsigned char **current_target_expand, **current_target_org_expand; //画像の周りに500ピクセルだけ黒の領域を設ける(念のため)
            unsigned char **current_ref_expand, **current_ref_org_expand;    //f_expandと同様

            current_ref_expand        = expand_image[filter_num][step][0];
            current_ref_org_expand    = expand_image[filter_num][step][1];
            current_target_expand     = expand_image[filter_num][step][2];
            current_target_org_expand = expand_image[filter_num][step][3];

            int spread = 16; // 探索範囲は16までなので16に戻す

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
                double magnification = 2.0;
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

                // 平行移動
                magnification = 2.0;
                while ( (p0.x + tmp_mv_parallel.x * magnification < -scaled_spread && p0.x + tmp_mv_parallel.x * magnification > current_target_image.cols - 1 + scaled_spread) &&
                        (p1.x + tmp_mv_parallel.x * magnification < -scaled_spread && p1.x + tmp_mv_parallel.x * magnification > current_target_image.cols - 1 + scaled_spread) &&
                        (p2.x + tmp_mv_parallel.x * magnification < -scaled_spread && p2.x + tmp_mv_parallel.x * magnification > current_target_image.cols - 1 + scaled_spread) &&
                        (p0.y + tmp_mv_parallel.y * magnification < -scaled_spread && p0.y + tmp_mv_parallel.y * magnification > current_target_image.rows - 1 + scaled_spread) &&
                        (p1.y + tmp_mv_parallel.y * magnification < -scaled_spread && p1.y + tmp_mv_parallel.y * magnification > current_target_image.rows - 1 + scaled_spread) &&
                        (p2.y + tmp_mv_parallel.y * magnification < -scaled_spread && p2.y + tmp_mv_parallel.y * magnification > current_target_image.rows - 1 + scaled_spread) ) {
                    if(magnification <= 1)break;
                    magnification -= 0.1;
                }
                tmp_mv_parallel *= magnification;
            }
            v_stack_parallel.clear();
            v_stack_warping.clear();

            double lambda_warp = 1.0, lambda_parallel = 1.0;
            double error_warp_prev = 1e9, error_parallel_prev = 1e9;
            // 11回ガウス・ニュートン法をやる
            for(int gaussIterateNum = 0 ; gaussIterateNum < 1000 ; gaussIterateNum++) {
                if(gaussIterateNum == 10 && step == 3) {
                    for(int i = 0 ; i < tmp_mv_warping.size(); i++){
                        tmp_mv_warping[i].x = 0.0;
                        tmp_mv_warping[i].y = 0.0;
                    }
                    tmp_mv_parallel.x = 0.0;
                    tmp_mv_parallel.y = 0.0;
                }

                // 移動後の座標を格納する
                std::vector<cv::Point2f> ref_coordinates_warping;
                std::vector<cv::Point2f> ref_coordinates_parallel;

                ref_coordinates_warping.emplace_back(p0);
                ref_coordinates_warping.emplace_back(p1);
                ref_coordinates_warping.emplace_back(p2);

                ref_coordinates_parallel.emplace_back(p0);
                ref_coordinates_parallel.emplace_back(p1);
                ref_coordinates_parallel.emplace_back(p2);

                cv::Point2f a = p2 - p0;
                cv::Point2f b = p1 - p0;
                double det = a.x * b.y - a.y * b.x;
                cv::Point2f c = tmp_mv_warping[2] - tmp_mv_warping[0];
                cv::Point2f d = tmp_mv_warping[1] - tmp_mv_warping[0];

                double area_before_move = 0.5 * fabs(det); // 移動前の面積
                double area_after_move = 0.5 * fabs((b.x + d.x)*(a.y + c.y) - (a.x + c.x)*(b.y + d.y)); // 移動後の面積

                double S[6];
                S[0] = -0.5*(a.y + c.y - b.y - d.y);
                S[1] = -0.5*(b.x + d.x - a.x - c.x);
                S[2] = 0.5*(a.y + c.y);
                S[3] = -0.5*(a.x + c.x);
                S[4] = -0.5*(b.y + d.y);
                S[5] = 0.5*(b.x + d.x);

                MSE_parallel = MSE_warping = 0.0;
                gg_warping = cv::Mat::zeros(warping_matrix_dim, warping_matrix_dim, CV_64F);
                B_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F);
                delta_uv_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F);

                gg_parallel = cv::Mat::zeros(parallel_matrix_dim, parallel_matrix_dim, CV_64F);
                B_parallel = cv::Mat::zeros(parallel_matrix_dim, 1, CV_64F);
                delta_uv_parallel = cv::Mat::zeros(parallel_matrix_dim, 1, CV_64F);

                double delta_g_warping[warping_matrix_dim] = {0};
                double delta_g_parallel[parallel_matrix_dim] = {0};

                cv::Point2f X;
                for(auto pixel : pixels_in_triangle) {
                    X.x = pixel.x - p0.x;
                    X.y = pixel.y - p0.y;

                    double alpha = (X.x * b.y - X.y * b.x) / det;
                    double beta = (a.x * X.y - a.y * X.x)/ det;
                    X.x += p0.x;
                    X.y += p0.y;

                    int x_integer = (int)floor(X.x);
                    int y_integer = (int)floor(X.y);
                    int x_decimal = X.x - x_integer;
                    int y_decimal = X.y - y_integer;

                    // 参照フレームの前進差分（平行移動）
                    double g_x_parallel;
                    double g_y_parallel;
                    cv::Point2f X_later_parallel, X_later_warping;

                    for(int i = 0 ; i < 6 ; i++) {
                        // 移動後の頂点を計算し格納
                        ref_coordinates_warping[0] = p0 + tmp_mv_warping[0];
                        ref_coordinates_warping[1] = p1 + tmp_mv_warping[1];
                        ref_coordinates_warping[2] = p2 + tmp_mv_warping[2];
                        ref_coordinates_parallel[0] = p0 + tmp_mv_parallel;
                        ref_coordinates_parallel[1] = p1 + tmp_mv_parallel;
                        ref_coordinates_parallel[2] = p2 + tmp_mv_parallel;

                        std::vector<cv::Point2f> triangle_later_warping;
                        std::vector<cv::Point2f> triangle_later_parallel;
                        triangle_later_warping.emplace_back(ref_coordinates_warping[0]);
                        triangle_later_warping.emplace_back(ref_coordinates_warping[1]);
                        triangle_later_warping.emplace_back(ref_coordinates_warping[2]);
                        triangle_later_parallel.emplace_back(ref_coordinates_parallel[0]);
                        triangle_later_parallel.emplace_back(ref_coordinates_parallel[1]);
                        triangle_later_parallel.emplace_back(ref_coordinates_parallel[2]);

                        cv::Point2f a_later_warping, a_later_parallel;
                        cv::Point2f b_later_warping, b_later_parallel;

                        a_later_warping.x = triangle_later_warping[2].x - triangle_later_warping[0].x;
                        a_later_warping.y = triangle_later_warping[2].y - triangle_later_warping[0].y;
                        a_later_parallel = triangle_later_parallel[2] - triangle_later_parallel[0];
                        b_later_warping.x = triangle_later_warping[1].x - triangle_later_warping[0].x;
                        b_later_warping.y = triangle_later_warping[1].y - triangle_later_warping[0].y;
                        b_later_parallel = triangle_later_parallel[1] - triangle_later_parallel[0];
                        X_later_warping.x = alpha * a_later_warping.x + beta * b_later_warping.x + triangle_later_warping[0].x;
                        X_later_warping.y = alpha * a_later_warping.y + beta * b_later_warping.y + triangle_later_warping[0].y;
                        X_later_parallel = alpha * a_later_parallel + beta * b_later_parallel + triangle_later_parallel[0];

                        if(X_later_warping.x >= current_ref_image.cols - 1 + scaled_spread) X_later_warping.x = current_ref_image.cols - 1.00 + scaled_spread;
                        if(X_later_warping.y >= current_ref_image.rows - 1 + scaled_spread) X_later_warping.y = current_ref_image.rows - 1.00 + scaled_spread;
                        if(X_later_warping.x < -scaled_spread) X_later_warping.x = -scaled_spread;
                        if(X_later_warping.y < -scaled_spread) X_later_warping.y = -scaled_spread;

                        if((current_ref_image.cols - 1 + scaled_spread) <= X_later_parallel.x) X_later_parallel.x = current_ref_image.cols - 1 + scaled_spread;
                        if(X_later_parallel.x < -scaled_spread) X_later_parallel.x = -scaled_spread;
                        if((current_ref_image.rows - 1 + scaled_spread) <= X_later_parallel.y) X_later_parallel.y = current_ref_image.rows - 1 + scaled_spread;
                        if(X_later_parallel.y < -scaled_spread) X_later_parallel.y = -scaled_spread;

                        // 頂点を動かしたときのパッチ内の変動量x軸y軸独立に計算(delta_gを求めるために必要)
                        double delta_x, delta_y;
                        switch (i) {//頂点ごとにxy軸独立に偏微分
                            case 0:
                                delta_x = 1 - alpha - beta;
                                delta_y = 0;
                                break;
                            case 1:
                                delta_x = 0;
                                delta_y = 1 - alpha - beta;
                                break;
                            case 2:
                                delta_x = beta;
                                delta_y = 0;
                                break;
                            case 3:
                                delta_x = 0;
                                delta_y = beta;
                                break;
                            case 4:
                                delta_x = alpha;
                                delta_y = 0;
                                break;
                            case 5:
                                delta_x = 0;
                                delta_y = alpha;
                                break;
                            default:
                                break;
                        }

                        // 参照フレームの前進差分を計算
                        double g_x   = current_ref_expand[(int) X_later_warping.x  + 1][(int) X_later_warping.y     ] - current_ref_expand[(int) X_later_warping.x ][(int) X_later_warping.y ];//前進差分
                        double g_y   = current_ref_expand[(int) X_later_warping.x     ][(int) X_later_warping.y  + 1] - current_ref_expand[(int) X_later_warping.x ][(int) X_later_warping.y ];
                        g_x_parallel = current_ref_expand[(int) X_later_parallel.x + 1][(int) X_later_parallel.y    ] - current_ref_expand[(int) X_later_parallel.x][(int) X_later_parallel.y];
                        g_y_parallel = current_ref_expand[(int) X_later_parallel.x    ][(int) X_later_parallel.y + 1] - current_ref_expand[(int) X_later_parallel.x][(int) X_later_parallel.y];

                        delta_g_warping[i] = g_x * delta_x + g_y * delta_y;
                    }
                    delta_g_parallel[0] = g_x_parallel;
                    delta_g_parallel[1] = g_y_parallel;

                    int x0_later_warping_integer = (int)floor(X_later_warping.x);
                    int y0_later_warping_integer = (int)floor(X_later_warping.y);
                    int x0_later_parallel_integer = (int)floor(X_later_parallel.x);
                    int y0_later_parallel_integer = (int)floor(X_later_parallel.y);

                    double f = bicubic_interpolation(current_target_expand, X.x, X.y);
                    double f_org = bicubic_interpolation(current_target_org_expand, X.x, X.y);

                    double g_warping = bicubic_interpolation(current_ref_expand, X_later_warping.x, X_later_warping.y);
                    double g_parallel = bicubic_interpolation(current_ref_expand, X_later_parallel.x, X_later_parallel.y);

                    double g_org_warping = bicubic_interpolation(current_ref_org_expand, X_later_warping.x, X_later_warping.y);
                    double g_org_parallel = bicubic_interpolation(current_ref_org_expand, X_later_parallel.x, X_later_parallel.y);

                    for (int row = 0; row < warping_matrix_dim; row++) {
                        for (int col = 0; col < warping_matrix_dim; col++) {
                            if(col == row) {
                                gg_warping.at<double>(row, col) += (1 + lambda_warp) * delta_g_warping[row] * delta_g_warping[col];//A_0の行列を生成(左辺の6x6の行列に相当)
                            }else{
                                gg_warping.at<double>(row, col) += delta_g_warping[row] * delta_g_warping[col];//A_0の行列を生成(左辺の6x6の行列に相当)
                            }

                        }
                        B_warping.at<double>(row, 0) += (f - g_warping) * delta_g_warping[row];//bの行列を生成(右辺の6x1のベクトルに相当)
                    }
                    for (int row = 0; row < 2; row++) {
                        for (int col = 0; col < 2; col++) {
                            if(row == col) {
                                gg_parallel.at<double>(row, col) += (1 + lambda_parallel) * delta_g_parallel[row] * delta_g_parallel[col];
                            }else {
                                gg_parallel.at<double>(row, col) += delta_g_parallel[row] * delta_g_parallel[col];
                            }
                        }
                        B_parallel.at<double>(row, 0) += (f - g_parallel) * delta_g_parallel[row];
                    }

                    MSE_warping += fabs(f_org - g_org_warping);   // * (f_org - g_org_warping);
                    MSE_parallel += fabs(f_org - g_org_parallel); // * (f_org - g_org_parallel);
//                    MSE_warping  += (f - g_warping)  * (f - g_warping);
//                    MSE_parallel += (f - g_parallel) * (f - g_parallel);
                }

                double mu = 10;
                for(int row = 0 ; row < warping_matrix_dim ; row++){
                    for(int col = 0 ; col < warping_matrix_dim ; col++) {
                        gg_warping.at<double>(row, col) += mu * ((area_after_move * area_after_move * area_after_move - area_before_move * area_before_move * area_after_move) / (area_before_move * area_before_move * area_before_move * area_before_move) * S[row]);
                    }
                }

                double Error_warping = MSE_warping;
                double Error_parallel = MSE_parallel;

//                std::cout << "Error_warping:" << Error_warping << " error_warp_prev:" << error_warp_prev << std::endl;
//                std::cout << "Error_parallel:" << Error_parallel << " error_parallel_prev:" << error_parallel_prev << std::endl;
                if(fabs(Error_warping - error_warp_prev) < 1e-5 && fabs(Error_parallel - error_parallel_prev) < 1e-5){
                    break;
                }
                if(Error_warping > error_warp_prev){
                    gg_warping = gg_warping_prev.clone();
                    B_warping = B_warping_prev.clone();
                    lambda_warp *= 10.0;
                }else{
                    gg_warping_prev = gg_warping.clone();
                    B_warping_prev = B_warping.clone();

                    cv::solve(gg_warping, B_warping, delta_uv_warping); //6x6の連立方程式を解いてdelta_uvに格納

                    v_stack_warping.emplace_back(tmp_mv_warping, Error_warping);

                    for (int k = 0; k < 6; k++) {

                        if (k % 2 == 0) {
                            if ((-scaled_spread <= scaled_coordinates[(int) (k / 2)].x + tmp_mv_warping[(int) (k / 2)].x + delta_uv_warping.at<double>(k, 0)) &&
                                (target_images[0][step].cols - 1 + scaled_spread >=
                                 scaled_coordinates[(int) (k / 2)].x + tmp_mv_warping[(int) (k / 2)].x + delta_uv_warping.at<double>(k, 0))) {
                                tmp_mv_warping[(int) (k / 2)].x = tmp_mv_warping[(int) (k / 2)].x + delta_uv_warping.at<double>(k, 0);//動きベクトルを更新(画像の外に出ないように)
                            }
                        } else {
                            if ((-scaled_spread <= scaled_coordinates[(int) (k / 2)].y + tmp_mv_warping[(int) (k / 2)].y + delta_uv_warping.at<double>(k, 0)) &&
                                (target_images[0][step].rows - 1 + scaled_spread >=
                                 scaled_coordinates[(int) (k / 2)].y + tmp_mv_warping[(int) (k / 2)].y + delta_uv_warping.at<double>(k, 0))) {
                                tmp_mv_warping[(int) (k / 2)].y = tmp_mv_warping[(int) (k / 2)].y + delta_uv_warping.at<double>(k, 0);
                            }
                        }
                    }

                    lambda_warp *= 0.1;
                    error_warp_prev = Error_warping;
                }

                if(Error_parallel > error_parallel_prev){
                    gg_parallel = gg_parallel_prev.clone();
                    B_parallel = B_parallel_prev.clone();
                    lambda_parallel *= 10.0;
                }else{
                    gg_parallel_prev = gg_parallel.clone();
                    B_parallel_prev = B_parallel.clone();

                    cv::solve(gg_parallel, B_parallel, delta_uv_parallel);

                    v_stack_parallel.emplace_back(tmp_mv_parallel, Error_parallel);

                    for (int k = 0; k < 2; k++) {
                        if (k % 2 == 0) {
                            if ((-scaled_spread <= scaled_coordinates[0].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) && (target_images[0][step].cols - 1 + scaled_spread >= scaled_coordinates[0].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) &&
                                (-scaled_spread <= scaled_coordinates[1].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) && (target_images[0][step].cols - 1 + scaled_spread >= scaled_coordinates[1].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) &&
                                (-scaled_spread <= scaled_coordinates[2].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) && (target_images[0][step].cols - 1 + scaled_spread >= scaled_coordinates[2].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0))) {
                                tmp_mv_parallel.x = tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0);
                            }
                        } else {
                            if ((-scaled_spread <= scaled_coordinates[0].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                                (target_images[0][step].rows - 1 + scaled_spread >=
                                 scaled_coordinates[0].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                                (-scaled_spread <= scaled_coordinates[1].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                                (target_images[0][step].rows - 1 + scaled_spread >=
                                 scaled_coordinates[1].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                                (-scaled_spread <= scaled_coordinates[2].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                                (target_images[0][step].rows - 1 + scaled_spread >=
                                 scaled_coordinates[2].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0))) {
                                tmp_mv_parallel.y = tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0);
                            }
                        }
                    }
                    error_parallel_prev = Error_parallel;
                    lambda_parallel *= 0.1;
                }

                error_warp_prev = Error_warping;
                error_parallel_prev = Error_parallel;
//                std::cout << "lambda_warp:" << lambda_warp << std::endl;
            }

            std::sort(v_stack_warping.begin(), v_stack_warping.end(), [](std::pair<std::vector<cv::Point2f>,double> a, std::pair<std::vector<cv::Point2f>,double> b){
              return a.second < b.second;
            });

            std::sort(v_stack_parallel.begin(), v_stack_parallel.end(), [](std::pair<cv::Point2f,double> a, std::pair<cv::Point2f,double> b){
              return a.second < b.second;
            });

            tmp_mv_warping = v_stack_warping[0].first;//一番良い動きベクトルを採用
            double Error_warping = v_stack_warping[0].second;
            tmp_mv_parallel = v_stack_parallel[0].first;
            double Error_parallel = v_stack_parallel[0].second;
            MSE_warping = Error_warping / (double)pixels_in_triangle.size();
            MSE_parallel = Error_parallel / (double)pixels_in_triangle.size();
            double PSNR_warping = 10 * log10((255 * 255) / MSE_warping);
            double PSNR_parallel = 10 * log10((255 * 255) / MSE_parallel);

//            std::cout << "min_parallel_error:" << Error_parallel << std::endl;
            if(step == 3) {//一番下の階層で
                if(PSNR_parallel >= max_PSNR_parallel){//2種類のボケ方で良い方を採用
                    max_PSNR_parallel = PSNR_parallel;
                    min_error_parallel = Error_parallel;
                    max_v_parallel = tmp_mv_parallel;
                }
                if (PSNR_warping >= max_PSNR_warping) {
                    max_PSNR_warping = PSNR_warping;
                    min_error_warping = Error_warping;
                    max_v_warping = tmp_mv_warping;
                }

                if (fabs(max_PSNR_warping - max_PSNR_parallel) <= 0.5 || max_PSNR_parallel > max_PSNR_warping) {//ワーピングと平行移動でRDのようなことをする
                    parallel_flag = true;//平行移動を採用
                } else{
                    parallel_flag = false;//ワーピングを採用
                }
            }
        }
    }

    parallel_flag = true;
    // 量子化
    double quantize_offset = 0.125;
    if(max_v_parallel.x < 0) {
        max_v_parallel.x = ((int)((max_v_parallel.x - quantize_offset) * 4) / 4.0);
    }else{
        max_v_parallel.x = ((int)((max_v_parallel.x + quantize_offset) * 4) / 4.0);
    }

    if(max_v_parallel.y < 0) {
        max_v_parallel.y = ((int) ((max_v_parallel.y - quantize_offset) * 4) / 4.0);
    }else{
        max_v_parallel.y = ((int) ((max_v_parallel.y + quantize_offset) * 4) / 4.0);
    }

    for(int i = 0 ; i < 3 ; i++){
        if(max_v_warping[i].x < 0) max_v_warping[i].x -= quantize_offset;
        else max_v_warping[i].x += quantize_offset;

        if(max_v_warping[i].y < 0) max_v_warping[i].y -= quantize_offset;
        else max_v_warping[i].y += quantize_offset;

        max_v_warping[i].x = ((int)((max_v_warping[i].x) * 4) / 4.0);
        max_v_warping[i].y = ((int)((max_v_warping[i].y) * 4) / 4.0);
    }

    double error = 0.0;
    if(parallel_flag) {
        error = min_error_parallel; // / (double)pixels_in_triangle.size();
    }else{
        error = min_error_warping; // / (double)pixels_in_triangle.size();
    }
    return std::make_tuple(std::vector<cv::Point2f>{max_v_warping[0], max_v_warping[1], max_v_warping[2]}, max_v_parallel, error, pixels_in_triangle.size(),true);
}

/**
 * @fn std::vector<cv::Point2f> getReferenceImageCoordinates(const cv::Mat& ref, const cv::Mat& target, const std::vector<cv::Point2f>& target_corners, cv::Mat& debug)
 * @breif Block-based Matchingベースの特徴点追跡
 * @param ref リファレンス画像
 * @param target ターゲット画像
 * @param target_corners  ターゲット画像上での頂点
 * @param debug デバッグ画像
 * @return ref_corners 参照画像上での頂点(corners[i]の移動先がref_corners[i]に格納されている)
 */
std::pair<std::vector<cv::Point2f>, std::priority_queue<int>> getReferenceImageCoordinates(const cv::Mat &ref,
                                                                                           const cv::Mat &target,
                                                                                           const std::vector<cv::Point2f> &target_corners,
                                                                                           cv::Mat &debug) {
    int SX = 160, SY = 160;
//  int SX = 80, SY = 80;
    std::vector<cv::Point2f> ref_corners(target_corners.size(), cv::Point2f(0.0, 0.0));
    std::priority_queue<int> pqueue;

#pragma omp parallel for
    for(int point_no = 0 ; point_no < (int)target_corners.size() ; point_no++){

        cv::Point2f pt = target_corners[point_no];

        // ブロックを作る
        int sx = std::max(static_cast<int>(pt.x - 20), 0);
        int sy = std::max(static_cast<int>(pt.y - 20), 0);
        int lx = std::min(static_cast<int>(pt.x + 20), ref.cols) - sx;
        int ly = std::min(static_cast<int>(pt.y + 20), ref.rows) - sy;

        cv::Rect rect = cv::Rect(sx, sy, lx, ly);
        //drawRectangle(debug, cv::Point2f(rect.x, rect.y), cv::Point2f(rect.x + rect.width, rect.y), cv::Point2f(rect.x + rect.width, rect.y + rect.height), cv::Point2f(rect.x, rect.y + rect.height));

        cv::Point2f mv(0, 0);
        double error = 1 << 20;

        cv::Mat expansion_ref = bilinearInterpolation(ref.clone());

        for (int j = -SY / 2; j <= SY / 2; j++) {
            for (int i = -SX / 2; i <= SX / 2; i++) {
                int nx = 2 * rect.x + i;
                int ny = 2 * rect.y + j;

                if(nx < 0 || expansion_ref.cols <= nx + rect.width * 2 || ny < 0 || expansion_ref.rows <= ny + rect.height * 2) continue; // 一部がはみ出ている

                double error_tmp = 0.0;
                for(int k = 0 ; k < rect.height ; k++){
                    for(int l = 0 ; l < rect.width; l++){
                        error_tmp += fabs(M(expansion_ref, nx + 2 * l, ny + 2 * k) - M(target, rect.x + l, rect.y + k));
                    }
                }
                error_tmp /= (rect.width * rect.height);
                if(error > error_tmp){
                    error = error_tmp;
                    mv.x = i; mv.y = j;
                }
            }
        }
        ref_corners[point_no] = mv + 2 * pt;

        if(error > 80 && !(pt.x == 0.0 || pt.x == WIDTH-1 || pt.y == 0.0 || pt.y == HEIGHT-1)){
            pqueue.emplace(point_no);
        }
    }
    return std::make_pair(ref_corners, pqueue);
}
