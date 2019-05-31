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
 * @param[in]  tr               三角形を表す3点の座標
 * @param[in]  expansion_image  2倍に拡大した(補間した)画像
 * @details
 *
 */
void block_matching(const cv::Mat& prev, const cv::Mat& current, double &error, cv::Point2f &mv, Point3Vec tr, cv::Mat expansion_image) {
    double sx, sy, lx, ly;
    cv::Point2f tp1, tp2, tp3;
    tp1 = tr.p1;
    tp2 = tr.p2;
    tp3 = tr.p3;

    sx = std::min({tp1.x, tp2.x, tp3.x});
    sy = std::min({tp1.y, tp2.y, tp3.y});
    lx = std::max({tp1.x, tp2.x, tp3.x});
    ly = std::max({tp1.y, tp2.y, tp3.y});

    cv::Point2f mv_tmp(0.0, 0.0);
    int SX = 100; // ブロックマッチングの探索範囲(X)
    int SY = 100; // ブロックマッチングの探索範囲(Y)

    double error_tmp, error_min;
    int error_count;

    error_min = 1 << 20;
    cv::Point2d xp(0.0, 0.0);

    for (int j = -SY / 2; j <= SY / 2; j++) {
        for (int i = -SX / 2; i <= SX / 2; i++) {
            error_tmp = 0.0;
            error_count = 0;

            int nx = static_cast<int>(round(sx) + i);
            int ny = static_cast<int>(round(sy) + j);

            // 範囲外の場合
            if(nx < 0 || prev.cols <= nx + (lx - sx) || ny < 0 || prev.rows <= ny + (ly - sy)) continue;

            for (int m = (int) (round(sy) - 1); m <= round(ly) + 1; m++) {
                for (int n = (int) (round(sx) - 1); n <= round(lx) + 1; n++) {
                    xp.x = (double) n;
                    xp.y = (double) m;

                    // xpが三角形trの中かどうか判定
                    if (isInTriangle(tr, xp)) {
                        // 現在のフレームとの差分
                        error_tmp += fabs(M(prev, n + i, m + j) - M(current, n, m));
                        error_count++;
                    }
                }
            }

            error_tmp = error_count > 0 ? (error_tmp / (double) error_count) : 1e6;
            if (error_tmp == error_min && error_count > 0) {
                if (abs(i) < abs(mv_tmp.x) && abs(j) < abs(mv_tmp.y)) {
                    mv_tmp.x = (float) i;
                    mv_tmp.y = (float) j;
                }
            }

            if (error_min > error_tmp && error_count > 0) {
                error_min = error_tmp;
                mv_tmp.x = (float) i;
                mv_tmp.y = (float) j;
            }
        }
    }

    error = error_min;

    error_min = 1 << 20;
    xp.x = 0.0; xp.y = 0.0;

    mv.x = mv_tmp.x * 2;
    mv.y = mv_tmp.y * 2;

    SX = 5; SY = 5;
    for (int j = -SY / 2; j <= SY / 2; j++) {
        for (int i = -SX / 2; i <= SX / 2; i++) {
            error_tmp = 0.0;
            error_count = 0;
            if (0 <= 2 * round(sx) - 1 + i + mv.x && 2 * (round(lx) + 1) + i + mv.x < expansion_image.cols && 0 <= 2 * round(sy) - 1 + j + mv.y && 2 * (round(ly) + 1) + j + mv.y < expansion_image.rows) {
                for (int m = (int) (round(sy) - 1); m <= round(ly) + 1; m++) {
                    for (int n = (int) (round(sx) - 1); n <= round(lx) + 1; n++) {
                        xp.x = (double) n;
                        xp.y = (double) m;

                        // xpが三角形trの中かどうか判定
                        if (isInTriangle(tr, xp)) {
                            // 現在のフレームとの差分
                            error_tmp += fabs(M(expansion_image, 2 * n + i + (int)mv.x, 2 * m + j + (int)mv.y) - M(current, n, m));
                            error_count++;
                        }
                    }
                }

                error_tmp = error_count > 0 ? (error_tmp / (double) error_count) : 1e6;
                if (error_tmp == error_min && error_count > 0) {
                    if (abs(i) < abs(mv_tmp.x) && abs(j) < abs(mv_tmp.y)) {
                        mv_tmp.x = (float) i;
                        mv_tmp.y = (float) j;
                    }
                }

                if (error_min > error_tmp && error_count > 0) {
                    error_min = error_tmp;
                    mv_tmp.x = (float) i;
                    mv_tmp.y = (float) j;
                }
            }
        }
    }

    error = error_min;
    mv.x = mv.x + mv_tmp.x;
    mv.y = mv.y + mv_tmp.y;


    // フルペルで探索して得られた動きベクトルで, ハーフペル画像を探索する.（2段探索）
    // 周囲8画素だけやれば良さそう
    //
    // ✕  ✕  ✕  ✕  ✕  ✕ ✕
    //
    // ✕  ○  ✕  ○  ✕  ○  ✕
    //
    // ✕  ✕  ✕  ✕  ✕  ✕ ✕
    //
    // ✕  ○  ✕  ○  ✕  ○  ✕
    //
    // ✕  ✕  ✕  ✕  ✕ ✕  ✕
    //
//
//  SX = 181; SY = 181;
//  mv = cv::Point2f(0.0, 0.0);
//  for (int j = -SY / 2; j <= SY / 2; j++) {
//    for (int i = -SX / 2; i <= SX / 2; i++) {
//      error_tmp = 0.0;
//      error_count = 0;
//      if (0 <= 2 * round(sx) - 1 + i && 2 * (round(lx) + 1) + i < expansion_image.cols && 0 <= 2 * round(sy) - 1 + j && 2 * (round(ly) + 1) + j < expansion_image.rows) {
//        for (int m = (int) (round(sy) - 1); m <= round(ly) + 1; m++) {
//          for (int n = (int) (round(sx) - 1); n <= round(lx) + 1; n++) {
//            xp.x = (double) n;
//            xp.y = (double) m;
//
//            // xpが三角形trの中かどうか判定
//            if (isInTriangle(tr, xp)) {
//              // 現在のフレームとの差分
//              error_tmp += fabs(M(expansion_image, 2 * n + i, 2 * m + j) - M(current, n, m));
//              error_count++;
//            }
//          }
//        }
//
//        error_tmp = error_count > 0 ? (error_tmp / (double) error_count) : 1e6;
//        if (error_tmp == error_min && error_count > 0) {
//          if (abs(i) < abs(mv_tmp.x) && abs(j) < abs(mv_tmp.y)) {
//            mv_tmp.x = (float) i;
//            mv_tmp.y = (float) j;
//          }
//        }
//
//        if (error_min > error_tmp && error_count > 0) {
//          error_min = error_tmp;
//          mv_tmp.x = (float) i;
//          mv_tmp.y = (float) j;
//        }
//      }
//    }
//  }
//
//  error = error_min;
//
//  mv.x = mv.x + mv_tmp.x;
//  mv.y = mv.y + mv_tmp.y;
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
double getPredictedImage(cv::Mat& ref_image, cv::Mat& target_image, cv::Mat& output_image, Point3Vec& triangle, std::vector<cv::Point2f>& mv, bool parallel_flag) {
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
 * @param[in] ref_image 参照画像（QPは変動）
 * @param[in] target_image 対象画像
 * @param[in] gauss_ref_image ガウス・ニュートン法で使用する参照画像（常にQP=22の参照画像）
 * @param[in] target_corners 対象画像上の三角パッチの座標
 * @return ワーピングの動きベクトル・平行移動の動きベクトル・予測残差・面積・平行移動のフラグのtuple
 */
std::tuple<std::vector<cv::Point2f>, cv::Point2f, double, int, bool> GaussNewton(cv::Mat ref_image, cv::Mat target_image, cv::Mat gauss_ref_image, Point3Vec target_corners){
    // 画像の初期化 vector[filter][picture_number]
    std::vector<std::vector<cv::Mat>> ref_images;
    std::vector<std::vector<cv::Mat>> target_images;

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

    ref_images.emplace_back(ref1_levels);
    ref_images.emplace_back(ref2_levels);
    target_images.emplace_back(target1_levels);
    target_images.emplace_back(target2_levels);


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
    const int expand = 500;

    for(int filter_num = 0 ; filter_num < static_cast<int>(ref_images.size()) ; filter_num++){
        std::vector<cv::Point2f> tmp_mv_warping(3, cv::Point2f(0.0, 0.0));
        cv::Point2f tmp_mv_parallel(0.0, 0.0);

        for(int step = 0 ; step < static_cast<int>(ref_images[filter_num].size()) ; step++){

            double scale = pow(2, 3 - step);
            cv::Mat current_ref_image = mv_filter(ref_images[filter_num][step],2);
            cv::Mat current_target_image = mv_filter(target_images[filter_num][step],2);

            unsigned char **current_target_expand, **current_target_org_expand; //画像の周りに500ピクセルだけ黒の領域を設ける(念のため)
            unsigned char **current_ref_expand, **current_ref_org_expand;    //f_expandと同様
            current_target_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (current_target_image.cols + expand * 2));
            current_target_expand += expand;
            current_target_org_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (current_target_image.cols + expand * 2));
            current_target_org_expand += expand;

            for (int j = -expand; j < current_target_image.cols + expand; j++) {
                current_target_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (current_target_image.rows + expand * 2));
                current_target_expand[j] += expand;

                current_target_org_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (current_target_image.rows + expand * 2));
                current_target_org_expand[j] += expand;
            }

            current_ref_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (current_target_image.cols + expand * 2));
            current_ref_expand += expand;
            current_ref_org_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (current_target_image.cols + expand * 2));
            current_ref_org_expand += expand;
            for (int j = -expand; j < current_ref_image.cols + expand; j++) {
                if ((current_ref_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (current_target_image.rows + expand * 2))) == NULL) {
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

                        current_target_org_expand[i][j] = M(target_images[filter_num][step], i, j);
                        current_ref_org_expand[i][j] = M(ref_images[filter_num][step], i, j);
                    } else {
                        current_target_expand[i][j] = 0;
                        current_ref_expand[i][j] = 0;
                        current_target_org_expand[i][j] = 0;
                        current_ref_org_expand[i][j] = 0;
                    }
                }
            }
            int k = 2;//画像の周り2ピクセルだけ折り返し
            for (int j = 0; j < current_target_image.rows; j++) {
                for (int i = 1; i <= k; i++) {
                    current_target_expand[-i][j] = current_target_expand[i][j];
                    current_target_expand[current_target_image.cols - 1 + i][j] = current_target_expand[current_target_image.cols - 1 - i][j];
                    current_ref_expand[-i][j] = current_ref_expand[i][j];
                    current_ref_expand[current_target_image.cols - 1 + i][j] = current_ref_expand[current_target_image.cols - 1 - i][j];
                    current_target_org_expand[-i][j] = current_target_org_expand[i][j];
                    current_target_org_expand[current_target_image.cols - 1 + i][j] = current_target_org_expand[current_target_image.cols - 1 - i][j];
                    current_ref_org_expand[-i][j] = current_ref_org_expand[i][j];
                    current_ref_org_expand[current_target_image.cols - 1 + i][j] = current_ref_org_expand[current_target_image.cols - 1 - i][j];
                }
            }
            for (int i = -k; i < current_target_image.cols + k; i++) {
                for (int j = 1; j <= k; j++) {
                    current_target_expand[i][-j] = current_target_expand[i][j];
                    current_target_expand[i][current_target_image.rows - 1 + j] = current_target_expand[i][current_target_image.rows - 1 - j];
                    current_ref_expand[i][-j] = current_ref_expand[i][j];
                    current_ref_expand[i][current_target_image.rows - 1 + j] = current_ref_expand[i][current_target_image.rows - 1 - j];

                    current_target_org_expand[i][-j] = current_target_org_expand[i][j];
                    current_target_org_expand[i][current_target_image.rows - 1 + j] = current_target_org_expand[i][current_target_image.rows - 1 - j];
                    current_ref_org_expand[i][-j] = current_ref_org_expand[i][j];
                    current_ref_org_expand[i][current_target_image.rows - 1 + j] = current_ref_org_expand[i][current_target_image.rows - 1 - j];
                }
            }


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

            std::vector<cv::Point2f> scaled_coordinates{p0, p1, p2};

            if(step != 0) {
                // 画面外にはみ出してる場合、２倍からだんだん小さく縮小していく

                // ワーピング
                double magnification = 2.0;
                while ( (p0.x + tmp_mv_warping[0].x * magnification < 0 && p0.x + tmp_mv_warping[0].x * magnification > current_target_image.cols - 1) &&
                        (p1.x + tmp_mv_warping[1].x * magnification < 0 && p1.x + tmp_mv_warping[1].x * magnification > current_target_image.cols - 1) &&
                        (p2.x + tmp_mv_warping[2].x * magnification < 0 && p2.x + tmp_mv_warping[2].x * magnification > current_target_image.cols - 1) &&
                        (p0.y + tmp_mv_warping[0].y * magnification < 0 && p0.y + tmp_mv_warping[0].y * magnification > current_target_image.rows - 1) &&
                        (p1.y + tmp_mv_warping[1].y * magnification < 0 && p1.y + tmp_mv_warping[1].y * magnification > current_target_image.rows - 1) &&
                        (p2.y + tmp_mv_warping[2].y * magnification < 0 && p2.y + tmp_mv_warping[2].y * magnification > current_target_image.rows - 1) ) {
                    if(magnification <= 1)break;
                    magnification -= 0.1;
                }
                for (int s = 0; s < 3; s++) tmp_mv_warping[s] *= magnification;

                // 平行移動
                magnification = 2.0;
                while ( (p0.x + tmp_mv_parallel.x * magnification < 0 && p0.x + tmp_mv_parallel.x * magnification > current_target_image.cols - 1) &&
                        (p1.x + tmp_mv_parallel.x * magnification < 0 && p1.x + tmp_mv_parallel.x * magnification > current_target_image.cols - 1) &&
                        (p2.x + tmp_mv_parallel.x * magnification < 0 && p2.x + tmp_mv_parallel.x * magnification > current_target_image.cols - 1) &&
                        (p0.y + tmp_mv_parallel.y * magnification < 0 && p0.y + tmp_mv_parallel.y * magnification > current_target_image.rows - 1) &&
                        (p1.y + tmp_mv_parallel.y * magnification < 0 && p1.y + tmp_mv_parallel.y * magnification > current_target_image.rows - 1) &&
                        (p2.y + tmp_mv_parallel.y * magnification < 0 && p2.y + tmp_mv_parallel.y * magnification > current_target_image.rows - 1) ) {
                    if(magnification <= 1)break;
                    magnification -= 0.1;
                }
                tmp_mv_parallel *= magnification;
            }
            v_stack_parallel.clear();
            v_stack_warping.clear();

            Point3Vec current_triangle_coordinates(p0, p1, p2);
            double sx = std::min({(int) p0.x, (int) p1.x, (int) p2.x});
            double lx = std::max({(int) p0.x, (int) p1.x, (int) p2.x});
            double sy = std::min({(int) p0.y, (int) p1.y, (int) p2.y});
            double ly = std::max({(int) p0.y, (int) p1.y, (int) p2.y});

            pixels_in_triangle.clear();
            cv::Point2f xp;
            for (int j = (int) (round(sy) - 1); j <= round(ly) + 1; j++) {
                for (int i = (int) (round(sx) - 1); i <= round(lx) + 1; i++) {
                    xp.x = (float) i;
                    xp.y = (float) j;
                    if (isInTriangle(current_triangle_coordinates, xp) == 1) {
                        pixels_in_triangle.emplace_back(xp);//三角形の内部のピクセルを格納
                    }
                }
            }

            // 11回ガウス・ニュートン法をやる
            for(int gaussIterateNum = 0 ; gaussIterateNum < 11 ; gaussIterateNum++) {
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

                        if (X_later_warping.x >= current_ref_image.cols - 1) X_later_warping.x = current_ref_image.cols - 1.001;
                        if (X_later_warping.y >= current_ref_image.rows - 1) X_later_warping.y = current_ref_image.rows - 1.001;
                        if(X_later_warping.x < 0) X_later_warping.x = 0;
                        if(X_later_warping.y < 0) X_later_warping.y = 0;

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
                    double x0_later_warping_decimal = X_later_warping.x - x0_later_warping_integer;
                    double y0_later_warping_decimal = X_later_warping.y - y0_later_warping_integer;
                    double x0_later_parallel_decimal = X_later_parallel.x - x0_later_parallel_integer;
                    double y0_later_parallel_decimal = X_later_parallel.y - y0_later_parallel_integer;

                    double f = current_target_expand[x_integer][y_integer    ] * (1 - x_decimal) * (1 - y_decimal) + current_target_expand[x_integer + 1][y_integer    ] * x_decimal * (1 - y_decimal) +
                               current_target_expand[x_integer][y_integer + 1] * (1 - x_decimal) * y_decimal       + current_target_expand[x_integer + 1][y_integer + 1] * x_decimal * y_decimal;

                    double f_org = current_target_org_expand[x_integer][y_integer    ] * (1 - x_decimal) * (1 - y_decimal) + current_target_org_expand[x_integer + 1][y_integer    ] * x_decimal * (1 - y_decimal) +
                                   current_target_org_expand[x_integer][y_integer + 1] * (1 - x_decimal) * y_decimal       + current_target_org_expand[x_integer + 1][y_integer + 1] * x_decimal * y_decimal;


                    double g_warping = current_ref_expand[x0_later_warping_integer    ][y0_later_warping_integer    ] * (1 - x0_later_warping_decimal) * (1 - y0_later_warping_decimal) +
                                       current_ref_expand[x0_later_warping_integer + 1][y0_later_warping_integer    ] * x0_later_warping_decimal       * (1 - y0_later_warping_decimal) +
                                       current_ref_expand[x0_later_warping_integer    ][y0_later_warping_integer + 1] * (1 - x0_later_warping_decimal) * y0_later_warping_decimal       +
                                       current_ref_expand[x0_later_warping_integer + 1][y0_later_warping_integer + 1] * x0_later_warping_decimal       * y0_later_warping_decimal;//頂点を移動させた後のワーピングの参照フレームの輝度値

                    double g_parallel = current_ref_expand[x0_later_parallel_integer    ][y0_later_parallel_integer    ] * (1 - x0_later_parallel_decimal) * (1 - y0_later_parallel_decimal) +
                                        current_ref_expand[x0_later_parallel_integer + 1][y0_later_parallel_integer    ] *      x0_later_parallel_decimal  * (1 - y0_later_parallel_decimal) +
                                        current_ref_expand[x0_later_parallel_integer    ][y0_later_parallel_integer + 1] * (1 - x0_later_parallel_decimal) * (    y0_later_parallel_decimal)  +
                                        current_ref_expand[x0_later_parallel_integer + 1][y0_later_parallel_integer + 1] *      x0_later_parallel_decimal  * (    y0_later_parallel_decimal);//頂点を移動させた後の平行移動の参照フレームの輝度値

                    double g_org_warping = current_ref_org_expand[x0_later_warping_integer    ][y0_later_warping_integer    ] * (1 - x0_later_warping_decimal) * (1 - y0_later_warping_decimal) +
                                           current_ref_org_expand[x0_later_warping_integer + 1][y0_later_warping_integer    ] * x0_later_warping_decimal       * (1 - y0_later_warping_decimal) +
                                           current_ref_org_expand[x0_later_warping_integer    ][y0_later_warping_integer + 1] * (1 - x0_later_warping_decimal) * y0_later_warping_decimal       +
                                           current_ref_org_expand[x0_later_warping_integer + 1][y0_later_warping_integer + 1] * x0_later_warping_decimal       * y0_later_warping_decimal;//頂点を移動させた後のワーピングの参照フレームの輝度値

                    double g_org_parallel = current_ref_org_expand[x0_later_parallel_integer    ][y0_later_parallel_integer    ] * (1 - x0_later_parallel_decimal) * (1 - y0_later_parallel_decimal) +
                                            current_ref_org_expand[x0_later_parallel_integer + 1][y0_later_parallel_integer    ] *      x0_later_parallel_decimal  * (1 - y0_later_parallel_decimal) +
                                            current_ref_org_expand[x0_later_parallel_integer    ][y0_later_parallel_integer + 1] * (1 - x0_later_parallel_decimal) * (    y0_later_parallel_decimal)  +
                                            current_ref_org_expand[x0_later_parallel_integer + 1][y0_later_parallel_integer + 1] *      x0_later_parallel_decimal  * (    y0_later_parallel_decimal);//頂点を移動させた後の平行移動の参照フレームの輝度値

                    for (int row = 0; row < warping_matrix_dim; row++) {
                        for (int col = 0; col < warping_matrix_dim; col++) {
                            gg_warping.at<double>(row, col) += delta_g_warping[row] * delta_g_warping[col];//A_0の行列を生成(左辺の6x6の行列に相当)
                        }
                        B_warping.at<double>(row, 0) += (f - g_warping) * delta_g_warping[row];//bの行列を生成(右辺の6x1のベクトルに相当)
                    }
                    for (int row = 0; row < 2; row++) {
                        for (int col = 0; col < 2; col++) {
                            gg_parallel.at<double>(row, col) +=  delta_g_parallel[row] * delta_g_parallel[col];
                        }
                        B_parallel.at<double>(row, 0) += (f - g_parallel) * delta_g_parallel[row];
                    }

                    MSE_warping += (f_org - g_org_warping) * (f_org - g_org_warping);
                    MSE_parallel += (f_org - g_org_parallel) * (f_org - g_org_parallel);
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

                cv::solve(gg_warping, B_warping, delta_uv_warping);//6x6の連立方程式を解いてdelta_uvに格納
                cv::solve(gg_parallel, B_parallel, delta_uv_parallel);

                std::pair<std::vector<cv::Point2f>, double> v_pair_warping;
                std::pair<cv::Point2f, double> v_pair_parallel;
                v_pair_warping.first = tmp_mv_warping;//動きベクトルと予測残差のpairをスタックに格納
                v_pair_warping.second = Error_warping;
                v_stack_warping.emplace_back(v_pair_warping);
                v_pair_parallel.first = tmp_mv_parallel;
                v_pair_parallel.second = Error_parallel;
                v_stack_parallel.emplace_back(v_pair_parallel);

                for (int k = 0; k < 6; k++) {

                    if (k % 2 == 0) {
                        if ((0 <= scaled_coordinates[(int) (k / 2)].x + tmp_mv_warping[(int) (k / 2)].x + delta_uv_warping.at<double>(k, 0)) &&
                            (target_images[0][step].cols - 1 >=
                             scaled_coordinates[(int) (k / 2)].x + tmp_mv_warping[(int) (k / 2)].x + delta_uv_warping.at<double>(k, 0))) {
                            tmp_mv_warping[(int) (k / 2)].x = tmp_mv_warping[(int) (k / 2)].x + delta_uv_warping.at<double>(k, 0);//動きベクトルを更新(画像の外に出ないように)
                        }
                    } else {
                        if ((0 <= scaled_coordinates[(int) (k / 2)].y + tmp_mv_warping[(int) (k / 2)].y + delta_uv_warping.at<double>(k, 0)) &&
                            (target_images[0][step].rows - 1 >=
                             scaled_coordinates[(int) (k / 2)].y + tmp_mv_warping[(int) (k / 2)].y + delta_uv_warping.at<double>(k, 0))) {
                            tmp_mv_warping[(int) (k / 2)].y = tmp_mv_warping[(int) (k / 2)].y + delta_uv_warping.at<double>(k, 0);
                        }
                    }
                }

                for (int k = 0; k < 2; k++) {
                    if (k % 2 == 0) {
                        if ((0 <= scaled_coordinates[0].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) &&
                            (target_images[0][step].cols - 1 >=
                             scaled_coordinates[0].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) &&
                            (0 <= scaled_coordinates[1].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) &&
                            (target_images[0][step].cols - 1 >=
                             scaled_coordinates[1].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) &&
                            (0 <= scaled_coordinates[2].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0)) &&
                            (target_images[0][step].cols - 1 >=
                             scaled_coordinates[2].x + tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0))) {
                            tmp_mv_parallel.x = tmp_mv_parallel.x + delta_uv_parallel.at<double>(k, 0);
                        }
                    } else {
                        if ((0 <= scaled_coordinates[0].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                            (target_images[0][step].rows - 1 >=
                             scaled_coordinates[0].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                            (0 <= scaled_coordinates[1].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                            (target_images[0][step].rows - 1 >=
                             scaled_coordinates[1].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                            (0 <= scaled_coordinates[2].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0)) &&
                            (target_images[0][step].rows - 1 >=
                             scaled_coordinates[2].y + tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0))) {
                            tmp_mv_parallel.y = tmp_mv_parallel.y + delta_uv_parallel.at<double>(k, 0);
                        }
                    }
                }
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

            for(int d = -expand ;d < current_target_image.cols + expand;d++){
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
//    std::cout << "min_error_parallel:" << min_error_parallel << " min_error_warping:" << min_error_warping << std::endl;
    return std::make_tuple(std::vector<cv::Point2f>{max_v_warping[0], max_v_warping[1], max_v_warping[2]}, max_v_parallel, error, pixels_in_triangle.size(),true);
}

/**
 * @fn double Gauss_Newton(const cv::Mat& prev_color, const cv::Mat& current_color,const cv::Mat& intra,Point3Vec target_corners, Point3Vec& ref_corners,int &triangle_size)
 * @brief ガウスニュートン法で動き推定して予測残差のみを返す(画像生成はしてない)
 * @param[in]  prev_color     参照画像のグレースケール画像
 * @param[in]  current_color   対象画像のグレースケール画像
 * @param[in]  intra     イントラ符号化した参照画像のカラー画像
 * @param[in]  target_corners  三角形を構成する3つの頂点
 * @param[in]  ref_corners     参照フレーム上の3つの頂点(使ってない)
 * @param[out]  triangle_size        三角形の面積(正確には三角形の内部と判定された画素数)
 * @return double 予測残差
 */

double Gauss_Newton(const cv::Mat& prev_color, const cv::Mat& current_color,const cv::Mat& intra,
                    Point3Vec target_corners, Point3Vec& ref_corners,int &triangle_size) {
    //階層化の準備
    std::vector<std::vector<cv::Mat>> f_,g_;//f_ = 2種類のぼかしの対象フレームの階層化を格納 g_ = 2種類のぼかしの参照フレームの階層化を格納
    std::vector<cv::Mat>f_0,g_0;//f_0 = 対象フレームの階層化を格納 g_0 = 参照フレームの階層化を格納
    std::vector<cv::Mat>f_1,g_1,f_2,g_2;
    cv::Mat f0x1,f0x2,f0x4,f0x8,g0x1,g0x2,g0x4,g0x8;

    // 対象画像の縮小
    f0x1 = current_color;//第0階層の対象フレーム
    f0x2 = half(f0x1,2); //第1階層の対象フレーム
    f0x4 = half(f0x2,2); //第2階層の対象フレーム
    f0x8 = half(f0x4,2); //第3階層の対象フレーム
    f_0.emplace_back(f0x8);
    f_0.emplace_back(f0x4);
    f_0.emplace_back(f0x2);
    f_0.emplace_back(f0x1);

    // 参照画像の縮小
    g0x1 = prev_color;   //第0階層の参照フレーム
    g0x2 = half(g0x1,2); //第1階層の参照フレーム(1/2)
    g0x4 = half(g0x2,2); //第2階層の参照フレーム(1/4)
    g0x8 = half(g0x4,2); //第3階層の参照フレーム(1/8)
    g_0.emplace_back(g0x8);
    g_0.emplace_back(g0x4);
    g_0.emplace_back(g0x2);
    g_0.emplace_back(intra);//一番上の階層はイントラ符号化した画像を使用する(ガウスニュートン法が局所解に収束しないようにするため)

    cv::Mat fx1,fx2,fx4,fx8,gx1,gx2,gx4,gx8;

    // 上と同様にフィルタのパラメータを変えて階層化を行う
    // 参照画像
    fx1 = current_color;
    fx2 = half(fx1,2);
    fx4 = half(fx2,1);
    fx8 = half(fx4,1);
    f_1.emplace_back(fx8);
    f_1.emplace_back(fx4);
    f_1.emplace_back(fx2);
    f_1.emplace_back(fx1);

    // 対象画像
    gx1 = prev_color;
    gx2 = half(gx1,2);
    gx4 = half(gx2,1);
    gx8 = half(gx4,1);
    g_1.emplace_back(gx8);
    g_1.emplace_back(gx4);
    g_1.emplace_back(gx2);
    g_1.emplace_back(intra);//一番上の階層はイントラ符号化した画像を使用する(ガウスニュートン法が局所解に収束しないようにするため)

    // 移動平均のパラメタkを0にして縮小
    // 参照画像
    cv::Mat f2x1,f2x2,f2x4,f2x8,g2x1,g2x2,g2x4,g2x8;
    f2x1 = current_color;
    f2x2 = half(f2x1,0);
    f2x4 = half(f2x2,0);
    f2x8 = half(f2x4,0);
    f_2.emplace_back(f2x8);
    f_2.emplace_back(f2x4);
    f_2.emplace_back(f2x2);
    f_2.emplace_back(f2x1);

    // 対象画像
    g2x1 = prev_color;
    g2x2 = half(g2x1,0);
    g2x4 = half(g2x2,0);
    g2x8 = half(g2x4,0);
    g_2.emplace_back(g2x8);
    g_2.emplace_back(g2x4);
    g_2.emplace_back(g2x2);
    g_2.emplace_back(intra);//一番上の階層はイントラ符号化した画像を使用する(ガウスニュートン法が局所解に収束しないようにするため)

    // 参照画像と対象画像のvector
    f_.emplace_back(f_0);
    f_.emplace_back(f_1);
    f_.emplace_back(f_2);
    g_.emplace_back(g_0);
    g_.emplace_back(g_1);
    g_.emplace_back(g_2);


    const int dim = 6;
    double f, g, g2, delta_ek[dim],g_para;
    cv::Mat gg = cv::Mat_<double>(dim, dim);         // ガウスニュートン法の6x6の左辺の行列(ワーピング)
    cv::Mat gg_para = cv::Mat_<double>(2,2);         // ガウスニュートン法の2x2の左辺の行列(平行移動)
    cv::Mat B = cv::Mat_<double>(dim, 1);            // ガウスニュートン法の6x1の右辺のベクトル(ワーピング)
    cv::Mat B_para = cv::Mat_<double>(2,1);          // ガウスニュートン法の6x1の右辺のベクトル(平行移動)
    cv::Mat delta_uv = cv::Mat_<double>(dim, 1);     // ガウスニュートン法の6x1の左辺のベクトル(ワーピング)
    cv::Mat delta_uv_para = cv::Mat_<double>(2,1);   // ガウスニュートン法の6x1の左辺のベクトル(平行移動)
    double delta_g[dim] = {0},delta_g_para[2] = {0}; // 論文の付録で求めた予測パッチの動きに対する偏微分(6x6または2x2の行列を構成するために必要)
    bool parallel_flag;                              // 平行移動 = true, ワーピング = false
    unsigned char *warp;
    unsigned char *para;
    const double th = 0.5;                           //ワーピングか平行移動を選択するための閾値
    double MSE,MSE_para,Error,Error_min = 1E6,Error_para,Error_para_min = 1E6; //予測残差諸々
    double PSNR,PSNR_max = 0,PSNR_para,PSNR_para_max = 0;

    float delta_x, delta_y;//頂点を動かしたときのパッチ内の変動量x軸y軸独立に計算(delta_gを求めるために必要)
    std::vector<cv::Point2f> triangle, triangle_later, triangle_later_para;//三角形の頂点を格納
    std::vector<cv::Point2f> in_triangle_pixels, in_triangle_pixels_later;//三角形の内部の座標を格納
    cv::Point2f xp(0.0, 0.0), p, p0, p1, p2;
    std::vector<cv::Point2i> mv,mv_diff;
    cv::Point2f v_para,v_para_max;//平行移動の動きベクトルを格納
    std::vector<std::pair<std::vector<cv::Point2f>,double>> v_stack;//ガウスニュートン法を繰り返す度にワーピングの動きベクトルと予測残差のpairをスタックする(一番良い動きベクトルv_maxを採用するため)
    std::vector<std::pair<cv::Point2f,double>> v_stack_para;
    std::pair<std::vector<cv::Point2f>,double> v_pair;//動きベクトルと予測残差のpair
    std::pair<cv::Point2f,double> v_pair_para;

    std::vector<cv::Point2f> v_max(3, cv::Point2f(0.0, 0.0)); // ワーピングでPSNRが最大になる動きベクトルを格納
    p0 = target_corners.p1;
    p1 = target_corners.p2;
    p2 = target_corners.p3;


    for(int blare = 0;blare < 2;blare++) {//2種類のぼかし方ごとに
        std::vector<cv::Point2f> v(3, cv::Point2f(0.0, 0.0));
        v_para.x = 0; // 平行移動の動きベクトル(x要素)
        v_para.y = 0; // 平行移動の動きベクトル(y要素)
        v_para_max.x = 0;
        v_para_max.y = 0;

        p0 = target_corners.p1;
        p1 = target_corners.p2;
        p2 = target_corners.p3;
        for (int z = 0; z < 4; z++) {//各階層ごとに
            double scale = pow(2, 3-z);//各階層のスケーリングの値
            cv::Mat f_img = mv_filter(f_[blare][z], 2);//対照画像
            cv::Mat g_img = mv_filter(g_[blare][z], 2);//参照画像
            const int expand = 500;
            unsigned char **f_expand;//画像の周りに500ピクセルだけ黒の領域を設ける(念のため)
            unsigned char **g_expand;//f_expandと同様
            f_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (f_img.cols + expand * 2));
            f_expand += expand;
            for (int j = -expand; j < f_img.cols + expand; j++) {
                f_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (f_img.rows + expand * 2));
                f_expand[j] += expand;
            }
            g_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (f_img.cols + expand * 2));
            g_expand += expand;
            for (int j = -expand; j < g_img.cols + expand; j++) {
                if ((g_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (f_img.rows + expand * 2))) == NULL) {
                }
                g_expand[j] += expand;
            }
            for (int j = -expand; j < f_img.rows + expand; j++) {
                for (int i = -expand; i < f_img.cols + expand; i++) {
                    if (j >= 0 && j < f_img.rows && i >= 0 && i < f_img.cols) {
                        f_expand[i][j] = M(f_img, i, j);
                        g_expand[i][j] = M(g_img, i, j);
                    } else {
                        f_expand[i][j] = 0;
                        g_expand[i][j] = 0;
                    }
                }
            }
            int k = 2;//画像の周り2ピクセルだけ折り返し
            for (int j = 0; j < f_img.rows; j++) {
                for (int i = 1; i <= k; i++) {
                    f_expand[-i][j] = f_expand[i][j];
                    f_expand[f_img.cols - 1 + i][j] = f_expand[f_img.cols - 1 - i][j];
                    g_expand[-i][j] = g_expand[i][j];
                    g_expand[f_img.cols - 1 + i][j] = g_expand[f_img.cols - 1 - i][j];
                }
            }
            for (int i = -k; i < f_img.cols + k; i++) {
                for (int j = 1; j <= k; j++) {
                    f_expand[i][-j] = f_expand[i][j];
                    f_expand[i][f_img.rows - 1 + j] = f_expand[i][f_img.rows - 1 - j];
                    g_expand[i][-j] = g_expand[i][j];
                    g_expand[i][f_img.rows - 1 + j] = g_expand[i][f_img.rows - 1 - j];
                }
            }
            p0 = target_corners.p1 / scale;//三角形の頂点をスケーリング
            p1 = target_corners.p2 / scale;
            p2 = target_corners.p3 / scale;

//            //////////////ここからは頂点の距離が近い場合にバグってたときに迷走してました(近い距離に頂点を取らないようにしてバグを回避したため頂点を分割していく場合またバグが発生する可能性大)
//            cv::Point2f a_,b_;
//            double S_prev,S_later;//パッチの面積(デバッグ変数)
//            a_.x = p2.x - p0.x;
//            a_.y = p2.y - p0.y;
//            b_.x = p1.x - p0.x;
//            b_.y = p1.y - p0.y;
//            S_prev = a_.x * b_.y - a_.y * b_.x;
//            if(fabs((int)a_.x * (int)b_.y - (int)a_.y * (int)b_.x) <= 4){//パッチの面積が小さすぎる場合は動きベクトルをゼロにする
//                for (int s = 0; s < 3; s++) {
//                    v[s].x = 0;
//                    v[s].y = 0;
//                }
//                v_para.x = 0;
//                v_para.y = 0;
//                continue;
//            }
//            a_.x = p2.x - p1.x;
//            a_.y = p2.y - p1.y;
//            b_.x = p0.x - p1.x;
//            b_.y = p0.y - p1.y;
//            if(fabs((int)a_.x * (int)b_.y - (int)a_.y * (int)b_.x) <= 4){//パッチの面積が小さすぎる場合は動きベクトルをゼロにする
//                for (int s = 0; s < 3; s++) {
//                    v[s].x = 0;
//                    v[s].y = 0;
//                }
//                v_para.x = 0;
//                v_para.y = 0;
//                continue;
//            }
//            a_.x = (p2.x + v[2].x) - (p0.x + v[0].x);
//            a_.y = (p2.y + v[2].y) - (p0.y + v[0].y);
//            b_.x = (p1.x + v[1].x) - (p0.x + v[0].x);
//            b_.y = (p1.y + v[1].y) - (p0.y + v[0].y);
//            S_later = a_.x * b_.y - a_.y * b_.x;
//
//            ///////////迷走終わり
            if (target_corners.p1.x == f_[0][3].cols - 1)p0.x = f_[0][z].cols - 1;//端の頂点の調整
            if (target_corners.p1.y == f_[0][3].rows - 1)p0.y = f_[0][z].rows - 1;
            if (target_corners.p2.x == f_[0][3].cols - 1)p1.x = f_[0][z].cols - 1;
            if (target_corners.p2.y == f_[0][3].rows - 1)p1.y = f_[0][z].rows - 1;
            if (target_corners.p3.x == f_[0][3].cols - 1)p2.x = f_[0][z].cols - 1;
            if (target_corners.p3.y == f_[0][3].rows - 1)p2.y = f_[0][z].rows - 1;
            cv::Point2f a_,b_;
            a_.x = p2.x - p0.x;
            a_.y = p2.y - p0.y;
            b_.x = p1.x - p0.x;
            b_.y = p1.y - p0.y;
            if(fabs(a_.x * b_.y - a_.y * b_.x) <= 0)break;
            triangle.clear();
            triangle.emplace_back(p0);
            triangle.emplace_back(p1);
            triangle.emplace_back(p2);

            bool length2_flag = false;
            if(z != 0) {
                double length = 2;
                while ( (p0.x + v[0].x * 2 < 0 && p0.x + v[0].x * 2 > f_img.cols - 1) &&
                        (p1.x + v[1].x * 2 < 0 && p1.x + v[1].x * 2 > f_img.cols - 1) &&
                        (p2.x + v[2].x * 2 < 0 && p2.x + v[2].x * 2 > f_img.cols - 1) &&
                        (p0.y + v[0].y * 2 < 0 && p0.y + v[0].y * 2 > f_img.rows - 1) &&
                        (p1.y + v[1].y * 2 < 0 && p1.y + v[1].y * 2 > f_img.rows - 1) &&
                        (p2.y + v[2].y * 2 < 0 && p2.y + v[2].y * 2 > f_img.rows - 1) ) {
                    if(length <= 1)break;
                    length -= 0.1;
                }
                for (int s = 0; s < 3; s++) v[s] *= length;

                length = 2;
                while ( (p0.x + v_para.x * length < 0 && p0.x + v_para.x * length > f_img.cols - 1) &&
                        (p1.x + v_para.x * length < 0 && p1.x + v_para.x * length > f_img.cols - 1) &&
                        (p2.x + v_para.x * length < 0 && p2.x + v_para.x * length > f_img.cols - 1) &&
                        (p0.y + v_para.y * length < 0 && p0.y + v_para.y * length > f_img.rows - 1) &&
                        (p1.y + v_para.y * length < 0 && p1.y + v_para.y * length > f_img.rows - 1) &&
                        (p2.y + v_para.y * length < 0 && p2.y + v_para.y * length > f_img.rows - 1) ) {
                    if(length <= 1)break;
                    length -= 0.1;
                }
                v_para *= length;
            }

            v_stack.clear();
            v_stack_para.clear();

            Point3Vec triangleVec = Point3Vec(p0, p1, p2);
            double sx = std::min({(int) p0.x, (int) p1.x, (int) p2.x});
            double lx = std::max({(int) p0.x, (int) p1.x, (int) p2.x});
            double sy = std::min({(int) p0.y, (int) p1.y, (int) p2.y});
            double ly = std::max({(int) p0.y, (int) p1.y, (int) p2.y});
            if(lx - sx == 0 || ly - sy == 0){
                std::cout << "baund = 0" << std::endl;
            }
            in_triangle_pixels.clear();
            for (int j = (int) (round(sy) - 1); j <= round(ly) + 1; j++) {
                for (int i = (int) (round(sx) - 1); i <= round(lx) + 1; i++) {
                    xp.x = (float) i;
                    xp.y = (float) j;
                    if (isInTriangle(triangleVec, xp) == 1) {
                        in_triangle_pixels.emplace_back(xp);//三角形の内部のピクセルを格納
                    }
                }
            }
            for (int q = 0; q < 11; q++) {//ガウスニュートン法を11回繰り返す
                if (q == 10 && z == 3) {//11回目は動きベクトルをゼロにする
                    for (int i = 0; i < 3; i++) {
                        v[i].x = 0;
                        v[i].y = 0;
                    }
                    v_para.x = 0;
                    v_para.y = 0;
                }
                //while(Residual_Error >= th) {
                p0 = target_corners.p1 / scale;//三角形の頂点をスケーリング
                p1 = target_corners.p2 / scale;
                p2 = target_corners.p3 / scale;
                if (target_corners.p1.x == f_[0][3].cols - 1)p0.x = f_[0][z].cols - 1;//端の頂点の調整
                if (target_corners.p1.y == f_[0][3].rows - 1)p0.y = f_[0][z].rows - 1;
                if (target_corners.p2.x == f_[0][3].cols - 1)p1.x = f_[0][z].cols - 1;
                if (target_corners.p2.y == f_[0][3].rows - 1)p1.y = f_[0][z].rows - 1;
                if (target_corners.p3.x == f_[0][3].cols - 1)p2.x = f_[0][z].cols - 1;
                if (target_corners.p3.y == f_[0][3].rows - 1)p2.y = f_[0][z].rows - 1;

                cv::Point2f a, b, X, c, d;
                cv::Point2f a_later, b_later, X_later, a_later_para, b_later_para, X_later_para;
                std::vector<cv::Point2f> pp,pp_para;//移動後の座標を格納
                pp.clear();
                pp.emplace_back(p0);
                pp.emplace_back(p1);
                pp.emplace_back(p2);
                pp_para.clear();
                pp_para.emplace_back(p0);
                pp_para.emplace_back(p1);
                pp_para.emplace_back(p2);
                int x0, y0, x0_later, y0_later,x0_later_para,y0_later_para;
                double d_x, d_y, d_x_later, d_y_later,d_x_later_para,d_y_later_para;
                double alpha, beta, det;
                double g_x, g_y,g_x_para,g_y_para;//参照フレームの前進差分
                double S0,S1,S_[6];//面積の制約に関する諸々
                double myu = 10;//面積の制約の重み
                a = p2 - p0;
                b = p1 - p0;
                det = a.x * b.y - a.y * b.x;
                c = v[2] - v[0];
                d = v[1] - v[0];
                S0 = 0.5*fabs(det);//移動前の面積
                S1 = 0.5*fabs((b.x + d.x)*(a.y + c.y) - (a.x + c.x)*(b.y + d.y));//移動後の面積
                S_[0] = -0.5*(a.y + c.y - b.y - d.y);
                S_[1] = -0.5*(b.x + d.x - a.x - c.x);
                S_[2] = 0.5*(a.y + c.y);
                S_[3] = -0.5*(a.x + c.x);
                S_[4] = -0.5*(b.y + d.y);
                S_[5] = 0.5*(b.x + d.x);

                MSE = 0;
                MSE_para = 0;
                for (int k = 0; k < dim; k++) {
                    for (int j = 0; j < dim; j++) {
                        gg.at<double>(k, j) = 0;
                    }
                    B.at<double>(k, 0) = 0;
                    delta_ek[k] = 0;
                }
                for (int k = 0; k < 2; k++) {
                    for (int j = 0; j < 2; j++) {
                        gg_para.at<double>(k, j) = 0;
                    }
                    B_para.at<double>(k, 0) = 0;
                }
                for (int m = 0; m < (int) in_triangle_pixels.size(); m++) {//パッチ内の画素ごとに
                    X.x = in_triangle_pixels[m].x - p0.x;
                    X.y = in_triangle_pixels[m].y - p0.y;
                    alpha = (X.x * b.y - X.y * b.x) / det;
                    beta = (a.x * X.y - a.y * X.x) / det;

                    X.x += p0.x;
                    X.y += p0.y;
                    x0 = (int) floor(X.x);
                    y0 = (int) floor(X.y);
                    d_x = X.x - x0;
                    d_y = X.y - y0;

                    for (int i = 0; i < 6; i++) {
                        pp[0] = p0 + v[0];
                        pp[1] = p1 + v[1];
                        pp[2] = p2 + v[2];
                        triangle_later.clear();
                        triangle_later.emplace_back(pp[0]);//移動後の頂点を格納
                        triangle_later.emplace_back(pp[1]);
                        triangle_later.emplace_back(pp[2]);
                        pp_para[0] = p0 + v_para;
                        pp_para[1] = p1 + v_para;
                        pp_para[2] = p2 + v_para;
                        triangle_later_para.clear();
                        triangle_later_para.emplace_back(pp_para[0]);
                        triangle_later_para.emplace_back(pp_para[1]);
                        triangle_later_para.emplace_back(pp_para[2]);

                        a_later.x = triangle_later[2].x - triangle_later[0].x;
                        a_later.y = triangle_later[2].y - triangle_later[0].y;
                        a_later_para = triangle_later_para[2] - triangle_later_para[0];
                        b_later.x = triangle_later[1].x - triangle_later[0].x;
                        b_later.y = triangle_later[1].y - triangle_later[0].y;
                        b_later_para = triangle_later_para[1] - triangle_later_para[0];
                        X_later.x = alpha * a_later.x + beta * b_later.x + triangle_later[0].x;
                        X_later.y = alpha * a_later.y + beta * b_later.y + triangle_later[0].y;
                        X_later_para = alpha * a_later_para + beta * b_later_para + triangle_later_para[0];

                        if (X_later.x >= g_img.cols - 1) {
                            X_later.x = g_img.cols - 1.001;
                        }
                        if (X_later.y >= g_img.rows - 1) {
                            X_later.y = g_img.rows - 1.001;
                        }
                        if(X_later.x < 0) {
                            X_later.x = 0;
                        }
                        if(X_later.y < 0) {
                            X_later.y = 0;
                        }
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
                        }
                        g_x = g_expand[(int) X_later.x + 1][(int) X_later.y] - g_expand[(int) X_later.x][(int) X_later.y];//前進差分
                        g_y = g_expand[(int) X_later.x][(int) X_later.y + 1] - g_expand[(int) X_later.x][(int) X_later.y];
                        g_x_para = g_expand[(int) X_later_para.x + 1][(int) X_later_para.y] - g_expand[(int) X_later_para.x][(int) X_later_para.y];
                        g_y_para = g_expand[(int) X_later_para.x][(int) X_later_para.y + 1] - g_expand[(int) X_later_para.x][(int) X_later_para.y];

                        delta_g[i] = g_x * delta_x + g_y * delta_y;

                    }
                    delta_g_para[0] = g_x_para;
                    delta_g_para[1] = g_y_para;


                    x0_later = (int) floor(X_later.x);
                    y0_later = (int) floor(X_later.y);
                    x0_later_para = (int)floor(X_later_para.x);
                    y0_later_para = (int)floor(X_later_para.y);
                    d_x_later = X_later.x - x0_later;
                    d_y_later = X_later.y - y0_later;
                    d_x_later_para = X_later_para.x - x0_later_para;
                    d_y_later_para = X_later_para.y - y0_later_para;
                    f = f_expand[x0][y0    ] * (1 - d_x) * (1 - d_y) + f_expand[x0 + 1][y0    ] * d_x * (1 - d_y) +
                        f_expand[x0][y0 + 1] * (1 - d_x) * d_y       + f_expand[x0 + 1][y0 + 1] * d_x * d_y;

                    g = g_expand[x0_later    ][y0_later    ] * (1 - d_x_later) * (1 - d_y_later) +
                        g_expand[x0_later + 1][y0_later    ] * d_x_later       * (1 - d_y_later) +
                        g_expand[x0_later    ][y0_later + 1] * (1 - d_x_later) * d_y_later       +
                        g_expand[x0_later + 1][y0_later + 1] * d_x_later       * d_y_later;//頂点を移動させた後のワーピングの参照フレームの輝度値

                    g_para = g_expand[x0_later_para    ][y0_later_para    ] * (1 - d_x_later_para) * (1 - d_y_later_para) +
                             g_expand[x0_later_para + 1][y0_later_para    ] * d_x_later_para       * (1 - d_y_later_para) +
                             g_expand[x0_later_para    ][y0_later_para + 1] * (1 - d_x_later_para) * d_y_later_para       +
                             g_expand[x0_later_para + 1][y0_later_para + 1] * d_x_later_para       * d_y_later_para;//頂点を移動させた後の平行移動の参照フレームの輝度値

                    g2 = g + delta_g[0] * delta_uv.at<double>(0, 0) + delta_g[1] * delta_uv.at<double>(1, 0) +
                         delta_g[2] * delta_uv.at<double>(2, 0) +
                         delta_g[3] * delta_uv.at<double>(3, 0) + delta_g[4] * delta_uv.at<double>(4, 0) +
                         delta_g[5] * delta_uv.at<double>(5, 0);//gを偏微分と微小変動量から近似したもの(デバッグ用)
                    for (int t = 0; t < dim; t++) {
                        delta_ek[t] += (f - g2) * delta_g[t];//近似値からbを生成(デバッグ用)この値が0に収束していればガウスニュートン法が正しく作用している
                    }
                    for (int k = 0; k < dim; k++) {
                        for (int j = 0; j < dim; j++) {
                            gg.at<double>(k, j) += delta_g[k] * delta_g[j];//A_0の行列を生成(左辺の6x6の行列に相当)
                        }
                        B.at<double>(k, 0) += (f - g) * delta_g[k];//bの行列を生成(右辺の6x1のベクトルに相当)
                    }
                    for (int k = 0; k < 2; k++) {
                        for (int j = 0; j < 2; j++) {
                            gg_para.at<double>(k, j) +=  delta_g_para[k] * delta_g_para[j];
                        }
                        B_para.at<double>(k, 0) += (f - g_para) * delta_g_para[k];
                    }

                    MSE += (f - g) * (f - g);
                    MSE_para += (f - g_para)*(f - g_para);

                }
                for (int k = 0; k < dim; k++) {
                    for (int j = 0; j < dim; j++) {
                        gg.at<double>(k, j) += myu*((S1*S1*S1 - S0*S0*S1)/(S0*S0*S0*S0)*S_[k]);//A_1の行列を足しこむ(面積の制約に相当)
                    }
                }
                for (int t = 0; t < dim; t++) {
                    delta_ek[t] *= -2;
                }
                Error = MSE;//パッチ全体の予測残差の和
                Error_para = MSE_para;
                triangle_size = (int)in_triangle_pixels.size();
                MSE = (in_triangle_pixels.size() == 0 ? MSE : MSE / in_triangle_pixels.size());//パッチ内の平均2乗誤差
                MSE_para = (in_triangle_pixels.size() == 0 ? MSE_para : MSE_para / in_triangle_pixels.size());
                PSNR = 10 * log10((255 * 255) / MSE);//パッチ内のPSNR
                PSNR_para = 10 * log10((255 * 255) / MSE_para);

                cv::solve(gg, B, delta_uv);//6x6の連立方程式を解いてdelta_uvに格納
                cv::solve(gg_para, B_para, delta_uv_para);
                v_pair.first = v;//動きベクトルと予測残差のpairをスタックに格納
                v_pair.second = Error;
                v_stack.emplace_back(v_pair);
                v_pair_para.first = v_para;
                v_pair_para.second = Error_para;
                v_stack_para.emplace_back(v_pair_para);
                for (int k = 0; k < 6; k++) {
                    if (k % 2 == 0) {
                        if ((0 <= triangle[(int) (k / 2)].x + v[(int) (k / 2)].x + delta_uv.at<double>(k, 0)) &&
                            (f_[0][z].cols - 1 >=
                             triangle[(int) (k / 2)].x + v[(int) (k / 2)].x + delta_uv.at<double>(k, 0))) {
                            v[(int) (k / 2)].x = v[(int) (k / 2)].x + delta_uv.at<double>(k, 0);//動きベクトルを更新(画像の外に出ないように)
                        }
                    } else {
                        if ((0 <= triangle[(int) (k / 2)].y + v[(int) (k / 2)].y + delta_uv.at<double>(k, 0)) &&
                            (f_[0][z].rows - 1 >=
                             triangle[(int) (k / 2)].y + v[(int) (k / 2)].y + delta_uv.at<double>(k, 0))) {
                            v[(int) (k / 2)].y = v[(int) (k / 2)].y + delta_uv.at<double>(k, 0);
                        }
                    }
                }
                for (int k = 0; k < 2; k++) {
                    if (k % 2 == 0) {
                        if ((0 <= triangle[0].x + v_para.x + delta_uv_para.at<double>(k, 0)) &&
                            (f_[0][z].cols - 1 >= triangle[0].x + v_para.x + delta_uv_para.at<double>(k, 0)) &&
                            (0 <= triangle[1].x + v_para.x + delta_uv_para.at<double>(k, 0)) &&
                            (f_[0][z].cols - 1 >=triangle[1].x + v_para.x + delta_uv_para.at<double>(k, 0)) &&
                            (0 <= triangle[2].x + v_para.x + delta_uv_para.at<double>(k, 0)) &&
                            (f_[0][z].cols - 1 >= triangle[2].x + v_para.x + delta_uv_para.at<double>(k, 0))) {
                            v_para.x = v_para.x + delta_uv_para.at<double>(k, 0);
                        }
                    } else {
                        if ((0 <= triangle[0].y + v_para.y + delta_uv_para.at<double>(k, 0)) &&
                            (f_[0][z].rows - 1 >=
                             triangle[0].y + v_para.y + delta_uv_para.at<double>(k, 0)) &&
                            (0 <= triangle[1].y + v_para.y + delta_uv_para.at<double>(k, 0)) &&
                            (f_[0][z].rows - 1 >=
                             triangle[1].y + v_para.y + delta_uv_para.at<double>(k, 0)) &&
                            (0 <= triangle[2].y + v_para.y + delta_uv_para.at<double>(k, 0)) &&
                            (f_[0][z].rows - 1 >=
                             triangle[2].y + v_para.y + delta_uv_para.at<double>(k, 0))){
                            v_para.y = v_para.y + delta_uv_para.at<double>(k, 0);
                        }
                    }
                }
            }

            std::sort(v_stack.begin(), v_stack.end(), [](std::pair<std::vector<cv::Point2f>,double> a, std::pair<std::vector<cv::Point2f>,double> b){
                return a.second < b.second;
            });

            std::sort(v_stack_para.begin(), v_stack_para.end(), [](std::pair<cv::Point2f,double> a, std::pair<cv::Point2f,double> b){
                return a.second < b.second;
            });
//            for(int i = 0;i < (int)v_stack_para.size();i++){
//                std::cout << "Gauss_newton1_v_stack_para[" << i << "][blur=" << blare << "]:[z=" << z << "]:"  << v_stack_para[i].first << " Error:" << v_stack_para[i].second << std::endl;
//            }

            v = v_stack[0].first;//一番良い動きベクトルを採用
            Error = v_stack[0].second;
            v_para = v_stack_para[0].first;
            Error_para = v_stack_para[0].second;
            MSE = Error / (double)in_triangle_pixels.size();
            MSE_para = Error_para / (double)in_triangle_pixels.size();
            PSNR = 10 * log10((255 * 255) / MSE);
            PSNR_para = 10 * log10((255 * 255) / MSE_para);

            bool flag_blare = false,flag_blare_para = false;//2種類のボケ方の採用時に使用
            if(z == 3) {//一番下の階層で
                if(PSNR_para >= PSNR_para_max){//2種類のボケ方で良い方を採用
                    PSNR_para_max = PSNR_para;
                    Error_para_min = Error_para;
                    v_para_max = v_para;
                    flag_blare_para = true;
                }
                if (PSNR >= PSNR_max) {
                    PSNR_max = PSNR;
                    Error_min = Error;
                    v_max = v;
                    flag_blare = true;
                }
                parallel_flag = true;
//                if (fabs(PSNR_max - PSNR_para_max) <= th || PSNR_para_max > PSNR_max) {//ワーピングと平行移動でRDのようなことをする
//                    parallel_flag = true;//平行移動を採用
//                } else{
//                    parallel_flag = false;//ワーピングを採用
//                }
            }
            //ここからは画像の生成と動きベクトルの処理(この関数内では使ってないので無視してください)
            double alpha, beta, det;

            cv::Point2f X, a, b;
            cv::Point2f X_later,X_later_para, a_later,a_later_para, b_later,b_later_para;
            cv::Point2f pp0,pp0_para, pp1,pp1_para, pp2,pp2_para;

            std::vector<cv::Point2f> pp,pp_para, mv_tmp;
            pp0.x = triangle[0].x + v[0].x;
            pp0.y = triangle[0].y + v[0].y;
            pp1.x = triangle[1].x + v[1].x;
            pp1.y = triangle[1].y + v[1].y;
            pp2.x = triangle[2].x + v[2].x;
            pp2.y = triangle[2].y + v[2].y;
            if(z == 3) {
                pp0_para.x = triangle[0].x + v_para.x;
                pp0_para.y = triangle[0].y + v_para.y;
                pp1_para.x = triangle[1].x + v_para.x;
                pp1_para.y = triangle[1].y + v_para.y;
                pp2_para.x = triangle[2].x + v_para.x;
                pp2_para.y = triangle[2].y + v_para.y;
            }
            double Quant = 4;
            std::vector<cv::Point2f> mv2, mv3, mv4, mv_diff_tmp;
            cv::Point2f ave_v;
            pp.clear();
            pp.emplace_back(pp0);
            pp.emplace_back(pp1);
            pp.emplace_back(pp2);
            pp_para.clear();
            pp_para.emplace_back(pp0_para);
            pp_para.emplace_back(pp1_para);
            pp_para.emplace_back(pp2_para);
            if(z == 3) {
                ave_v.x = (v_max[0].x + v_max[1].x + v_max[2].x) / 3;
                ave_v.y = (v_max[0].y + v_max[1].y + v_max[2].y) / 3;
                ave_v.x = (int) floor(ave_v.x + 0.5);
                ave_v.y = (int) floor(ave_v.y + 0.5);
                cv::Point2f v0, v1, v2;
                v0.x = floor(ref_corners.p1.x / 2 + 0.5) - p0.x;
                v0.y = floor(ref_corners.p1.y / 2 + 0.5) - p0.y;
                v1.x = floor(ref_corners.p2.x / 2 + 0.5) - p1.x;
                v1.y = floor(ref_corners.p2.y / 2 + 0.5) - p1.y;
                v2.x = floor(ref_corners.p3.x / 2 + 0.5) - p2.x;
                v2.y = floor(ref_corners.p3.y / 2 + 0.5) - p2.y;
                mv2.clear();
                mv2.emplace_back(v0);
                mv2.emplace_back(v1);
                mv2.emplace_back(v2);
                mv_tmp.clear();
                mv_tmp.emplace_back(v[0]);
                mv_tmp.emplace_back(v[1]);
                mv_tmp.emplace_back(v[2]);
                if (parallel_flag == false) {
                    mv_diff_tmp.clear();
                    mv_diff_tmp.emplace_back(v_max[0]);
                    mv_diff_tmp.emplace_back(v_max[1]);
                    mv_diff_tmp.emplace_back(v_max[2]);
                }
                mv.clear();
                mv.emplace_back(v[0]);
                mv.emplace_back(v[1]);
                mv.emplace_back(v[2]);
                mv3.clear();
                mv3.emplace_back(v_max[0]);
                mv3.emplace_back(v_max[1]);
                mv3.emplace_back(v_max[2]);
                mv4.clear();
                mv4.emplace_back(v_para_max);
                mv4.emplace_back(v_para_max);
                mv4.emplace_back(v_para_max);
                pp.clear();
                pp.emplace_back(pp0);
                pp.emplace_back(pp1);
                pp.emplace_back(pp2);
                pp_para.clear();
                pp_para.emplace_back(pp0_para);
                pp_para.emplace_back(pp1_para);
                pp_para.emplace_back(pp2_para);
                if (parallel_flag == false) {
                    for (int j = 0; j < 3; j++) {
                        //mv3[j] -= ave_v;
                        int x0, y0;
                        double d_x, d_y;
                        mv3[j] *= Quant;
                        mv3[j].x = (int)mv3[j].x;
                        mv3[j].y = (int)mv3[j].y;
                        mv3[j] /= 2;
                        x0 = (int) floor(mv3[j].x);
                        y0 = (int) floor(mv3[j].y);
                        d_x = mv3[j].x - x0;
                        d_y = mv3[j].y - y0;
                        //mv3[j] *= Quant;
                        mv3[j].x = (int) floor(mv3[j].x + 0.5);
                        mv3[j].y = (int) floor(mv3[j].y + 0.5);
                        mv[j] = mv3[j];
                        cv::Point2i dv(d_x, d_y);
                        mv.emplace_back(dv);
                    }
                }
                else{
                    mv4[0] = v_para_max;
                    int x0,y0;
                    double d_x,d_y;
                    mv4[0] *= Quant;
                    mv4[0].x = (int)mv4[0].x;
                    mv4[0].y = (int)mv4[0].y;
                    mv4[0] /= 2;
                    x0 = (int)mv4[0].x;
                    y0 = (int)mv4[0].y;
                    d_x = (mv4[0].x - x0)*2; // ハーフペル相当になっている
                    d_y = (mv4[0].y - y0)*2;
                    mv[0].x = x0; mv[0].y = y0;
                    mv[1].x = x0; mv[1].y = y0;
                    mv[2].x = x0; mv[2].y = y0;
                    cv::Point2i dv(d_x,d_y);
                    mv.emplace_back(dv); // 安全のために小数部を3つ追加
                    mv.emplace_back(dv);
                    mv.emplace_back(dv);
//                    std::cout << "Gauss_newton1_mv:" << mv << std::endl;
                }


                for (int j = 0; j < 3; j++) {
                    if (pp[j].x < 0) {
                        mv_tmp[j].x -= pp[j].x;
                        pp[j].x = 0;
                    } else if (pp[j].x > g_img.cols - 1) {
                        mv_tmp[j].x -= g_img.cols - 1 - pp[j].x;
                        pp[j].x = g_img.cols - 1;
                    }
                    if (pp[j].y < 0) {
                        mv_tmp[j].y -= pp[j].y;
                        pp[j].y = 0;
                    } else if (pp[j].y > g_img.rows - 1) {
                        mv_tmp[j].y -= g_img.rows - 1 - pp[j].y;
                        pp[j].y = g_img.rows - 1;
                    }
                }
                for (int j = 0; j < 3; j++) {
                    double tmp_x, tmp_y;
                    tmp_x = pp[j].x * Quant;
                    tmp_y = pp[j].y * Quant;
                    tmp_x = floor(tmp_x + 0.5);
                    tmp_y = floor(tmp_y + 0.5);
                    pp[j].x = tmp_x / Quant;
                    pp[j].y = tmp_y / Quant;
                    tmp_x = pp_para[j].x * Quant;
                    tmp_y = pp_para[j].y * Quant;
                    tmp_x = floor(tmp_x + 0.5);
                    tmp_y = floor(tmp_y + 0.5);
                    pp_para[j].x = tmp_x / Quant;
                    pp_para[j].y = tmp_y / Quant;
                    tmp_x = mv_tmp[j].x * Quant;
                    tmp_y = mv_tmp[j].y * Quant;

                }
            }
            a.x = triangle[2].x - triangle[0].x;
            a.y = triangle[2].y - triangle[0].y;
            b.x = triangle[1].x - triangle[0].x;
            b.y = triangle[1].y - triangle[0].y;
            det = a.x * b.y - a.y * b.x;

            if(blare == 0) {
                warp = (unsigned char *) malloc(sizeof(unsigned char) * (int) in_triangle_pixels.size());
                para = (unsigned char *) malloc(sizeof(unsigned char) * (int) in_triangle_pixels.size());
            }
            for (int i = 0; i < (int) in_triangle_pixels.size(); i++) {
                double d_x, d_y, d_x_para, d_y_para;
                int x0, y0, x0_para, y0_para;
                X.x = in_triangle_pixels[i].x - triangle[0].x;
                X.y = in_triangle_pixels[i].y - triangle[0].y;
                alpha = (X.x * b.y - X.y * b.x) / det;
                beta = (a.x * X.y - a.y * X.x) / det;
                X.x += triangle[0].x;
                X.y += triangle[0].y;

                a_later.x = pp[2].x - pp[0].x;
                a_later.y = pp[2].y - pp[0].y;
                a_later_para = pp_para[2] - pp_para[0];
                b_later.x = pp[1].x - pp[0].x;
                b_later.y = pp[1].y - pp[0].y;
                b_later_para = pp_para[1] - pp_para[0];
                X_later.x = alpha * a_later.x + beta * b_later.x + pp[0].x;
                X_later.y = alpha * a_later.y + beta * b_later.y + pp[0].y;
                X_later_para = alpha * a_later_para + beta * b_later_para + pp_para[0];

                if (X_later.x >= g_img.cols - 1) {
                    X_later.x = g_img.cols - 1.001;
                }
                if (X_later.y >= g_img.rows - 1) {
                    X_later.y = g_img.rows - 1.001;
                }

                if (X_later.x < 0) {
                    X_later.x = 0;
                }
                if (X_later.y < 0) {
                    X_later.y = 0;
                }
                if (X_later_para.x >= g_img.cols - 1) {
                    X_later_para.x = g_img.cols - 1.001;
                }
                if (X_later_para.y >= g_img.rows - 1) {
                    X_later_para.y = g_img.rows - 1.001;
                }

                if (X_later_para.x < 0) {
                    X_later_para.x = 0;
                }
                if (X_later_para.y < 0) {
                    X_later_para.y = 0;
                }
                x0 = floor(X_later.x);
                d_x = X_later.x - x0;
                y0 = floor(X_later.y);
                d_y = X_later.y - y0;
                x0_para = floor(X_later_para.x);
                d_x_para = X_later_para.x - x0_para;
                y0_para = floor(X_later_para.y);
                d_y_para = X_later_para.y - y0_para;
                int y, y_para;

                y = (int) floor((M(g_[blare][z], (int) x0, (int) y0) * (1 - d_x) * (1 - d_y) +
                                 M(g_[blare][z], (int) x0 + 1, (int) y0) * d_x * (1 - d_y)
                                 + M(g_[blare][z], (int) x0, (int) y0 + 1) * (1 - d_x) * d_y +
                                 M(g_[blare][z], (int) x0 + 1, (int) y0 + 1) * d_x * d_y) + 0.5);
                y_para = (int) floor((M(g_[blare][z], (int) x0_para, (int) y0_para) * (1 - d_x_para) * (1 - d_y_para) +
                                      M(g_[blare][z], (int) x0_para + 1, (int) y0_para) * d_x_para * (1 - d_y_para)
                                      + M(g_[blare][z], (int) x0_para, (int) y0_para + 1) * (1 - d_x_para) * d_y_para +
                                      M(g_[blare][z], (int) x0_para + 1, (int) y0_para + 1) * d_x_para * d_y_para)+ 0.5);
                if(z == 3){
                    if(flag_blare_para){
                        para[i] = (unsigned char) y_para;
                    }
                    else{
                        warp[i] = (unsigned char) y;
                    }
                }
            }

            for (int j = 0; j < 3; j++) {
                pp[j].x -= triangle[j].x;
                pp[j].y -= triangle[j].y;
            }

            for(int d = -expand ;d < f_img.cols + expand;d++){
                f_expand[d] -= expand;
                g_expand[d] -= expand;
                free(f_expand[d]);
                free(g_expand[d]);
            }
            f_expand -= expand;
            g_expand -= expand;
            free(f_expand);
            free(g_expand);

        }
    }
    double squaredError = 0.0;
    for (int i = 0; i < (int) in_triangle_pixels.size(); i++) {
        unsigned char y;
        if (parallel_flag == true) {
            y = para[i];
        }
        else{
            y = warp[i];
        }
        cv::Point2f X = in_triangle_pixels[i];
        squaredError += (M(current_color,(int)X.x,(int)X.y) - y) * (M(current_color,(int)X.x,(int)X.y) - y);
    }
    return squaredError;
}

/**
 * @fn std::vector<cv::Point2i> Gauss_Newton2(const cv::Mat& prev_color,const cv::Mat& current_color,const cv::Mat& intra, std::vector<cv::Mat>& predict_buf,cv::Mat&predict_warp,cv::Mat&predict_para,cv::Mat& color,
                                       double &error_warp, Point3Vec target_corners, Point3Vec& ref_corners, std::ofstream &tri_list,bool *flag,std::vector<cv::Point2f> &add_corners,int *add_count,int t,const cv::Mat& residual_ref ,int &in_triangle_size,bool add_corner_flag,double erase_th_global)
 * @brief ガウスニュートン法で動き推定して予測画像と動きベクトルを返す
 * @param[in]   prev_color                参照画像のグレースケール画像
 * @param[in]   current_color             対象画像のグレースケール画像
 * @param[in]   intra                     イントラ符号化した参照画像のカラー画像
 * @param[out]  predict_buf               予測画像を各階層ごとに格納するバッファ
 * @param[out]  predict_warp              ワーピングの予測画像を格納(デバッグ用)
 * @param[out]  predict_para              平行移動の予測画像を格納(デバッグ用)
 * @param[out]  error_warp                予測残差を格納(ワーピングと平行移動選択された方)
 * @param[in]   target_corners            三角形を構成する3つの頂点
 * @param[in]   ref_corners               参照フレーム上の3つの頂点(使ってない)
 * @param[out]  flag                      ワーピングと平行移動を表すフラグ
 * @param[out]  add_corners               頂点を追加するためのバッファ(頂点追加はやってないので使ってない)
 * @param[in]   t                         何番目の三角パッチかを見たかった変数(デバッグ用)
 * @param[in]   residual_ref              対象フレームと参照フレームとの差分画像を入力(パッチごとの差分画像を見たかったため)
 * @param[out]  in_triangle_size          三角形の面積(正確には三角形の内部と判定された画素数)
 * @param[in]   erase_th_global           消したいパッチの残差の閾値をもらう(パッチ全体のMSEの平均と標準偏差の和が入る)
 * @return      std::vector<cv::Point2i>  推定した動きベクトルをクォーターペル精度で量子化して1bit左シフトしたものを返す
 */
std::vector<cv::Point2i> Gauss_Newton2(const cv::Mat& prev_color,const cv::Mat& current_color,const cv::Mat& intra, std::vector<cv::Mat>& predict_buf,cv::Mat &predict_warp,cv::Mat &predict_para,
                                       double &error_warp, Point3Vec target_corners, Point3Vec& ref_corners, bool *flag,int t,const cv::Mat& residual_ref ,int &in_triangle_size ,double erase_th_global){

    void block_matching_full(const cv::Mat& prev, const cv::Mat& current, double &error, cv::Point2f &mv, Point3Vec tr);
    std::vector<std::vector<cv::Mat>> f_,g_;
    std::vector<cv::Mat>f_0,g_0;
    std::vector<cv::Mat>f_1,g_1,f_2,g_2;
    cv::Mat f0x1,f0x2,f0x4,f0x8,g0x1,g0x2,g0x4,g0x8;
    f0x1 = current_color;
    f0x2 = half(f0x1,2);
    f0x4 = half(f0x2,2);
    f0x8 = half(f0x4,2);
    f_0.emplace_back(f0x8);
    f_0.emplace_back(f0x4);
    f_0.emplace_back(f0x2);
    f_0.emplace_back(f0x1);

    g0x1 = prev_color;
    g0x2 = half(g0x1,2);
    g0x4 = half(g0x2,2);
    g0x8 = half(g0x4,2);
    g_0.emplace_back(g0x8);
    g_0.emplace_back(g0x4);
    g_0.emplace_back(g0x2);
    g_0.emplace_back(intra);

    cv::Mat fx1,fx2,fx4,fx8,gx1,gx2,gx4,gx8;
    fx1 = current_color;
    fx2 = half(fx1,2);
    fx4 = half(fx2,1);
    fx8 = half(fx4,1);
    f_1.emplace_back(fx8);
    f_1.emplace_back(fx4);
    f_1.emplace_back(fx2);
    f_1.emplace_back(fx1);

    gx1 = prev_color;
    gx2 = half(gx1,2);
    gx4 = half(gx2,1);
    gx8 = half(gx4,1);
    g_1.emplace_back(gx8);
    g_1.emplace_back(gx4);
    g_1.emplace_back(gx2);
    g_1.emplace_back(intra);

    cv::Mat f2x1,f2x2,f2x4,f2x8,g2x1,g2x2,g2x4,g2x8;
    f2x1 = current_color;
    f2x2 = half(f2x1,0);
    f2x4 = half(f2x2,0);
    f2x8 = half(f2x4,0);
    g2x1 = prev_color;
    g2x2 = half(g2x1,0);
    g2x4 = half(g2x2,0);
    g2x8 = half(g2x4,0);
    f_2.emplace_back(f2x8);
    f_2.emplace_back(f2x4);
    f_2.emplace_back(f2x2);
    f_2.emplace_back(f2x1);
    g_2.emplace_back(g2x8);
    g_2.emplace_back(g2x4);
    g_2.emplace_back(g2x2);
    g_2.emplace_back(intra);
    f_.emplace_back(f_0);
    f_.emplace_back(f_1);
    f_.emplace_back(f_2);
    g_.emplace_back(g_0);
    g_.emplace_back(g_1);
    g_.emplace_back(g_2);

    cv::Point2f v0(0.0, 0.0), v1(0.0, 0.0), v2(0.0, 0.0);
    const int dim = 6;
    double f, g, g1, g2, ek_tmp, ek, delta_ek[dim],g_para;
    cv::Mat gg = cv::Mat_<double>(dim, dim);
    cv::Mat gg_para = cv::Mat_<double>(2,2);
    cv::Mat B = cv::Mat_<double>(dim, 1);
    cv::Mat B_para = cv::Mat_<double>(2,1);
    cv::Mat delta_uv = cv::Mat_<double>(dim, 1);
    cv::Mat delta_uv_para = cv::Mat_<double>(2,1);
    cv::Mat residual = cv::Mat::zeros(current_color.rows,current_color.cols,CV_8UC1);
    double Residual_Error = 10000, prev_error = 10000, current_error = 0, current_error_2 = 0;
    double prev_PSNR = 0,prev_PSNR_para = 0;
    double delta_g[dim] = {0},delta_g_para[2] = {0};
    unsigned char *warp;
    unsigned char *para;
    const double delta = 0.1;
    const double th = 0.5;
    double MSE,MSE_para,Error,Error_para,Error_min = 1E06,Error_para_min = 1E06,MSE_min = 1E06,MSE_para_min = 1E06;
    double PSNR,PSNR_max = 0,PSNR_para,PSNR_para_max = 0;
    int img_ip(unsigned char **img, int xs, int ys, double x, double y, int mode);
    double w(double x);

    float delta_x, delta_y;
    std::vector<cv::Point2f> triangle, triangle_later, triangle_delta,triangle_later_para;
    std::vector<cv::Point2f> in_triangle_pixels, in_triangle_pixels_later;
    cv::Point2f xp(0.0, 0.0), p0, p1, p2, p;
    std::vector<cv::Point2f> corners, corners_org;
    std::vector<cv::Point2i> mv,mv_diff;
    std::vector<cv::Point2f> v,v_max;
    std::vector<std::pair<std::vector<cv::Point2f>,double>> v_stack;
    std::vector<std::pair<cv::Point2f,double>> v_stack_para;
    std::pair<std::vector<cv::Point2f>,double> v_pair;
    std::pair<cv::Point2f,double> v_pair_para;

    cv::Point2f v_para,v_para_max;
    Point3Vec target_corner = target_corners;
    v.clear();
    v.emplace_back(v0);
    v.emplace_back(v1);
    v.emplace_back(v2);
    v_max.clear();
    v_max.emplace_back(v0);
    v_max.emplace_back(v1);
    v_max.emplace_back(v2);
    std::vector<cv::Point2f> v_prev(v);
    p0 = target_corners.p1;
    p1 = target_corners.p2;
    p2 = target_corners.p3;
    double sx = std::min({(int) p0.x, (int) p1.x, (int) p2.x});
    double lx = std::max({(int) p0.x, (int) p1.x, (int) p2.x});
    double sy = std::min({(int) p0.y, (int) p1.y, (int) p2.y});
    double ly = std::max({(int) p0.y, (int) p1.y, (int) p2.y});
    double Nx =0,Ny = 0,NT = 16;
    Nx = lx - sx;
    Ny = ly - sy;
    int kx,ky;
    kx = std::min({(int)floor(log2(Nx/NT)),3});
    ky = std::min({(int)floor(log2(Ny/NT)),3});


    if(target_corners.p1.x == f_[0][3].cols - 1)target_corner.p1.x = f_[0][3].cols / 8 -1;
    else target_corner.p1.x /= 8;
    if(target_corners.p1.y == f_[0][3].rows - 1)target_corner.p1.y = f_[0][3].rows / 8 -1;
    else target_corner.p1.y /= 8;
    if(target_corners.p2.x == f_[0][3].cols - 1)target_corner.p2.x = f_[0][3].cols / 8 -1;
    else target_corner.p2.x /= 8;
    if(target_corners.p2.y == f_[0][3].rows - 1)target_corner.p2.y = f_[0][3].rows / 8 -1;
    else target_corner.p2.y /= 8;
    if(target_corners.p3.x == f_[0][3].cols - 1)target_corner.p3.x = f_[0][3].cols / 8 -1;
    else target_corner.p3.x /= 8;
    if(target_corners.p3.y == f_[0][3].rows - 1)target_corner.p3.y = f_[0][3].rows / 8 -1;
    else target_corner.p3.y /= 8;

    for(int blare = 0;blare < 2;blare++) {
        cv::Point2f v0(0.0, 0.0), v1(0.0, 0.0), v2(0.0, 0.0);
        v.clear();
        v.emplace_back(v0);
        v.emplace_back(v1);
        v.emplace_back(v2);
        v_para.x = 0;
        v_para.y = 0;
        p0 = target_corners.p1;
        p1 = target_corners.p2;
        p2 = target_corners.p3;
        for (int z = 0; z < 4; z++) {
            double scale = pow(2, 3-z),scale_x = scale,scale_y = scale;
            cv::Mat f_img = f_[blare][z].clone();//対照画像
            cv::Mat g_img = g_[blare][z].clone();//参照画像
            f_img = mv_filter(f_img,2);
            g_img = mv_filter(g_img,2);
            const int expand = 500;
            unsigned char **f_expand,**f_org_expand;
            unsigned char **g_expand,**g_org_expand;
            f_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (f_img.cols + expand * 2));
            f_expand += expand;
            f_org_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (f_img.cols + expand * 2));
            f_org_expand += expand;
            for (int j = -expand; j < f_img.cols + expand; j++) {
                f_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (f_img.rows + expand * 2));
                f_expand[j] += expand;
                f_org_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (f_img.rows + expand * 2));
                f_org_expand[j] += expand;
            }
            g_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (f_img.cols + expand * 2));
            g_expand += expand;
            g_org_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (f_img.cols + expand * 2));
            g_org_expand += expand;
            for (int j = -expand; j < g_img.cols + expand; j++) {
                if ((g_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (f_img.rows + expand * 2))) == NULL) {
                }
                g_expand[j] += expand;
                if ((g_org_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (f_img.rows + expand * 2))) == NULL) {
                }
                g_org_expand[j] += expand;
            }
            for (int j = -expand; j < f_img.rows + expand; j++) {
                for (int i = -expand; i < f_img.cols + expand; i++) {
                    if (j >= 0 && j < f_img.rows && i >= 0 && i < f_img.cols) {
                        f_expand[i][j] = M(f_img, i, j);
                        g_expand[i][j] = M(g_img, i, j);
                        f_org_expand[i][j] = M(f_[blare][z], i, j);
                        g_org_expand[i][j] = M(g_[blare][z], i, j);
                    } else {
                        f_expand[i][j] = 0;
                        g_expand[i][j] = 0;
                        f_org_expand[i][j] = 0;
                        g_org_expand[i][j] = 0;
                    }
                }
            }
            int k = 2;
            for (int j = 0; j < f_img.rows; j++) {
                for (int i = 1; i <= k; i++) {
                    f_expand[-i][j] = f_expand[i][j];
                    f_expand[f_img.cols - 1 + i][j] = f_expand[f_img.cols - 1 - i][j];
                    g_expand[-i][j] = g_expand[i][j];
                    g_expand[f_img.cols - 1 + i][j] = g_expand[f_img.cols - 1 - i][j];
                    f_org_expand[-i][j] = f_org_expand[i][j];
                    f_org_expand[f_img.cols - 1 + i][j] = f_org_expand[f_img.cols - 1 - i][j];
                    g_org_expand[-i][j] = g_org_expand[i][j];
                    g_org_expand[f_img.cols - 1 + i][j] = g_org_expand[f_img.cols - 1 - i][j];
                }
            }
            for (int i = -k; i < f_img.cols + k; i++) {
                for (int j = 1; j <= k; j++) {
                    f_expand[i][-j] = f_expand[i][j];
                    f_expand[i][f_img.rows - 1 + j] = f_expand[i][f_img.rows - 1 - j];
                    g_expand[i][-j] = g_expand[i][j];
                    g_expand[i][f_img.rows - 1 + j] = g_expand[i][f_img.rows - 1 - j];
                    f_org_expand[i][-j] = f_org_expand[i][j];
                    f_org_expand[i][f_img.rows - 1 + j] = f_org_expand[i][f_img.rows - 1 - j];
                    g_org_expand[i][-j] = g_org_expand[i][j];
                    g_org_expand[i][f_img.rows - 1 + j] = g_org_expand[i][f_img.rows - 1 - j];
                }
            }
            p0.x = target_corners.p1.x / scale_x;
            p0.y = target_corners.p1.y / scale_y;
            p1.x = target_corners.p2.x / scale_x;
            p1.y = target_corners.p2.y / scale_y;
            p2.x = target_corners.p3.x / scale_x;
            p2.y = target_corners.p3.y / scale_y;

            if (target_corners.p1.x == f_[0][3].cols - 1)p0.x = f_[0][z].cols - 1;
            if (target_corners.p1.y == f_[0][3].rows - 1)p0.y = f_[0][z].rows - 1;
            if (target_corners.p2.x == f_[0][3].cols - 1)p1.x = f_[0][z].cols - 1;
            if (target_corners.p2.y == f_[0][3].rows - 1)p1.y = f_[0][z].rows - 1;
            if (target_corners.p3.x == f_[0][3].cols - 1)p2.x = f_[0][z].cols - 1;
            if (target_corners.p3.y == f_[0][3].rows - 1)p2.y = f_[0][z].rows - 1;
            cv::Point2f a_,b_;
            a_.x = p2.x - p0.x;
            a_.y = p2.y - p0.y;
            b_.x = p1.x - p0.x;
            b_.y = p1.y - p0.y;
            if(fabs(a_.x * b_.y - a_.y * b_.x) <= 0)break;
            triangle.clear();
            triangle.emplace_back(p0);
            triangle.emplace_back(p1);
            triangle.emplace_back(p2);


            bool length2_flag = false;
            if(z != 0) {
                double length = 2;
                while ( (p0.x + v[0].x * length < 0 && p0.x + v[0].x * length > f_img.cols - 1) &&
                        (p1.x + v[1].x * length < 0 && p1.x + v[1].x * length > f_img.cols - 1) &&
                        (p2.x + v[2].x * length < 0 && p2.x + v[2].x * length > f_img.cols - 1) &&
                        (p0.y + v[0].y * length < 0 && p0.y + v[0].y * length > f_img.rows - 1) &&
                        (p1.y + v[1].y * length < 0 && p1.y + v[1].y * length > f_img.rows - 1) &&
                        (p2.y + v[2].y * length < 0 && p2.y + v[2].y * length > f_img.rows - 1) ) {
                    if(length <= 1)break;
                    length -= 0.1;
                }
                for (int s = 0; s < 3; s++) v[s] *= length;

                length = 2;
                while ( (p0.x + v_para.x * length < 0 && p0.x + v_para.x * length > f_img.cols - 1) &&
                        (p1.x + v_para.x * length < 0 && p1.x + v_para.x * length > f_img.cols - 1) &&
                        (p2.x + v_para.x * length < 0 && p2.x + v_para.x * length > f_img.cols - 1) &&
                        (p0.y + v_para.y * length < 0 && p0.y + v_para.y * length > f_img.rows - 1) &&
                        (p1.y + v_para.y * length < 0 && p1.y + v_para.y * length > f_img.rows - 1) &&
                        (p2.y + v_para.y * length < 0 && p2.y + v_para.y * length > f_img.rows - 1) ) {
                    if(length <= 1)break;
                    length -= 0.1;
                }
                v_para *= length;
            }


            v_stack.clear();
            v_stack_para.clear();

            Point3Vec triangleVec = Point3Vec(p0, p1, p2);
            double sx = std::min({(int) p0.x, (int) p1.x, (int) p2.x});
            double lx = std::max({(int) p0.x, (int) p1.x, (int) p2.x});
            double sy = std::min({(int) p0.y, (int) p1.y, (int) p2.y});
            double ly = std::max({(int) p0.y, (int) p1.y, (int) p2.y});
            in_triangle_pixels.clear();
            for (int j = (int) (round(sy) - 1); j <= round(ly) + 1; j++) {
                for (int i = (int) (round(sx) - 1); i <= round(lx) + 1; i++) {
                    xp.x = (float) i;
                    xp.y = (float) j;
                    if (isInTriangle(triangleVec, xp) == 1) {
                        in_triangle_pixels.emplace_back(xp);
                    }
                }
            }
            if(z == 5){
                double block_error;
                cv::Point2f block_mv;
                block_matching_full(g_img,f_img,block_error,block_mv,target_corner);
                v[0] = v[1] = v[2] = v_para = block_mv;
            }
            else {
                for (int q = 0; q < 11; q++) {

                    //while(Residual_Error >= th) {
                    if (q == 10 && z == 3) {
                        for (int i = 0; i < 3; i++) {
                            v[i].x = 0;
                            v[i].y = 0;
                        }
                        v_para.x = 0;
                        v_para.y = 0;
                    }
                    p0.x = target_corners.p1.x / scale_x;
                    p0.y = target_corners.p1.y / scale_y;
                    p1.x = target_corners.p2.x / scale_x;
                    p1.y = target_corners.p2.y / scale_y;
                    p2.x = target_corners.p3.x / scale_x;
                    p2.y = target_corners.p3.y / scale_y;
                    if (target_corners.p1.x == f_[0][3].cols - 1)p0.x = f_[0][z].cols - 1;
                    if (target_corners.p1.y == f_[0][3].rows - 1)p0.y = f_[0][z].rows - 1;
                    if (target_corners.p2.x == f_[0][3].cols - 1)p1.x = f_[0][z].cols - 1;
                    if (target_corners.p2.y == f_[0][3].rows - 1)p1.y = f_[0][z].rows - 1;
                    if (target_corners.p3.x == f_[0][3].cols - 1)p2.x = f_[0][z].cols - 1;
                    if (target_corners.p3.y == f_[0][3].rows - 1)p2.y = f_[0][z].rows - 1;

                    cv::Point2f a, b, X, c, d;
                    cv::Point2f a_later, b_later, X_later, a_later_para, b_later_para, X_later_para;
                    cv::Point2f a_later_delta, b_later_delta, X_later_delta;
                    std::vector<cv::Point2f> pp, pp_delta, pp_para;
                    pp.clear();
                    pp.emplace_back(p0);
                    pp.emplace_back(p1);
                    pp.emplace_back(p2);
                    pp_para.clear();
                    pp_para.emplace_back(p0);
                    pp_para.emplace_back(p1);
                    pp_para.emplace_back(p2);
                    pp_delta.clear();
                    pp_delta.emplace_back(p0);
                    pp_delta.emplace_back(p1);
                    pp_delta.emplace_back(p2);
                    int x0, y0, x0_later, y0_later, x0_delta, y0_delta, x0_later_para, y0_later_para;
                    double d_x, d_y, d_x_later, d_y_later, d_x_delta, d_y_delta, d_x_later_para, d_y_later_para;
                    double alpha, beta, det, tri_S0;
                    double g_x, g_y, g_x_para, g_y_para;
                    double S0,S1,S_[6];
                    double myu = 10;
                    a.x = p2.x - p0.x;
                    a.y = p2.y - p0.y;
                    b.x = p1.x - p0.x;
                    b.y = p1.y - p0.y;
                    det = a.x * b.y - a.y * b.x;
                    c = v[2] - v[0];
                    d = v[1] - v[0];
                    S0 = 0.5*fabs(det);
                    S1 = 0.5*fabs((b.x + d.x)*(a.y + c.y) - (a.x + c.x)*(b.y + d.y));
                    S_[0] = -0.5*(a.y + c.y - b.y - d.y);
                    S_[1] = -0.5*(b.x + d.x - a.x - c.x);
                    S_[2] = 0.5*(a.y + c.y);
                    S_[3] = -0.5*(a.x + c.x);
                    S_[4] = -0.5*(b.y + d.y);
                    S_[5] = 0.5*(b.x + d.x);
                    tri_S0 = fabs(det);

                    MSE = 0;
                    MSE_para = 0;
                    current_error = 0;
                    current_error_2 = 0;
                    ek = 0;
                    for (int k = 0; k < dim; k++) {
                        for (int j = 0; j < dim; j++) {
                            gg.at<double>(k, j) = 0;
                        }
                        B.at<double>(k, 0) = 0;
                        delta_ek[k] = 0;
                    }
                    for (int k = 0; k < 2; k++) {
                        for (int j = 0; j < 2; j++) {
                            gg_para.at<double>(k, j) = 0;
                        }
                        B_para.at<double>(k, 0) = 0;
                    }

                    for (int m = 0; m < (int) in_triangle_pixels.size(); m++) {
                        X.x = in_triangle_pixels[m].x - p0.x;
                        X.y = in_triangle_pixels[m].y - p0.y;

                        alpha = (X.x * b.y - X.y * b.x) / det;
                        beta = (a.x * X.y - a.y * X.x) / det;

                        X.x += p0.x;
                        X.y += p0.y;
                        x0 = (int) floor(X.x);
                        y0 = (int) floor(X.y);
                        d_x = X.x - x0;
                        d_y = X.y - y0;

                        ek_tmp = 0;
                        for (int i = 0; i < 6; i++) {
                            pp[0] = p0 + v[0];
                            pp[1] = p1 + v[1];
                            pp[2] = p2 + v[2];
                            triangle_later.clear();
                            triangle_later.emplace_back(pp[0]);
                            triangle_later.emplace_back(pp[1]);
                            triangle_later.emplace_back(pp[2]);
                            pp_para[0] = p0 + v_para;
                            pp_para[1] = p1 + v_para;
                            pp_para[2] = p2 + v_para;
                            triangle_later_para.clear();
                            triangle_later_para.emplace_back(pp_para[0]);
                            triangle_later_para.emplace_back(pp_para[1]);
                            triangle_later_para.emplace_back(pp_para[2]);
                            a_later.x = triangle_later[2].x - triangle_later[0].x;
                            a_later.y = triangle_later[2].y - triangle_later[0].y;
                            a_later_para = triangle_later_para[2] - triangle_later_para[0];
                            b_later.x = triangle_later[1].x - triangle_later[0].x;
                            b_later.y = triangle_later[1].y - triangle_later[0].y;
                            b_later_para = triangle_later_para[1] - triangle_later_para[0];
                            X_later.x = alpha * a_later.x + beta * b_later.x + triangle_later[0].x;
                            X_later.y = alpha * a_later.y + beta * b_later.y + triangle_later[0].y;
                            X_later_para = alpha * a_later_para + beta * b_later_para + triangle_later_para[0];
                            switch (i) {
                                case 0:
                                    pp[0].x += delta;
                                    delta_x = 1 - alpha - beta;
                                    delta_y = 0;
                                    break;
                                case 1:
                                    pp[0].y += delta;
                                    delta_x = 0;
                                    delta_y = 1 - alpha - beta;
                                    break;
                                case 2:
                                    pp[1].x += delta;
                                    delta_x = beta;
                                    delta_y = 0;
                                    break;
                                case 3:
                                    pp[1].y += delta;
                                    delta_x = 0;
                                    delta_y = beta;
                                    break;
                                case 4:
                                    pp[2].x += delta;
                                    delta_x = alpha;
                                    delta_y = 0;
                                    break;
                                case 5:
                                    pp[2].y += delta;
                                    delta_x = 0;
                                    delta_y = alpha;
                                    break;
                            }
                            triangle_delta.clear();
                            triangle_delta.emplace_back(pp[0]);
                            triangle_delta.emplace_back(pp[1]);
                            triangle_delta.emplace_back(pp[2]);
                            a_later_delta.x = triangle_delta[2].x - triangle_delta[0].x;
                            a_later_delta.y = triangle_delta[2].y - triangle_delta[0].y;
                            b_later_delta.x = triangle_delta[1].x - triangle_delta[0].x;
                            b_later_delta.y = triangle_delta[1].y - triangle_delta[0].y;
                            X_later_delta.x = alpha * a_later_delta.x + beta * b_later_delta.x + triangle_delta[0].x;
                            X_later_delta.y = alpha * a_later_delta.y + beta * b_later_delta.y + triangle_delta[0].y;
                            x0_delta = (int) floor(X_later_delta.x);
                            y0_delta = (int) floor(X_later_delta.y);
                            d_x_delta = X_later_delta.x - x0_delta;
                            d_y_delta = X_later_delta.y - y0_delta;
                            g_x = g_expand[(int) X_later.x + 1][(int) X_later.y] -
                                  g_expand[(int) X_later.x][(int) X_later.y];
                            g_y = g_expand[(int) X_later.x][(int) X_later.y + 1] -
                                  g_expand[(int) X_later.x][(int) X_later.y];
                            g_x_para = g_expand[(int) X_later_para.x + 1][(int) X_later_para.y] -
                                       g_expand[(int) X_later_para.x][(int) X_later_para.y];
                            g_y_para = g_expand[(int) X_later_para.x][(int) X_later_para.y + 1] -
                                       g_expand[(int) X_later_para.x][(int) X_later_para.y];
                            if (i % 2 == 0) {
                                ek_tmp += g_x * delta_x * delta_uv.at<double>(i, 0);
                            } else {
                                ek_tmp += g_y * delta_y * delta_uv.at<double>(i, 0);
                            }
                            delta_g[i] = g_x * delta_x + g_y * delta_y;
                        }
                        delta_g_para[0] = g_x_para;
                        delta_g_para[1] = g_y_para;


                        x0_later = (int) floor(X_later.x);
                        y0_later = (int) floor(X_later.y);
                        x0_later_para = (int) floor(X_later_para.x);
                        y0_later_para = (int) floor(X_later_para.y);
                        d_x_later = X_later.x - x0_later;
                        d_y_later = X_later.y - y0_later;
                        d_x_later_para = X_later_para.x - x0_later_para;
                        d_y_later_para = X_later_para.y - y0_later_para;
                        if(x0_later < 0 || x0_later > f_img.cols - 1 || y0_later < 0 || y0_later > f_img.rows -1) {
                            std::cout << "x0_later = " << x0_later << " y0_later = " << y0_later << std::endl;
                        }
                        if(x0_later_para < 0 || x0_later_para > f_img.cols - 1 || y0_later_para < 0 || y0_later_para > f_img.rows -1) {
                            std::cout << "x0_later_para = " << x0_later_para << " y0_later_para = " << y0_later_para << std::endl;
                        }

                        f = f_expand[x0][y0] * (1 - d_x) * (1 - d_y) + f_expand[x0 + 1][y0] * d_x * (1 - d_y) +
                            f_expand[x0][y0 + 1] * (1 - d_x) * d_y + f_expand[x0 + 1][y0 + 1] * d_x * d_y;

                        double f_org = f_org_expand[x0][y0] * (1 - d_x) * (1 - d_y) + f_org_expand[x0 + 1][y0] * d_x * (1 - d_y) +
                                       f_org_expand[x0][y0 + 1] * (1 - d_x) * d_y + f_org_expand[x0 + 1][y0 + 1] * d_x * d_y;

                        g1 = g_expand[x0_later][y0_later] * (1 - d_x_later) * (1 - d_y_later) +
                             g_expand[x0_later + 1][y0_later] * d_x_later * (1 - d_y_later) +
                             g_expand[x0_later][y0_later + 1] * (1 - d_x_later) * d_y_later +
                             g_expand[x0_later + 1][y0_later + 1] * d_x_later * d_y_later;

                        g = g_expand[x0_later][y0_later] * (1 - d_x_later) * (1 - d_y_later) +
                            g_expand[x0_later + 1][y0_later] * d_x_later * (1 - d_y_later) +
                            g_expand[x0_later][y0_later + 1] * (1 - d_x_later) * d_y_later +
                            g_expand[x0_later + 1][y0_later + 1] * d_x_later * d_y_later;

                        double g_org = g_org_expand[x0_later][y0_later] * (1 - d_x_later) * (1 - d_y_later) +
                               g_org_expand[x0_later + 1][y0_later] * d_x_later * (1 - d_y_later) +
                               g_org_expand[x0_later][y0_later + 1] * (1 - d_x_later) * d_y_later +
                               g_org_expand[x0_later + 1][y0_later + 1] * d_x_later * d_y_later;

                        g_para = g_expand[x0_later_para    ][y0_later_para    ] * (1 - d_x_later_para) * (1 - d_y_later_para) +
                                 g_expand[x0_later_para + 1][y0_later_para    ] * d_x_later_para       * (1 - d_y_later_para) +
                                 g_expand[x0_later_para    ][y0_later_para + 1] * (1 - d_x_later_para) * d_y_later_para +
                                 g_expand[x0_later_para + 1][y0_later_para + 1] * d_x_later_para       * d_y_later_para;

                        double g_para_org = g_org_expand[x0_later_para    ][y0_later_para    ] * (1 - d_x_later_para) * (1 - d_y_later_para) +
                                            g_org_expand[x0_later_para + 1][y0_later_para    ] * d_x_later_para       * (1 - d_y_later_para) +
                                            g_org_expand[x0_later_para    ][y0_later_para + 1] * (1 - d_x_later_para) * d_y_later_para +
                                            g_org_expand[x0_later_para + 1][y0_later_para + 1] * d_x_later_para       * d_y_later_para;

                        g2 = g + delta_g[0] * delta_uv.at<double>(0, 0) + delta_g[1] * delta_uv.at<double>(1, 0) +
                             delta_g[2] * delta_uv.at<double>(2, 0) +
                             delta_g[3] * delta_uv.at<double>(3, 0) + delta_g[4] * delta_uv.at<double>(4, 0) +
                             delta_g[5] * delta_uv.at<double>(5, 0);
                        for (int t = 0; t < dim; t++) {
                            delta_ek[t] += (f - g2) * delta_g[t];
                        }
                        for (int k = 0; k < dim; k++) {
                            for (int j = 0; j < dim; j++) {
                                gg.at<double>(k, j) += delta_g[k] * delta_g[j];
                            }
                            B.at<double>(k, 0) += (f - g) * delta_g[k];
                        }

                        for (int k = 0; k < 2; k++) {
                            for (int j = 0; j < 2; j++) {
                                gg_para.at<double>(k, j) += delta_g_para[k] * delta_g_para[j];
                            }
                            B_para.at<double>(k, 0) += (f - g_para) * delta_g_para[k];
                        }

                        current_error += (f - g1) * (f - g1);
                        current_error_2 += (f - g2) * (f - g2);

                        ek += (f - g - ek_tmp) * (f - g - ek_tmp);
                        MSE += (f_org - g_org) * (f_org - g_org);
                        MSE_para += (f_org - g_para_org) * (f_org - g_para_org);

                        // TODO: 予測残差はホンモノの画像で撮ったほうがいいのでは？
                    }

                    for (int k = 0; k < dim; k++) {
                        for (int j = 0; j < dim; j++) {
                            gg.at<double>(k, j) += myu*((S1*S1*S1 - S0*S0*S1)/(S0*S0*S0*S0)*S_[k]);
                        }
                    }
                    double tri_S1 = fabs(a_later.x * b_later.y - b_later.x * a_later.y);
                    double tri_per = (tri_S1 / tri_S0) * 100;
                    for (int t = 0; t < dim; t++) {
                        delta_ek[t] *= -2;
                    }
                    Error = MSE;
                    Error_para = MSE_para;
                    MSE = (in_triangle_pixels.size() == 0 ? MSE : MSE / in_triangle_pixels.size());
                    MSE_para = (in_triangle_pixels.size() == 0 ? MSE_para : MSE_para / in_triangle_pixels.size());
                    in_triangle_size = in_triangle_pixels.size();
                    PSNR = 10 * log10((255 * 255) / MSE);
                    PSNR_para = 10 * log10((255 * 255) / MSE_para);
                    cv::solve(gg, B, delta_uv);
                    cv::solve(gg_para, B_para, delta_uv_para);

                    v_pair.first = v;
                    v_pair.second = Error;
                    v_stack.emplace_back(v_pair);
                    v_pair_para.first = v_para;
                    v_pair_para.second = Error_para;
                    v_stack_para.emplace_back(v_pair_para);
                    for (int k = 0; k < 6; k++) {
                        if (k % 2 == 0) {
                            if ((0 <= triangle[(int) (k / 2)].x + v[(int) (k / 2)].x + delta_uv.at<double>(k, 0)) &&
                                (f_[0][z].cols - 1 >=
                                 triangle[(int) (k / 2)].x + v[(int) (k / 2)].x + delta_uv.at<double>(k, 0))) {
                                v[(int) (k / 2)].x = v[(int) (k / 2)].x + delta_uv.at<double>(k, 0);
                            }
                        } else {
                            if ((0 <= triangle[(int) (k / 2)].y + v[(int) (k / 2)].y + delta_uv.at<double>(k, 0)) &&
                                (f_[0][z].rows - 1 >=
                                 triangle[(int) (k / 2)].y + v[(int) (k / 2)].y + delta_uv.at<double>(k, 0))) {
                                v[(int) (k / 2)].y = v[(int) (k / 2)].y + delta_uv.at<double>(k, 0);
                            }
                        }
                    }
                    for (int k = 0; k < 2; k++) {
                        if (k % 2 == 0) {
                            if ((0 <= triangle[0].x + v_para.x + delta_uv_para.at<double>(k, 0)) &&
                                (f_[0][z].cols - 1 >= triangle[0].x + v_para.x + delta_uv_para.at<double>(k, 0)) &&
                                 (0 <= triangle[1].x + v_para.x + delta_uv_para.at<double>(k, 0)) &&
                                (f_[0][z].cols - 1 >=triangle[1].x + v_para.x + delta_uv_para.at<double>(k, 0)) &&
                                 (0 <= triangle[2].x + v_para.x + delta_uv_para.at<double>(k, 0)) &&
                                (f_[0][z].cols - 1 >= triangle[2].x + v_para.x + delta_uv_para.at<double>(k, 0))) {
                                v_para.x = v_para.x + delta_uv_para.at<double>(k, 0);
                            }
                        } else {
                            if ((0 <= triangle[0].y + v_para.y + delta_uv_para.at<double>(k, 0)) &&
                                (f_[0][z].rows - 1 >=
                                 triangle[0].y + v_para.y + delta_uv_para.at<double>(k, 0)) &&
                                 (0 <= triangle[1].y + v_para.y + delta_uv_para.at<double>(k, 0)) &&
                                (f_[0][z].rows - 1 >=
                                 triangle[1].y + v_para.y + delta_uv_para.at<double>(k, 0)) &&
                                 (0 <= triangle[2].y + v_para.y + delta_uv_para.at<double>(k, 0)) &&
                                (f_[0][z].rows - 1 >=
                                 triangle[2].y + v_para.y + delta_uv_para.at<double>(k, 0))){
                                v_para.y = v_para.y + delta_uv_para.at<double>(k, 0);
                            }
                        }
                    }
                    Residual_Error = fabs(prev_error - current_error);
                    prev_error = current_error;
                    prev_PSNR = PSNR;
                    v_prev = v;

                }


                std::sort(v_stack.begin(), v_stack.end(), [](std::pair<std::vector<cv::Point2f>,double> a, std::pair<std::vector<cv::Point2f>,double> b){
                    return a.second < b.second;
                });

                std::sort(v_stack_para.begin(), v_stack_para.end(), [](std::pair<cv::Point2f,double> a, std::pair<cv::Point2f,double> b){
                    return a.second < b.second;
                });

                v = v_stack[0].first;
                Error = v_stack[0].second;
                v_para = v_stack_para[0].first;
                Error_para = v_stack_para[0].second;
                MSE = Error / (double)in_triangle_pixels.size();
                MSE_para = Error_para / (double)in_triangle_pixels.size();
                PSNR = 10.0 * log10((255 * 255) / MSE);
                PSNR_para = 10.0 * log10((255 * 255) / MSE_para);
            }

            bool flag_blare = false,flag_blare_para = false;
            if(z == 3) {
                if(PSNR_para >= PSNR_para_max){
                    PSNR_para_max = PSNR_para;
                    Error_para_min = Error_para;
                    v_para_max = v_para;
                    flag_blare_para = true;
                    MSE_para_min = MSE_para;
                }
                if (PSNR >= PSNR_max) {
                    PSNR_max = PSNR;
                    Error_min = Error;
                    v_max = v;
                    flag_blare = true;
                    MSE_min = MSE;
                }
//                *flag = true;
                if (fabs(PSNR_max - PSNR_para_max) <= th || PSNR_para_max > PSNR_max) {
                    *flag = true;
                } else{
                    *flag = false;
                }
                *flag = true;
            }

            double alpha, beta, det;

            cv::Point2f X, a, b;
            cv::Point2f X_later,X_later_para, a_later,a_later_para, b_later,b_later_para;
            cv::Point2f pp0,pp0_para, pp1,pp1_para, pp2,pp2_para;

            std::vector<cv::Point2f> pp,pp_para, mv_tmp;

            pp0.x = triangle[0].x + v[0].x;
            pp0.y = triangle[0].y + v[0].y;
            pp1.x = triangle[1].x + v[1].x;
            pp1.y = triangle[1].y + v[1].y;
            pp2.x = triangle[2].x + v[2].x;
            pp2.y = triangle[2].y + v[2].y;
            if(z == 3) {
                pp0_para.x = triangle[0].x + v_para.x;
                pp0_para.y = triangle[0].y + v_para.y;
                pp1_para.x = triangle[1].x + v_para.x;
                pp1_para.y = triangle[1].y + v_para.y;
                pp2_para.x = triangle[2].x + v_para.x;
                pp2_para.y = triangle[2].y + v_para.y;
            }
            double Quant = 4;
            std::vector<cv::Point2f> mv2, mv3,mv4,mv_diff_tmp;
            cv::Point2f ave_v;
            pp.clear();
            pp.emplace_back(pp0);
            pp.emplace_back(pp1);
            pp.emplace_back(pp2);
            pp_para.clear();
            pp_para.emplace_back(pp0_para);
            pp_para.emplace_back(pp1_para);
            pp_para.emplace_back(pp2_para);
            if(z == 3) {
                ave_v.x = (v_max[0].x + v_max[1].x + v_max[2].x) / 3;
                ave_v.y = (v_max[0].y + v_max[1].y + v_max[2].y) / 3;
                ave_v.x = (int) floor(ave_v.x + 0.5);
                ave_v.y = (int) floor(ave_v.y + 0.5);
                v0.x = floor(ref_corners.p1.x / 2 + 0.5) - p0.x;
                v0.y = floor(ref_corners.p1.y / 2 + 0.5) - p0.y;
                v1.x = floor(ref_corners.p2.x / 2 + 0.5) - p1.x;
                v1.y = floor(ref_corners.p2.y / 2 + 0.5) - p1.y;
                v2.x = floor(ref_corners.p3.x / 2 + 0.5) - p2.x;
                v2.y = floor(ref_corners.p3.y / 2 + 0.5) - p2.y;
                mv2.clear();
                mv2.emplace_back(v0);
                mv2.emplace_back(v1);
                mv2.emplace_back(v2);
                mv_tmp.clear();
                mv_tmp.emplace_back(v[0]);
                mv_tmp.emplace_back(v[1]);
                mv_tmp.emplace_back(v[2]);
                if (*flag == false) {
                    mv_diff_tmp.clear();
                    mv_diff_tmp.emplace_back(v_max[0]);
                    mv_diff_tmp.emplace_back(v_max[1]);
                    mv_diff_tmp.emplace_back(v_max[2]);
                }
                mv.clear();
                mv.emplace_back(v[0]);
                mv.emplace_back(v[1]);
                mv.emplace_back(v[2]);
                mv3.clear();
                mv3.emplace_back(v_max[0]);
                mv3.emplace_back(v_max[1]);
                mv3.emplace_back(v_max[2]);
                mv4.clear();
                mv4.emplace_back(v_para_max);
                mv4.emplace_back(v_para_max);
                mv4.emplace_back(v_para_max);
                pp.clear();
                pp.emplace_back(pp0);
                pp.emplace_back(pp1);
                pp.emplace_back(pp2);
                pp_para.clear();
                pp_para.emplace_back(pp0_para);
                pp_para.emplace_back(pp1_para);
                pp_para.emplace_back(pp2_para);
                if (*flag == false) {
                    for (int j = 0; j < 3; j++) {
                        //mv3[j] -= ave_v;
                        int x0, y0;
                        double d_x, d_y;
                        if(mv3[j].x < 0) mv3[j].x -= 0.125;
                        else mv3[j].y += 0.125;
                        if(mv3[0].y < 0) mv3[0].y -= 0.125;
                        else mv3[0].y += 0.125;
                        mv3[j] *= Quant;
                        mv3[j].x = (int)(mv3[j].x);
                        mv3[j].y = (int)(mv3[j].y);
                        mv3[j] /= 2.0;
                        x0 = (int) mv3[j].x;
                        y0 = (int) mv3[j].y;
                        d_x = (mv3[j].x - x0) * 2.0;
                        d_y = (mv3[j].y - y0) * 2.0;
                        mv[j].x = x0;
                        mv[j].y = y0;
                        cv::Point2i dv(d_x, d_y);
                        mv.emplace_back(dv);
                    }
                }
                else{
                    mv4[0] = v_para_max;
                    int x0,y0;
                    double d_x,d_y;
                    if(mv4[0].x < 0) mv4[0].x -= 0.125;
                    else mv4[0].x += 0.125;
                    if(mv4[0].y < 0) mv4[0].y -= 0.125;
                    else mv4[0].y += 0.125;
                    mv4[0] *= Quant;
                    mv4[0].x = (int)(mv4[0].x);
                    mv4[0].y = (int)(mv4[0].y);
                    mv4[0] /= 2.0;
                    x0 = (int)mv4[0].x;
                    y0 = (int)mv4[0].y;
                    d_x = (mv4[0].x - x0)*2.0; // ハーフペル相当になっている
                    d_y = (mv4[0].y - y0)*2.0;
                    mv[0].x = x0; mv[0].y = y0;
                    mv[1].x = x0; mv[1].y = y0;
                    mv[2].x = x0; mv[2].y = y0;
                    cv::Point2i dv(d_x,d_y);
                    mv.emplace_back(dv); // 安全のために小数部を3つ追加
                    mv.emplace_back(dv);
                    mv.emplace_back(dv);
                }
                for (int j = 0; j < 3; j++) {
                    if (pp[j].x < 0) {
                        mv_tmp[j].x -= pp[j].x;
                        pp[j].x = 0;
                    } else if (pp[j].x > g_img.cols - 1) {
                        mv_tmp[j].x -= g_img.cols - 1 - pp[j].x;
                        pp[j].x = g_img.cols - 1;
                    }
                    if (pp[j].y < 0) {
                        mv_tmp[j].y -= pp[j].y;
                        pp[j].y = 0;
                    } else if (pp[j].y > g_img.rows - 1) {
                        mv_tmp[j].y -= g_img.rows - 1 - pp[j].y;
                        pp[j].y = g_img.rows - 1;
                    }
                }
                for (int j = 0; j < 3; j++) {
                    double tmp_x, tmp_y;
                    tmp_x = pp[j].x * Quant;
                    tmp_y = pp[j].y * Quant;
                    tmp_x = floor(tmp_x + 0.5);
                    tmp_y = floor(tmp_y + 0.5);
                    pp[j].x = tmp_x / Quant;
                    pp[j].y = tmp_y / Quant;
                    tmp_x = pp_para[j].x * Quant;
                    tmp_y = pp_para[j].y * Quant;
                    tmp_x = floor(tmp_x + 0.5);
                    tmp_y = floor(tmp_y + 0.5);
                    pp_para[j].x = tmp_x / Quant;
                    pp_para[j].y = tmp_y / Quant;
                    tmp_x = mv_tmp[j].x * Quant;
                    tmp_y = mv_tmp[j].y * Quant;
                }
            }
            a.x = triangle[2].x - triangle[0].x;
            a.y = triangle[2].y - triangle[0].y;
            b.x = triangle[1].x - triangle[0].x;
            b.y = triangle[1].y - triangle[0].y;
            det = a.x * b.y - a.y * b.x;

            if(blare == 0) {
                warp = (unsigned char *) malloc(sizeof(unsigned char) * (int) in_triangle_pixels.size());
                para = (unsigned char *) malloc(sizeof(unsigned char) * (int) in_triangle_pixels.size());
            }
            for (int i = 0; i < (int) in_triangle_pixels.size(); i++) {
                double d_x, d_y,d_x_para,d_y_para;
                int x0, y0,x0_para,y0_para;
                X.x = in_triangle_pixels[i].x - triangle[0].x;
                X.y = in_triangle_pixels[i].y - triangle[0].y;
                alpha = (X.x * b.y - X.y * b.x) / det;
                beta = (a.x * X.y - a.y * X.x) / det;
                X.x += triangle[0].x;
                X.y += triangle[0].y;
                //std::cout<<"X = " << X.x << ", " << X.y << std::endl;
                a_later.x = pp[2].x - pp[0].x;
                a_later.y = pp[2].y - pp[0].y;
                a_later_para = pp_para[2] - pp_para[0];
                b_later.x = pp[1].x - pp[0].x;
                b_later.y = pp[1].y - pp[0].y;
                b_later_para = pp_para[1] - pp_para[0];
                X_later.x = alpha * a_later.x + beta * b_later.x + pp[0].x;
                X_later.y = alpha * a_later.y + beta * b_later.y + pp[0].y;
                X_later_para = alpha * a_later_para + beta * b_later_para + pp_para[0];

                if (X_later.x >= g_img.cols - 1) {
                    X_later.x = g_img.cols - 1.001;
                }
                if (X_later.y >= g_img.rows - 1) {
                    X_later.y = g_img.rows - 1.001;
                }

                if (X_later.x < 0) {
                    X_later.x = 0;
                }
                if (X_later.y < 0) {
                    X_later.y = 0;
                }
                if (X_later_para.x >= g_img.cols - 1) {
                    X_later_para.x = g_img.cols - 1.001;
                }
                if (X_later_para.y >= g_img.rows - 1) {
                    X_later_para.y = g_img.rows - 1.001;
                }

                if (X_later_para.x < 0) {
                    X_later_para.x = 0;
                }
                if (X_later_para.y < 0) {
                    X_later_para.y = 0;
                }
                x0 = floor(X_later.x);
                d_x = X_later.x - x0;
                y0 = floor(X_later.y);
                d_y = X_later.y - y0;
                x0_para = floor(X_later_para.x);
                d_x_para = X_later_para.x - x0_para;
                y0_para = floor(X_later_para.y);
                d_y_para = X_later_para.y - y0_para;
                int y, y_para;

                y = (int) floor((M(g_[blare][z], (int) x0, (int) y0) * (1 - d_x) * (1 - d_y) +
                                 M(g_[blare][z], (int) x0 + 1, (int) y0) * d_x * (1 - d_y)
                                 + M(g_[blare][z], (int) x0, (int) y0 + 1) * (1 - d_x) * d_y +
                                 M(g_[blare][z], (int) x0 + 1, (int) y0 + 1) * d_x * d_y) + 0.5);
                y_para = (int) floor((M(g_[blare][z], (int) x0_para, (int) y0_para) * (1 - d_x_para) * (1 - d_y_para) +
                                      M(g_[blare][z], (int) x0_para + 1, (int) y0_para) * d_x_para * (1 - d_y_para)
                                      + M(g_[blare][z], (int) x0_para, (int) y0_para + 1) * (1 - d_x_para) * d_y_para +
                                      M(g_[blare][z], (int) x0_para + 1, (int) y0_para + 1) * d_x_para * d_y_para) + 0.5);
                double val = 0;
                for (int nx = -1; nx <= 2; nx++) {
                    for (int ny = -1; ny <= 2; ny++) {
                        val += g_expand[(int) x0 + nx][(int) y0 + ny] * w(nx - d_x) * w(ny - d_y);
                        if ((int) (M(g_img, (int) x0, (int) y0)) != (int) g_expand[(int) x0][(int) y0]) {
                            std::cout << "M = " << M(g_img, (int) x0, (int) y0) << " g_expand = "
                                      << (int) g_expand[(int) x0][(int) y0] << std::endl;
                            exit(1);
                        }
                    }
                }

                if (y < 0) {
                    y = 0;
                } else if (y > 255) {
                    y = 255;
                }
                if(z == 3){
                    if (flag_blare_para == true) {
                        para[i] = (unsigned char) y_para;
                        R(predict_para, (int) X.x, (int) X.y) = (unsigned char) y_para;
                        G(predict_para, (int) X.x, (int) X.y) = (unsigned char) y_para;
                        B(predict_para, (int) X.x, (int) X.y) = (unsigned char) y_para;
                    }
                    if (flag_blare == true) {
                        warp[i] = (unsigned char) y;
                        R(predict_warp, (int) X.x, (int) X.y) = (unsigned char) y;
                        G(predict_warp, (int) X.x, (int) X.y) = (unsigned char) y;
                        B(predict_warp, (int) X.x, (int) X.y) = (unsigned char) y;
                    }
                }else {
                    R(predict_buf[z], (int) X.x, (int) X.y) = (unsigned char) y;
                    G(predict_buf[z], (int) X.x, (int) X.y) = (unsigned char) y;
                    B(predict_buf[z], (int) X.x, (int) X.y) = (unsigned char) y;
                }
                float alpha_color;
                if (40 <= PSNR) {
                    alpha_color = 1;
                } else if (PSNR >= 39 && PSNR < 40) {
                    alpha_color = 0.9;
                } else if (PSNR >= 38 && PSNR < 39) {
                    alpha_color = 0.8;
                } else if (PSNR >= 37 && PSNR < 38) {
                    alpha_color = 0.7;
                } else if (PSNR >= 36 && PSNR < 37) {
                    alpha_color = 0.6;
                } else if (PSNR >= 35 && PSNR < 36) {
                    alpha_color = 0.5;
                } else if (PSNR >= 34 && PSNR < 35) {
                    alpha_color = 0.4;
                } else if (PSNR >= 33 && PSNR < 34) {
                    alpha_color = 0.3;
                } else if (PSNR >= 32 && PSNR < 33) {
                    alpha_color = 0.2;
                } else if (PSNR >= 31 && PSNR < 32) {
                    alpha_color = 0.1;
                } else if (PSNR < 31) {
                    alpha_color = 0.0;
                }

            }

            for (int j = 0; j < 3; j++) {
                pp[j].x -= triangle[j].x;
                pp[j].y -= triangle[j].y;
            }
            for(int d = -expand ;d < f_img.cols + expand;d++){
                f_expand[d] -= expand;
                g_expand[d] -= expand;
                free(f_expand[d]);
                free(g_expand[d]);
            }
            f_expand -= expand;
            g_expand -= expand;
            free(f_expand);
            free(g_expand);
        }
    }

    double squaredError = 0.0;
    for (int i = 0; i < (int) in_triangle_pixels.size(); i++) {
        unsigned char y;
        int Y;
        if (*flag == true) {
            y = para[i];
        }
        else{
            y = warp[i];
        }
        cv::Point2f X = in_triangle_pixels[i];
        R(predict_buf[3], (int) X.x, (int) X.y) = (unsigned char) y;
        G(predict_buf[3], (int) X.x, (int) X.y) = (unsigned char) y;
        B(predict_buf[3], (int) X.x, (int) X.y) = (unsigned char) y;
        Y = sqrt(residual_ref.at<unsigned char>((int)X.y,(int)X.x)*abs((int)M(current_color,(int)X.x,(int)X.y) - y)) + 0.5;
        Y = abs((int)M(current_color,(int)X.x,(int)X.y) - y);
        squaredError += Y * Y;
        if(Y > 255) {
            Y = 255;
        }
        residual.at<unsigned char>(X.y,X.x) = Y;
    }
    cv::imwrite("residual_" + std::to_string(t) + ".bmp",residual);
    if(*flag == true) {
        PSNR_max = PSNR_para_max;
    }
    cv::Point2f a, b;
    p0 = target_corners.p1;
    p1 = target_corners.p2;
    p2 = target_corners.p3;
    a = p2 - p0;
    b = p1 - p0;


    if (!*flag) {
        error_warp = Error_min;
    } else {
        error_warp = Error_para_min;
    }

//    error_warp = squaredError;
    return mv;
}


int img_ip(unsigned char **img, int xs, int ys, double x, double y, int mode)
{
    int x0, y0;          /* 補間点 (x, y) の整数部分 */
    double dx, dy;       /* 補間点 (x, y) の小数部分 */
    int nx, ny;          /* 双3次補間用のループ変数 */
    double val = 0.0, w(double);

    /*** 補間点(x, y)が原画像の領域外なら, 範囲外を示す -1 を返す ***/
    if (x < 0.0 || x > xs - 1 || y < 0 || y > ys - 1) exit(1);

    /*** 補間点(x, y)の整数部分(x0, y0), 小数部分(dx, dy)を求める ***/
    x0 = (int) floor(x);
    y0 = (int) floor(y);
    dx = x - (double) x0;
    dy = y - (double) y0;

    /*** mode で指定された補間法に従って補間し，値を val に保存 ***/
    switch(mode) { /* mode = 0 : 最近傍, 1 : 双1次, 2 : 双3次 */
        case 0:  /* 最近傍補間法 --- 式(9.4) */
            if (dx <= 0.5 && dy <= 0.5)     val = img[x0  ][y0  ];
            else if (dx > 0.5 && dy <= 0.5) val = img[x0+1][y0  ];
            else if (dx <= 0.5 && dy > 0.5) val = img[x0  ][y0+1];
            else                            val = img[x0+1][y0+1];
            break;
        case 1: /* 双1次補間法 --- 式(9.8) */
            val = img[x0  ][y0  ] * (1.0 - dx) * (1.0 - dy) +
                  img[x0+1][y0  ] * dx         * (1.0 - dy) +
                  img[x0  ][y0+1] * (1.0 - dx) * dy         +
                  img[x0+1][y0+1] * dx         * dy;
            break;
        case 2: /* 3次補間法 --- 式(9.13) */
            val = 0.0;
            for(ny = -1 ; ny <= 2 ; ny++) {
                for(nx = -1 ; nx <= 2 ; nx++) {
                    val += img[x0 + nx][y0 + ny] * w(nx - dx) * w(ny - dy);
                }
            }
            break;
        default:
            break;
    }

    /*** リミッタを掛けて return ***/
    if (val >= 255.5) return 255;
    else if (val < -0.5) return 0;
    else return (int) (val + 0.5);
}
double w(double x)
{
    double absx = fabs(x);

    if (absx <= 1.0) {
        return absx * absx * absx - 2 * absx * absx + 1;
    } else if (absx <= 2.0) {
        return - absx * absx * absx + 5 * absx * absx - 8 * absx + 4;
    } else {
        return 0.0;
    }
}
double getDistance2(const cv::Point2d& a, const cv::Point2d& b){
    cv::Point2d v = a - b;
    return sqrt(v.x * v.x + v.y * v.y);
}
//パッチをもらってその動きベクトルの符号量を返す
std::vector<cv::Point2i> Gauss_Golomb(Triangle triangle, bool *flag, std::vector<cv::Point2i> &ev, std::vector<cv::Point2f> corners, DelaunayTriangulation md,std::vector<cv::Point2i> mv,const cv::Mat& target,const cv::Mat& targetx8,bool &para_flag){
    std::vector<cv::Point2i> p;
    std::vector<int> index;
    std::vector<cv::Point2i> ret_mv;
    ret_mv.clear();
    cv::Point2f diff(0,0);
    index.clear();
    index.emplace_back(triangle.p1_idx);
    index.emplace_back(triangle.p2_idx);
    index.emplace_back(triangle.p3_idx);
    p.clear();
    p.emplace_back(corners[index[0]]);
    p.emplace_back(corners[index[1]]);
    p.emplace_back(corners[index[2]]);
    for(int i = 0;i < (int)corners.size();i++){
        std::cout << "flag[" << i << "] = " << flag[i] << std::endl;
        //std::cout << "p[" << i << "] = " << p[i].x << "," << p[i].y << std::endl;
    }
    for(int i = 0;i < 3;i++){
        //std::cout << "p[" << i << "] = " << p[i].x << "," << p[i].y << std::endl;
        //std::cout << "index[" << i << "] = " << index[i] << std::endl;
        std::cout << "mv[" << i << "] = " << mv[i] << std::endl;
        if(flag[index[i]] == true){
            diff = mv[i] - ev[index[i]];
        }
        else{
            double init_Distance = 10E05;
            double min_Distance = init_Distance;
            int min_num = 0;
            std::vector<int> neighbor = md.getNeighborVertexNum(index[i]);
            std::vector<cv::Point2f> neighbor_cood = md.getNeighborVertex(index[i]);
            for(int k = 0;k < (int)neighbor.size();k++){
                std::cout << "neighbor[" << k << "] = " << neighbor[k] - 4<< std::endl;
                std::cout << "neighbor_cood[" << k << "] = " << neighbor_cood[k] << std::endl;
                if(flag[neighbor[k] - 4]){
                    double Distance = getDistance2(p[i],neighbor_cood[k]);
                    std::cout << "Distance = " << Distance << std::endl;
                    if(min_Distance > Distance){
                        min_Distance = Distance;
                        min_num = neighbor[k] - 4;
                    }
                }
            }
            std::cout << "min_num = " << min_num << std::endl;
            if(min_Distance != init_Distance){
                if(!para_flag) {
                    diff = mv[i] - ev[min_num];
                }else{
                    diff = mv[0] - ev[min_num];
                }
                flag[min_num] = true;
            }
            else{
                if(!para_flag) {
                    diff = mv[i];
                }else{
                    diff = mv[0];
                }
                ev[index[i]] = mv[i];
                flag[index[i]] = true;
            }
        }
        std::cout << "diff = " << diff << std::endl;
        ret_mv.emplace_back(diff);
    }
    return ret_mv;
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

void block_matching_full(const cv::Mat& prev, const cv::Mat& current, double &error, cv::Point2f &mv, Point3Vec tr) {
    double sx, sy, lx, ly;
    cv::Point2f tp1, tp2, tp3;
    tp1 = tr.p1;
    tp2 = tr.p2;
    tp3 = tr.p3;

    sx = std::min({tp1.x, tp2.x, tp3.x});
    sy = std::min({tp1.y, tp2.y, tp3.y});
    lx = std::max({tp1.x, tp2.x, tp3.x});
    ly = std::max({tp1.y, tp2.y, tp3.y});

    cv::Point2f mv_tmp(0.0, 0.0);
    int SX = 30; // ブロックマッチングの探索範囲(X)
    int SY = 30; // ブロックマッチングの探索範囲(Y)

    double error_tmp, error_min;
    int error_count;

    error_min = 1 << 20;
    cv::Point2d xp(0.0, 0.0);

    for (int j = -SY / 2; j <= SY / 2; j++) {
        for (int i = -SX / 2; i <= SX / 2; i++) {
            error_tmp = 0.0;
            error_count = 0;

            int nx = static_cast<int>(round(sx) + i);
            int ny = static_cast<int>(round(sy) + j);

            // 範囲外の場合
            if(nx < 0 || prev.cols <= nx + (lx - sx) || ny < 0 || prev.rows <= ny + (ly - sy)) continue;

            for (int m = (int) (round(sy) - 1); m <= round(ly) + 1; m++) {
                for (int n = (int) (round(sx) - 1); n <= round(lx) + 1; n++) {
                    xp.x = (double) n;
                    xp.y = (double) m;

                    // xpが三角形trの中かどうか判定
                    if (isInTriangle(tr, xp)) {
                        // 現在のフレームとの差分
                        error_tmp += (M(prev, n + i, m + j) - M(current, n, m)) * (M(prev, n + i, m + j) - M(current, n, m));
                        error_count++;
                    }
                }
            }

            error_tmp = error_count > 0 ? (error_tmp / (double) error_count) : 1e6;
            if (error_tmp == error_min && error_count > 0) {
                if (abs(i) < abs(mv_tmp.x) && abs(j) < abs(mv_tmp.y)) {
                    mv_tmp.x = (float) i;
                    mv_tmp.y = (float) j;
                }
            }

            if (error_min > error_tmp && error_count > 0) {
                error_min = error_tmp;
                mv_tmp.x = (float) i;
                mv_tmp.y = (float) j;
            }
        }
    }
    std::cout << "check_point" << std::endl;
    error = error_min;
    mv.x = mv_tmp.x;
    mv.y = mv_tmp.y;
    std::cout << "mv = " << mv.x << mv.y << std::endl;
    std::cout << "block_error = " << error << std::endl;
}
