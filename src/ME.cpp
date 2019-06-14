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

    // 拡大画像の取得
    unsigned char **expand_ref;
    int offset = 32;
    expand_ref = (unsigned char **)malloc((ref_image.cols + offset * 2) * sizeof(unsigned char *));
    expand_ref += offset;
    for(int i = -offset ; i < ref_image.cols + offset ; i++) {
        expand_ref[i] = (unsigned char *)malloc((ref_image.rows + offset * 2) * sizeof(unsigned char));
        expand_ref[i] += offset;
    }

    for(int y = 0 ; y < ref_image.rows ; y++){
        for(int x = 0 ; x < ref_image.cols ; x++){
            expand_ref[x][y] = M(ref_image, x, y);
        }
    }

    for(int y = 0 ; y < ref_image.rows ; y++){
        for(int x = -offset ; x < 0 ; x++){
            expand_ref[x][y] = M(ref_image, 0, y);
            expand_ref[ref_image.cols + offset + x][y] = M(ref_image, ref_image.cols - 1, y);
        }
    }

    for(int y = -offset ; y < 0 ; y++){
        for(int x = -offset ; x < ref_image.cols + offset ; x++){
            expand_ref[x][y] = M(ref_image, x, 0);
            expand_ref[x][ref_image.rows + offset + y] = M(ref_image, x, ref_image.rows - 1);
        }
    }

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

        int y = bicubic_interpolation(expand_ref, X_later.x, X_later.y);

        R(output_image, (int)pixel.x, (int)pixel.y) = y;
        G(output_image, (int)pixel.x, (int)pixel.y) = y;
        B(output_image, (int)pixel.x, (int)pixel.y) = y;

        squared_error += pow((M(target_image, (int)pixel.x, (int)pixel.y) - (0.299 * y + 0.587 * y + 0.114 * y)), 2);
    }

    // メモリの開放
    for(int i = -offset ; i < ref_image.cols + offset ; i++) {
        expand_ref[i] -= offset;
        free(expand_ref[i]);
    }

    expand_ref -= offset;
    free(expand_ref);

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
std::tuple<std::vector<cv::Point2f>, cv::Point2f, double, int, bool> GaussNewton(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, std::vector<std::vector<std::vector<unsigned char **>>> expand_image, Point3Vec target_corners){
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

    int bm_x_offset = 10;
    int bm_y_offset = 10;
    double error_min = 1e9;

    for(int by = -bm_y_offset ; by < bm_y_offset ; by++){
        for(int bx = -bm_x_offset ; bx < bm_x_offset ; bx++){
            if(sx + bx < -16 || ref_images[0][3].cols <= (lx + bx + 16) || sy + by < -16 || ref_images[0][3].rows <=(ly + by + 16)) continue;
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

//    std::cout << target_corners.p1 << " " << target_corners.p2 << " " << target_corners.p3 << std::endl;
//    std::cout << initial_vector << std::endl;
    initial_vector /= 2.0;
    for(int filter_num = 0 ; filter_num < static_cast<int>(ref_images.size()) ; filter_num++){
        std::vector<cv::Point2f> tmp_mv_warping(3, cv::Point2f(initial_vector.x, initial_vector.y));
        cv::Point2f tmp_mv_parallel(initial_vector.x, initial_vector.y);

        for(int step = 3 ; step < static_cast<int>(ref_images[filter_num].size()) ; step++){

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
                    double x0_later_warping_decimal = X_later_warping.x - x0_later_warping_integer;
                    double y0_later_warping_decimal = X_later_warping.y - y0_later_warping_integer;
                    double x0_later_parallel_decimal = X_later_parallel.x - x0_later_parallel_integer;
                    double y0_later_parallel_decimal = X_later_parallel.y - y0_later_parallel_integer;

                    double f = bicubic_interpolation(current_target_expand, X.x, X.y);
                    double f_org = bicubic_interpolation(current_target_org_expand, X.x, X.y);

                    double g_warping = bicubic_interpolation(current_ref_expand, X_later_warping.x, X_later_warping.y);
                    double g_parallel = bicubic_interpolation(current_ref_expand, X_later_parallel.x, X_later_parallel.y);

                    double g_org_warping = bicubic_interpolation(current_ref_org_expand, X_later_warping.x, X_later_warping.y);
                    double g_org_parallel = bicubic_interpolation(current_ref_org_expand, X_later_parallel.x, X_later_parallel.y);

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
