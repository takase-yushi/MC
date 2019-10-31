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
 * @fn std::tuple<std::vector<cv::Point2f>, double> block_matching(Point4Vec sq, const cv::Mat& current, cv::Mat expansion_image)
 * @brief ブロックマッチングを行い, 動きベクトルを求める
 * @param[in]  sq             四角形を表す4点の座標
 * @param[in]  current          対象画像
 * @param[in]  expansion_image  4倍に拡大した(補間した)画像
 * @details
 *
 */
std::tuple<std::vector<cv::Point2f>, std::vector<double>> blockMatching(Point4Vec square, const cv::Mat& target_image, cv::Mat expansion_ref_image) {
        double sx, sy, lx, ly;
        cv::Point2f sp1, sp4;

        sp1 = square.p1;
        sp4 = square.p4;

        sx = 4 * sp1.x;
        sy = 4 * sp1.y;
        lx = 4 * sp4.x + 3;
        ly = 4 * sp4.y + 3;

        cv::Point2f mv_tmp(0.0, 0.0); //ブロックの動きベクトル
        int SX = SERACH_RANGE;                 // ブロックマッチングの探索範囲(X)
        int SY = SERACH_RANGE;                 // ブロックマッチングの探索範囲(Y)
        int neighbor_pixels = BLOCKMATCHING_NEIGHBOR_PIXELS;     //1 : 近傍 1 画素,  2 : 近傍 2 画素,   n : 近傍 n 画素

        double e;
        double e_min = 1e9;

        cv::Point2f mv_min;
        int spread_quarter = 4 * SERACH_RANGE;
        int s = 4;                   //4 : Full-pel, 2 : Half-pel, 1 : Quarter-pel
        std::vector<cv::Point2f> pixels = getPixelsInSquare(square);

        for(int j = -SY * 4 ; j <= SY * 4 ; j += s) {            //j : y方向のMV
            for(int i = -SX * 4 ; i <= SX * 4 ; i += s) {        //i : x方向のMV
                //探索範囲が画像上かどうか判定
                if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
                   && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                    e = 0.0;
                    for (int y = (int) (round(sy) / 4); y <= (int) (round(ly) / 4); y++) {
                        for (int x = (int) (round(sx) / 4); x <= (int) (round(lx) / 4); x++) {
                            e += fabs(R(expansion_ref_image, i + 4 * x + spread_quarter, j + 4 * y + spread_quarter) - R(target_image, x, y));
                        }
                    }
                    if(e_min > e){
                        e_min = e;
                        mv_min.x = (double)i / 4.0;
                        mv_min.y = (double)j / 4.0;
                    }
                }
            }
        }

        std::vector<cv::Point2f> mvs;
        std::vector<double> errors;
        mvs.emplace_back(mv_min.x, mv_min.y);
        errors.emplace_back(e_min);

        mv_tmp.x = mv_min.x * 4;
        mv_tmp.y = mv_min.y * 4;

        s = 2;
        for(int j = - neighbor_pixels * s + mv_tmp.y ; j <= neighbor_pixels * s + mv_tmp.y ; j += s){            //j : y方向のMV
            for(int i = - neighbor_pixels * s + mv_tmp.x ; i <= neighbor_pixels * s + mv_tmp.x ; i += s){        //i : x方向のMV
                if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
                   && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                    e = 0.0;
                    for (int y = (int) (round(sy) / 4); y <= (int) (round(ly) / 4); y++) {
                        for (int x = (int) (round(sx) / 4); x <= (int) (round(lx) / 4); x++) {
                            e += fabs(R(expansion_ref_image, i + 4 * x + spread_quarter, j + 4 * y + spread_quarter) - R(target_image, x, y));
                        }
                    }
                    if(e_min > e){
                        e_min = e;
                        mv_min.x = (double)i / 4.0;
                        mv_min.y = (double)j / 4.0;
                    }
                }
            }
        }

        mvs.emplace_back(mv_min.x, mv_min.y);
        errors.emplace_back(e_min);
        mv_tmp.x = mv_min.x * 4;
        mv_tmp.y = mv_min.y * 4;

        s = 1;

        for(int j = - neighbor_pixels * s + mv_tmp.y ; j <= neighbor_pixels * s + mv_tmp.y ; j += s){            //j : y方向のMV
            for(int i = - neighbor_pixels * s + mv_tmp.x ; i <= neighbor_pixels * s + mv_tmp.x ; i += s){        //i : x方向のMV
                if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
                   && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                    e = 0.0;
                    for (int y = (int) (round(sy) / 4); y <= (int) (round(ly) / 4); y++) {
                        for (int x = (int) (round(sx) / 4); x <= (int) (round(lx) / 4); x++) {
                            e += fabs(R(expansion_ref_image, i + 4 * x + spread_quarter, j + 4 * y + spread_quarter) - R(target_image, x, y));
                        }
                    }
                    if(e_min > e){
                        e_min = e;
                        mv_min.x = (double)i / 4.0;
                        mv_min.y = (double)j / 4.0;
                    }
                }
            }
        }

        errors.emplace_back(e_min);
        mvs.emplace_back(mv_min.x, mv_min.y);

        return std::make_tuple(mvs, errors);
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
 * @fn double getPredictedImage(unsigned char **expand_ref, cv::Mat& target_image, cv::Mat& output_image, Point4Vec& square, cv::Point2f& mv, unsigned char **ref_hevc)
 * @brief 動きベクトルをもらって、out_imageに画像を書き込む
 * @param[in] ref_image
 * @param[in] target_image
 * @param[out] output_image
 * @param[in] triangle
 * @param[in] mv
 * @param[in] parallel_flag
 * @return 2乗誤差
 */
double getPredictedImage(unsigned char **expand_ref, cv::Mat& target_image, cv::Mat& output_image, Point4Vec& square, std::vector<cv::Point2f>& mv, unsigned char **ref_hevc) {
    cv::Point2f pp0, pp1, pp2;

    pp0 = square.p1 + mv[0];
    pp1 = square.p2 + mv[1];
    pp2 = square.p3 + mv[2];

    std::vector<cv::Point2f> in_square_pixels = getPixelsInSquare(square);

    cv::Point2f X, a, b, X_later, a_later, b_later;
    double alpha, beta, det;

    double sse = 0.0;

    a = square.p3 - square.p1;
    b = square.p2 - square.p1;
    //変形前の四角形の面積を求める
    det = a.x * b.y - a.y * b.x;

    double squared_error = 0.0;

    for(const auto& pixel : in_square_pixels) {
        //ある画素までのベクトル
        X = pixel - square.p1;
        //変形前のα,βを求める
        alpha = (X.x * b.y - X.y * b.x) / det;
        beta = (a.x * X.y - a.y * X.x) / det;
        //変形後のa,bを求める
        a_later = pp2 - pp0;
        b_later = pp1 - pp0;
        //変形後の画素の座標を求める
        X_later = alpha * a_later + beta * b_later + pp0;

        int y;
        if(ref_hevc != nullptr){
            y = img_ip(ref_hevc, cv::Rect(-4 * SERACH_RANGE, -4 * SERACH_RANGE, 4 * (target_image.cols + 2 * SERACH_RANGE), 4 * (target_image.rows + 2 * SERACH_RANGE)), 4 * X_later.x, 4 * X_later.y, 1);
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
 * @fn std::pair<std::vector<cv::Point2f>, cv::Point2f> Square_GaussNewton(cv::Mat ref_image, cv::Mat target_mage, cv::Mat gauss_ref_image, Point3Vec target_corners)
 * @brief ガウス・ニュートン法を行い、動きベクトル・予測残差・面積を返す
 * @param ref_images
 * @param target_images
 * @param expand_image
 * @param target_corners
 * @param square_index
 * @param ctu
 * @return ワーピングの動きベクトル・平行移動の動きベクトル・予測残差・面積・平行移動のフラグのtuple
 */
std::tuple<std::vector<cv::Point2f>, cv::Point2f, double, double, int> Square_GaussNewton(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, std::vector<std::vector<std::vector<unsigned char **>>> expand_image, Point4Vec target_corners, int square_index, CodingTreeUnit *ctu, cv::Point2f init_vector, unsigned char **ref_hevc){
    // 画像の初期化 vector[filter][picture_number]

    const int warping_matrix_dim = 6; // 方程式の次元
    const int translation_matrix_dim = 2;
    cv::Mat gg_warping = cv::Mat::zeros(warping_matrix_dim, warping_matrix_dim, CV_64F); // 式(45)の左辺6×6行列
    cv::Mat gg_translation = cv::Mat::zeros(translation_matrix_dim, translation_matrix_dim, CV_64F); // 式(52)の左辺2×2行列
    cv::Mat B_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F); // 式(45)の右辺
    cv::Mat B_translation = cv::Mat::zeros(translation_matrix_dim, 1, CV_64F); // 式(52)の右辺
    cv::Mat delta_uv_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F); // 式(45)の左辺 delta
    cv::Mat delta_uv_translation = cv::Mat::zeros(translation_matrix_dim, 1, CV_64F); // 式(52)の右辺 delta

    double MSE_warping, MSE_translation;
    double min_error_warping = 1E6, min_error_translation = 1E6;
    double max_PSNR_warping = -1, max_PSNR_translation = -1;

    cv::Point2f p0, p1, p2, p3;
    std::vector<cv::Point2f> max_v_warping;
    cv::Point2f max_v_translation;

    std::vector<std::pair<std::vector<cv::Point2f>,double>> v_stack_warping;
    std::vector<std::pair<cv::Point2f,double>> v_stack_translation;
    std::vector<cv::Point2f> pixels_in_square;

    bool translation_flag = true;

    cv::Point2f initial_vector(0.0, 0.0);

    p0 = target_corners.p1;
    p1 = target_corners.p2;
    p2 = target_corners.p3;
    p3 = target_corners.p4;

    Point4Vec current_triangle_coordinates(p0, p1, p2, p3);

    pixels_in_square = getPixelsInSquare(current_triangle_coordinates);

    double sx = p0.x;
    double sy = p0.y;
    double lx = p3.x;
    double ly = p3.y;

    int SX = SERACH_RANGE;                 // ブロックマッチングの探索範囲(X)
    int SY = SERACH_RANGE;                 // ブロックマッチングの探索範囲(Y)
    double error_min = 1e9;

    if(init_vector.x == -1000 && init_vector.y == -1000) {
        for (int by = -SY; by < SY; by++) {
            for (int bx = -SX; bx < SX; bx++) {
                if (sx + bx < -SERACH_RANGE || ref_images[0][3].cols + SERACH_RANGE <= (lx + bx) || sy + by < -SERACH_RANGE || ref_images[0][3].rows + SERACH_RANGE <= (ly + by)) continue;
                double error_tmp = 0.0;
                for (const auto &pixel : pixels_in_square) {
#if GAUSS_NEWTON_HEVC_IMAGE
                    error_tmp += abs(expand_image[0][3][1][4 * (int) (pixel.x + bx)][4 * (int) (pixel.y + by)] -
                                     expand_image[0][3][3][4 * (int) (pixel.x)][4 * (int) (pixel.y)]);
#else
                    error_tmp += abs(expand_image[0][3][1][(int) (pixel.x + bx)][(int) (pixel.y + by)] -
                                                     expand_image[0][3][3][(int) (pixel.x)][(int) (pixel.y)]);
#endif
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
        cv::Point2f tmp_mv_translation(initial_vector.x, initial_vector.y);

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

            int spread = SERACH_RANGE; // 探索範囲は16までなので16に戻す

            int scaled_spread = spread / scale;
            p0 = target_corners.p1 / scale;
            p1 = target_corners.p2 / scale;
            p2 = target_corners.p3 / scale;
            p3 = target_corners.p4 / scale;

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
            current_triangle_coordinates.p4 = p3;
            pixels_in_square = getPixelsInSquare(current_triangle_coordinates);
//TODO なにこれ
            std::vector<cv::Point2f> scaled_coordinates{p0, p1, p2, p3};

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
            v_stack_warping.clear();

            double prev_error_warping = 1e6, prev_error_translation = 1e6;
            cv::Point2f prev_mv_translation;
            std::vector<cv::Point2f> prev_mv_warping;
            bool warping_update_flag = true, translation_update_flag = true;

            int iterate_counter = 0;
            while(true){
                // 移動後の座標を格納する
                std::vector<cv::Point2f> ref_coordinates_warping;
                std::vector<cv::Point2f> ref_coordinates_translation;

                ref_coordinates_warping.emplace_back(p0);
                ref_coordinates_warping.emplace_back(p1);
                ref_coordinates_warping.emplace_back(p2);
                ref_coordinates_warping.emplace_back(p3);

                ref_coordinates_translation.emplace_back(p0);
                ref_coordinates_translation.emplace_back(p1);
                ref_coordinates_translation.emplace_back(p2);
                ref_coordinates_translation.emplace_back(p3);

                cv::Point2f a = p2 - p0;
                cv::Point2f b = p1 - p0;
                double det = a.x * b.y - a.y * b.x;
                // tmp_mv_warping, tmp_mv_translationは現在の動きベクトル
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

                MSE_translation = MSE_warping = 0.0;
                gg_warping = cv::Mat::zeros(warping_matrix_dim, warping_matrix_dim, CV_64F);
                B_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F);
                delta_uv_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F);

                gg_translation = cv::Mat::zeros(translation_matrix_dim, translation_matrix_dim, CV_64F);
                B_translation = cv::Mat::zeros(translation_matrix_dim, 1, CV_64F);
                delta_uv_translation = cv::Mat::zeros(translation_matrix_dim, 1, CV_64F);

                double delta_g_warping[warping_matrix_dim] = {0};
                double delta_g_translation[translation_matrix_dim] = {0};

                cv::Point2f X;
                double RMSE_warping_filter = 0;
                double RMSE_translation_filter = 0;
                for(auto pixel : pixels_in_square) {
                    X = pixel - p0;

                    double alpha = (X.x * b.y - X.y * b.x) / det;
                    double beta = (a.x * X.y - a.y * X.x)/ det;
                    X += p0;

//                    int x_integer = (int)floor(X.x);
//                    int y_integer = (int)floor(X.y);
//                    int x_decimal = X.x - x_integer;
//                    int y_decimal = X.y - y_integer;

                    // 参照フレームの中心差分（平行移動）
                    double g_x_translation;
                    double g_y_translation;
                    cv::Point2f X_later_translation, X_later_warping;
                    //TODO ここのfor文もうちょい下じゃないと無駄に計算しないですか？
                    for(int i = 0 ; i < 6 ; i++) {
                        // 移動後の頂点を計算し格納
                        ref_coordinates_warping[0] = p0 + tmp_mv_warping[0];
                        ref_coordinates_warping[1] = p1 + tmp_mv_warping[1];
                        ref_coordinates_warping[2] = p2 + tmp_mv_warping[2];
                        ref_coordinates_translation[0] = p0 + tmp_mv_translation;
                        ref_coordinates_translation[1] = p1 + tmp_mv_translation;
                        ref_coordinates_translation[2] = p2 + tmp_mv_translation;

                        std::vector<cv::Point2f> square_later_warping;
                        std::vector<cv::Point2f> square_later_translation;
                        square_later_warping.emplace_back(ref_coordinates_warping[0]);
                        square_later_warping.emplace_back(ref_coordinates_warping[1]);
                        square_later_warping.emplace_back(ref_coordinates_warping[2]);
                        square_later_translation.emplace_back(ref_coordinates_translation[0]);
                        square_later_translation.emplace_back(ref_coordinates_translation[1]);
                        square_later_translation.emplace_back(ref_coordinates_translation[2]);

                        cv::Point2f a_later_warping, a_later_translation;
                        cv::Point2f b_later_warping, b_later_translation;

                        //変形後のa',b'を求める
                        a_later_warping  =  square_later_warping[2] -  square_later_warping[0];
                        a_later_translation = square_later_translation[2] - square_later_translation[0];
                        b_later_warping  =  square_later_warping[1] -  square_later_warping[0];
                        b_later_translation = square_later_translation[1] - square_later_translation[0];
                        //変形後の座標を求める
                        X_later_warping  = alpha *  a_later_warping + beta *  b_later_warping +  square_later_warping[0];
                        X_later_translation = alpha * a_later_translation + beta * b_later_translation + square_later_translation[0];

                        if(X_later_warping.x >= current_ref_image.cols - 1 + scaled_spread) X_later_warping.x = current_ref_image.cols - 1.00 + scaled_spread;
                        if(X_later_warping.y >= current_ref_image.rows - 1 + scaled_spread) X_later_warping.y = current_ref_image.rows - 1.00 + scaled_spread;
                        if(X_later_warping.x < -scaled_spread) X_later_warping.x = -scaled_spread;
                        if(X_later_warping.y < -scaled_spread) X_later_warping.y = -scaled_spread;

                        if(X_later_translation.x >= (current_ref_image.cols - 1 + scaled_spread)) X_later_translation.x = current_ref_image.cols - 1 + scaled_spread;
                        if(X_later_translation.y >= (current_ref_image.rows - 1 + scaled_spread)) X_later_translation.y = current_ref_image.rows - 1 + scaled_spread;
                        if(X_later_translation.x < -scaled_spread) X_later_translation.x = -scaled_spread;
                        if(X_later_translation.y < -scaled_spread) X_later_translation.y = -scaled_spread;

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
                        double g_x, g_y;
#if GAUSS_NEWTON_HEVC_IMAGE
                        g_x          = (img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_warping.x  + 1), 4 * (X_later_warping.y     ), 1) - img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_warping.x  - 1), 4 * (X_later_warping.y     ), 1)) / 2.0;  // (current_ref_expand[x_warping_tmp + 4 ][y_warping_tmp     ] - current_ref_expand[x_warping_tmp - 4 ][y_warping_tmp     ]) / 2.0;
                        g_y          = (img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_warping.x     ), 4 * (X_later_warping.y  + 1), 1) - img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_warping.x     ), 4 * (X_later_warping.y  - 1), 1)) / 2.0;  // (current_ref_expand[x_warping_tmp     ][y_warping_tmp + 4 ] - current_ref_expand[x_warping_tmp     ][y_warping_tmp - 4 ]) / 2.0;
                        g_x_translation = (img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_translation.x + 1), 4 * (X_later_translation.y    ), 1) - img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_translation.x - 1), 4 * (X_later_translation.y    ), 1)) / 2.0;  // (current_ref_expand[x_translation_tmp + 4][y_translation_tmp    ] - current_ref_expand[x_translation_tmp - 4][y_translation_tmp    ]) / 2.0;
                        g_y_translation = (img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_translation.x    ), 4 * (X_later_translation.y + 1), 1) - img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_translation.x    ), 4 * (X_later_translation.y - 1), 1)) / 2.0;  // (current_ref_expand[x_translation_tmp    ][y_translation_tmp + 4] - current_ref_expand[x_translation_tmp    ][y_translation_tmp - 4]) / 2.0;
#else
                        g_x   = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x  + 1 , X_later_warping.y    , 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x  - 1, X_later_warping.y     , 1)) / 2.0;  // (current_ref_expand[x_warping_tmp + 4 ][y_warping_tmp     ] - current_ref_expand[x_warping_tmp - 4 ][y_warping_tmp     ]) / 2.0;
                            g_y   = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x     , X_later_warping.y  + 1, 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x     , X_later_warping.y  - 1, 1)) / 2.0;  // (current_ref_expand[x_warping_tmp     ][y_warping_tmp + 4 ] - current_ref_expand[x_warping_tmp     ][y_warping_tmp - 4 ]) / 2.0;
                            g_x_translation = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x + 1, X_later_translation.y    , 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x - 1, X_later_translation.y    , 1)) / 2.0;  // (current_ref_expand[x_translation_tmp + 4][y_translation_tmp    ] - current_ref_expand[x_translation_tmp - 4][y_translation_tmp    ]) / 2.0;
                            g_y_translation = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x    , X_later_translation.y + 1, 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x    , X_later_translation.y - 1, 1)) / 2.0;  // (current_ref_expand[x_translation_tmp    ][y_translation_tmp + 4] - current_ref_expand[x_translation_tmp    ][y_translation_tmp - 4]) / 2.0;
#endif
                        spread-=1;

                        // 式(28)～(33)
                        delta_g_warping[i] = g_x * delta_x + g_y * delta_y;
                    }
                    delta_g_translation[0] = g_x_translation;
                    delta_g_translation[1] = g_y_translation;

                    double f;
                    double f_org;
                    double g_warping;
                    double g_translation;

#if GAUSS_NEWTON_HEVC_IMAGE
                    f              = img_ip(current_target_expand    , cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *                X.x, 4 *                X.y, 1);
                    f_org          = img_ip(current_target_org_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *                X.x, 4 *                X.y, 1);
                    g_warping      = img_ip(current_ref_expand       , cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *  X_later_warping.x, 4 *  X_later_warping.y, 1);
                    g_translation     = img_ip(current_ref_expand       , cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * X_later_translation.x, 4 * X_later_translation.y, 1);
#else
                    f              = img_ip(current_target_expand    , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),                X.x,                X.y, 2);
                    f_org          = img_ip(current_target_org_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),                X.x,                X.y, 2);
                    g_warping      = img_ip(current_ref_expand       , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),  X_later_warping.x,  X_later_warping.y, 2);
                    g_translation     = img_ip(current_ref_expand       , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x, X_later_translation.y, 2);
#endif
                    double g_org_warping;
                    double g_org_translation;
                    RMSE_warping_filter += fabs(f - g_warping);
                    RMSE_translation_filter += fabs(f - g_translation);

                    cv::Point2f tmp_X_later_warping, tmp_X_later_translation;
                    tmp_X_later_warping.x = X_later_warping.x;
                    tmp_X_later_warping.y = X_later_warping.y;
                    tmp_X_later_translation.x = X_later_translation.x;
                    tmp_X_later_translation.y = X_later_translation.y;

                    tmp_X_later_warping = roundVecQuarter(tmp_X_later_warping);
                    tmp_X_later_translation = roundVecQuarter(tmp_X_later_translation);

                    if(ref_hevc != nullptr) {
                        g_org_warping  = img_ip(ref_hevc, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * tmp_X_later_warping.x,  4 * tmp_X_later_warping.y, 1);
                        g_org_translation = img_ip(ref_hevc, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * tmp_X_later_translation.x, 4 * tmp_X_later_translation.y, 1);
                    }else {
#if GAUSS_NEWTON_HEVC_IMAGE
                        g_org_warping  = img_ip(current_ref_org_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *  tmp_X_later_warping.x, 4 *  tmp_X_later_warping.y, 1);
                        g_org_translation = img_ip(current_ref_org_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * tmp_X_later_translation.x, 4 * tmp_X_later_translation.y, 1);
#else
                        g_org_warping  = img_ip(current_ref_org_expand, cv::Rect(-spread, -spread, current_target_image.cols + 2 * spread, current_target_image.rows + 2 * spread),  tmp_X_later_warping.x, tmp_X_later_warping.y, 2);
                            g_org_translation = img_ip(current_ref_org_expand, cv::Rect(-spread, -spread, current_target_image.cols + 2 * spread, current_target_image.rows + 2 * spread), tmp_X_later_translation.x, tmp_X_later_translation.y, 2);
#endif
                    }

                    if(iterate_counter > 4){
                        f = f_org;
                        g_warping = g_org_warping;
                        g_translation = g_org_translation;
                    }

                    for (int row = 0; row < warping_matrix_dim; row++) {
                        for (int col = 0; col < warping_matrix_dim; col++) {
                            gg_warping.at<double>(row, col) += delta_g_warping[row] * delta_g_warping[col];//A_0の行列を生成(左辺の6x6の行列に相当)
                        }
                        B_warping.at<double>(row, 0) += (f - g_warping) * delta_g_warping[row];//bの行列を生成(右辺の6x1のベクトルに相当)
                    }
                    for (int row = 0; row < 2; row++) {
                        for (int col = 0; col < 2; col++) {
                            gg_translation.at<double>(row, col) += delta_g_translation[row] * delta_g_translation[col];
                        }
                        B_translation.at<double>(row, 0) += (f - g_translation) * delta_g_translation[row];
                    }


                    MSE_warping += fabs(f_org - g_org_warping);   // * (f_org - g_org_warping);
                    MSE_translation += fabs(f_org - g_org_translation); // * (f_org - g_org_translation);
                }

                double mu = 10;
                for(int row = 0 ; row < warping_matrix_dim ; row++){
                    for(int col = 0 ; col < warping_matrix_dim ; col++) {
                        gg_warping.at<double>(row, col) += mu * ((area_after_move * area_after_move * area_after_move - area_before_move * area_before_move * area_after_move) / (area_before_move * area_before_move * area_before_move * area_before_move) * S[row]);
                    }
                }

                double Error_warping = MSE_warping;
                double Error_translation = MSE_translation;
                double mu2 = pixels_in_square.size() * 0.0001;
                gg_translation.at<double>(0, 0) += 4 * mu2 * tmp_mv_translation.x * tmp_mv_translation.x;
                gg_translation.at<double>(0, 1) += 4 * mu2 * tmp_mv_translation.x * tmp_mv_translation.y;
                gg_translation.at<double>(1, 0) += 4 * mu2 * tmp_mv_translation.y * tmp_mv_translation.x;
                gg_translation.at<double>(1, 1) += 4 * mu2 * tmp_mv_translation.y * tmp_mv_translation.y;
                B_translation.at<double>(0, 0) -= 2 * mu2 * tmp_mv_translation.x * (tmp_mv_translation.x * tmp_mv_translation.x + tmp_mv_translation.y * tmp_mv_translation.y);
                B_translation.at<double>(1, 0) -= 2 * mu2 * tmp_mv_translation.y * (tmp_mv_translation.x * tmp_mv_translation.x + tmp_mv_translation.y * tmp_mv_translation.y);
                cv::solve(gg_warping, B_warping, delta_uv_warping); //6x6の連立方程式を解いてdelta_uvに格納
                v_stack_warping.emplace_back(tmp_mv_warping, Error_warping);

                for (int k = 0; k < 6; k++) {

                    if (k % 2 == 0) {
                        if ((-scaled_spread <= scaled_coordinates[(int) (k / 2)].x + tmp_mv_warping[(int) (k / 2)].x +delta_uv_warping.at<double>(k, 0)) &&
                            (target_images[0][step].cols - 1 + scaled_spread >=scaled_coordinates[(int) (k / 2)].x + tmp_mv_warping[(int) (k / 2)].x + delta_uv_warping.at<double>(k, 0))) {
                            tmp_mv_warping[(int) (k / 2)].x = tmp_mv_warping[(int) (k / 2)].x + delta_uv_warping.at<double>(k, 0);//動きベクトルを更新(画像の外に出ないように)
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

                cv::solve(gg_translation, B_translation, delta_uv_translation);
                v_stack_translation.emplace_back(tmp_mv_translation, Error_translation);

                for (int k = 0; k < 2; k++) {
                    if (k % 2 == 0) {
                        if ((-scaled_spread <=
                             scaled_coordinates[0].x + tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0)) &&
                            (target_images[0][step].cols - 1 + scaled_spread >=
                             scaled_coordinates[0].x + tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0)) &&
                            (-scaled_spread <=
                             scaled_coordinates[1].x + tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0)) &&
                            (target_images[0][step].cols - 1 + scaled_spread >=
                             scaled_coordinates[1].x + tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0)) &&
                            (-scaled_spread <=
                             scaled_coordinates[2].x + tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0)) &&
                            (target_images[0][step].cols - 1 + scaled_spread >=
                             scaled_coordinates[2].x + tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0))) {
                            tmp_mv_translation.x = tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0);
                        }
                    } else {
                        if ((-scaled_spread <=
                             scaled_coordinates[0].y + tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0)) &&
                            (target_images[0][step].rows - 1 + scaled_spread >=
                             scaled_coordinates[0].y + tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0)) &&
                            (-scaled_spread <=
                             scaled_coordinates[1].y + tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0)) &&
                            (target_images[0][step].rows - 1 + scaled_spread >=
                             scaled_coordinates[1].y + tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0)) &&
                            (-scaled_spread <=
                             scaled_coordinates[2].y + tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0)) &&
                            (target_images[0][step].rows - 1 + scaled_spread >=
                             scaled_coordinates[2].y + tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0))) {
                            tmp_mv_translation.y = tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0);
                        }
                    }
                }

                double eps = 1e-3;
                if(((fabs(prev_error_translation - MSE_translation) / MSE_translation) < eps && (fabs(prev_error_warping - MSE_warping) / MSE_warping < eps)) || (!translation_update_flag && !warping_update_flag) || iterate_counter > 20){
                    break;
                }

                prev_error_translation = MSE_translation;
                prev_error_warping = MSE_warping;
                prev_mv_translation = tmp_mv_translation;
                prev_mv_warping = tmp_mv_warping;
                iterate_counter++;
            }

            std::sort(v_stack_warping.begin(), v_stack_warping.end(), [](std::pair<std::vector<cv::Point2f>,double> a, std::pair<std::vector<cv::Point2f>,double> b){
                return a.second < b.second;
            });

            std::sort(v_stack_translation.begin(), v_stack_translation.end(), [](std::pair<cv::Point2f,double> a, std::pair<cv::Point2f,double> b){
                return a.second < b.second;
            });

            tmp_mv_warping = v_stack_warping[0].first;//一番良い動きベクトルを採用
            double Error_warping = v_stack_warping[0].second;
            tmp_mv_translation = v_stack_translation[0].first;
            double Error_translation = v_stack_translation[0].second;
            MSE_warping = Error_warping / (double)pixels_in_square.size();
            MSE_translation = Error_translation / (double)pixels_in_square.size();
            double PSNR_warping = 10 * log10((255 * 255) / MSE_warping);
            double PSNR_translation = 10 * log10((255 * 255) / MSE_translation);

            if(step == 3) {//一番下の階層で
                if(PSNR_translation >= max_PSNR_translation){//2種類のボケ方で良い方を採用
                    max_PSNR_translation = PSNR_translation;
                    min_error_translation = Error_translation;
                    max_v_translation = tmp_mv_translation;
                }
                if (PSNR_warping >= max_PSNR_warping) {
                    max_PSNR_warping = PSNR_warping;
                    min_error_warping = Error_warping;
                    max_v_warping = tmp_mv_warping;
                }

                if (fabs(max_PSNR_warping - max_PSNR_translation) <= 0.5 || max_PSNR_translation > max_PSNR_warping) {//ワーピングと平行移動でRDのようなことをする
                    translation_flag = true;//平行移動を採用
                } else{
                    translation_flag = false;//ワーピングを採用
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


    return std::make_tuple(std::vector<cv::Point2f>{max_v_warping[0], max_v_warping[1], max_v_warping[2]}, max_v_translation, min_error_warping, min_error_translation, pixels_in_square.size());
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
std::tuple<std::vector<cv::Point2f>, cv::Point2f, double, double, int> GaussNewton(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, std::vector<std::vector<std::vector<unsigned char **>>> expand_image, Point3Vec target_corners, const std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, int block_size_x, int block_size_y, cv::Point2f init_vector, unsigned char **ref_hevc){
    // 画像の初期化 vector[filter][picture_number]

    const int warping_matrix_dim = 6; // 方程式の次元
    const int translation_matrix_dim = 2;
    cv::Mat gg_warping = cv::Mat::zeros(warping_matrix_dim, warping_matrix_dim, CV_64F); // 式(45)の左辺6×6行列
    cv::Mat gg_translation = cv::Mat::zeros(translation_matrix_dim, translation_matrix_dim, CV_64F); // 式(52)の左辺2×2行列
    cv::Mat B_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F); // 式(45)の右辺
    cv::Mat B_translation = cv::Mat::zeros(translation_matrix_dim, 1, CV_64F); // 式(52)の右辺
    cv::Mat delta_uv_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F); // 式(45)の左辺 delta
    cv::Mat delta_uv_translation = cv::Mat::zeros(translation_matrix_dim, 1, CV_64F); // 式(52)の右辺 delta

    double MSE_warping, MSE_translation;
    double min_error_warping = 1E6, min_error_translation = 1E6;
    double max_PSNR_warping = -1, max_PSNR_translation = -1;

    cv::Point2f p0, p1, p2;
    std::vector<cv::Point2f> max_v_warping;
    cv::Point2f max_v_translation;

    std::vector<std::pair<std::vector<cv::Point2f>,double>> v_stack_warping;
    std::vector<std::pair<cv::Point2f,double>> v_stack_translation;
    std::vector<cv::Point2f> pixels_in_triangle;

    bool translation_flag = true;

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
#if GAUSS_NEWTON_HEVC_IMAGE
                    error_tmp += abs(expand_image[0][3][1][4 * (int) (pixel.x + bx)][4 * (int) (pixel.y + by)] -
                                     expand_image[0][3][3][4 * (int) (pixel.x)][4 * (int) (pixel.y)]);
#else
                    error_tmp += abs(expand_image[0][3][1][(int) (pixel.x + bx)][(int) (pixel.y + by)] -
                                                     expand_image[0][3][3][(int) (pixel.x)][(int) (pixel.y)]);
#endif
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
        cv::Point2f tmp_mv_translation(initial_vector.x, initial_vector.y);

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

            int spread = SERACH_RANGE; // 探索範囲は16までなので16に戻す

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
            v_stack_warping.clear();

            double prev_error_warping = 1e6, prev_error_translation = 1e6;
            cv::Point2f prev_mv_translation;
            std::vector<cv::Point2f> prev_mv_warping;
            bool warping_update_flag = true, translation_update_flag = true;

            int iterate_counter = 0;
            while(true){
                // 移動後の座標を格納する
                std::vector<cv::Point2f> ref_coordinates_warping;
                std::vector<cv::Point2f> ref_coordinates_translation;

                ref_coordinates_warping.emplace_back(p0);
                ref_coordinates_warping.emplace_back(p1);
                ref_coordinates_warping.emplace_back(p2);

                ref_coordinates_translation.emplace_back(p0);
                ref_coordinates_translation.emplace_back(p1);
                ref_coordinates_translation.emplace_back(p2);

                cv::Point2f a = p2 - p0;
                cv::Point2f b = p1 - p0;
                double det = a.x * b.y - a.y * b.x;
                // tmp_mv_warping, tmp_mv_translationは現在の動きベクトル
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

                MSE_translation = MSE_warping = 0.0;
                gg_warping = cv::Mat::zeros(warping_matrix_dim, warping_matrix_dim, CV_64F);
                B_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F);
                delta_uv_warping = cv::Mat::zeros(warping_matrix_dim, 1, CV_64F);

                gg_translation = cv::Mat::zeros(translation_matrix_dim, translation_matrix_dim, CV_64F);
                B_translation = cv::Mat::zeros(translation_matrix_dim, 1, CV_64F);
                delta_uv_translation = cv::Mat::zeros(translation_matrix_dim, 1, CV_64F);

                double delta_g_warping[warping_matrix_dim] = {0};
                double delta_g_translation[translation_matrix_dim] = {0};

                cv::Point2f X;
                double RMSE_warping_filter = 0;
                double RMSE_translation_filter = 0;
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
                    double g_x_translation;
                    double g_y_translation;
                    cv::Point2f X_later_translation, X_later_warping;

                    for(int i = 0 ; i < 6 ; i++) {
                        // 移動後の頂点を計算し格納
                        ref_coordinates_warping[0] = p0 + tmp_mv_warping[0];
                        ref_coordinates_warping[1] = p1 + tmp_mv_warping[1];
                        ref_coordinates_warping[2] = p2 + tmp_mv_warping[2];
                        ref_coordinates_translation[0] = p0 + tmp_mv_translation;
                        ref_coordinates_translation[1] = p1 + tmp_mv_translation;
                        ref_coordinates_translation[2] = p2 + tmp_mv_translation;

                        std::vector<cv::Point2f> triangle_later_warping;
                        std::vector<cv::Point2f> triangle_later_translation;
                        triangle_later_warping.emplace_back(ref_coordinates_warping[0]);
                        triangle_later_warping.emplace_back(ref_coordinates_warping[1]);
                        triangle_later_warping.emplace_back(ref_coordinates_warping[2]);
                        triangle_later_translation.emplace_back(ref_coordinates_translation[0]);
                        triangle_later_translation.emplace_back(ref_coordinates_translation[1]);
                        triangle_later_translation.emplace_back(ref_coordinates_translation[2]);

                        cv::Point2f a_later_warping, a_later_translation;
                        cv::Point2f b_later_warping, b_later_translation;

                        a_later_warping  =  triangle_later_warping[2] -  triangle_later_warping[0];
                        a_later_translation = triangle_later_translation[2] - triangle_later_translation[0];
                        b_later_warping  =  triangle_later_warping[1] -  triangle_later_warping[0];
                        b_later_translation = triangle_later_translation[1] - triangle_later_translation[0];
                        X_later_warping  = alpha *  a_later_warping + beta *  b_later_warping +  triangle_later_warping[0];
                        X_later_translation = alpha * a_later_translation + beta * b_later_translation + triangle_later_translation[0];

                        if(X_later_warping.x >= current_ref_image.cols - 1 + scaled_spread) X_later_warping.x = current_ref_image.cols - 1.00 + scaled_spread;
                        if(X_later_warping.y >= current_ref_image.rows - 1 + scaled_spread) X_later_warping.y = current_ref_image.rows - 1.00 + scaled_spread;
                        if(X_later_warping.x < -scaled_spread) X_later_warping.x = -scaled_spread;
                        if(X_later_warping.y < -scaled_spread) X_later_warping.y = -scaled_spread;

                        if(X_later_translation.x >= (current_ref_image.cols - 1 + scaled_spread)) X_later_translation.x = current_ref_image.cols - 1 + scaled_spread;
                        if(X_later_translation.y >= (current_ref_image.rows - 1 + scaled_spread)) X_later_translation.y = current_ref_image.rows - 1 + scaled_spread;
                        if(X_later_translation.x < -scaled_spread) X_later_translation.x = -scaled_spread;
                        if(X_later_translation.y < -scaled_spread) X_later_translation.y = -scaled_spread;

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
                        double g_x, g_y;
#if GAUSS_NEWTON_HEVC_IMAGE
                        g_x          = (img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_warping.x  + 1), 4 * (X_later_warping.y     ), 1) - img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_warping.x  - 1), 4 * (X_later_warping.y     ), 1)) / 2.0;  // (current_ref_expand[x_warping_tmp + 4 ][y_warping_tmp     ] - current_ref_expand[x_warping_tmp - 4 ][y_warping_tmp     ]) / 2.0;
                        g_y          = (img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_warping.x     ), 4 * (X_later_warping.y  + 1), 1) - img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_warping.x     ), 4 * (X_later_warping.y  - 1), 1)) / 2.0;  // (current_ref_expand[x_warping_tmp     ][y_warping_tmp + 4 ] - current_ref_expand[x_warping_tmp     ][y_warping_tmp - 4 ]) / 2.0;
                        g_x_translation = (img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_translation.x + 1), 4 * (X_later_translation.y    ), 1) - img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_translation.x - 1), 4 * (X_later_translation.y    ), 1)) / 2.0;  // (current_ref_expand[x_translation_tmp + 4][y_translation_tmp    ] - current_ref_expand[x_translation_tmp - 4][y_translation_tmp    ]) / 2.0;
                        g_y_translation = (img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_translation.x    ), 4 * (X_later_translation.y + 1), 1) - img_ip(current_ref_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * (X_later_translation.x    ), 4 * (X_later_translation.y - 1), 1)) / 2.0;  // (current_ref_expand[x_translation_tmp    ][y_translation_tmp + 4] - current_ref_expand[x_translation_tmp    ][y_translation_tmp - 4]) / 2.0;
#else
                        g_x   = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x  + 1 , X_later_warping.y    , 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x  - 1, X_later_warping.y     , 1)) / 2.0;  // (current_ref_expand[x_warping_tmp + 4 ][y_warping_tmp     ] - current_ref_expand[x_warping_tmp - 4 ][y_warping_tmp     ]) / 2.0;
                            g_y   = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x     , X_later_warping.y  + 1, 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_warping.x     , X_later_warping.y  - 1, 1)) / 2.0;  // (current_ref_expand[x_warping_tmp     ][y_warping_tmp + 4 ] - current_ref_expand[x_warping_tmp     ][y_warping_tmp - 4 ]) / 2.0;
                            g_x_translation = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x + 1, X_later_translation.y    , 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x - 1, X_later_translation.y    , 1)) / 2.0;  // (current_ref_expand[x_translation_tmp + 4][y_translation_tmp    ] - current_ref_expand[x_translation_tmp - 4][y_translation_tmp    ]) / 2.0;
                            g_y_translation = (img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x    , X_later_translation.y + 1, 1) - img_ip(current_ref_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x    , X_later_translation.y - 1, 1)) / 2.0;  // (current_ref_expand[x_translation_tmp    ][y_translation_tmp + 4] - current_ref_expand[x_translation_tmp    ][y_translation_tmp - 4]) / 2.0;
#endif
                        spread-=1;

                        // 式(28)～(33)
                        delta_g_warping[i] = g_x * delta_x + g_y * delta_y;
                    }
                    delta_g_translation[0] = g_x_translation;
                    delta_g_translation[1] = g_y_translation;

                    double f;
                    double f_org;
                    double g_warping;
                    double g_translation;

#if GAUSS_NEWTON_HEVC_IMAGE
                    f              = img_ip(current_target_expand    , cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *                X.x, 4 *                X.y, 1);
                    f_org          = img_ip(current_target_org_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *                X.x, 4 *                X.y, 1);
                    g_warping      = img_ip(current_ref_expand       , cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *  X_later_warping.x, 4 *  X_later_warping.y, 1);
                    g_translation     = img_ip(current_ref_expand       , cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * X_later_translation.x, 4 * X_later_translation.y, 1);
#else
                    f              = img_ip(current_target_expand    , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),                X.x,                X.y, 2);
                    f_org          = img_ip(current_target_org_expand, cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),                X.x,                X.y, 2);
                    g_warping      = img_ip(current_ref_expand       , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)),  X_later_warping.x,  X_later_warping.y, 2);
                    g_translation     = img_ip(current_ref_expand       , cv::Rect(-spread, -spread, (current_target_image.cols + 2 * spread), (current_target_image.rows + 2 * spread)), X_later_translation.x, X_later_translation.y, 2);
#endif
                    double g_org_warping;
                    double g_org_translation;
                    RMSE_warping_filter += fabs(f - g_warping);
                    RMSE_translation_filter += fabs(f - g_translation);

                    cv::Point2f tmp_X_later_warping, tmp_X_later_translation;
                    tmp_X_later_warping.x = X_later_warping.x;
                    tmp_X_later_warping.y = X_later_warping.y;
                    tmp_X_later_translation.x = X_later_translation.x;
                    tmp_X_later_translation.y = X_later_translation.y;

                    tmp_X_later_warping = roundVecQuarter(tmp_X_later_warping);
                    tmp_X_later_translation = roundVecQuarter(tmp_X_later_translation);

                    if(ref_hevc != nullptr) {
                        g_org_warping  = img_ip(ref_hevc, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * tmp_X_later_warping.x,  4 * tmp_X_later_warping.y, 1);
                        g_org_translation = img_ip(ref_hevc, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * tmp_X_later_translation.x, 4 * tmp_X_later_translation.y, 1);
                    }else {
#if GAUSS_NEWTON_HEVC_IMAGE
                        g_org_warping  = img_ip(current_ref_org_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 *  tmp_X_later_warping.x, 4 *  tmp_X_later_warping.y, 1);
                        g_org_translation = img_ip(current_ref_org_expand, cv::Rect(-4 * spread, -4 * spread, 4 * (current_target_image.cols + 2 * spread), 4 * (current_target_image.rows + 2 * spread)), 4 * tmp_X_later_translation.x, 4 * tmp_X_later_translation.y, 1);
#else
                        g_org_warping  = img_ip(current_ref_org_expand, cv::Rect(-spread, -spread, current_target_image.cols + 2 * spread, current_target_image.rows + 2 * spread),  tmp_X_later_warping.x, tmp_X_later_warping.y, 2);
                            g_org_translation = img_ip(current_ref_org_expand, cv::Rect(-spread, -spread, current_target_image.cols + 2 * spread, current_target_image.rows + 2 * spread), tmp_X_later_translation.x, tmp_X_later_translation.y, 2);
#endif
                    }

                    if(iterate_counter > 4){
                        f = f_org;
                        g_warping = g_org_warping;
                        g_translation = g_org_translation;
                    }

                    for (int row = 0; row < warping_matrix_dim; row++) {
                        for (int col = 0; col < warping_matrix_dim; col++) {
                            gg_warping.at<double>(row, col) += delta_g_warping[row] * delta_g_warping[col];//A_0の行列を生成(左辺の6x6の行列に相当)
                        }
                        B_warping.at<double>(row, 0) += (f - g_warping) * delta_g_warping[row];//bの行列を生成(右辺の6x1のベクトルに相当)
                    }
                    for (int row = 0; row < 2; row++) {
                        for (int col = 0; col < 2; col++) {
                            gg_translation.at<double>(row, col) += delta_g_translation[row] * delta_g_translation[col];
                        }
                        B_translation.at<double>(row, 0) += (f - g_translation) * delta_g_translation[row];
                    }

                    MSE_warping += fabs(f_org - g_org_warping);   // * (f_org - g_org_warping);
                    MSE_translation += fabs(f_org - g_org_translation); // * (f_org - g_org_translation);
                }

                double mu = 10;
                for(int row = 0 ; row < warping_matrix_dim ; row++){
                    for(int col = 0 ; col < warping_matrix_dim ; col++) {
                        gg_warping.at<double>(row, col) += mu * ((area_after_move * area_after_move * area_after_move - area_before_move * area_before_move * area_after_move) / (area_before_move * area_before_move * area_before_move * area_before_move) * S[row]);
                    }
                }

                double Error_warping = MSE_warping;
                double Error_translation = MSE_translation;
                double mu2 = pixels_in_triangle.size() * 0.0001;
                gg_translation.at<double>(0, 0) += 4 * mu2 * tmp_mv_translation.x * tmp_mv_translation.x;
                gg_translation.at<double>(0, 1) += 4 * mu2 * tmp_mv_translation.x * tmp_mv_translation.y;
                gg_translation.at<double>(1, 0) += 4 * mu2 * tmp_mv_translation.y * tmp_mv_translation.x;
                gg_translation.at<double>(1, 1) += 4 * mu2 * tmp_mv_translation.y * tmp_mv_translation.y;
                B_translation.at<double>(0, 0) -= 2 * mu2 * tmp_mv_translation.x * (tmp_mv_translation.x * tmp_mv_translation.x + tmp_mv_translation.y * tmp_mv_translation.y);
                B_translation.at<double>(1, 0) -= 2 * mu2 * tmp_mv_translation.y * (tmp_mv_translation.x * tmp_mv_translation.x + tmp_mv_translation.y * tmp_mv_translation.y);
                cv::solve(gg_warping, B_warping, delta_uv_warping); //6x6の連立方程式を解いてdelta_uvに格納
                v_stack_warping.emplace_back(tmp_mv_warping, Error_warping);

                for (int k = 0; k < 6; k++) {

                    if (k % 2 == 0) {
                        if ((-scaled_spread <= scaled_coordinates[(int) (k / 2)].x + tmp_mv_warping[(int) (k / 2)].x +delta_uv_warping.at<double>(k, 0)) &&
                            (target_images[0][step].cols - 1 + scaled_spread >=scaled_coordinates[(int) (k / 2)].x + tmp_mv_warping[(int) (k / 2)].x + delta_uv_warping.at<double>(k, 0))) {
                            tmp_mv_warping[(int) (k / 2)].x = tmp_mv_warping[(int) (k / 2)].x + delta_uv_warping.at<double>(k, 0);//動きベクトルを更新(画像の外に出ないように)
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

                cv::solve(gg_translation, B_translation, delta_uv_translation);
                v_stack_translation.emplace_back(tmp_mv_translation, Error_translation);

                for (int k = 0; k < 2; k++) {
                    if (k % 2 == 0) {
                        if ((-scaled_spread <=
                             scaled_coordinates[0].x + tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0)) &&
                            (target_images[0][step].cols - 1 + scaled_spread >=
                             scaled_coordinates[0].x + tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0)) &&
                            (-scaled_spread <=
                             scaled_coordinates[1].x + tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0)) &&
                            (target_images[0][step].cols - 1 + scaled_spread >=
                             scaled_coordinates[1].x + tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0)) &&
                            (-scaled_spread <=
                             scaled_coordinates[2].x + tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0)) &&
                            (target_images[0][step].cols - 1 + scaled_spread >=
                             scaled_coordinates[2].x + tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0))) {
                            tmp_mv_translation.x = tmp_mv_translation.x + delta_uv_translation.at<double>(k, 0);
                        }
                    } else {
                        if ((-scaled_spread <=
                             scaled_coordinates[0].y + tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0)) &&
                            (target_images[0][step].rows - 1 + scaled_spread >=
                             scaled_coordinates[0].y + tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0)) &&
                            (-scaled_spread <=
                             scaled_coordinates[1].y + tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0)) &&
                            (target_images[0][step].rows - 1 + scaled_spread >=
                             scaled_coordinates[1].y + tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0)) &&
                            (-scaled_spread <=
                             scaled_coordinates[2].y + tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0)) &&
                            (target_images[0][step].rows - 1 + scaled_spread >=
                             scaled_coordinates[2].y + tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0))) {
                            tmp_mv_translation.y = tmp_mv_translation.y + delta_uv_translation.at<double>(k, 0);
                        }
                    }
                }

                double eps = 1e-3;
                if(((fabs(prev_error_translation - MSE_translation) / MSE_translation) < eps && (fabs(prev_error_warping - MSE_warping) / MSE_warping < eps)) || (!translation_update_flag && !warping_update_flag) || iterate_counter > 20){
                    break;
                }

                prev_error_translation = MSE_translation;
                prev_error_warping = MSE_warping;
                prev_mv_translation = tmp_mv_translation;
                prev_mv_warping = tmp_mv_warping;
                iterate_counter++;
            }

            std::sort(v_stack_warping.begin(), v_stack_warping.end(), [](std::pair<std::vector<cv::Point2f>,double> a, std::pair<std::vector<cv::Point2f>,double> b){
                return a.second < b.second;
            });

            std::sort(v_stack_translation.begin(), v_stack_translation.end(), [](std::pair<cv::Point2f,double> a, std::pair<cv::Point2f,double> b){
                return a.second < b.second;
            });

            tmp_mv_warping = v_stack_warping[0].first;//一番良い動きベクトルを採用
            double Error_warping = v_stack_warping[0].second;
            tmp_mv_translation = v_stack_translation[0].first;
            double Error_translation = v_stack_translation[0].second;
            MSE_warping = Error_warping / (double)pixels_in_triangle.size();
            MSE_translation = Error_translation / (double)pixels_in_triangle.size();
            double PSNR_warping = 10 * log10((255 * 255) / MSE_warping);
            double PSNR_translation = 10 * log10((255 * 255) / MSE_translation);

            if(step == 3) {//一番下の階層で
                if(PSNR_translation >= max_PSNR_translation){//2種類のボケ方で良い方を採用
                    max_PSNR_translation = PSNR_translation;
                    min_error_translation = Error_translation;
                    max_v_translation = tmp_mv_translation;
                }
                if (PSNR_warping >= max_PSNR_warping) {
                    max_PSNR_warping = PSNR_warping;
                    min_error_warping = Error_warping;
                    max_v_warping = tmp_mv_warping;
                }

                if (fabs(max_PSNR_warping - max_PSNR_translation) <= 0.5 || max_PSNR_translation > max_PSNR_warping) {//ワーピングと平行移動でRDのようなことをする
                    translation_flag = true;//平行移動を採用
                } else{
                    translation_flag = false;//ワーピングを採用
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
