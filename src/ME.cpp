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
#include "../includes/DelaunayTriangulation.hpp"
#include "../includes/Utils.h"

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

    const double th = 0.5;                           //ワーピングか平行移動を選択するための閾値
    double MSE,MSE_para,Error,Error_min = 0,Error_para,Error_para_min = 0; //予測残差諸々
    double PSNR,PSNR_max = 0,PSNR_para,PSNR_para_max = 0;

    float delta_x, delta_y;//頂点を動かしたときのパッチ内の変動量x軸y軸独立に計算(delta_gを求めるために必要)
    std::vector<cv::Point2f> triangle, triangle_later,triangle_later_para;//三角形の頂点を格納
    std::vector<cv::Point2f> in_triangle_pixel, in_triangle_later,in_triangle_later_para;//三角形の内部の座標を格納
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

        p0 = target_corners.p1;
        p1 = target_corners.p2;
        p2 = target_corners.p3;
        for (int z = 0; z < 4; z++) {//各階層ごとに
            double scale = pow(2, 3-z);//各階層のスケーリングの値
            cv::Mat f_img = mv_filter(f_[blare][z], 2);//対照画像
            cv::Mat g_img = mv_filter(g_[blare][z], 2);//参照画像
            cv::Mat point;
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
            //////////////ここからは頂点の距離が近い場合にバグってたときに迷走してました(近い距離に頂点を取らないようにしてバグを回避したため頂点を分割していく場合またバグが発生する可能性大)
            cv::Point2f a_,b_;
            double S_prev,S_later;//パッチの面積(デバッグ変数)
            a_.x = p2.x - p0.x;
            a_.y = p2.y - p0.y;
            b_.x = p1.x - p0.x;
            b_.y = p1.y - p0.y;
            S_prev = a_.x * b_.y - a_.y * b_.x;
            if(fabs((int)a_.x * (int)b_.y - (int)a_.y * (int)b_.x) <= 4){//パッチの面積が小さすぎる場合は動きベクトルをゼロにする
                for (int s = 0; s < 3; s++) {
                    v[s].x = 0;
                    v[s].y = 0;
                }
                v_para.x = 0;
                v_para.y = 0;
                continue;
            }
            a_.x = p2.x - p1.x;
            a_.y = p2.y - p1.y;
            b_.x = p0.x - p1.x;
            b_.y = p0.y - p1.y;
            if(fabs((int)a_.x * (int)b_.y - (int)a_.y * (int)b_.x) <= 4){//パッチの面積が小さすぎる場合は動きベクトルをゼロにする
                for (int s = 0; s < 3; s++) {
                    v[s].x = 0;
                    v[s].y = 0;
                }
                v_para.x = 0;
                v_para.y = 0;
                continue;
            }
            a_.x = (p2.x + v[2].x) - (p0.x + v[0].x);
            a_.y = (p2.y + v[2].y) - (p0.y + v[0].y);
            b_.x = (p1.x + v[1].x) - (p0.x + v[0].x);
            b_.y = (p1.y + v[1].y) - (p0.y + v[0].y);
            S_later = a_.x * b_.y - a_.y * b_.x;

            ///////////迷走終わり
            if (target_corners.p1.x == f_[0][3].cols - 1)p0.x = f_[0][z].cols - 1;//端の頂点の調整
            if (target_corners.p1.y == f_[0][3].rows - 1)p0.y = f_[0][z].rows - 1;
            if (target_corners.p2.x == f_[0][3].cols - 1)p1.x = f_[0][z].cols - 1;
            if (target_corners.p2.y == f_[0][3].rows - 1)p1.y = f_[0][z].rows - 1;
            if (target_corners.p3.x == f_[0][3].cols - 1)p2.x = f_[0][z].cols - 1;
            if (target_corners.p3.y == f_[0][3].rows - 1)p2.y = f_[0][z].rows - 1;
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

                Point3Vec triangleVec = Point3Vec(p0, p1, p2);
                double sx = std::min({(int) p0.x, (int) p1.x, (int) p2.x});
                double lx = std::max({(int) p0.x, (int) p1.x, (int) p2.x});
                double sy = std::min({(int) p0.y, (int) p1.y, (int) p2.y});
                double ly = std::max({(int) p0.y, (int) p1.y, (int) p2.y});
                if(lx - sx == 0 || ly - sy == 0){
                    std::cout << "baund = 0" << std::endl;
                }
                in_triangle_pixel.clear();
                for (int j = (int) (round(sy) - 1); j <= round(ly) + 1; j++) {
                    for (int i = (int) (round(sx) - 1); i <= round(lx) + 1; i++) {
                        xp.x = (float) i;
                        xp.y = (float) j;
                        if (isInTriangle(triangleVec, xp) == 1) {
                            in_triangle_pixel.emplace_back(xp);//三角形の内部のピクセルを格納
                        }
                    }
                }

                cv::Point2f a, b, X,c,d;
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
                for (int m = 0; m < (int) in_triangle_pixel.size(); m++) {//パッチ内の画素ごとに
                    X.x = in_triangle_pixel[m].x - p0.x;
                    X.y = in_triangle_pixel[m].y - p0.y;
                    alpha = (X.x * b.y - X.y * b.x) / det;
                    beta = (a.x * X.y - a.y * X.x) / det;

                    X.x += p0.x;
                    X.y += p0.y;
                    x0 = (int) floor(X.x);
                    y0 = (int) floor(X.y);
                    d_x = X.x - x0;
                    d_y = X.y - y0;

                    for (int i = 0; i < 6; i++) {
                        pp[0].x = p0.x + v[0].x;
                        pp[0].y = p0.y + v[0].y;
                        pp[1].x = p1.x + v[1].x;
                        pp[1].y = p1.y + v[1].y;
                        pp[2].x = p2.x + v[2].x;
                        pp[2].y = p2.y + v[2].y;
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

                    x0_later = (int) floor(X_later.x);
                    y0_later = (int) floor(X_later.y);
                    x0_later_para = (int)floor(X_later_para.x);
                    y0_later_para = (int)floor(X_later_para.y);
                    d_x_later = X_later.x - x0_later;
                    d_y_later = X_later.y - y0_later;
                    d_x_later_para = X_later_para.x - x0_later_para;
                    d_y_later_para = X_later_para.y - y0_later_para;
                    f = f_expand[x0][y0] * (1 - d_x) * (1 - d_y) + f_expand[x0 + 1][y0] * d_x * (1 - d_y) +
                        f_expand[x0][y0 + 1] * (1 - d_x) * d_y + f_expand[x0 + 1][y0 + 1] * d_x * d_y;

                    g = g_expand[x0_later][y0_later] * (1 - d_x_later) * (1 - d_y_later) +
                        g_expand[x0_later + 1][y0_later] * d_x_later * (1 - d_y_later) +
                        g_expand[x0_later][y0_later + 1] * (1 - d_x_later) * d_y_later +
                        g_expand[x0_later + 1][y0_later + 1] * d_x_later * d_y_later;//頂点を移動させた後のワーピングの参照フレームの輝度値

                    g_para = g_expand[x0_later_para][y0_later_para] * (1 - d_x_later_para) * (1 - d_y_later_para) +
                             g_expand[x0_later_para + 1][y0_later_para] * d_x_later_para * (1 - d_y_later_para) +
                             g_expand[x0_later_para][y0_later_para + 1] * (1 - d_x_later_para) * d_y_later_para +
                             g_expand[x0_later_para + 1][y0_later_para + 1] * d_x_later_para * d_y_later_para;//頂点を移動させた後の平行移動の参照フレームの輝度値

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
                triangle_size = (int)in_triangle_pixel.size();
                MSE = (in_triangle_pixel.size() == 0 ? MSE : MSE / in_triangle_pixel.size());//パッチ内の平均2乗誤差
                MSE_para = (in_triangle_pixel.size() == 0 ? MSE_para : MSE_para / in_triangle_pixel.size());
                PSNR = 10 * log10((255 * 255) / MSE);//パッチ内のPSNR
                PSNR_para = 10 * log10((255 * 255) / MSE_para);

                cv::solve(gg, B, delta_uv);//6x6の連立方程式を解いてdelta_uvに格納
                cv::solve(gg_para,B_para,delta_uv_para);
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


            v = v_stack[0].first;//一番良い動きベクトルを採用
            Error = v_stack[0].second;
            v_para = v_stack_para[0].first;
            Error_para = v_stack_para[0].second;
            MSE = Error / (double)in_triangle_pixel.size();
            MSE_para = Error_para / (double)in_triangle_pixel.size();
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
                if (fabs(PSNR_max - PSNR_para_max) <= th || PSNR_para_max > PSNR_max) {//ワーピングと平行移動でRDのようなことをする
                    parallel_flag = true;//平行移動を採用
                } else{
                    parallel_flag = false;//ワーピングを採用
                }
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
            std::vector<cv::Point2f> mv2, mv3,mv_diff_tmp;
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
                //ave_v *= Quant;
                ave_v.x = (int) floor(ave_v.x + 0.5);
                ave_v.y = (int) floor(ave_v.y + 0.5);
                //ave_v /= Quant;
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
                mv3.emplace_back(v_max[0]);
                mv3.emplace_back(v_max[1]);
                mv3.emplace_back(v_max[2]);
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
                    mv[0] = v_para_max;
                    double d_x,d_y;
                    d_x = v_para_max.x - mv[0].x;
                    d_y = v_para_max.y - mv[0].y;
                    d_x *= Quant;
                    d_y *= Quant;
                    d_x = (int)floor(d_x + 0.5);
                    d_y = (int)floor(d_y + 0.5);
                    cv::Point2i dv(d_x,d_y);
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

            for (int i = 0; i < (int) in_triangle_pixel.size(); i++) {
                double d_x, d_y;
                int x0, y0;
                X.x = in_triangle_pixel[i].x - triangle[0].x;
                X.y = in_triangle_pixel[i].y - triangle[0].y;
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

    if(parallel_flag == true){
        Error_min = Error_para_min;
    }
    return Error_min;
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
    g0x1 = prev_color;
    g0x2 = half(g0x1,2);
    g0x4 = half(g0x2,2);
    g0x8 = half(g0x4,2);
    f_0.emplace_back(f0x8);
    f_0.emplace_back(f0x4);
    f_0.emplace_back(f0x2);
    f_0.emplace_back(f0x1);
    g_0.emplace_back(g0x8);
    g_0.emplace_back(g0x4);
    g_0.emplace_back(g0x2);
    g_0.emplace_back(intra);
    cv::Mat fx1,fx2,fx4,fx8,gx1,gx2,gx4,gx8;
    fx1 = current_color;
    fx2 = half(fx1,2);
    fx4 = half(fx2,1);
    fx8 = half(fx4,1);
    gx1 = prev_color;
    gx2 = half(gx1,2);
    gx4 = half(gx2,1);
    gx8 = half(gx4,1);
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
    f_1.emplace_back(fx8);
    f_1.emplace_back(fx4);
    f_1.emplace_back(fx2);
    f_1.emplace_back(fx1);
    g_1.emplace_back(gx8);
    g_1.emplace_back(gx4);
    g_1.emplace_back(gx2);
    g_1.emplace_back(intra);
    f_.emplace_back(f_0);
    f_.emplace_back(f_1);
    f_.emplace_back(f_2);
    g_.emplace_back(g_0);
    g_.emplace_back(g_1);
    g_.emplace_back(g_2);
    cv::Point2f v0(0.0, 0.0), v1(0.0, 0.0), v2(0.0, 0.0);
    const int dim = 6;
    double f, g, g1, g2,g3, ek_tmp, ek, delta_ek[dim],g_para;
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
    int blare_max = 0;
    int img_ip(unsigned char **img, int xs, int ys, double x, double y, int mode);
    double w(double x);
    std::string str;
    float delta_x, delta_y;
    std::vector<cv::Point2f> triangle, triangle_later, triangle_delta,triangle_later_para;
    std::vector<cv::Point2f> in_triangle_pixels, in_triangle_later,in_triangle_later_para;
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
        v_para_max.x = 0;
        v_para_max.y = 0;
        p0 = target_corners.p1;
        p1 = target_corners.p2;
        p2 = target_corners.p3;
        for (int z = 0; z < 4; z++) {
            cv::Point2f aaa = target_corners.p2 - target_corners.p1;
            cv::Point2f bbb = target_corners.p3 - target_corners.p2;
            if(aaa.x == 32 || bbb.x == 32) {
                for(int vectors = 0 ; vectors < v.size() ; vectors++) {
                    v[vectors].x = 0.0; v[vectors].y = 0.0;
                }
                v_para.x = 0.0; v_para.y = 0.0;
                puts("hoge");
                continue;
            }

            if(z == 0 && blare == 0){
                for (int s = 0; s < 3; s++) {
                    v[s].x = 0;
                    v[s].y = 0;
                }
            }
            if(z == 0 && blare == 1){
                for (int s = 0; s < 3; s++) {
                    v[s].x = 0;
                    v[s].y = 0;
                }
            }
            if(z == 1 && blare == 2){
                for (int s = 0; s < 3; s++) {
                    v[s].x = 0;
                    v[s].y = 0;
                }
            }
            int scale = pow(2, 3-z),scale_x = scale,scale_y = scale;
            cv::Mat f_img = f_[blare][z];//対照画像
            cv::Mat g_img = g_[blare][z];//参照画像
            cv::Mat point;
            f_img = mv_filter(f_img,2);
            g_img = mv_filter(g_img,2);
            const int expand = 500;
            unsigned char **f_expand;
            unsigned char **g_expand;
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
            int k = 2;
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

                    cv::Point2f a, b, X,c,d;
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
                            pp[0].x = p0.x + v[0].x;
                            pp[0].y = p0.y + v[0].y;
                            pp[1].x = p1.x + v[1].x;
                            pp[1].y = p1.y + v[1].y;
                            pp[2].x = p2.x + v[2].x;
                            pp[2].y = p2.y + v[2].y;

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
                        //if(t >= 32 && t <= 33)std::cout << "check 20" << std::endl;
                        a_later.x = triangle_later[2].x - triangle_later[0].x;
                        a_later.y = triangle_later[2].y - triangle_later[0].y;
                        a_later_para = triangle_later_para[2] - triangle_later_para[0];
                        //if(t >= 32 && t <= 33)  std::cout << "check 21" << std::endl;
                        //std::cout << "a_later = " << a_later.x << std::endl;
                        b_later.x = triangle_later[1].x - triangle_later[0].x;
                        b_later.y = triangle_later[1].y - triangle_later[0].y;
                        b_later_para = triangle_later_para[1] - triangle_later_para[0];
                        //if(t >= 32 && t <= 33) std::cout << "check 22" << std::endl;
                        X_later.x = alpha * a_later.x + beta * b_later.x + triangle_later[0].x;
                        X_later.y = alpha * a_later.y + beta * b_later.y + triangle_later[0].y;
                        X_later_para = alpha * a_later_para + beta * b_later_para + triangle_later_para[0];

                        x0_later = (int) floor(X_later.x);
                        y0_later = (int) floor(X_later.y);
                        x0_later_para = (int) floor(X_later_para.x);
                        y0_later_para = (int) floor(X_later_para.y);
                        //if(t >= 32 && t <= 33)  std::cout << "check 24" << std::endl;
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

                        g1 = g_expand[x0_later][y0_later] * (1 - d_x_later) * (1 - d_y_later) +
                             g_expand[x0_later + 1][y0_later] * d_x_later * (1 - d_y_later) +
                             g_expand[x0_later][y0_later + 1] * (1 - d_x_later) * d_y_later +
                             g_expand[x0_later + 1][y0_later + 1] * d_x_later * d_y_later;

                        g = g_expand[x0_later][y0_later] * (1 - d_x_later) * (1 - d_y_later) +
                            g_expand[x0_later + 1][y0_later] * d_x_later * (1 - d_y_later) +
                            g_expand[x0_later][y0_later + 1] * (1 - d_x_later) * d_y_later +
                            g_expand[x0_later + 1][y0_later + 1] * d_x_later * d_y_later;

                        g_para = g_expand[x0_later_para][y0_later_para] * (1 - d_x_later_para) * (1 - d_y_later_para) +
                                 g_expand[x0_later_para + 1][y0_later_para] * d_x_later_para * (1 - d_y_later_para) +
                                 g_expand[x0_later_para][y0_later_para + 1] * (1 - d_x_later_para) * d_y_later_para +
                                 g_expand[x0_later_para + 1][y0_later_para + 1] * d_x_later_para * d_y_later_para;

                        g2 = g + delta_g[0] * delta_uv.at<double>(0, 0) + delta_g[1] * delta_uv.at<double>(1, 0) +
                             delta_g[2] * delta_uv.at<double>(2, 0) +
                             delta_g[3] * delta_uv.at<double>(3, 0) + delta_g[4] * delta_uv.at<double>(4, 0) +
                             delta_g[5] * delta_uv.at<double>(5, 0);
                        g3 = g_expand[x0_later_para][y0_later_para] * (1 - d_x_later_para) * (1 - d_y_later_para) +
                             g_expand[x0_later_para + 1][y0_later_para] * d_x_later_para * (1 - d_y_later_para) +
                             g_expand[x0_later_para][y0_later_para + 1] * (1 - d_x_later_para) * d_y_later_para +
                             g_expand[x0_later_para + 1][y0_later_para + 1] * d_x_later_para * d_y_later_para;
                        for (int t = 0; t < dim; t++) {
                            delta_ek[t] += (f - g2) * delta_g[t];
                        }
                        for (int k = 0; k < dim; k++) {
                            for (int j = 0; j < dim; j++) {
                                gg.at<double>(k, j) += delta_g[k] * delta_g[j];
                            }
                            B.at<double>(k, 0) += (f - g) * delta_g[k];
                        }
                        //if(t >= 32 && t <= 33)  std::cout << "check 33" << std::endl;
                        for (int k = 0; k < 2; k++) {
                            for (int j = 0; j < 2; j++) {
                                gg_para.at<double>(k, j) += delta_g_para[k] * delta_g_para[j];
                            }
                            B_para.at<double>(k, 0) += (f - g3) * delta_g_para[k];
                        }
                        //if(t >= 32 && t <= 33) std::cout << "check 34" << std::endl;
                        current_error += (f - g1) * (f - g1);
                        current_error_2 += (f - g2) * (f - g2);

                        ek += (f - g - ek_tmp) * (f - g - ek_tmp);
                        MSE += (f - g) * (f - g);
                        MSE_para += (f - g_para) * (f - g_para);

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
                    /* if(prev_PSNR > PSNR){
                         v = v_prev;
                         break;
                     }*/
                    v_pair.first = v;
                    v_pair.second = PSNR;
                    v_stack.emplace_back(v_pair);
                    v_pair_para.first = v_para;
                    v_pair_para.second = PSNR_para;
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
                *flag = true;
//                if (fabs(PSNR_max - PSNR_para_max) <= th || PSNR_para_max > PSNR_max) {
//                    *flag = true;
//                } else{
//                    *flag = false;
//                }
            }
            //tri_list << "PSNR, " << PSNR << ", PSNR_para," << PSNR_para << std::endl;
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
                        mv3[j] *= Quant;
                        mv3[j].x = (int)mv3[j].x;
                        mv3[j].y = (int)mv3[j].y;
                        mv3[j] /= 2;
                        x0 = (int) mv3[j].x;
                        y0 = (int) mv3[j].y;
                        d_x = mv3[j].x - x0;
                        d_y = mv3[j].y - y0;
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
                    mv4[0] *= Quant;
                    mv4[0].x = (int)mv4[0].x;
                    mv4[0].y = (int)mv4[0].y;
                    mv4[0] /= 2;
                    x0 = (int)mv4[0].x;
                    y0 = (int)mv4[0].y;
                    d_x = mv4[0].x - x0;
                    d_y = mv4[0].y - y0;
                    mv[0].x = x0;
                    mv[0].y = y0;
                    cv::Point2i dv(d_x,d_y);
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
                int y, y1,y_para;

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
    double S = 0.5 * fabs(a.x*b.y - b.x*a.y);

    if (!*flag) {
        error_warp = Error_min;
    } else {
        error_warp = Error_para_min;
    }
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
