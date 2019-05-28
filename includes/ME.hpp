/**
 * @file ME.h
 * @brief ME.cpp用のヘッダファイル
 * @author Keisuke KAMIYA
 */

#ifndef ENCODER_ME_H
#define ENCODER_ME_H

#include <opencv2/core/mat.hpp>
#include "config.h"
#include <vector>
#include <queue>
#include "Utils.h"
#include "../includes/DelaunayTriangulation.hpp"

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
void block_matching(const cv::Mat& prev, const cv::Mat& current, double &error, cv::Point2f &mv, Point3Vec tr, cv::Mat expansion_image);

/**
 * @fn std::vector<cv::Point2f> warping(cv::Mat &prev_gray, cv::Mat &current_gray, cv::Mat &prev_color, cv::Mat &current_color,
                                 double &error_warp, Point3Vec target_corners, Point3Vec ref_corners, std::vector<uchar> prev_status)
 * @brief ワーピングを行い、動きベクトルと誤差を返す
 * @param[in]  prev_gray      参照画像のグレースケール画像
 * @param[in]  current_gray   対象画像のグレースケール画像
 * @param[in]  prev_color     参照画像のカラー画像
 * @param[in]  current_color  対象画像のグレースケール画像
 * @param[out] error_warp     ワーピングの誤差
 * @param[in]  corners        三角点を構成する3つの頂点
 * @return 三角形3点の移動先座標prev_cornersを返す. corners[i]の動きベクトルはprev_corners[i]に格納される.
 */
std::vector<cv::Point2f> warping(const cv::Mat& prev_color, const cv::Mat& current_color,
                                 double &error_warp, Point3Vec target_corners, Point3Vec& ref_corners);

double Gauss_Newton(const cv::Mat& prev_color, const cv::Mat& current_color,const cv::Mat& intra,
                                      Point3Vec target_corners, Point3Vec& ref_corners,int &triangle_size);

std::vector<cv::Point2i> Gauss_Newton2(const cv::Mat& prev_color,const cv::Mat& current_color,const cv::Mat& intra,std::vector<cv::Mat>& predict_buf,cv::Mat&predict_warp,cv::Mat&predict_para,
                                       double &error_warp, Point3Vec target_corners, Point3Vec& ref_corners,
                                       bool *flag,int t,const cv::Mat& residual_ref,int& in_triangle_size,double erase_th_global);

std::vector<cv::Point2i> Gauss_Golomb(Triangle triangle, bool *flag, std::vector<cv::Point2i> &ev, std::vector<cv::Point2f> corners, DelaunayTriangulation md,std::vector<cv::Point2i> mv,const cv::Mat& target,const cv::Mat& targetx8,bool &para_flag);
/**
 * @fn std::vector<cv::Point2f> getReferenceImageCoordinates(const cv::Mat& image, const std::vector<cv::Point2f>& target_corners, const int side_point_nums);
 * @brief 特徴点が参照画像でどの位置に属するのかを求める.(ブロックマッチングでフローをもとめる）
 * @param image
 * @param target_corners
 * @return std::vector<cv::Point2f> ref_corners 参照画像の特徴点群
 */
std::pair<std::vector<cv::Point2f>, std::priority_queue<int>> getReferenceImageCoordinates(const cv::Mat &ref,
                                                                                           const cv::Mat &target,
                                                                                           const std::vector<cv::Point2f> &target_corners,
                                                                                           cv::Mat &debug);

double getPredictedImage(cv::Mat& ref_image, cv::Mat& target_image, cv::Mat& output_image, Point3Vec& triangle, std::vector<cv::Point2f>& mv, bool parallel_flag);
std::tuple<std::vector<cv::Point2f>, cv::Point2f, double, int, bool> GaussNewton(cv::Mat ref_image, cv::Mat target_image, cv::Mat gauss_ref_image, Point3Vec target_corners);

#endif //ENCODER_ME_H
