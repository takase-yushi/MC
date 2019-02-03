/**
 * @file psnr.h
 * @brief psnr.cppのヘッダファイル
 * @author Keisuke KAMIYA
 */


#ifndef DELAUNAY_PSNR_H
#define DELAUNAY_PSNR_H

#include <opencv2/core.hpp>


/**
 * @fn double getMSE(cv::Mat in1, cv::Mat in2)
 * @brief MSEを求める
 * @param[in] in1 Mat型の画像データ1
 * @param[in] in2 Mat型の画像データ2
 * @details
 *  画像1と2の平均二乗誤差を返す。
 */
double getMSE(const cv::Mat& in1, const cv::Mat& in2, cv::Rect rect);

/**
 * @fn double getPSNR(cv::Mat in1, cv::Mat in2)
 * @brief PSNRを求める
 * @param[in] in1 Mat型の画像データ1
 * @param[in] in2 Mat型の画像データ2
 * @return double PSNRを返す
 * @details
 *  2つの画像のPSNR値(Peak Signal-to-Noise Ratio)を返す
 */
double getPSNR(const cv::Mat& in1, const cv::Mat& in2, const cv::Rect& rect = cv::Rect(0, 0, 1920, 1024));

#endif //DELAUNAY_PSNR_H
