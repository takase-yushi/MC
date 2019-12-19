/**
 * @file psnr.cpp
 * @brief PSNRを測定する関数群
 * @author Keisuke KAMIYA
 */
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include "../includes/psnr.h"
#include "../includes/Utils.h"


/**
 * @fn double getMSE(cv::Mat in1, cv::Mat in2)
 * @brief MSEを求める
 * @param[in] in1 Mat型の画像データ1
 * @param[in] in2 Mat型の画像データ2
 * @details
 *  画像1と2の平均二乗誤差を返す。
 */
double getMSE(const cv::Mat& in1, const cv::Mat& in2, cv::Rect rect ){
  double sum = 0.0;

  assert(in1.size == in2.size); // sizeが同じかどうかチェック

  int width = std::min(rect.width, in1.cols);
  int height = std::min(rect.height, in1.rows);

  for(int j = 0 ; j < height ; j++){
    for(int i = 0 ; i < width ; i++){
      sum += (R(in1, rect.x + i, rect.y + j) - R(in2, rect.x + i, rect.y + j)) * (R(in1, rect.x + i, rect.y + j) - R(in2, rect.x + i, rect.y + j));
      sum += (G(in1, rect.x + i, rect.y + j) - G(in2, rect.x + i, rect.y + j)) * (G(in1, rect.x + i, rect.y + j) - G(in2, rect.x + i, rect.y + j));
      sum += (B(in1, rect.x + i, rect.y + j) - B(in2, rect.x + i, rect.y + j)) * (B(in1, rect.x + i, rect.y + j) - B(in2, rect.x + i, rect.y + j));
    }
  }

  return sum / (3.0 * width * height);
}

/**
 * @fn double getPSNR(cv::Mat in1, cv::Mat in2)
 * @brief PSNRを求める
 * @param[in] in1 Mat型の画像データ1
 * @param[in] in2 Mat型の画像データ2
 * @return double PSNRを返す
 * @details
 *  2つの画像のPSNR値(Peak Signal-to-Noise Ratio)を返す
 */
double getPSNR(const cv::Mat& in1, const cv::Mat& in2, const cv::Rect& rect){
  return 10 * std::log10(255.0 * 255.0 / getMSE(in1, in2, rect));
}

