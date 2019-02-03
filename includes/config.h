/**
 * @file config.h
 * @brief 各種パラメータの設定ファイル
 * @author Keisuke KAMIYA
 */

#ifndef ENCODER_CONFIG_H
#define ENCODER_CONFIG_H

#include <opencv2/core.hpp>

/**
 * @typedef std::pair<double, cv::Point2f> pdp
 * @brief Pair<Double Point2f>の略
 * @details
 *  pair<double, cv::Point2f> それぞれの頭文字をとってpdfとした. このようなことは本来してはいけない.
 */
typedef std::pair<double, cv::Point2f> pdp;

/**
 * @def GFTT_QUAULITY
 * @brief goodFeaturesToTrack()に使用するクオリティ
 * @details
 *  特徴点追跡に向いた特徴点抽出であるgoodFeaturesToTrack()に設定するクオリティレベル
 */
#define GFTT_QUAULITY 0.03

/**
 * @def QUANTIZE
 * @brief 量子化ステップ幅
 * @details
 *  特徴点を量子化する際のステップ幅Δを定義
 */
#define QUANTIZE 4

/**
 * @def BM_AVAILABLE
 * @brief ブロックマッチング有効化フラグ
 * @details
 *  ブロックマッチングを有効にする場合はtrue, そうでない場合はfalseを設定する
 */
#define BM_AVAILABLE true

/**
 * @def WARP_AVAILABLE
 * @brief ワーピング有効化フラグ
 * @details
 *  ワーピングを有効にする場合はtrue, そうでない場合はfalseを設定する
 */
#define WARP_AVAILABLE true

/**
 * @def EPS
 * @brief 誤差を表す定数
 * @details
 *  誤差を表す定数EPS(EPSILON). 1×10^(-9)を表す.
 */
#define EPS 1e-9

#endif //ENCODER_CONFIG_H
