/**
 * @file Utils.h
 * @brief マクロや構造体等々を定義
 * @author Keisuke KAMIYA
 */

#ifndef ENCODER_UTILS_H
#define ENCODER_UTILS_H

#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include "CodingTreeUnit.h"
#include "Config.h"

/**
 * @def R(img, x, y) ((img).data[((int)img.step) * (y) + (x) * 3 + 2])
 * @brief x,y座標に格納されているRの値を返す.
 * @details
 *  Mat型に格納されている画像の(x,y)画素のRの値を返す.
 */
#define R(img, x, y) ((img).data[((int)img.step) * (y) + (x) * 3 + 2])

/**
 * @def G(img, x, y) ((img).data[((int)img.step) * (y) + (x) * 3 + 1])
 * @brief x,y座標に格納されているGの値を返す.
 * @details
 * Mat型に格納されている画像の(x,y)画素のGの値を返す.
 */
#define G(img, x, y) ((img).data[((int)(img.step)) * (y) + (x) * 3 + 1])

/**
 * @def B(img, x, y) ((img)).data[((int)img.step) * (y) + (x) * 3 + 0])
 * @brief x,y座標に格納されているBの値を返す.
 * @details
 *  Mat型に格納されている画像の(x,y)画素のBの値を返す.
 */
#define B(img, x, y) ((img).data[((int)img.step) * (y) + (x) * 3 + 0])

/**
 * @def BLUE  (cv::Scalar(255, 0, 0))
 * @brief 青を表す定数
 * @details
 *  OpenCVで使用する色指定クラスScalarで青色を表したもの.
 */
#define BLUE  (cv::Scalar(255, 0, 0))

/**
 * @def GREEN (cv::Scalar(0, 255, 0))
 * @brief 緑を表す定数
 * @details
 *  OpenCVで使用する色指定クラスScalarで緑色を表したもの.
 */
#define GREEN (cv::Scalar(0, 255, 0))

/**
 * @def RED (cv::Scalar(0, 0, 255))
 * @brief 赤を表す定数
 * @details
 *  OpenCVで使用する色指定クラスScalarで赤色を表したもの.
 */
#define RED   (cv::Scalar(0, 0, 255))

#define  WHITE (cv::Scalar(255, 255, 255))

#define YELLOW (cv::Scalar(0, 255, 255))

#define LIGHT_BLUE (cv::Scalar(255, 255, 0))

#define PURPLE (cv::Scalar(255, 0, 255))

/**
 * @def M(img, x, y) ((0.299 * (double)(R(img, (x), (y))) + 0.587 * (double)(G(img, (x), (y))) + 0.114 * (double)(B(img, (x), (y)))))
 * @brief 指定画素の輝度値を返す.
 * @details
 *  画像imgの(x, y)画素のRGB値から輝度値を計算する. 輝度値は M = 0.299 × R + 0.587 × G + 0.114 × Bとして計算される.
 */
#define M(img, x, y) (int)round(0.299 * (double)(R(img, (x), (y))) + 0.587 * (double)(G(img, (x), (y))) + 0.114 * (double)(B(img, (x), (y))))

/**
 * @def F(img, x, y, f, width) (unsigned char)img[((x) + f) + ((y) * (2 * (f) + (width))]
 * @brief unsigned charの画像（一次元配列）から画素値を返す
 *
 */
#define F(img, x, y, offset, width) img[((x) + (offset)) + (((y) + (offset)) * (2 * (offset) + (width)))]


enum {
    BM,
    NEWTON,
};

#define TEST_MODE false

/**
 * ガウス・ニュートン関連
 */
#define PRED_MODE NEWTON
#define GAUSS_NEWTON_TRANSLATION_ONLY false
#define GAUSS_NEWTON_INIT_VECTOR false
#define GAUSS_NEWTON_HEVC_IMAGE true
#define HEVC_REF_IMAGE true
#define USE_BM_TRANSLATION_MV false
#define MV_LIST_MAX_NUM 5
#define SEARCH_RANGE 16
#define TRANSLATION_COST_RATIO 1.0
#define WARPING_COST_RATIO 1.0
#define WARPING_LIMIT 2
#define MIN_CTU_SIZE 8

/**
 * 各種モードのON/OFF
 */
#define MERGE_MODE true
#define INTRA_MODE false
#define MERGE2_ENABLE true
#define COLLOCATED_ENABLE false
#define INTRA_LIMIT_MODE false // イントラが選択されたらそれ以降は割らないモード

/**
 * ログファイル表示関連
 */
#define MVD_DEBUG_LOG false
#define DISPLAY_RD_COST false
#define DISPLAY_MERGE_LOG false

/**
 * ログファイル書き出し関連
 */
#define STORE_DISTRIBUTION_LOG true
#define STORE_MVD_DISTRIBUTION_LOG false
#define STORE_IMG_LOG true
#define STORE_NEWTON_LOG false
#define STORE_MERGE_LOG false

// 2画素以上のアップデートが来たら打ち切る
#define DELTA_UV_THRESHOLD 2

#define ADAPTIVE_SPLIT false

const std::string OS = "Ubuntu";

/**
 * @struct struct Point3Vec
 * @brief 三点を保持する構造体
 */
struct Point3Vec{
    cv::Point2f p1; ///< 三角形の頂点1の座標
    cv::Point2f p2; ///< 三角形の頂点2の座標
    cv::Point2f p3; ///< 三角形の頂点3の座標


    /**
     * コンストラクタ
     * @param p1 三角形の頂点1の座標
     * @param p2 三角形の頂点2の座標
     * @param p3 三角形の頂点3の座標
     */
    Point3Vec(const cv::Point2f &p1, const cv::Point2f &p2, const cv::Point2f &p3) : p1(p1), p2(p2), p3(p3) {}

    Point3Vec() {}

    bool operator==(const Point3Vec &rhs) const {
        return p1 == rhs.p1 &&
               p2 == rhs.p2 &&
               p3 == rhs.p3;
    }

    bool operator!=(const Point3Vec &rhs) const {
        return !(rhs == *this);
    }
};

/**
 *   @struct struct Triangle
 *   @brief 三角形の座標のインデックス3点を保存する構造体
 */
struct Triangle{
    int p1_idx; ///< 三角形の頂点1のインデックス
    int p2_idx; ///< 三角形の頂点2のインデックス
    int p3_idx; ///< 三角形の頂点3のインデックス
    int idx;
    int depth;

    /**
     * コンストラクタ
     * @param p1_idx 三角形の頂点1のインデックス
     * @param p2_idx 三角形の頂点2のインデックス
     * @param p3_idx 三角形の頂点3のインデックス
     */
    Triangle(int p1_idx, int p2_idx, int p3_idx) : p1_idx(p1_idx), p2_idx(p2_idx), p3_idx(p3_idx) {}
    Triangle(int p1_idx, int p2_idx, int p3_idx, int idx) : p1_idx(p1_idx), p2_idx(p2_idx), p3_idx(p3_idx), idx(idx) {}
};

void drawTriangle(cv::Mat &img, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Scalar color);

double intersectM(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4);

void drawPoint(cv::Mat &img, cv::Point2f p, cv::Scalar color, int size);

std::string getProjectDirectory(std::string os);

std::string replaceBackslash(std::string str);

std::string getVersionOfOpenCV();

/**
 * @fn inline bool isInTriangle(const Point3Vec& trig, const cv::Point2d& p)
 * @brief 座標pが三角形trigの内部にあるか判定する.
 * @details 座標pが三角形trigの内部にあるか判定する.
 * @return  1 ある点が三角形内の点（境界線も含む）である,
 * @return -1 外の点である
 */
inline bool isInTriangle(const Point3Vec& trig, const cv::Point2d& p) {
    cv::Point2f tp1, tp2, tp3;

    tp1 = trig.p1;
    tp2 = trig.p2;
    tp3 = trig.p3;

    if ((tp1.x - tp2.x) * (tp1.y - tp3.y) == (tp1.y - tp2.y) * (tp1.x - tp3.x)) return false;

    cv::Point2f ret;

    ret.x = (float) ((tp1.x + tp2.x + tp3.x) / 3.0);
    ret.y = (float) ((tp1.y + tp2.y + tp3.y) / 3.0);

    return !(intersectM(tp1, tp2, p, ret) < 0 || intersectM(tp1, tp3, p, ret) < 0 || intersectM(tp2, tp3, p, ret) < 0);
}

bool isPointOnTheLine(cv::Point2f a, cv::Point2f b, cv::Point2f p);

cv::Point2f roundVecQuarter(const cv::Point2f &p);

std::vector<std::string> splitString(const std::string &s, char delim);

cv::Mat half(cv::Mat &in,int k);

std::string getCurrentTimestamp();

int getNumberOfDigits(int num);

#endif //ENCODER_UTILS_H
