/**
 * @file Utils.h
 * @brief マクロや構造体等々を定義
 * @author Keisuke KAMIYA
 */

#ifndef ENCODER_UTILS_H
#define ENCODER_UTILS_H

#include <opencv2/core/types.hpp>
#include "config.h"
#include "CodingTreeUnit.h"

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
 *
 */
#define SIDE_X_MIN 500

/**
 *
 */
#define SIDE_Y_MIN 400


enum {
    BM,
    NEWTON,
};

#define PRED_MODE NEWTON
#define GAUSS_NEWTON_PARALLEL_ONLY true
#define GAUSS_NEWTON_INIT_VECTOR false
#define GAUSS_NEWTON_HEVC_IMAGE true
#define HEVC_REF_IMAGE true

#define MVD_DEBUG_LOG false

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
class TRIANGLE{
    int p1_idx; ///< 三角形の頂点1のインデックス
    int p2_idx; ///< 三角形の頂点2のインデックス
    int p3_idx; ///< 三角形の頂点3のインデックス

    /**
     * コンストラクタ
     * @param p1_idx 三角形の頂点1のインデックス
     * @param p2_idx 三角形の頂点2のインデックス
     * @param p3_idx 三角形の頂点3のインデックス
     */
public: TRIANGLE(Triangle t) { p1_idx = t.p1_idx; p2_idx = t.p2_idx; p3_idx = t.p3_idx; }

};
//double log_2(double num);

void drawTriangle(cv::Mat &img, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Scalar color);

void drawTriangle_residual(cv::Mat &img, const cv::Point2f p1, const cv::Point2f p2, const cv::Point2f p3, const cv::Scalar color,cv::Mat &residual);

void interpolation(cv::Mat &in, double x, double y, unsigned char& rr1, unsigned char& gg1, unsigned char& bb1);

void drawRectangle(cv::Mat &img, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4);

bool check_coordinate(cv::Point2f coordinate, cv::Vec4f range);

double intersectM(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4);

void drawPoint(cv::Mat &img, cv::Point2f p, cv::Scalar color, int size);

double round2(double dIn, int nLen);

cv::Mat bilinearInterpolation(cv::Mat src);

std::string getProjectDirectory(std::string os);

std::string replaceBackslash(std::string str);

std::string getVersionOfOpenCV();

void storeGnuplotFile(const std::string& out_file_path, const std::string& xlable, const std::string& ylabel, const std::string& data_name);

/**
 *  @fn inline int RR(cv::Mat &img, int i, int j)
 *  @brief 座標値が適当なときのみ, その画素のR値を返す.
 *  @return int RGB値のRを返す.
 *  @return 0 指定座標が画像外の場合
 *  @details
 *   imgの(x, y)画素のR値を返す. ただし, (i, j)が画像以外を刺した場合, 0が返る.
 */
inline int RR(cv::Mat &img, int i, int j) {
    if (i < 0 || img.cols <= i || j < 0 || img.rows <= j) return 0;
    return (unsigned char) R(img, i, j);
}

/**
 *  @fn inline int GG(cv::Mat &img, int i, int j)
 *  @brief 座標値が適当なときのみ, その画素のG値を返す.
 *  @return int RGB値のGを返す.
 *  @return 0 指定座標が画像外の場合
 *  @details
 *   imgの(x, y)画素のG値を返す. ただし, (i, j)が画像以外を刺した場合, 0が返る.
 */
inline int GG(cv::Mat &img, int i, int j) {
    if (i < 0 || img.cols <= i || j < 0 || img.rows <= j) return 0;
    return (unsigned char) G(img, i, j);
}

/**
 *  @fn inline int BB(cv::Mat &img, int i, int j)
 *  @brief 座標値が適当なときのみ, その画素のB値を返す.
 *  @return int RGB値のBを返す.
 *  @return 0 指定座標が画像外の場合
 *  @details
 *   imgの(x, y)画素のB値を返す. ただし, (i, j)が画像以外を刺した場合, 0が返る.
 */
inline int BB(cv::Mat &img, int i, int j) {
    if (i < 0 || img.cols <= i || j < 0 || img.rows <= j) return 0;
    return (unsigned char) B(img, i, j);
}

/**
 *  @fn inline int MM(cv::Mat &img, int i, int j)
 *  @brief 座標値が適当なときのみ, その画素の輝度値を返す.
 *  @return int 指定座標の輝度値を返す.
 *  @return 0 指定座標が画像外の場合
 *  @details
 *   imgの(x, y)画素のB値を返す. ただし, (i, j)が画像以外を刺した場合, 0が返る.
 */
inline double MM(cv::Mat &img, int i, int j) {
    return 0.299 * RR(img, i, j) + 0.587 * GG(img, i, j) + 0.114 * BB(img, i, j);
}

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

inline int myRound(double x, int delta){
    return static_cast<int>((x / delta) + (x > 0 ? 0.5 : -0.5));
}

cv::Point2f roundVecQuarter(const cv::Point2f &p);

std::vector<std::string> splitString(const std::string &s, char delim);

cv::Mat half(cv::Mat &in);

cv::Mat half(cv::Mat &in,int k);

cv::Mat half_MONO(cv::Mat &in,int k);

cv::Mat half_x(cv::Mat &in,int k);

cv::Mat half_y(cv::Mat &in,int k);

cv::Mat half_2(cv::Mat &in);

cv::Mat half_sharp(cv::Mat &in);

cv::Mat mv_filter(cv::Mat& in);

cv::Mat mv_filter(cv::Mat& in,int k);

cv::Mat sobel_filter(cv::Mat &in);

cv::Mat sobel_filter_x(cv::Mat &in);

cv::Mat sobel_filter_y(cv::Mat &in);

void bubbleSort(std::vector<std::pair<std::vector<cv::Point2f>,double>> &sort_cornes, int array_size);

void bubbleSort(std::vector<std::pair<cv::Point2f,double>> &sort_cornes, int array_size);

void bubbleSort(std::vector<std::tuple<cv::Point2f,double,std::vector<Triangle>>> &sort_cornes, int array_size);

std::vector<Triangle> inter_div(std::vector<Triangle> &triangles, std::vector<cv::Point2f> &corners,cv::Point2f add_corner, int t);

void add_corner_edge(std::vector<cv::Point2f> &corners,cv::Mat &canny,double r1,double r2);

std::vector<cv::Point2f> slide_corner_edge(std::vector<cv::Point2f> &corners,cv::Mat &canny,double r1);

std::string getCurrentTimestamp();

#endif //ENCODER_UTILS_H
