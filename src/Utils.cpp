#include <utility>

/**
 * @file Utils.cpp
 * @brief 点を書いたり線を引いたり...
 * @author Keisuke KAMIYA
 */
#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/imgproc.hpp>
#include "../includes/Utils.h"
#include "../includes/CodingTreeUnit.h"
#include <unistd.h>
#include <fstream>
#include <tuple>
#include <iostream>

/**
 * @fn void drawPoint(cv::Mat &img, const cv::Point2f p, const cv::Scalar color, int size)
 * @brief img上の座標pにcolorで指定した色の点を描画する
 * @param[out] img   点を打ちたい画像
 * @param[in]  p     座標
 * @param[in]  color 色
 * @param[in]  size  点のサイズ
 * @details
 * img上に半径sizeでcolor色の円を描画する.
 */
void drawPoint(cv::Mat &img, const cv::Point2f p, const cv::Scalar color, int size) {
  cv::circle(img, p, size, color, -1);
}

/**
 * @fn void drawTriangle(cv::Mat &img, const cv::Point2f p1, const cv::Point2f p2, const cv::Point2f p3, const cv::Scalar color)
 * @brief 三角形を描画する
 * @param[out] img 描画対象の画像データ
 * @param[in] p1 頂点1
 * @param[in] p2 頂点2
 * @param[in] p3 頂点3
 * @param[in] color 線の色
 * @details
 * Mat型のimg上に, 3点(p1, p2, p3)から構成される三角形をcolor色で描画する.
 */
void drawTriangle(cv::Mat &img, const cv::Point2f p1, const cv::Point2f p2, const cv::Point2f p3, const cv::Scalar color){

  drawPoint(img, p1, RED, 4);
  drawPoint(img, p2, RED, 4);
  drawPoint(img, p3, RED, 4);

  cv::line(img, p1, p2, color, 2);
  cv::line(img, p2, p3, color, 2);
  cv::line(img, p3, p1, color, 2);

}

/**
 * @fn void drawRectangle(cv::Mat &img, const cv::Point2f p1, const cv::Point2f p2, const cv::Point2f p3, const cv::Point2f p4)
 * @brief p1, p2, p3, p4で形成される四角形を描画する
 * @param[in] img 入力画像
 * @param[in] p1  点1
 * @param[in] p2  点2
 * @param[in] p3  点3
 * @param[in] p4  点4
 * @details
 *  画像データ(cv::Mat)にp1, p2, p3, p4から構成される四角形を描画する. p1, p2, p3, p4は左上の点から時計回りに与える必要がある.
 *
 *  p1               p2
 *   +---------------+
 *   |               |
 *   |               |
 *   |               |
 *   +---------------+
 *  p4               p3
 *
 */
void drawRectangle(cv::Mat &img, const cv::Point2f p1, const cv::Point2f p2, const cv::Point2f p3, const cv::Point2f p4) {
  assert(p1.x < p2.x && p1.y == p2.y);
  assert(p2.y < p3.y && p2.x == p3.x);
  assert(p4.x < p3.x && p3.y == p4.y);
  assert(p1.y < p4.y && p1.x == p4.x);
  cv::line(img, p1, p2, RED); //  upper  left ->  upper right
  cv::line(img, p2, p3, RED); //  upper right -> bottom right
  cv::line(img, p3, p4, RED); // bottom right -> bottom  left
  cv::line(img, p4, p1, RED); // bottom right ->  upper  left
}

/**
 * @fn cv::Point2f roundVecQuarter(const cv::Point2f p)
 * @brief 動きベクトルをハーフペル精度に丸める
 * @param p 丸めたい実数精度の動きベクトル
* @return 丸められた動きベクトル
 */
cv::Point2f roundVecQuarter(const cv::Point2f &p){
    cv::Point2f ret;
    double quantize_offset = 0.125;
    if(p.x < 0) {
        ret.x = ((int)((p.x - quantize_offset) * 4) / 4.0);
    }else{
        ret.x = ((int)((p.x + quantize_offset) * 4) / 4.0);
    }

    if(p.y < 0) {
        ret.y = ((int) ((p.y - quantize_offset) * 4) / 4.0);
    }else{
        ret.y = ((int) ((p.y + quantize_offset) * 4) / 4.0);
    }

    return ret;
}

/**
 * @fn double outerProduct(cv::Point2f a, cv::Point2f b)
 * @brief 外積の計算
 * @param a ベクトル1
 * @param b ベクトル2
 * @return 計算結果
 */
double outerProduct(cv::Point2f a, cv::Point2f b){
    return (a.x * b.y - a.y * b.x);
}

/**
 * @fn bool isPointOnTheLine
 * @brief 点pが線分a,b上に乗っているか判定する
 * @param a 始点
 * @param b 終点
 * @param p 判定したい点
 * @return 線に乗っていればtrue, そうでなければfalseを返す
 */
bool isPointOnTheLine(cv::Point2f a, cv::Point2f b, cv::Point2f p){
    cv::Point2f v1 = b - a;
    cv::Point2f v2 = p - a;
    return outerProduct(v1, v2) == 0;
}


/**
 * @fn double intersectM(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4)
 * @brief 線分の交差判定
 * @param[in] p1 線1の点1
 * @param[in] p2 線1の点2
 * @param[in] p3 線2の点1
 * @param[in] p4 線2の点2
 * @return 0  直線上に線分の1点 or 2点がある
 * @return 正 直線と線分は交差しない
 * @return 負 直線と線分が交差する
 * @details
 *  座標 p1,p2 を通る直線と座標 p3,p4 を結ぶ線分が交差しているかを調べる
 */
double intersectM(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4) {
  return ((p1.x - p2.x) * (p3.y - p1.y) + (p1.y - p2.y) * (p1.x - p3.x)) *
         ((p1.x - p2.x) * (p4.y - p1.y) + (p1.y - p2.y) * (p1.x - p4.x));
}

/**
 * @fn void interpolation(cv::Mat &in, double x, double y, unsigned char &rr1, unsigned char &gg1, unsigned char &bb1)
 * @brief バイリニア補間をする.
 * @param[in]   in  入力画像
 * @param[in]   x   元のx座標
 * @param[in]   y   元のy座標
 * @param[out]  rr1 補間したRの値
 * @param[out]  gg1 補間したGの値
 * @param[out]  bb1 補間したBの値
 * @details
 *  xとyの座標を補間してrr1, gg1, bb1に入れる
 */
void interpolation(cv::Mat &in, double x, double y, unsigned char& rr1, unsigned char& gg1, unsigned char& bb1) {
  int i, j;
  double alpha, beta;

  i = (int) x;
  j = (int) y;
  alpha = x - (double) i;
  beta = y - (double) j;

  if (i < 0 || in.cols <= i + 1 || j < 0 || in.rows <= j + 1) {
    rr1 = (unsigned char) RR(in, i, j);
    gg1 = (unsigned char) GG(in, i, j);
    bb1 = (unsigned char) BB(in, i, j);
  } else {
    rr1 = (unsigned char) (alpha * beta * (unsigned char) RR(in, i + 1, j + 1) +
                           alpha * (1.0 - beta) * (unsigned char) RR(in, i + 1, j) +
                           (1.0 - alpha) * beta * (unsigned char) RR(in, i, j + 1) +
                           (1.0 - alpha) * (1.0 - beta) * (unsigned char) RR(in, i, j));

    gg1 = (unsigned char) (alpha * beta * (unsigned char) GG(in, i + 1, j + 1) +
                           alpha * (1.0 - beta) * (unsigned char) GG(in, i + 1, j) +
                           (1.0 - alpha) * beta * (unsigned char) GG(in, i, j + 1) +
                           (1.0 - alpha) * (1.0 - beta) * (unsigned char) GG(in, i, j));

    bb1 = (unsigned char) (alpha * beta * (unsigned char) BB(in, i + 1, j + 1) +
                           alpha * (1.0 - beta) * (unsigned char) BB(in, i + 1, j) +
                           (1.0 - alpha) * beta * (unsigned char) BB(in, i, j + 1) +
                           (1.0 - alpha) * (1.0 - beta) * (unsigned char) BB(in, i, j));
  }
}

/**
 * @fn cv::Mat bilinearInterpolation(cv::Mat src)
 * @brief bilinear補間を用いて, 2倍に拡大した画像を作成する.
 * @param[in] src 入力画像
 * @return 補間を用いて2倍に拡大されたsrcを返す
 */
cv::Mat bilinearInterpolation(cv::Mat src){
  cv::Mat dst = cv::Mat::zeros(src.rows * 2, src.cols * 2, CV_8UC3);

  double kx = 0.5;
  double ky = 0.5;

  for(int j = 0 ; j < dst.rows ; j++) {
    for (int i = 0 ; i < dst.cols ; i++) {
      int origin_x = static_cast<int>(i * kx);
      int origin_y = static_cast<int>(j * ky);
      double alpha = i * kx - origin_x;
      double beta = j * ky - origin_y;

      if(i <= 0 || src.cols <= origin_x + 1|| j <= 0 || src.rows <= origin_y + 1 ){
        R(dst, i, j) = (unsigned char) RR(src, origin_x, origin_y);
        G(dst, i, j) = (unsigned char) GG(src, origin_x, origin_y);
        B(dst, i, j) = (unsigned char) BB(src, origin_x, origin_y);
      }else{
        R(dst, i, j) = (unsigned char) (alpha * beta * (unsigned char) RR(src, origin_x + 1, origin_y + 1) +
                                        alpha * (1.0 - beta) * (unsigned char) RR(src, origin_x + 1, origin_y) +
                                        (1.0 - alpha) * beta * (unsigned char) RR(src, origin_x, origin_y + 1) +
                                        (1.0 - alpha) * (1.0 - beta) * (unsigned char) RR(src, origin_x, origin_y));

        G(dst, i, j) = (unsigned char) (alpha * beta * (unsigned char) GG(src, origin_x + 1, origin_y + 1) +
                                        alpha * (1.0 - beta) * (unsigned char) GG(src, origin_x + 1, origin_y) +
                                        (1.0 - alpha) * beta * (unsigned char) GG(src, origin_x, origin_y + 1) +
                                        (1.0 - alpha) * (1.0 - beta) * (unsigned char) GG(src, origin_x, origin_y));

        B(dst, i, j) = (unsigned char) (alpha * beta * (unsigned char) BB(src, origin_x + 1, origin_y + 1) +
                                        alpha * (1.0 - beta) * (unsigned char) BB(src, origin_x + 1, origin_y) +
                                        (1.0 - alpha) * beta * (unsigned char) BB(src, origin_x, origin_y + 1) +
                                        (1.0 - alpha) * (1.0 - beta) * (unsigned char) BB(src, origin_x, origin_y));
      }
    }
  }

  return dst;
}

/**
 * @fn std::string getProjectDirectory
 * @brief プロジェクトディレクトリを返す
 * @return cwd プロジェクトディレクトリを表す文字列(std::string)
 */
std::string getProjectDirectory(std::string os){
  char cwd[1024];
  getcwd(cwd, sizeof(cwd));
  std::string current_directory = std::string(cwd);
  if(OS == "Win") {
      return current_directory.substr(0, current_directory.rfind('\\'));
  }else{
      return current_directory.substr(0, current_directory.rfind('/'));
  }

}

/**
 * @fn std::string replaceBackslash(std::string str)
 * @brief windowsようにバックスラッシュを置換する
 * @param str バックスラッシュを含む文字列
 * @return 置換後の文字列
 */
std::string replaceBackslash(std::string str){
    std::replace(str.begin(), str.end(), '/', '\\');
    return str;
}

/**
 * @fn std::string getVersionOfOpenCV
 * @brief 使用しているOpenCVのバージョンを返す
 * @return CV_VERSION OpenCVのバージョン
 */
std::string getVersionOfOpenCV(){
  return CV_VERSION;
}

std::vector<std::string> splitString(const std::string &s, char delim) {
  std::vector<std::string> elems;
  std::string item;
  for (char ch: s) {
    if (ch == delim) {
      if (!item.empty())
        elems.emplace_back(item);
      item.clear();
    }
    else {
      item += ch;
    }
  }
  if (!item.empty())
    elems.emplace_back(item);
  return elems;
}

/**
 * @fn cv::Mat half(cv::Mat &in,int k)
 * @brief 移動平均フィルタを使用して、パラメタkで画像を縮小する
 * @param in cv::Mat 縮小したい画像
 * @param k  int パラメタ
 * @return cv::Mat
 */
cv::Mat half(cv::Mat &in,int k) {
    cv::Mat out = cv::Mat::zeros((int) (in.rows/2), (int) (in.cols/2), CV_8UC3);
    double **g;
    g = (double **) malloc(sizeof(double *) * (in.cols + k*2));
    g += k;
    for (int i = -k; i < in.cols + k; i++) {
        g[i] = (double *) malloc(sizeof(double) * (in.rows + k*2));
        g[i] += k;
    }
    for (int j = 0; j < in.rows; j++) {
        for (int i = 0; i < in.cols; i++) {
            g[i][j] = M(in, i, j);
        }
    }
    for (int j = 0; j < in.rows; j++) {
        for(int i = 1;i <= k;i++) {
            g[-i][j] = g[i][j];
            g[in.cols-1+i][j] = g[in.cols -1 -i][j];
        }
    }
    for (int i = -k; i < in.cols + k; i++) {
        for(int j = 1;j <= k;j++) {
            g[i][-j] = g[i][j];
            g[i][in.rows - 1+j] = g[i][in.rows  -1-j];
        }
    }
    for (int j = 0; j < in.rows; j += 2) {
        for (int i = 0; i < in.cols; i += 2) {
            double sum = 0;
            for (int l = -k; l <= k; l++) {
                for (int m = -k; m <= k; m++) {
//std::cout << "j+l = " << j+l << "i+k = " << i+k << std::endl;
                    sum += g[i + m][j + l];
                }
            }
            sum /= (2*k+1)*(2*k+1);
            R(out, (int)(i/2), (int) (j/2)) = (unsigned char) (sum + 0.5);
            G(out, (int)(i/2), (int) (j/2)) = (unsigned char) (sum + 0.5);
            B(out, (int)(i/2), (int) (j/2)) = (unsigned char) (sum + 0.5);
        }
    }
    for(int i = -k;i < in.cols + k;i++){
        g[i] -= k;
        free(g[i]);
    }
    g -= k;
    free(g);
    return out;
}


/***
 * @fn cv::Mat mv_filter(cv::Mat &in, int k)
 * @brief 画像に対してkタップの移動平均フィルタを追加する
 * @param in 入力画像
 * @return フィルタをかけた画像
 */
cv::Mat mv_filter(cv::Mat &in,int k){
    cv::Mat out = cv::Mat::zeros((int) (in.rows), (int) (in.cols), CV_8UC3);
    double **g;
    g = (double **) malloc(sizeof(double *) * (in.cols + k*2));
    g += k;
    for (int i = -k; i < in.cols + k; i++) {
        g[i] = (double *) malloc(sizeof(double) * (in.rows + k*2));
        g[i] += k;
    }
    for (int j = 0; j < in.rows; j++) {
        for (int i = 0; i < in.cols; i++) {
            g[i][j] = M(in,i,j);
        }
    }
    for (int j = 0; j < in.rows; j++) {
        for(int i = 1;i <= k;i++) {
            g[-i][j] = g[i][j];
            g[in.cols-1+i][j] = g[in.cols -1 -i][j];
        }
    }
    for (int i = -k; i < in.cols + k; i++) {
        for(int j = 1;j <= k;j++) {
            g[i][-j] = g[i][j];
            g[i][in.rows - 1+j] = g[i][in.rows  -1-j];
        }
    }
    for (int j = 0; j < in.rows; j++) {
        for (int i = 0; i < in.cols; i++) {
            double sum = 0;
            for (int l = -k; l <= k; l++) {
                for (int m = -k; m <= k; m++) {
//std::cout << "j+l = " << j+l << "i+k = " << i+k << std::endl;
                    sum += g[i + m][j + l];
                }
            }
            sum /= (2*k+1)*(2*k+1);

            R(out,i,j) = (unsigned char)( sum + 0.5);
            G(out,i,j) = (unsigned char)( sum + 0.5);
            B(out,i,j) = (unsigned char)( sum + 0.5);
        }
    }
    for(int i = -k;i < in.cols + k;i++){
        g[i] -= k;
        free(g[i]);
    }
    g -= k;
    free(g);
    return out;
}

void bubbleSort(std::vector<std::pair<cv::Point2f,double>> &sort_cornes, int array_size) {
    int i, j;
    std::pair<cv::Point2f,double> tmp;

    for (i = 0; i < (array_size - 1); i++) {
        for (j = (array_size - 1); j > i; j--) {
            if (sort_cornes[j - 1].second > sort_cornes[j].second) {
                tmp = sort_cornes[j - 1];
                sort_cornes[j - 1] = sort_cornes[j];
                sort_cornes[j] = tmp;
            }
        }
    }
}

/**
 * @fn std::string getCurrentTimestamp
 * @brief 現在のタイムスタンプをすべてつなげた文字列を返す
 * @return タイムスタンプ文字列
 */
std::string getCurrentTimestamp(){
    time_t timer;
    struct tm* tm;
    char datetime[30];
    timer = time(NULL);
    tm = localtime(&timer);
    strftime(datetime, 30, "%Y%m%d%H%M%S",tm );

    return std::string(datetime);
}