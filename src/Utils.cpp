/**
 * @file Utils.cpp
 * @brief 点を書いたり線を引いたり...
 * @author Keisuke KAMIYA
 */
#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <opencv2/imgproc.hpp>
#include "../includes/Utils.h"
#include "../includes/GnuplotFileGenerator.hpp"
#include <unistd.h>
#include <fstream>
#include <iostream>

/**
 * @fn double log_2(double num)
 * @brief 値をバリデートして, log2の値を返す関数
 * @param  num    真数
 * @return double log2(num)を返す.
 * @details
 *  1) 負の値, 小さすぎるときは0.0
 *  2) それ以外はlog2(num)を返す
 */
//double log_2(double num){
//  if(fabs(num) < EPS || num < 0) return 0.0;
//  return log2(num);
//}

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
void drawTriangle_residual(cv::Mat &img, const cv::Point2f p1, const cv::Point2f p2, const cv::Point2f p3, const cv::Scalar color,cv::Mat &residual){
    if(p1.x < 0 || p1.x > img.cols-1||p1.y < 0 || p1.y > img.rows-1||
            p2.x < 0 || p2.x > img.cols-1||p2.y < 0 || p2.y > img.rows-1||
            p3.x < 0 || p3.x > img.cols-1||p3.y < 0 || p3.y > img.rows-1){
        return;
    }
    cv::Scalar p1_color,p2_color,p3_color;
    if(100 < M(residual,(int)p1.x,(int)p1.y))p1_color = cv::Scalar(0,0,255);
    //else if(20 > M(residual,(int)p1.x,(int)p1.y))p1_color = cv::Scalar(0,0,0);
    else if(10 >= M(residual,(int)p1.x,(int)p1.y))p1_color = cv::Scalar(0,255,0);
    else p1_color = cv::Scalar(0,0,(double)(M(residual,(int)p1.x,(int)p1.y))*255/90);
    if(100 < M(residual,(int)p2.x,(int)p2.y))p2_color = cv::Scalar(0,0,255);
    //else if(20 > M(residual,(int)p2.x,(int)p2.y))p2_color = cv::Scalar(0,0,0);
    else if(10 >= M(residual,(int)p2.x,(int)p2.y))p2_color = cv::Scalar(0,255,0);
    else p2_color = cv::Scalar(0,0,(double)(M(residual,(int)p2.x,(int)p2.y))*255/90);
    if(100 < M(residual,(int)p3.x,(int)p3.y))p3_color = cv::Scalar(0,0,255);
    //else if(20 > M(residual,(int)p3.x,(int)p3.y))p3_color = cv::Scalar(0,0,0);
    else if(10 >= M(residual,(int)p3.x,(int)p3.y))p3_color = cv::Scalar(0,255,0);
    else p3_color = cv::Scalar(0,0,(double)(M(residual,(int)p3.x,(int)p3.y))*255/90);
    drawPoint(img, p1, p1_color, 4);
    drawPoint(img, p2, p2_color, 4);
    drawPoint(img, p3, p3_color, 4);
    cv::line(img, p1, p2, color, 1);
    cv::line(img, p2, p3, color, 1);
    cv::line(img, p3, p1, color, 1);
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
 * @fn bool check_coordinate(cv::Point2f coordinate, cv::Vec4f range)
 * @brief 座標のチェックをする.
 * @param[in] coordinate チェックしたい座標
 * @param[in] range      値の範囲. x座標の範囲[range[0], range[1]), y座標の範囲[range[2], range[3])の形式で指定する.
 * @return bool 範囲内ならtrue, 範囲外ならfalseを返す
 * @details
 *  coordinateがrange内であるかどうかをチェックする. rangeはVec4fであり,
 *  range[0] <= coordinate.x < range[1]かつrange[2] <= coordinate.y < range[3]であればtrueを返す.
 */
bool check_coordinate(cv::Point2f coordinate, cv::Vec4f range) {
  return range[0] <= coordinate.x && coordinate.x < range[1]
         && range[2] <= coordinate.y && coordinate.y < range[3];
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
std::string getProjectDirectory(){
  char cwd[1024];
  getcwd(cwd, sizeof(cwd));
  std::string current_directory = std::string(cwd);
  return current_directory.substr(0, current_directory.rfind('\\'));
}

/**
 * @fn std::string getVersionOfOpenCV
 * @brief 使用しているOpenCVのバージョンを返す
 * @return CV_VERSION OpenCVのバージョン
 */
std::string getVersionOfOpenCV(){
  return CV_VERSION;
}

/**
 * @fn void storeGnuplotFile(const std::string& out_file_path, const std::string& xlable, const std::string& ylabel, const std::string& data_name)
 * @brief Gnuplotの設定ファイルのようなものを作る
 * @param out_file_path 設定ファイルを書き出す先のパス+ファイル名
 * @param xlable x軸のラベル
 * @param ylabel y軸のラベル
 * @param data_name プロットするデータのファイル名
 */
void storeGnuplotFile(const std::string& out_file_path, const std::string& xlabel, const std::string& ylabel, const std::string& data_name){
  ozi::GnuplotFileGenerator gp(out_file_path);

  gp.clearGraph();
  gp.setXLabel(xlabel);
  gp.setYLabel(ylabel);
  gp.unsetKey();
  gp.plotData(data_name, "#FF0000", ozi::POINTS);
  gp.replotData(data_name, "#FF0000", ozi::IMPULSES);
  gp.close();

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

double round2(double dIn, int nLen)
{
  double    dOut;

  dOut = dIn * pow(10.0, nLen);
  dOut = (double)(int)(dOut + 0.5);

  return dOut * pow(10.0, -nLen);
}
cv::Mat half(cv::Mat &in) {
    cv::Mat out = cv::Mat::zeros((int) (in.rows/2), (int) (in.cols/2), CV_8UC3);
    double **g;
    int k = 1;
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
cv::Mat half_MONO(cv::Mat &in,int k) {
    cv::Mat out = cv::Mat::zeros((int) (in.rows/2), (int) (in.cols/2), CV_8UC1);
    double **g;
    g = (double **) malloc(sizeof(double *) * (in.cols + k*2));
    g += k;
    for (int i = -k; i < in.cols + k; i++) {
        g[i] = (double *) malloc(sizeof(double) * (in.rows + k*2));
        g[i] += k;
    }
    for (int j = 0; j < in.rows; j++) {
        for (int i = 0; i < in.cols; i++) {
            g[i][j] = in.at<unsigned char>(j,i);
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
            out.at<unsigned char>((int)(j/2),(int)(i/2)) = (unsigned char) (sum + 0.5);
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
cv::Mat half_x(cv::Mat &in,int k) {
    cv::Mat out = cv::Mat::zeros((int) (in.rows), (int) (in.cols/2), CV_8UC3);
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
    for (int j = 0; j < in.rows; j += 1) {
        for (int i = 0; i < in.cols; i += 2) {
            double sum = 0;
            for (int l = -k; l <= k; l++) {
                for (int m = -k; m <= k; m++) {
//std::cout << "j+l = " << j+l << "i+k = " << i+k << std::endl;
                    sum += g[i + m][j + l];
                }
            }
            sum /= (2*k+1)*(2*k+1);
            R(out, (int)(i/2), (int) (j)) = (unsigned char) (sum + 0.5);
            G(out, (int)(i/2), (int) (j)) = (unsigned char) (sum + 0.5);
            B(out, (int)(i/2), (int) (j)) = (unsigned char) (sum + 0.5);
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
cv::Mat half_y(cv::Mat &in,int k) {
    cv::Mat out = cv::Mat::zeros((int) (in.rows/2), (int) (in.cols), CV_8UC3);
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
        for (int i = 0; i < in.cols; i += 1) {
            double sum = 0;
            for (int l = -k; l <= k; l++) {
                for (int m = -k; m <= k; m++) {
//std::cout << "j+l = " << j+l << "i+k = " << i+k << std::endl;
                    sum += g[i + m][j + l];
                }
            }
            sum /= (2*k+1)*(2*k+1);
            R(out, (int)(i), (int) (j/2)) = (unsigned char) (sum + 0.5);
            G(out, (int)(i), (int) (j/2)) = (unsigned char) (sum + 0.5);
            B(out, (int)(i), (int) (j/2)) = (unsigned char) (sum + 0.5);
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
cv::Mat half_2(cv::Mat &in) {
    cv::Mat out = cv::Mat::zeros((int) (in.rows/2), (int) (in.cols/2 ), CV_8UC3);
    double **g;
    int k = 2;
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
            R(out, (int)(i/2 ), (int) (j/2 )) = (unsigned char) (sum + 0.5);
            G(out, (int)(i/2 ), (int) (j/2 )) = (unsigned char) (sum + 0.5);
            B(out, (int)(i/2 ), (int) (j/2 )) = (unsigned char) (sum + 0.5);
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
cv::Mat half_sharp(cv::Mat &in) {

    cv::Mat out = cv::Mat::zeros((int) (in.rows / 2), (int) (in.cols / 2), CV_8UC3);
    double **g;
    g = (double **) malloc(sizeof(double *) * (in.cols + 2));
    g += 1;
    for (int i = -1; i < in.cols + 1; i++) {
        g[i] = (double *) malloc(sizeof(double) * (in.rows + 2));
        g[i] += 1;
    }
    for (int j = 0; j < in.rows; j++) {
        for (int i = 0; i < in.cols; i++) {
            g[i][j] = M(in, i, j);
        }
    }
    for (int j = 0; j < in.rows; j++) {
        g[-1][j] = g[1][j];
        g[in.cols][j] = g[in.cols - 2][j];
    }
    for (int i = -1; i < in.cols + 1; i++) {
        g[i][-1] = g[i][1];
        g[i][in.rows] = g[i][in.rows - 2];
    }
    for (int j = 0; j < in.rows; j += 2) {
        for (int i = 0; i < in.cols; i += 2) {
            double sum = 0;
            sum = g[i][j];
            R(out, (int)(i / 2), (int) (j / 2)) = (unsigned char) (sum + 0.5);
            G(out, (int)(i / 2), (int) (j / 2)) = (unsigned char) (sum + 0.5);
            B(out, (int)(i / 2), (int) (j / 2)) = (unsigned char) (sum + 0.5);
        }
    }
    for(int i = -1;i < in.cols + 1;i++){
        g[i] -= 1;
        free(g[i]);
    }
    g -= 1;
    free(g);
    return out;
}

cv::Mat mv_filter(cv::Mat &in){
    cv::Mat out = cv::Mat::zeros((int) (in.rows), (int) (in.cols), CV_8UC1);
    double **g;
    int k = 3;
    g = (double **) malloc(sizeof(double *) * (in.cols + k*2));
    g += k;
    for (int i = -k; i < in.cols + k; i++) {
        g[i] = (double *) malloc(sizeof(double) * (in.rows + k*2));
        g[i] += k;
    }
    for (int j = 0; j < in.rows; j++) {
        for (int i = 0; i < in.cols; i++) {
            g[i][j] = in.at<unsigned char>(j,i);
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

            out.at<unsigned char>(j,i) = (unsigned char)( sum + 0.5);
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
cv::Mat sobel_filter(cv::Mat &in){
    int hW[3][3] = { {-1,0,1},
                     {-2,0,2},
                     {-1,0,1}};
    int hH[3][3] = { {-1,-2,-1},
                     {0,0,0},
                     {1,2,1}};
    cv::Mat out = cv::Mat::zeros(in.size(), CV_8UC1);
    double **g;
    double fx,fy;
    int k = 3;
    g = (double **) malloc(sizeof(double *) * (in.cols + k*2));
    g += k;
    for (int i = -k; i < in.cols + k; i++) {
        g[i] = (double *) malloc(sizeof(double) * (in.rows + k*2));
        g[i] += k;
    }
    for (int j = 0; j < in.rows; j++) {
        for (int i = 0; i < in.cols; i++) {
            g[i][j] = in.at<unsigned char>(j,i);
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
            fx = 0;
            fy = 0;
            for (int l = -1; l <= 1; l++) {
                for (int m = -1; m <= 1; m++) {
//std::cout << "g = " << g[i-m][j-l] << "hW = " << hW[m][l] << std::endl;
                    fx += g[i - m][j - l]*hW[m + 1][l + 1];
                    fy += g[i - m][j - l]*hH[m + 1][l + 1];
                }
            }
            //std::cout << "fx = " << fx << " fy = " << fy << std::endl;
            sum = sqrt(fx * fx + fy * fy);
            //std::cout << "sum = " << sum << std::endl;
            if(sum > 255) {
                sum = 255;
            }
            else if(sum < 0){
                sum = 0;
            }
            out.at<unsigned char>(j,i) = (unsigned char)( sum + 0.5);
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
cv::Mat sobel_filter_x(cv::Mat &in){
    int hW[3][3] = { {-1,0,1},
                     {-2,0,2},
                     {-1,0,1}};
    int hH[3][3] = { {-1,-2,-1},
                     {0,0,0},
                     {1,2,1}};
    cv::Mat out = cv::Mat::zeros(in.size(), CV_8UC1);
    double **g;
    double fx,fy;
    int k = 3;
    g = (double **) malloc(sizeof(double *) * (in.cols + k*2));
    g += k;
    for (int i = -k; i < in.cols + k; i++) {
        g[i] = (double *) malloc(sizeof(double) * (in.rows + k*2));
        g[i] += k;
    }
    for (int j = 0; j < in.rows; j++) {
        for (int i = 0; i < in.cols; i++) {
            g[i][j] = in.at<unsigned char>(j,i);
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
            fx = 0;
            fy = 0;
            for (int l = -1; l <= 1; l++) {
                for (int m = -1; m <= 1; m++) {
//std::cout << "g = " << g[i-m][j-l] << "hW = " << hW[m][l] << std::endl;
                    fx += g[i - m][j - l]*hW[m + 1][l + 1];
                    //fy += g[i - m][j - l]*hH[m + 1][l + 1];
                }
            }
            //std::cout << "fx = " << fx << " fy = " << fy << std::endl;
            sum = sqrt(fx * fx + fy * fy);
            //std::cout << "sum = " << sum << std::endl;
            if(sum > 255) {
                sum = 255;
            }
            else if(sum < 0){
                sum = 0;
            }
            out.at<unsigned char>(j,i) = (unsigned char)( sum + 0.5);
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
cv::Mat sobel_filter_y(cv::Mat &in){
    int hW[3][3] = { {-1,0,1},
                     {-2,0,2},
                     {-1,0,1}};
    int hH[3][3] = { {-1,-2,-1},
                     {0,0,0},
                     {1,2,1}};
    cv::Mat out = cv::Mat::zeros(in.size(), CV_8UC1);
    double **g;
    double fx,fy;
    int k = 3;
    g = (double **) malloc(sizeof(double *) * (in.cols + k*2));
    g += k;
    for (int i = -k; i < in.cols + k; i++) {
        g[i] = (double *) malloc(sizeof(double) * (in.rows + k*2));
        g[i] += k;
    }
    for (int j = 0; j < in.rows; j++) {
        for (int i = 0; i < in.cols; i++) {
            g[i][j] = in.at<unsigned char>(j,i);
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
            fx = 0;
            fy = 0;
            for (int l = -1; l <= 1; l++) {
                for (int m = -1; m <= 1; m++) {
//std::cout << "g = " << g[i-m][j-l] << "hW = " << hW[m][l] << std::endl;
                    //fx += g[i - m][j - l]*hW[m + 1][l + 1];
                    fy += g[i - m][j - l]*hH[m + 1][l + 1];
                }
            }
            //std::cout << "fx = " << fx << " fy = " << fy << std::endl;
            sum = sqrt(fx * fx + fy * fy);
            //std::cout << "sum = " << sum << std::endl;
            if(sum > 255) {
                sum = 255;
            }
            else if(sum < 0){
                sum = 0;
            }
            out.at<unsigned char>(j,i) = (unsigned char)( sum + 0.5);
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
void bubbleSort(std::vector<std::pair<std::vector<cv::Point2f>,double>> &sort_cornes, int array_size) {
    int i, j;
    std::pair<std::vector<cv::Point2f>,double> tmp;

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
void bubbleSort(std::vector<std::tuple<cv::Point2f,double,std::vector<Triangle>>> &sort_cornes, int array_size) {
    int i, j;
    std::tuple<cv::Point2f,double,std::vector<Triangle>> tmp;

    for (i = 0; i < (array_size - 1); i++) {
        for (j = (array_size - 1); j > i; j--) {
            if (std::get<1>(sort_cornes[j - 1]) > std::get<1>(sort_cornes[j])) {
                tmp = sort_cornes[j - 1];
                sort_cornes[j - 1] = sort_cornes[j];
                sort_cornes[j] = tmp;
            }
        }
    }
}

std::vector<Triangle> inter_div(std::vector<Triangle> &triangles, std::vector<cv::Point2f> &corners,cv::Point2f add_corner, int t){
    Triangle triangle = triangles[t];
    std::vector<Triangle> add_triangles;
    add_triangles.clear();
    Point3Vec triangleVec(corners[triangle.p1_idx], corners[triangle.p2_idx], corners[triangle.p3_idx]);
    cv::Point2f p1 = triangleVec.p1;
    cv::Point2f p2 = triangleVec.p2;
    cv::Point2f p3 = triangleVec.p3;
    //corners.emplace_back(add_corner);

    Triangle triangle_p1((int)corners.size() - 1,triangle.p1_idx,triangle.p2_idx);
    Triangle triangle_p2((int)corners.size() - 1,triangle.p2_idx,triangle.p3_idx);
    Triangle triangle_p3((int)corners.size() - 1,triangle.p3_idx,triangle.p1_idx);
    triangles.erase(triangles.begin() + t);
    triangles.insert(triangles.begin() + t,triangle_p1);
    triangles.insert(triangles.begin() + t,triangle_p2);
    triangles.insert(triangles.begin() + t,triangle_p3);

    add_triangles.emplace_back(triangle_p1);
    add_triangles.emplace_back(triangle_p2);
    add_triangles.emplace_back(triangle_p3);
    return  add_triangles;
}

void add_corner_edge(std::vector<cv::Point2f> &corners,cv::Mat &canny,double r1,double r2){
    bool** flag;
    int crop_W = 32,crop_H = 32;
    flag = (bool**)malloc(sizeof(bool*)*canny.rows);
    for(int j = 0;j < (int)canny.rows;j++){
        flag[j] = (bool*)malloc(sizeof(bool)*canny.cols);
    }
    for(int j = 0;j < (int)canny.rows;j++){
        for(int i = 0;i < (int)canny.cols;i++){
            flag[j][i] = true;
        }
    }
    for(int k = 0;k < (int)corners.size();k++){
        for(int j = corners[k].y - r1;j < corners[k].y + r1;j++){
            for(int i = corners[k].x - r1;i < corners[k].x + r1;i++){
                if(i < 0 || j < 0 || i > canny.cols - 1 || j > canny.rows - 1)continue;
                if(fabs(corners[k].x - i) + fabs(corners[k].y - j) <= r1)flag[j][i] = false;
            }
        }
    }
    for(int j = crop_H;j < canny.rows - crop_H;j++){
        for(int i = crop_W;i < canny.cols - crop_W;i++){
            if(canny.at<unsigned char>(j,i) != 0) {
                if (flag[j][i] == true) {
                        std::cout << "add_corner(" << i << "," << j << ")"  << std::endl;
                    corners.emplace_back(cv::Point2f(i, j));
                    for (int y = j - r2; y < j + r2; y++) {
                        for (int x = i - r2; x < i + r2; x++) {
                            if (x < 0 || y < 0 || x > canny.cols - 1 || y > canny.rows - 1)continue;
                            if (fabs(i - x) + fabs(j - y) <= r2)flag[y][x] = false;
                        }
                    }
                }
            }
        }
    }
    for(int j = 0;j < (int)canny.rows;j++){
        free(flag[j]);
    }
    free(flag);
}

std::vector<cv::Point2f> slide_corner_edge(std::vector<cv::Point2f> &corners,cv::Mat &canny,double r1){
    std::vector<cv::Point2f> ret_corners = corners;

    for(int idx= 0;idx < (int)corners.size();idx++){
        bool flag = false;
        cv::Point2f later_point(corners[idx].x,corners[idx].y);
        ret_corners[idx] = corners[idx];
        for(int mv_distance = 1;mv_distance <= r1;mv_distance++) {
            for (int direct = 0; direct <= 8; direct++) {
                if (direct == 0) {
                    later_point.x = corners[idx].x;
                    later_point.y = corners[idx].y;
                } else if (direct == 1) {
                    later_point.x = corners[idx].x + mv_distance;
                    later_point.y = corners[idx].y;
                } else if (direct == 2) {
                    later_point.x = corners[idx].x;
                    later_point.y = corners[idx].y + mv_distance;
                } else if (direct == 3) {
                    later_point.x = corners[idx].x - mv_distance;
                    later_point.y = corners[idx].y;
                } else if (direct == 4) {
                    later_point.x = corners[idx].x;
                    later_point.y = corners[idx].y - mv_distance;
                } else if (direct == 5) {
                    later_point.x = corners[idx].x + mv_distance;
                    later_point.y = corners[idx].y + mv_distance;
                } else if (direct == 6) {
                    later_point.x = corners[idx].x - mv_distance;
                    later_point.y = corners[idx].y + mv_distance;
                } else if (direct == 7) {
                    later_point.x = corners[idx].x - mv_distance;
                    later_point.y = corners[idx].y - mv_distance;
                } else if (direct == 8) {
                    later_point.x = corners[idx].x + mv_distance;
                    later_point.y = corners[idx].y - mv_distance;
                }
                if (later_point.x < 0)later_point.x = 0;
                else if (later_point.x > canny.cols - 1)later_point.x = canny.cols - 1;
                if (later_point.y < 0)later_point.y = 0;
                else if (later_point.y > canny.rows - 1)later_point.y = canny.rows - 1;
                if(canny.at<unsigned char>(later_point.y,later_point.x) >= 50){
                    ret_corners[idx] = later_point;
                    flag = true;
                }
                if(flag)break;
            }
            if(flag)break;
        }
    }
    return ret_corners;
}