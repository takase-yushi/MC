/**
 * @file main.cpp
 * @brief 実行ファイル
 * @author Keisuke KAMIYA
 */

#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include <queue>
#include <fstream>
#include <ctime>
#include <utility>
#include <numeric>
#include "../includes/config.h"
#include "../includes/ME.hpp"
#include "../includes/DelaunayTriangulation.hpp"
#include "../includes/Vector.hpp"
#include "../includes/psnr.h"
#include "../includes/Golomb.hpp"

struct PredictedImageResult {
  cv::Mat out, mv_image;
  int freq_block, freq_warp, block_matching_pixel_nums, warping_pixel_nums, x_bits, y_bits;
  double block_matching_pixel_errors, warping_pixel_errors;

  PredictedImageResult(cv::Mat out,
                       cv::Mat mv_image,
                       int freq_block,
                       int freq_warp,
                       int block_matching_pixel_nums,
                       int warping_pixel_nums,
                       int xbits,
                       int ybits,
                       double block_matching_pixel_errors,
                       double warping_pixel_errors
  ) :
          out(std::move(out)),
          mv_image(std::move(mv_image)),
          freq_block(freq_block),
          freq_warp(freq_warp),
          block_matching_pixel_nums(block_matching_pixel_nums),
          warping_pixel_nums(warping_pixel_nums),
          x_bits(xbits),
          y_bits(ybits),
          block_matching_pixel_errors(block_matching_pixel_errors),
          warping_pixel_errors(warping_pixel_errors){}

  double getBlockMatchingFrequency() {
    return (double) freq_block / (freq_block + freq_warp) * 100;
  }

  double getWarpingFrequency() {
    return (double) freq_warp / (freq_block + freq_warp) * 100;
  }

  double getBlockMatchingPatchPSNR() {
    return 10.0 * log10(255.0 * 255.0 / (block_matching_pixel_errors / (3.0 * block_matching_pixel_nums)));
  }

  double getWarpingPatchPSNR() {
    return 10.0 * log10(255.0 * 255.0 / (warping_pixel_errors / (3.0 * warping_pixel_nums)));
  }

  int getXbits() {
    return x_bits;
  }

  int getYbits() {
    return y_bits;
  }

};

int golomb_mv_x = 0, golomb_mv_y = 0;

double erase_th_global = 0;

int addSideCorners(cv::Mat img, std::vector<cv::Point2f> &corners);

std::vector<cv::Point2f> cornersQuantization(std::vector<cv::Point2f> &corners, const cv::Mat &target);

PredictedImageResult
getPredictedImage(const cv::Mat &ref, const cv::Mat &refx2, const cv::Mat &refx4,const cv::Mat &refx8, const cv::Mat &target, const cv::Mat &targetx2, const cv::Mat &targetx4, const cv::Mat &targetx8, const cv::Mat &intra, std::vector<Triangle> &triangles,
                  const std::vector<cv::Point2f> &ref_corners, std::vector<cv::Point2f> &corners, DelaunayTriangulation md,std::vector<cv::Point2f> &add_corners,int *add_count,const cv::Mat& residual_ref,int &tmp_mv_x,int &tmp_mv_y,bool add_flag);

std::vector<cv::Point2f> uniqCoordinate(const std::vector<cv::Point2f> &corners);

void storeFrequency(const std::string &file_path, const std::vector<int> freq, int mid);

cv::Point2f getDifferenceVector(const Triangle &triangle, const std::vector<cv::Point2f> &corners,
                                const std::vector<cv::Point2f> &corners_mv, const cv::Point2f &mv);


// 問題は差分ベクトルどうするの…？って
std::vector<int> count_all_diff_x_mv(1001, 0);
std::vector<int> count_all_diff_y_mv(1001, 0);
cv::Mat triangle_error_img;

#define HARRIS false
#define THRESHOLD true
#define LAMBDA 0.2
#define INTER_DIV true

int main(int argc, char *argv[]) {
  std::cout << "OpenCV_version : " << getVersionOfOpenCV() << std::endl;

  const std::string file_path = getProjectDirectory();

  FILE *img_list;
  if ((img_list = fopen((file_path + "\\list.txt").c_str(), "r")) == NULL) {
    std::cerr << "Error : Can not open file" << std::endl;
    exit(1);
  }

  char buf[512];

  // 頂点復号ベクトルのカウント
  std::vector<int> count_all_x_coord(1001, 0);
  std::vector<int> count_all_y_coord(1001, 0);

  // 動きベクトル復元のカウント
  std::vector<int> count_all_x_mv(1001, 0);
  std::vector<int> count_all_y_mv(1001, 0);

  // 引き返す点数のカウント
  std::vector<int> count_all_prev_id(1001, 0);

  // BMの差分ベクトルはグローバルにおいてある

  double point_nums = 0.0;

  std::string graph_file_path = file_path + "\\graph\\";
  std::cout << "graph_file_path:" << graph_file_path << std::endl;


  std::vector<cv::Point2f> corners, corners_org;
  std::vector<cv::Point2f> ref_corners, ref_corners_org;


  // 全画像分ループ
  while (fgets(buf, sizeof(buf), img_list) != NULL) {
    if (buf[0] == '#') continue;
    char t_file_name[256], r_file_name[256], o_file_name[256], i_file_path[256], csv_prefix[256],r_intra_file_name[256],rex2_file_name[256],rex4_file_name[256],tx2_file_name[256],r_r_file_name[256],target_color_file_name[256],c_file_name[256];
    sscanf(buf, "%s %s %s %s %s %s %s %s", i_file_path, r_file_name, t_file_name, o_file_name,r_intra_file_name,r_r_file_name,target_color_file_name,c_file_name);

    std::string img_path = std::string(i_file_path);
    std::string img_directory = file_path + img_path;
    std::string target_file_name = std::string(t_file_name);
    std::string targetx2_file_name = std::string(tx2_file_name);
    std::string ref_r_file_name = std::string(r_r_file_name);
    std::string ref_file_name = std::string(r_file_name);
    std::string refx2_file_name = std::string(rex2_file_name);
    std::string refx4_file_name = std::string(rex4_file_name);
    std::string ref_intra_file_name = std::string(r_intra_file_name);
    std::string corner_file_name = std::string(c_file_name);
    std::string csv_file_prefix = std::string("aaa");
    std::string ref_file_path = file_path + img_path + ref_file_name;
    std::string target_file_path = file_path + img_path + target_file_name;
    std::string refx2_file_path = file_path + img_path + refx2_file_name;
    std::string targetx2_file_path = file_path + img_path + targetx2_file_name;
    std::string refx4_file_path = file_path + img_path + refx4_file_name;
    std::string r_r_file_path = file_path + img_path + r_r_file_name;
    std::string ref_intra_file_path = file_path + img_path + ref_intra_file_name;
    std::string target_color_file_path = file_path + img_path + target_color_file_name;

    std::vector<std::string> out_file = splitString(std::string(o_file_name), '.');

    std::cout << "img_path:" << img_path << std::endl;
    std::cout << "target_file_name:" << target_file_name << std::endl;
    std::cout << "ref_file_name:" << ref_file_name << std::endl;
    std::cout << "ref file path:" << ref_file_path << std::endl;
    std::cout << "target file path:" << target_file_path << std::endl;

    // 符号量
    std::ofstream code_amount;
    if (HARRIS) code_amount = std::ofstream(img_directory + "code_amount_harris.txt");
    else if (THRESHOLD) code_amount = std::ofstream(img_directory + "code_amount_THRESHOLD.txt");

    // PSNRとしきい値以下の点を取り除いたグラフ（頂点を1つずつ取り除いたやつ）
    std::ofstream psnr_points_efficience;
    if (HARRIS) psnr_points_efficience = std::ofstream(img_directory + "psnr_remove_corner_harris.txt");
    if (THRESHOLD) psnr_points_efficience = std::ofstream(img_directory + "psnr_remove_corner_threshold.txt");

    // 頂点数を制限して取った場合のPSNRとの関係
    std::ofstream corner_psnr;
    if (HARRIS) corner_psnr = std::ofstream(img_directory + "corner_psnr_harris.txt");
    if (THRESHOLD) corner_psnr = std::ofstream(img_directory + "corner_psnr_threshold.txt");

    std::ofstream corner_code_amount;
    if (HARRIS) corner_code_amount = std::ofstream(img_directory + "corner_code_amount_harris.txt");
    if (THRESHOLD) corner_code_amount = std::ofstream(img_directory + "corner_code_amount_threshold.txt");

    // 頂点数とMSE
    std::ofstream rate_mse;
    if (HARRIS) rate_mse = std::ofstream(img_directory + "rate_mse_harris.txt");
    else if (THRESHOLD) rate_mse = std::ofstream(img_directory + "rate_mse_threshold.txt");

    std::ofstream rate_psnr;
    if (HARRIS) rate_psnr = std::ofstream(img_directory + "rate_psnr_harris.txt");
    else if (THRESHOLD) rate_psnr = std::ofstream(img_directory + "rate_psnr_threshold.txt");

    //RD性能グラフにしたい
    std::ofstream rate_psnr_csv;
    rate_psnr_csv = std::ofstream(img_directory + target_file_name + "rate_psnr_csv.csv");

    // 時間計測
    clock_t start = clock();
    std::cout << "check1" << std::endl;
    // 準備 --------------------------------------------------------------------------------
    // 画像の読み込み
    cv::Mat ref, ref_gray;          // 参照フレーム
    cv::Mat ref_intra;
    cv::Mat target, target_gray;    // 対象フレーム
    cv::Mat refx2,refx4;
    cv::Mat targetx2,targetx4,targetx2_sharp,target_sharp,target_sharp_gray;
    cv::Mat refx8,targetx8;
    cv::Mat targetx4_sharp,targetx8_gray;
    cv::Mat ref_ref;
    cv::Mat target_bi,ref_bi;
    cv::Mat canny,canny_target;
    cv::Mat target_color;
    cv::Mat target_R,target_G,target_B,target_Y;
    cv::Mat targetx2_R,targetx2_G,targetx2_B;
    cv::Mat targetx4_R,targetx4_G,targetx4_B,targetx4_Y;
      std::vector<cv::Point2f> later_corners = corners;
      cv::Mat color = cv::Mat::zeros(target.size(),CV_8UC3);
      std::vector<cv::Point2f> add_corner_dummy;
//      int add_count_dummy = 0;
      cv::Mat predict_img0 = cv::Mat::zeros(targetx8.size(), CV_8UC3);
      cv::Mat predict_img1 = cv::Mat::zeros(targetx4.size(), CV_8UC3);
      cv::Mat predict_img2 = cv::Mat::zeros(targetx2.size(), CV_8UC3);
      cv::Mat predict_img3 = cv::Mat::zeros(target.size(), CV_8UC3);
      cv::Mat predict_warp = cv::Mat::zeros(target.size(),CV_8UC3);
      cv::Mat predict_para = cv::Mat::zeros(target.size(),CV_8UC3);
      cv::Point2f mv_diff,mv_prev;
      std::vector<cv::Mat> predict_buf;
      std::vector<std::vector<cv::Point2i>> buffer;
      std::vector<cv::Point2i> tmp;
      std::ofstream tri_list;
//      bool para_flag = false;
//      int Quant = 4;

      tri_list = std::ofstream("tri_list.csv");
      predict_buf.emplace_back(predict_img0);
      predict_buf.emplace_back(predict_img1);
      predict_buf.emplace_back(predict_img2);
      predict_buf.emplace_back(predict_img3);
    //cv::Mat residual_ref = cv::Mat::zeros(target.size(),CV_8UC1);
    /*
    std::vector<cv::Mat> ref_buff,target_buff,ref_tmp_buff,target_tmp_buff;
    cv::Mat refz = cv::imread(ref_file_path);
    cv::Mat targetz = cv::imread(target_file_path);
    ref_tmp_buff.emplace_back(refz);
    target_tmp_buff.emplace_back(targetz);
    const int level = 3;
    for(int z = 0;z < level-1;z++){
        cv::Mat ref_ = half(ref_tmp_buff[z]);
        ref_tmp_buff.emplace_back(ref_);
        cv::Mat target_ = half(target_tmp_buff[z]);
        target_tmp_buff.emplace_back(target_);
    }*/
/*
    refx8 = cv::imread(ref_file_path);
    targetx8 = cv::imread(target_file_path);
    refx4 = half(refx8);
    targetx4 = half(targetx8);
    refx2 = half(refx4);
    targetx2 = half(targetx4);
    ref = half(refx2);
    target = half(targetx2);
    targetx4_sharp = half_sharp(targetx8);
    targetx2_sharp = half_sharp(targetx4_sharp);
    target_sharp = half_sharp(targetx2_sharp);
*/
      std::cout << "check2" << std::endl;
      ref = cv::imread(ref_file_path);
      target = cv::imread(target_file_path);
      std::cout << "check3" << std::endl;
      ref_intra = cv::imread(ref_intra_file_path);
      target_color = cv::imread(target_color_file_path);
      refx2 = half_2(ref);
      targetx2 = half_2(target);
      refx4 = half_2(refx2);
      targetx4 = half_2(targetx2);
      refx8 = half_2(refx4);
      targetx8 = half_2(targetx4);
      targetx4_sharp = half_sharp(targetx8);
      targetx2_sharp = half_sharp(targetx4_sharp);
      target_sharp = half_sharp(targetx2_sharp);
      ref_ref = cv::imread(r_r_file_path);
      target_R = cv::Mat::zeros(target.size(),CV_8UC1);
      target_G = cv::Mat::zeros(target.size(),CV_8UC1);
      target_B = cv::Mat::zeros(target.size(),CV_8UC1);
      for(int j = 0;j < target_color.rows;j++){
          for(int i = 0;i < target_color.cols;i++){
              target_R.at<unsigned char>(j,i) = (unsigned char)R(target_color,i,j);
              target_G.at<unsigned char>(j,i) = (unsigned char)G(target_color,i,j);
              target_B.at<unsigned char>(j,i) = (unsigned char)B(target_color,i,j);
          }
      }
      targetx2_R = half_MONO(target_R,2);
      targetx2_G = half_MONO(target_G,2);
      targetx2_B = half_MONO(target_B,2);
      targetx4_R = half_MONO(targetx2_R,2);
      targetx4_G = half_MONO(targetx2_G,2);
      targetx4_B = half_MONO(targetx2_B,2);
      cv::Mat residual_ref = cv::Mat::zeros(target.size(),CV_8UC1);
      cv::Mat Maskx4 = cv::Mat::zeros(targetx4.size(),CV_8UC1);
      {
          int crop_W = 8,crop_H = 8;
          for (int j = crop_H;j < targetx4.rows - crop_H;j++){
              for(int i = crop_W;i < targetx4.cols - crop_W;i++){
                  Maskx4.at<unsigned char>(j,i) = 1;
              }
          }
      }
      cv::Mat residual_ref_bi;
      std::cout << "check" << std::endl;
      cvtColor(ref, ref_gray, cv::COLOR_BGR2GRAY);
      std::cout << "check point 2" << std::endl;
      cvtColor(target, target_gray, cv::COLOR_BGR2GRAY);
      target_Y = target_gray;
      cvtColor(targetx4, targetx4_Y, cv::COLOR_BGR2GRAY);
      cv::bilateralFilter(ref_gray,ref_bi,5,150,150,CV_HAL_BORDER_REFLECT_101);
      cv::bilateralFilter(target_gray,target_bi,5,150,150,CV_HAL_BORDER_REFLECT_101);
      double **y_;
      y_ = (double **)malloc(sizeof(double *)*target.rows);
      for(int j = 0;j < target.rows;j++){
          y_[j] = (double *)malloc(sizeof(double)*target.cols);
      }
      int y_max = 0,y_min = 255;
      for(int j = 0;j < target.rows;j++) {
          for (int i = 0; i < target.cols; i++) {
              int y = abs(target_bi.at<unsigned char>(j, i) - ref_bi.at<unsigned char>(j, i));
              if(y_max < y){
                  y_max = y;
              }
              else if(y_min > y){
                  y_min = y;
              }
          }
      }
      std::cout << "check2" << std::endl;
      double y_scale = y_max - y_min;
      for(int j = 0;j < target.rows;j++) {
          for (int i = 0; i < target.cols; i++) {
              int y = abs(target_bi.at<unsigned char>(j, i) - ref_bi.at<unsigned char>(j, i));
              y_[j][i] = (y - y_min)/y_scale;
              if(y < 0)
                  y = 0;
              else if(y > 255)
                  y = 255;

              /*
                int y = M(target,i,j) - M(ref_ref,i,j);
              if(y < -128)
                  y = -128;
              else if(y > 127)
                  y = 127;
              y += 128;
*/
              //std::cout << "i = " << i << "j = "<< j << std::endl;
              residual_ref.at<unsigned char>(j,i) = (unsigned char)y;
          }
      }
      cv::bilateralFilter(residual_ref,residual_ref_bi,5,150,150,CV_HAL_BORDER_REFLECT_101);
      cv::Mat residual_filter = mv_filter(residual_ref);
      cv::imwrite(img_directory + "residual_ref.bmp",residual_ref);
      cv::Mat residual_refx2 =   half_MONO(residual_ref,2);
      cv::Mat residual_refx4 = half_MONO(residual_refx2,2);
    std::cout << "check point 1" << std::endl;
    // グレイスケールに変換

    cv::imwrite("ref_bi.bmp",ref_bi);
    cv::imwrite("target_bi.bmp",target_bi);
      std::cout << "smooth end" << std::endl;
      for(int j = 0;j < target.rows;j++) {
          for (int i = 0; i < target.cols; i++) {
              target_gray.at<unsigned char>(j,i) = y_[j][i] * target_gray.at<unsigned char>(j,i);
          }
      }
    //cvtColor(target_sharp, target_sharp_gray, cv::COLOR_BGR2GRAY);
    cv::Mat sobel_target = cv::Mat::zeros(target.size(),CV_8UC1);
    sobel_target = sobel_filter(target_gray);
    cv::Mat sobel_target_x = sobel_filter_x(target_gray);
    cv::Mat sobel_target_y = sobel_filter_y(target_gray);
    cv::imwrite("sobel_target.bmp",sobel_target);
    cv::imwrite("sobel_target_x.bmp",sobel_target_x);
    cv::imwrite("sobel_target_y.bmp",sobel_target_y);
    std::cout << "check point 2" << std::endl;
    double high_th = 100;
    double low_th = 0;
    cv::Canny(residual_ref,canny,high_th,low_th);
    high_th = 100;
    low_th = 0;
    cv::Canny(target_Y,canny_target,high_th,low_th);
    cv::imwrite("canny.bmp", canny);
    cv::imwrite("canny_target.bmp", canny_target);
    cv::imwrite("reidal_ref_bi.bmp", residual_ref_bi);
    cv::imwrite("target4_R.bmp", targetx4_R);
      cv::imwrite("target4_G.bmp", targetx4_G);
      cv::imwrite("target4_B.bmp", targetx4_B);
    // ドロネー分割 -------------------------------------------------------------------------

    const int POINT_MAX = 250; // static_cast<int>(((ref.cols + ref.rows) * 4) / 5 * (corner_ratio / 10.0)); // 特徴点の最大個数

    corners.clear();
    std::vector<cv::Point2f> corners_R,corners_G,corners_B,corners_Y;
    // 特徴点抽出(GFTTDetector)
    cv::goodFeaturesToTrack(residual_ref, corners_org, POINT_MAX, GFTT_QUAULITY, 24,residual_ref, 3);
    cv::goodFeaturesToTrack(targetx4_Y, corners_Y, POINT_MAX, GFTT_QUAULITY, 16,Maskx4, 3);//8
    cv::goodFeaturesToTrack(targetx4_R, corners_R, POINT_MAX, GFTT_QUAULITY, 8,targetx4_R, 3);
    cv::goodFeaturesToTrack(targetx4_G, corners_G, POINT_MAX, GFTT_QUAULITY, 8,targetx4_G, 3);
    cv::goodFeaturesToTrack(targetx4_B, corners_B, POINT_MAX, GFTT_QUAULITY, 8,targetx4_B, 3);
     // image – 8ビットまたは浮動小数点型，シングルチャンネルの入力画像．
     // corners – 検出されたコーナーが出力されるベクトル．
     // maxCorners – 出力されるコーナーの最大数．これより多い数のコーナーが検出された場合，より強いコーナーが出力されます．
     // qualityLevel – 許容される画像コーナーの最低品質を決定します．このパラメータ値を，最良のコーナーを示す測度（ cornerMinEigenVal() で述べた最小固有値や， cornerHarris() で述べた Harris 関数の応答）に乗じます．その掛け合わされた値よりも品質度が低いコーナーは，棄却されます．例えば，コーナーの最高品質度 = 1500， qualityLevel=0.01 である場合，品質度が15より小さいすべてのコーナーが棄却されます．
     // minDistance – 出力されるコーナー間で許容される，最小ユークリッド距離．
      corners_org.clear();
      for(int i = 0;i < (int)corners_R.size();i++){
          bool flag = true;
          for(int j = 0;j < (int)corners_org.size();j++){
              if(corners_R[i].x == corners_org[j].x && corners_R[i].y == corners_org[j].y)flag = false;
          }
          if(flag) {
              //corners_org.emplace_back(corners_R[i]);
          }
      }
      std::cout << "check2" << std::endl;
      for(int i = 0;i < (int)corners_G.size();i++){
          bool flag = true;
          for(int j = 0;j < (int)corners_org.size();j++){
              if(corners_G[i].x == corners_org[j].x && corners_G[i].y == corners_org[j].y)flag = false;
          }
          if(flag) {
              //corners_org.emplace_back(corners_G[i]);
          }
      }
      std::cout << "check3" << std::endl;
      for(int i = 0;i < (int)corners_B.size();i++){
          bool flag = true;
          for(int j = 0;j < (int)corners_org.size();j++){
              if(corners_B[i].x == corners_org[j].x && corners_B[i].y == corners_org[j].y)flag = false;
          }
          if(flag) {
              //corners_org.emplace_back(corners_B[i]);
          }
      }
      for(int i = 0;i < (int)corners_Y.size();i++){
          bool flag = true;
          for(int j = 0;j < (int)corners_org.size();j++){
              if(corners_Y[i].x == corners_org[j].x && corners_Y[i].y == corners_org[j].y)flag = false;
          }
          if(flag) {
              corners_org.emplace_back(corners_Y[i]);
          }
      }
      std::cout << "check4" << std::endl;
      for(int i = 0;i < (int)corners_org.size();i++){
          if(residual_refx4.at<unsigned char>(corners_org[i].y,corners_org[i].x) <= 1){
              corners_org.erase(corners_org.begin() + i);
              i--;
          }
      }
      for(int i = 0;i < (int)corners_org.size();i++){
          corners_org[i] *= 4;
      }
    // 外周に点を打つ
    addSideCorners(target, corners_org);
    //add_corner_edge(corners_org,canny,16,100);

    // 頂点の量子化
    corners_org = cornersQuantization(corners_org, target);
    puts("Quantized");

    // 頂点の動きをブロックマッチングで求める -----------------------------------------------------------------------
    cv::Mat points = target.clone();
    std::pair<std::vector<cv::Point2f>, std::priority_queue<int>> ret_ref_corners = getReferenceImageCoordinates(
            ref, target, corners_org, points);
      ref_corners_org = ret_ref_corners.first;

     // std::vector<cv::Point2f> ret_ref_corners(corners_org.size(), cv::Point2f(0.0, 0.0));
     //ref_corners_org = ret_ref_corners;

    corners.clear();
    for (int i = 0; i < (int) corners_org.size(); i++) corners.emplace_back(corners_org[i]);
    ref_corners.clear();
    for (int i = 0; i < (int) ref_corners_org.size(); i++) ref_corners.emplace_back(ref_corners_org[i]);

    // for (int corner_ratio = 10; corner_ratio > 0; corner_ratio-=2) {
//    for(int corner_ratio = 1 ; corner_ratio <= 10 ; corner_ratio++) {
//    double corner_ratio = 10;
//    while(corner_ratio > 0){

    {

      double threshold = 17;
      double max_threshold = 17;
      double th = 0.05;
      while (threshold <= max_threshold) {

        std::string out_file_name;

        if (HARRIS) {
          out_file_name =
                  img_directory + out_file[0] + "_corners_size_" + std::to_string(corners.size()) + "." + out_file[1];
        } else if (THRESHOLD) {
          out_file_name = img_directory + out_file[0] + "_threshold_" + std::to_string(threshold) + "_lambda_" +
                          std::to_string(LAMBDA) + "." + out_file[1];
        }
        std::cout << "out_file_name:" << out_file_name << std::endl;

        std::cout << "target.cols = " << target.cols << "target.rows = "<< target.rows << std::endl;
        // Subdiv2Dの初期化
        cv::Size size = target.size();
        cv::Rect rect(0, 0, size.width, size.height);
        std::cout << "size.width = " << size.width << "size.height = " << size.height << std::endl;
        cv::Subdiv2D subdiv(rect);
        puts("Init subdiv2d");

        std::cout << "corners.size():" << corners.size() << std::endl;
        std::cout << "ref_corners's size :" << ref_corners.size() << std::endl;
        std::cout << "corners's size     :" << corners.size() << std::endl;

        // 頂点の動きベクトルを描画
        for (int i = 0; i < (int) corners.size(); i++) {
          drawPoint(points, corners[i], RED, 4);
          drawPoint(points, cv::Point2d(ref_corners[i].x / 2.0, ref_corners[i].y / 2.0), BLUE, 4);
          cv::line(points, corners[i], cv::Point2d(ref_corners[i].x / 2.0, ref_corners[i].y / 2.0), GREEN);
        }
        cv::imwrite(file_path + img_path + "points.png", points);

        puts("insert");
        // 点を追加
        subdiv.insert(corners);

        puts("insert md");

        DelaunayTriangulation md(Rectangle(0, 0, target.cols, target.rows));
        md.insert(corners);

        std::vector<cv::Vec6f> triangles_mydelaunay;
        md.getTriangleList(triangles_mydelaunay);
        std::cout << "check1" << std::endl;
        int cnt_erased_elem = 0;
        // 頂点を間引いています（11回ほどやる）
        for (int i = 0; i < 0; i++) {
          std::cout << "erase:" << i + 1 << std::endl;
          md = DelaunayTriangulation(Rectangle(0, 0, target.cols, target.rows));
          md.insert(corners);
          md.getTriangleList(triangles_mydelaunay);

          cv::Mat corner_reduction = target.clone();
          for(const cv::Vec6f t : triangles_mydelaunay){
            cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
            //drawTriangle(corner_reduction, p1, p2, p3, BLUE);
              drawTriangle_residual(corner_reduction, p1, p2, p3, BLUE,sobel_target);
          }
          cv::imwrite(img_directory + "corner_reduction_" + std::to_string(i) + ".png", corner_reduction);

          std::priority_queue<int> ret_unnecessary_pt = md.getUnnecessaryPoint(ref_corners, threshold, target);

          std::cout << "threshold:" << threshold << std::endl;
          std::cout << "erase elem size:" << ret_unnecessary_pt.size() << std::endl;

          cnt_erased_elem += ret_unnecessary_pt.size();

          while (!ret_unnecessary_pt.empty()) {
            int idx = ret_unnecessary_pt.top();
            ret_unnecessary_pt.pop();
            corners.erase(corners.begin() + idx);
            ref_corners.erase(ref_corners.begin() + idx);
          }
        }
          cv::Mat corner_residual_ref = residual_ref.clone();
        for(int j = 0;j < target.rows;j++){
            for(int i = 0;i < target.cols;i++){
                int y = 1*M(corner_residual_ref,i,j);
                if(y < 0)y = 0;
                else if(y > 255)y = 255;
                R(corner_residual_ref,i,j) = y;
                G(corner_residual_ref,i,j) = y;
                B(corner_residual_ref,i,j) = y;
            }
        }
          cv::Mat corner_ref = ref.clone();

          for(const cv::Vec6f t : triangles_mydelaunay){
              cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
              //drawTriangle(corner_reduction, p1, p2, p3, BLUE);
              drawTriangle(corner_ref, p1, p2, p3, BLUE);
              drawTriangle(corner_residual_ref, p1, p2, p3, BLUE);
          }
          cv::imwrite(img_directory + "corner_ref" + ".png", corner_ref);
          cv::imwrite(img_directory + "corner_residual_ref" + ".png", corner_residual_ref);
          cv::imwrite(img_directory + "target_gray" + ".png", target_gray);
/*
          corners.clear();
          corners.emplace_back(cv::Point2f(500,500));
          corners.emplace_back(cv::Point2f(0,0));
          corners.emplace_back(cv::Point2f(1919,0));
          corners.emplace_back(cv::Point2f(0,1023));
          corners.emplace_back(cv::Point2f(1919,1023));
         */
        //頂点削除
/*
        double erase_th_per = 0.6;
          md.Sort_Coners(corners);
          std::vector<double> sigma_tmp;
          std::vector<Triangle> triangles_t = md.Get_triangles(corners);
          double MSE = 0;
          double triangle_sum = 0;
          int fx[51] = {0};
          for (int t = 0; t < (int)triangles_t.size(); t++) {
              int triangle_size;
              double RMSE;
              double MSE_tmp = 0;
              cv::Point2f p1(corners[triangles_t[t].p1_idx]), p2(corners[triangles_t[t].p2_idx]), p3(
                      corners[triangles_t[t].p3_idx]);
              Point3Vec target_corers = Point3Vec(p1, p2, p3);
              Point3Vec prev_corners = Point3Vec(p1, p2, p3);
              MSE_tmp = Gauss_Newton(ref, target, ref_intra, target_corers, prev_corners, triangle_size);
              MSE += MSE_tmp;
              triangle_sum += triangle_size;
              sigma_tmp.emplace_back(MSE_tmp);
              RMSE = sqrt(MSE_tmp / triangle_size);
              std::cout << "t = " << t << "/" << triangles_t.size()  << " RMSE = " << RMSE << std::endl;
              if(RMSE < 50)fx[(int)RMSE]++;
              else fx[50]++;
          }
          double myu = sqrt(MSE / triangle_sum);
          double sigma = 0;
          for(int i = 0;i < (int)sigma_tmp.size();i++){
              sigma += (sqrt(sigma_tmp[i]) - myu) * (sqrt(sigma_tmp[i]) - myu);
          }
          sigma = sqrt(sigma/triangle_sum);
          std::cout << "myu = "<< myu << "sigma = " << sigma << std::endl;
          for(int i = 0;i < 51;i++) {
              std::cout << "fx[" << i << "] = " << fx[i] << std::endl;
          }

          //double erase_th = (MSE / triangle_sum) * 10 / pow(2, q);
          double erase_th = (myu + sigma) * (myu + sigma);
          //double erase_th = 100;
          erase_th_global = erase_th;
          std::cout << "erase_th = " << erase_th << std::endl;
          std::cout << "check2" << std::endl;
            for (int q = 0; q < 4; q++) {
*/
                /*
                md.insert(corners);
                md.getTriangleList(triangles_mydelaunay);
                md.Sort_Coners(corners);
                md.insert(corners);
                md.getTriangleList(triangles_mydelaunay);
                std::vector<double> sigma_tmp;
                std::vector<Triangle> triangles_t = md.Get_triangles(corners);
                double MSE = 0;
                double triangle_sum = 0;
                int fx[51] = {0};
                for (int t = 0; t < (int)triangles_t.size(); t++) {
                    int triangle_size;
                    double RMSE;
                    double MSE_tmp = 0;
                    cv::Point2f p1(corners[triangles_t[t].p1_idx]), p2(corners[triangles_t[t].p2_idx]), p3(
                            corners[triangles_t[t].p3_idx]);
                    Point3Vec target_corers = Point3Vec(p1, p2, p3);
                    Point3Vec prev_corners = Point3Vec(p1, p2, p3);
                    MSE_tmp = Gauss_Newton(ref, target, ref_intra, target_corers, prev_corners, triangle_size);
                    MSE += MSE_tmp;
                    triangle_sum += triangle_size;
                    sigma_tmp.emplace_back(MSE_tmp);
                    RMSE = sqrt(MSE_tmp / triangle_size);
                    std::cout << "t = " << t << "/" << triangles_t.size()  << " RMSE = " << RMSE << std::endl;
                    if(RMSE < 50)fx[(int)RMSE]++;
                    else fx[50]++;
                }
                double myu = sqrt(MSE / triangle_sum);
                double sigma = 0;
                for(int i = 0;i < (int)sigma_tmp.size();i++){
                    sigma += (sqrt(sigma_tmp[i]) - myu) * (sqrt(sigma_tmp[i]) - myu);
                }
                sigma = sqrt(sigma/triangle_sum);
                std::cout << "myu = "<< myu << "sigma = " << sigma << std::endl;
                for(int i = 0;i < 51;i++) {
                    std::cout << "fx[" << i << "] = " << fx[i] << std::endl;
                }

                //double erase_th = (MSE / triangle_sum) * 10 / pow(2, q);
                double erase_th = (myu + sigma) * (myu + sigma);
                //double erase_th = 100;
                erase_th_global = erase_th;
                std::cout << "erase_th = " << erase_th << std::endl;
                 */
/*
                for (int idx = 0; idx < (int) corners.size(); idx++) {
                    bool erase_flag = false;
                    int erase_cout = 0;
                    DelaunayTriangulation md_prev(Rectangle(0, 0, target.cols, target.rows));
                    md_prev.insert(corners);
                    std::vector<cv::Vec6f> triangles_mydelaunay;
                    md_prev.getTriangleList(triangles_mydelaunay);
                    cv::Mat corner_reduction = target.clone();
                    for (const cv::Vec6f t : triangles_mydelaunay) {
                        cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
                        //drawTriangle(corner_reduction, p1, p2, p3, BLUE);
                        drawTriangle_residual(corner_reduction, p1, p2, p3, BLUE, sobel_target);
                    }
                    cv::imwrite(img_directory + "corner_reduction_" + std::to_string(idx) + ".png", corner_reduction);
                    if (corners[idx].x <= 0.0 || corners[idx].y <= 0.0 || corners[idx].x >= target.cols - 1 ||
                        corners[idx].y >= target.rows - 1) {
                        bool skip_flag = false;
                        std::vector<cv::Point2f> corners_later(corners);
                        corners_later.erase(corners_later.begin() + idx);
                        DelaunayTriangulation md_later(Rectangle(0, 0, target.cols, target.rows));
                        md_later.insert(corners_later);
                        std::vector<cv::Vec6f> triangles_tmp;
                        md_later.getTriangleList(triangles_tmp);
                        md_later.serch_wrong(corners_later, target, &skip_flag);
                        if (skip_flag == true) {
                            continue;
                        }
                    }
                    std::vector<bool> flag_around = std::vector<bool>(corners.size(), false);
                    std::vector<cv::Point2f> corners_later(corners);
                    if ((corners[idx].x == 0.0 && corners[idx].y == 0.0) ||
                        (corners[idx].x == target.cols - 1 && corners[idx].y == 0.0)
                        || (corners[idx].x == target.cols - 1 && corners[idx].y == target.rows - 1) ||
                        (corners[idx].x == 0.0 && corners[idx].y == target.rows - 1)) {
                        continue;
                    }
                    std::cout << "idx = " << idx << "/ " << corners.size() << " q = " << q << "/ " << 4 << corners[idx]
                              << std::endl;
                    std::vector<Triangle> triangles_around;
                    triangles_around = md_prev.Get_triangles_around(idx, corners, flag_around);
                    double MSE_prev = 0, MSE_later = 0;
                    int triangle_size_sum_prev = 0, triangle_size_sum_later = 0;
//#pragma omp parallel for
                    for (int t = 0; t < (int) triangles_around.size(); t++) {
                        int triangle_size;
                        //double error_warp;
                        double MSE_tmp = 0;
                        Triangle triangle = triangles_around[t];
                        Point3Vec triangleVec(corners[triangle.p1_idx], corners[triangle.p2_idx],
                                              corners[triangle.p3_idx]);

                        Point3Vec prev_corners = Point3Vec(corners[triangle.p1_idx], corners[triangle.p2_idx],
                                                           corners[triangle.p3_idx]);
                        //std::cout << "prev_Gauss" << std::endl;
                        MSE_tmp = Gauss_Newton(ref, target, ref_intra, triangleVec, prev_corners, triangle_size);
                        //Gauss_Newton2(ref,target,ref_intra, predict_buf,predict_warp,predict_para, color, error_warp, triangleVec, prev_corners, tri_list,&para_flag,add_corner_dummy,&add_count_dummy,t,residual_ref,triangle_size, false,erase_th_global);
                        std::cout << "triangle_size = " << triangle_size << "MSE_tmp = " << MSE_tmp << std::endl;
                        MSE_prev += MSE_tmp;
                        triangle_size_sum_prev += triangle_size;
                        if (MSE_tmp / (double) triangle_size >= erase_th) {
                            erase_cout++;
                        }
                    }
                    //if(erase_cout/triangles_around.size() >= erase_th_per)erase_flag = true;
                    //double MSE_prev_sum = MSE_prev;
                    MSE_prev /= triangle_size_sum_prev;
                    std::cout << "MSE_prev = " << MSE_prev << std::endl;
                    //for (const bool flag:flag_around) {
                        //std::cout<< flag << std::endl;
                    //}
                    corners_later.erase(corners_later.begin() + idx);
                    flag_around.erase(flag_around.begin() + idx);

                    DelaunayTriangulation md_later(Rectangle(0, 0, target.cols, target.rows));
                    md_later.insert(corners_later);
                    std::vector<Triangle> triangles_later;
                    triangles_later = md_later.Get_triangles_later(md_later, idx, corners_later, flag_around);
//#pragma omp parallel for
                    for (int t = 0; t < (int) triangles_later.size(); t++) {
                        double MSE_tmp = 0;
                        int triangle_size;
                        Triangle triangle = triangles_later[t];
                        Point3Vec triangleVec(corners_later[triangle.p1_idx], corners_later[triangle.p2_idx],
                                              corners_later[triangle.p3_idx]);
                        Point3Vec prev_corners = Point3Vec(corners[triangle.p1_idx], corners[triangle.p2_idx],
                                                           corners[triangle.p3_idx]);
                        //std::cout << "later_Gauss" << std::endl;
                        MSE_tmp= Gauss_Newton(ref, target, ref_intra, triangleVec, prev_corners, triangle_size);
                        MSE_later += MSE_tmp;
                        triangle_size_sum_later += triangle_size;
                        std::cout << "triangle_size = " << triangle_size <<  "MSE_later = " << MSE_tmp << std::endl;
                    }
                    //double MSE_later_sum = MSE_later;
                    MSE_later /= (double)triangle_size_sum_later;
                    std::cout << "MSE_later = " << MSE_later << std::endl;
                    double RMSE_prev = sqrt(MSE_prev);
                    double RMSE_later = sqrt(MSE_later);
                    double S_per = (double) triangle_size_sum_later / (double) triangle_size_sum_prev;
                    std::cout << "RMSE_prev = " << RMSE_prev << " RMSE_later = " << RMSE_later << " RMSE_per = "
                              << (MSE_later - MSE_prev) / MSE_prev << " S_per = "
                              << S_per
                              << " erase_count = " << erase_cout << " / " << triangles_around.size()
                              << " erase_per = " << RMSE_later/RMSE_prev << std::endl;
                    std::cout << "MSE x S = " << (RMSE_later - RMSE_prev)*triangle_size_sum_later << std::endl;
                    if (erase_cout / triangles_around.size() >= erase_th_per && RMSE_later/RMSE_prev < 1.5)
                        erase_flag = true;
                    if ((fabs(MSE_prev - MSE_later)/MSE_prev < th ) || MSE_prev > MSE_later ||
                        erase_flag || (RMSE_later - RMSE_prev)*triangle_size_sum_later < 1000) {//6 10000
                        std::cout << "erased" << std::endl;
                        corners.erase(corners.begin() + idx);
                        idx--;
                    }
                }
            }
*/
            /*
            for (int idx = 0; idx < (int) corners.size(); idx++) {
                double min_distance = md.neighbor_distance(corners, idx);
                int mv_distance = std::min(8, (int) pow(2, (int) std::log2(sqrt(min_distance) / 2)));
                std::vector<cv::Point2f> later_corners = corners;
                for (int idx = 0; idx < (int) corners.size(); idx++) {
                    if ((corners[idx].x == 0.0 && corners[idx].y == 0.0) ||
                        (corners[idx].x == target.cols - 1 && corners[idx].y == 0.0)
                        || (corners[idx].x == target.cols - 1 && corners[idx].y == target.rows - 1) ||
                        (corners[idx].x == 0.0 && corners[idx].y == target.rows - 1)) {
                        continue;
                    }
                    std::vector<std::pair<cv::Point2f, double>> point_pairs;
                    std::pair<cv::Point2f, double> point_pair;
                    std::vector<bool> flag_around = std::vector<bool>(corners.size(), false);
                    for (int direct = 0; direct < 5; direct++) {
                        if (direct == 1) {
                            later_corners[idx].x = corners[idx].x + mv_distance;
                            later_corners[idx].y = corners[idx].y;
                        } else if (direct == 2) {
                            later_corners[idx].x = corners[idx].x;
                            later_corners[idx].y = corners[idx].y + mv_distance;
                        } else if (direct == 3) {
                            later_corners[idx].x = corners[idx].x - mv_distance;
                            later_corners[idx].y = corners[idx].y;
                        } else if (direct == 4) {
                            later_corners[idx].x = corners[idx].x;
                            later_corners[idx].y = corners[idx].y - mv_distance;
                        } else if (direct == 5) {
                            later_corners[idx].x = corners[idx].x + mv_distance;
                            later_corners[idx].y = corners[idx].y + mv_distance;
                        } else if (direct == 6) {
                            later_corners[idx].x = corners[idx].x - mv_distance;
                            later_corners[idx].y = corners[idx].y + mv_distance;
                        } else if (direct == 7) {
                            later_corners[idx].x = corners[idx].x - mv_distance;
                            later_corners[idx].y = corners[idx].y - mv_distance;
                        } else if (direct == 8) {
                            later_corners[idx].x = corners[idx].x + mv_distance;
                            later_corners[idx].y = corners[idx].y - mv_distance;
                        }
                        for (int c_idx = 0; c_idx < (int) corners.size(); c_idx++) {
                            if (later_corners[idx] == corners[c_idx]) {
                                later_corners[idx] = corners[idx];
                            }
                        }
                        if (later_corners[idx].x < 0)later_corners[idx].x = 0;
                        else if (later_corners[idx].x > target.cols - 1)later_corners[idx].x = target.cols;
                        if (later_corners[idx].y < 0)later_corners[idx].y = 0;
                        else if (later_corners[idx].y > target.rows - 1)later_corners[idx].y = target.rows;

                        point_pair.first = later_corners[idx];
                        DelaunayTriangulation md_later(Rectangle(0, 0, target.cols, target.rows));
                        md_later.insert(later_corners);
                        std::vector<Triangle> triangles_later;
                        int triangle_size_sum_later = 0;
                        double MSE_later = 0;
                        triangles_later = md_later.Get_triangles_around(idx, later_corners, flag_around);
#pragma omp parallel for
                        for (int t = 0; t < (int) triangles_later.size(); t++) {
                            int triangle_size;
                            Triangle triangle = triangles_later[t];
                            Point3Vec triangleVec(later_corners[triangle.p1_idx], later_corners[triangle.p2_idx],
                                                  later_corners[triangle.p3_idx]);
                            Point3Vec prev_corners = Point3Vec(ref_corners[triangle.p1_idx],
                                                               ref_corners[triangle.p2_idx],
                                                               ref_corners[triangle.p3_idx]);
                            //std::cout << "later_Gauss" << std::endl;
                            MSE_later += Gauss_Newton(ref, target, ref_intra, triangleVec, prev_corners, triangle_size);
                            triangle_size_sum_later += triangle_size;
                            //std::cout << "MSE_later = " << MSE_later << std::endl;
                        }
                        MSE_later /= triangle_size_sum_later;
                        point_pair.second = MSE_later;
                        point_pairs.emplace_back(point_pair);
                    }
                }
            }
*/
/*
        std::ofstream corner_list = std::ofstream("corner_list_" + corner_file_name + ".dat");
        for(const cv::Point2f point : corners){
            corner_list << point.x << " " << point.y << std::endl;
        }
*/
        std::ifstream in_corner_list = std::ifstream("corner_list_" + corner_file_name + ".dat");
         // std::ifstream in_corner_list = std::ifstream("corner_list_car_38_5_34.dat");
          std::string str1;
          int point_x,point_y;
          corners.clear();
          while (getline(in_corner_list, str1)) {
              sscanf(str1.data(), "%d %d", &point_x,&point_y);
              corners.emplace_back(cv::Point2f(point_x,point_y));
          }
/*
          cv::Mat color = cv::Mat::zeros(target.size(),CV_8UC3);
          std::vector<cv::Point2f> add_corner_dummy;
          int add_count_dummy = 0;
          cv::Mat predict_img0 = cv::Mat::zeros(targetx8.size(), CV_8UC3);
          cv::Mat predict_img1 = cv::Mat::zeros(targetx4.size(), CV_8UC3);
          cv::Mat predict_img2 = cv::Mat::zeros(targetx2.size(), CV_8UC3);
          cv::Mat predict_img3 = cv::Mat::zeros(target.size(), CV_8UC3);
          cv::Mat predict_warp = cv::Mat::zeros(target.size(),CV_8UC3);
          cv::Mat predict_para = cv::Mat::zeros(target.size(),CV_8UC3);
          cv::Point2f mv_diff,mv_prev;
          std::vector<cv::Mat> predict_buf;
          std::vector<std::vector<cv::Point2i>> buffer;
          std::vector<cv::Point2i> tmp;
          std::ofstream tri_list;
          bool para_flag = false;
          int Quant = 4;
          tri_list = std::ofstream("tri_list.csv");
          predict_buf.emplace_back(predict_img0);
          predict_buf.emplace_back(predict_img1);
          predict_buf.emplace_back(predict_img2);
          predict_buf.emplace_back(predict_img3);
          std::vector<cv::Point2f> edge_corners = slide_corner_edge(corners,canny_target,8);
          std::vector<cv::Point2f> later_corners = corners;
          for(int idx = 0;idx < (int)corners.size();idx++) {
              if (corners[idx].x == 0.0 || corners[idx].y == 0.0 ||
                  corners[idx].x == target.cols - 1 || corners[idx].y == target.rows - 1) {
                  continue;
              }
              std::vector<std::pair<cv::Point2f, double>> point_pairs;
              std::pair<cv::Point2f, double> point_pair;
              std::vector<bool> flag_around = std::vector<bool>(corners.size(), false);
              for(int i = 0;i <= 1;i++) {
                  if(i == 0)later_corners[idx] = corners[idx];
                  else if(i == 1)later_corners[idx] = edge_corners[idx];
                  for (int c_idx = 0; c_idx < (int) corners.size(); c_idx++) {
                      if (later_corners[idx] == corners[c_idx]) {
                          later_corners[idx] = corners[idx];
                      }
                  }
                  point_pair.first = later_corners[idx];
                  DelaunayTriangulation md_later(Rectangle(0, 0, target.cols, target.rows));
                  md_later.insert(later_corners);
                  std::vector<Triangle> triangles_later;
                  int triangle_size_sum_later = 0;
                  double MSE_later = 0;
                  triangles_later = md_later.Get_triangles_around(idx, later_corners, flag_around);
#pragma omp parallel for
                  for (int t = 0; t < (int) triangles_later.size(); t++) {
                      //std::cout << "t = " << t << " / " << triangles_later.size() << std::endl;
                      int triangle_size;
                      double error_warp;
                      Triangle triangle = triangles_later[t];
                      Point3Vec triangleVec(later_corners[triangle.p1_idx], later_corners[triangle.p2_idx],
                                            later_corners[triangle.p3_idx]);
                      Point3Vec prev_corners = Point3Vec(ref_corners[triangle.p1_idx], ref_corners[triangle.p2_idx],
                                                         ref_corners[triangle.p3_idx]);

                      //MSE_later += Gauss_Newton(ref, target, ref_intra, triangleVec, prev_corners, triangle_size);
                      Gauss_Newton2(ref, target, ref_intra, predict_buf, predict_warp, predict_para, color, error_warp,
                                    triangleVec, prev_corners, tri_list, &para_flag, add_corner_dummy, &add_count_dummy,
                                    t, residual_ref, triangle_size, false);
                      MSE_later += error_warp;
                      triangle_size_sum_later += triangle_size;
                  }
                  MSE_later /= triangle_size_sum_later;

                  point_pair.second = MSE_later;
                  if(i == 0)point_pair.second -= 0.5;
                  point_pairs.emplace_back(point_pair);
                  std::cout << "idx = " << idx << " / " << corners.size() << "i = " << i << "corners = "
                            << corners[idx] << "later_corners = " << later_corners[idx] << MSE_later << std::endl;
              }
              bubbleSort(point_pairs, point_pairs.size());
              corners[idx] = point_pairs[0].first;
          }
          */
/*
          std::vector<cv::Point2f> later_corners = corners;
          cv::Mat color = cv::Mat::zeros(target.size(),CV_8UC3);
          std::vector<cv::Point2f> add_corner_dummy;
          int add_count_dummy = 0;
          cv::Mat predict_img0 = cv::Mat::zeros(targetx8.size(), CV_8UC3);
          cv::Mat predict_img1 = cv::Mat::zeros(targetx4.size(), CV_8UC3);
          cv::Mat predict_img2 = cv::Mat::zeros(targetx2.size(), CV_8UC3);
          cv::Mat predict_img3 = cv::Mat::zeros(target.size(), CV_8UC3);
          cv::Mat predict_warp = cv::Mat::zeros(target.size(),CV_8UC3);
          cv::Mat predict_para = cv::Mat::zeros(target.size(),CV_8UC3);
          cv::Point2f mv_diff,mv_prev;
          std::vector<cv::Mat> predict_buf;
          std::vector<std::vector<cv::Point2i>> buffer;
          std::vector<cv::Point2i> tmp;
          std::ofstream tri_list;
          bool para_flag = false;
          int Quant = 4;

          tri_list = std::ofstream("tri_list.csv");
          predict_buf.emplace_back(predict_img0);
          predict_buf.emplace_back(predict_img1);
          predict_buf.emplace_back(predict_img2);
          predict_buf.emplace_back(predict_img3);
          for(int idx = 0;idx < (int)corners.size();idx++){
              if (corners[idx].x == 0.0 || corners[idx].y == 0.0 ||
                  corners[idx].x == target.cols - 1 || corners[idx].y == target.rows - 1) {
                  continue;
              }
              double min_distance = md.neighbor_distance(corners,idx);
              int mv_distance = std::min(4,(int)pow(2,(int)std::log2((min_distance/2))));
              std::cout << "min_distance = " << sqrt(min_distance) << std::endl;
              std::cout << "mv_distance = " << mv_distance << std::endl;
              std::vector<std::pair<cv::Point2f,double>> point_pairs;
              std::pair<cv::Point2f,double> point_pair;
              std::vector<bool> flag_around = std::vector<bool>(corners.size(), false);
              while(mv_distance >= 1) {
                  int triangle_size_sum_prev;
                  for (int direct = 0; direct < 9; direct++) {
                      if(direct == 0){
                          later_corners[idx].x = corners[idx].x;
                          later_corners[idx].y = corners[idx].y;
                      }
                      else if (direct == 1) {
                          later_corners[idx].x = corners[idx].x + mv_distance;
                          later_corners[idx].y = corners[idx].y;
                      } else if (direct == 2) {
                          later_corners[idx].x = corners[idx].x;
                          later_corners[idx].y = corners[idx].y + mv_distance;
                      } else if (direct == 3) {
                          later_corners[idx].x = corners[idx].x - mv_distance;
                          later_corners[idx].y = corners[idx].y;
                      } else if (direct == 4) {
                          later_corners[idx].x = corners[idx].x;
                          later_corners[idx].y = corners[idx].y - mv_distance;
                      } else if (direct == 5) {
                          later_corners[idx].x = corners[idx].x + mv_distance;
                          later_corners[idx].y = corners[idx].y + mv_distance;
                      } else if (direct == 6) {
                          later_corners[idx].x = corners[idx].x - mv_distance;
                          later_corners[idx].y = corners[idx].y + mv_distance;
                      } else if (direct == 7) {
                          later_corners[idx].x = corners[idx].x - mv_distance;
                          later_corners[idx].y = corners[idx].y - mv_distance;
                      } else if (direct == 8) {
                          later_corners[idx].x = corners[idx].x + mv_distance;
                          later_corners[idx].y = corners[idx].y - mv_distance;
                      }
                      for(int c_idx = 0;c_idx < (int)corners.size();c_idx++){
                          if(later_corners[idx] == corners[c_idx]){
                              later_corners[idx] = corners[idx];
                          }
                      }
                      if (later_corners[idx].x < 0)later_corners[idx].x = 0;
                      else if (later_corners[idx].x > target.cols - 1)later_corners[idx].x = target.cols - 1;
                      if (later_corners[idx].y < 0)later_corners[idx].y = 0;
                      else if (later_corners[idx].y > target.rows - 1)later_corners[idx].y = target.rows - 1;

                      point_pair.first = later_corners[idx];
                      DelaunayTriangulation md_later(Rectangle(0, 0, target.cols, target.rows));
                      md_later.insert(later_corners);
                      std::vector<Triangle> triangles_later;
                      int triangle_size_sum_later = 0;
                      double MSE_later = 0;
                      triangles_later = md_later.Get_triangles_around(idx, later_corners, flag_around);
#pragma omp parallel for
                      for (int t = 0; t < (int) triangles_later.size(); t++) {
                          //std::cout << "t = " << t << " / " << triangles_later.size() << std::endl;
                          int triangle_size;
                          double error_warp;
                          Triangle triangle = triangles_later[t];
                          Point3Vec triangleVec(later_corners[triangle.p1_idx], later_corners[triangle.p2_idx],
                                                later_corners[triangle.p3_idx]);
                          Point3Vec prev_corners = Point3Vec(ref_corners[triangle.p1_idx], ref_corners[triangle.p2_idx],
                                                             ref_corners[triangle.p3_idx]);

                          //MSE_later += Gauss_Newton(ref, target, ref_intra, triangleVec, prev_corners, triangle_size);
                          Gauss_Newton2(ref,target,ref_intra, predict_buf,predict_warp,predict_para, color, error_warp, triangleVec, prev_corners, tri_list,&para_flag,add_corner_dummy,&add_count_dummy,t,residual_ref,triangle_size, false);
                          MSE_later += error_warp;
                          triangle_size_sum_later += triangle_size;
                      }
                      if(direct == 0)triangle_size_sum_prev = triangle_size_sum_later;
                      double S_per = (double)triangle_size_sum_later/(double)triangle_size_sum_prev;
                      MSE_later /= triangle_size_sum_later;

                      point_pair.second = MSE_later;
                      if(direct == 0)point_pair.second -= 1;
                      point_pairs.emplace_back(point_pair);
                      std::cout << "idx = " << idx << " / " << corners.size() << "direct = " << direct << "corners = "
                                << corners[idx] << "later_corners = " << later_corners[idx] << MSE_later << std::endl;
                  }
                  bubbleSort(point_pairs, point_pairs.size());
                  corners[idx] = point_pairs[0].first;
                  mv_distance /= 2;
              }
          }
*/
          std::ofstream corner_list_later = std::ofstream("corner_list_" + corner_file_name + "_later.dat");
          for(const cv::Point2f point : corners){
              corner_list_later << point.x << " " << point.y << std::endl;
          }

/*
          corners.clear();
          corners.emplace_back(cv::Point2f(900,500));
          corners.emplace_back(cv::Point2f(0,0));
          corners.emplace_back(cv::Point2f(1919,0));
          corners.emplace_back(cv::Point2f(0,1023));
          corners.emplace_back(cv::Point2f(1919,1023));
*/
/*
          int W_num = 12,H_num = 8;
          int W_step = target.cols/W_num,H_step = target.rows/H_num;
          corners.clear();
          for(int j = 0;j <= H_num;j++){
              for(int i = 0;i <= W_num;i++){
                  int x = i*W_step,y = j*H_step;
                  if(x >= target.cols)x = target.cols - 1;
                  if(y >= target.rows)y = target.rows - 1;
                  corners.emplace_back(cv::Point2f(x,y));
              }
          }
*/
          for(int i = 0;i < (int)corners.size();i++){
              std::cout << "corner[" << i <<"] =" << corners[i] << std::endl;
          }
          std::cout << "corners's size :" << corners.size() << std::endl;
          std::cout << "ref_corners's size :" << ref_corners.size() << std::endl;
          ret_ref_corners = getReferenceImageCoordinates(ref, target, corners, points);

          ref_corners_org = ret_ref_corners.first;
          ref_corners.clear();
          for (int i = 0; i < (int) ref_corners_org.size(); i++) ref_corners.emplace_back(ref_corners_org[i]);
          std::cout << "corners's size :" << corners.size() << std::endl;
          std::cout << "ref_corners's size :" << ref_corners.size() << std::endl;
        /*
          md = DelaunayTriangulation(Rectangle(0, 0, target.cols, target.rows));
          md.insert(corners);
          md.getTriangleList(triangles_mydelaunay);
          corners = md.repair_around(corners,target);
          md = DelaunayTriangulation(Rectangle(0, 0, target.cols, target.rows));
          md.insert(corners);
          md.getTriangleList(triangles_mydelaunay);
          */
/*
        corners.clear();
          corners.emplace_back(cv::Point2f(0,0));
          corners.emplace_back(cv::Point2f(696,0));
          corners.emplace_back(cv::Point2f(1919,0));
          corners.emplace_back(cv::Point2f(704,460));
          corners.emplace_back(cv::Point2f(816,460));
          //corners.emplace_back(cv::Point2f(863,464));
          corners.emplace_back(cv::Point2f(920,472));
          corners.emplace_back(cv::Point2f(535,521));
          corners.emplace_back(cv::Point2f(531,500));
          corners.emplace_back(cv::Point2f(628,613));
          corners.emplace_back(cv::Point2f(651,455));
          corners.emplace_back(cv::Point2f(428,520));
          //corners.emplace_back(cv::Point2f(1119,620));
          //corners.emplace_back(cv::Point2f(1159,686));
          //corners.emplace_back(cv::Point2f(1154,633));
          //corners.emplace_back(cv::Point2f(628,655));
          corners.emplace_back(cv::Point2f(1078,631));
          //corners.emplace_back(cv::Point2f(1080,683));
          //corners.emplace_back(cv::Point2f(1152,694));
          corners.emplace_back(cv::Point2f(1105,713));
          corners.emplace_back(cv::Point2f(584,685));
          corners.emplace_back(cv::Point2f(582,634));
          corners.emplace_back(cv::Point2f(1032,544));
          corners.emplace_back(cv::Point2f(1208,568));//固定9
          corners.emplace_back(cv::Point2f(706,621));//670 635 708 621
          //corners.emplace_back(cv::Point2f(750,708));
         // corners.emplace_back(cv::Point2f(675,651));
          corners.emplace_back(cv::Point2f(848,626));
          //corners.emplace_back(cv::Point2f(48,70));
          corners.emplace_back(cv::Point2f(402,640));
          //corners.emplace_back(cv::Point2f(149,80));
          //corners.emplace_back(cv::Point2f(57,81));
          corners.emplace_back(cv::Point2f(640,704));
          //corners.emplace_back(cv::Point2f(108,90));
          // corners.emplace_back(cv::Point2f(140,89));
          corners.emplace_back(cv::Point2f(1225,725));
          corners.emplace_back(cv::Point2f(0,1023));
          corners.emplace_back(cv::Point2f(864,1023));
          corners.emplace_back(cv::Point2f(1919,1023));
          corners.emplace_back(cv::Point2f(1208,1023));
          //corners.emplace_back(cv::Point2f(0,320));
          corners.emplace_back(cv::Point2f(0,720));
          corners.emplace_back(cv::Point2f(1919,320));
          //corners.emplace_back(cv::Point2f(1919,90));
          corners.emplace_back(cv::Point2f(657,468));
          corners.emplace_back(cv::Point2f(1272,656));
          //corners.emplace_back(cv::Point2f(157,91));
          //corners.emplace_back(cv::Point2f(152,74));
          corners.emplace_back(cv::Point2f(432,712));
          corners.emplace_back(cv::Point2f(864,725));//864 736
          //corners.emplace_back(cv::Point2f(1023,711));
          //corners.emplace_back(cv::Point2f(865,710));
          //corners.emplace_back(cv::Point2f(816,576));
          corners.emplace_back(cv::Point2f(651,734));
          corners.emplace_back(cv::Point2f(1119,755));
          corners.emplace_back(cv::Point2f(1148,626));
        //corners.emplace_back(cv::Point2f(1054,521));
          //corners.emplace_back(cv::Point2f(156,100));
*/

        std::cout << "corner size(erased):" << corners.size() << std::endl;

        // 減らした点で細分割
        subdiv = cv::Subdiv2D(rect);
        subdiv.insert(corners);

        md = DelaunayTriangulation(Rectangle(0, 0, target.cols, target.rows));
        md.insert(corners);
        // 現状これやらないとneighbor_vtxがとれないので許して
        md.getTriangleList(triangles_mydelaunay);

        puts("insert done");

        // 三角網を描画します
        cv::Mat my_triangle = target.clone();
        triangle_error_img = target.clone();
        for (auto t:triangles_mydelaunay) {
          cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
          drawTriangle(my_triangle, p1, p2, p3, BLUE);
          drawTriangle(triangle_error_img, p1, p2, p3, BLUE);
        }

        if (HARRIS) {
          cv::imwrite(file_path + img_path + "my_triangle" + "_corner_" + std::to_string(corners.size()) + ".png",
                      my_triangle);
        } else if (THRESHOLD) {
          cv::imwrite(file_path + img_path + "my_triangle" + "_threshold_" + std::to_string(threshold) + "_lambda_" +
                      std::to_string(LAMBDA) + ".png",
                      my_triangle);
        }

        // 頂点の符号化の類
        std::vector<DelaunayTriangulation::PointCode> coded_coordinate = md.getPointCoordinateCode(corners, QUEUE);

        // 分布
        // たまに8の倍数からずれる事あるんだけど, それ多分右側と下側の座標が1919と1079なのでだとおもわれ
        // 1画素程度のズレを許容する気持ちで行く
        int min_coord_x = std::numeric_limits<int>::max(), min_coord_y = std::numeric_limits<int>::max();
        int max_coord_x = std::numeric_limits<int>::min(), max_coord_y = std::numeric_limits<int>::min();
        std::vector<int> prev_id_count(1001, 0);
        for (const auto &p : coded_coordinate) {
          prev_id_count[p.prev_id + 500]++;
          max_coord_x = std::max(max_coord_x, (int) p.coord.x);
          min_coord_x = std::min(min_coord_x, (int) p.coord.x);
          max_coord_y = std::max(max_coord_y, (int) p.coord.y);
          min_coord_y = std::min(min_coord_y, (int) p.coord.y);
        }

        int offset = QUANTIZE - (std::abs(min_coord_x) % QUANTIZE);

        // 8の倍数でない場合は適当にずらす
        if (min_coord_x % QUANTIZE != 0) {
          min_coord_x = min_coord_x < 0 ? min_coord_x - offset : min_coord_x + offset;
          std::cout << "offset:" << offset << std::endl;
        }
        if (max_coord_x % QUANTIZE != 0) {
          max_coord_x = max_coord_x < 0 ? max_coord_x - offset : max_coord_x + offset;
          std::cout << "offset:" << offset << std::endl;
        }
        if (min_coord_y % QUANTIZE != 0) {
          min_coord_y = min_coord_y < 0 ? min_coord_y - offset : min_coord_y + offset;
          std::cout << "offset:" << offset << std::endl;
        }
        if (max_coord_y % QUANTIZE != 0) {
          max_coord_y = max_coord_y < 0 ? max_coord_y - offset : max_coord_y + offset;
          std::cout << "offset:" << offset << std::endl;
        }

        min_coord_x = (abs(min_coord_x) / QUANTIZE);
        min_coord_y = (abs(min_coord_y) / QUANTIZE);
        max_coord_x = (abs(max_coord_x) / QUANTIZE);
        max_coord_y = (abs(max_coord_y) / QUANTIZE);
        std::vector<int> freq_coord_x(max_coord_x + min_coord_x + 1, 0);
        std::vector<int> freq_coord_y(max_coord_y + min_coord_y + 1, 0);
        std::cout << "freq_coord_x.size = " << freq_coord_x.size() << std::endl;
        std::cout << "freq_coord_y.size = " << freq_coord_y.size() << std::endl;
        // 頻度を求める奴
        for (const auto &p : coded_coordinate) {
          point_nums += 1.0;
          if (static_cast<int>(p.coord.x) % QUANTIZE != 0) {
            offset = QUANTIZE - (std::abs((int) p.coord.x) % QUANTIZE);
            if (p.coord.x < 0) offset *= -1;
            freq_coord_x[(p.coord.x + offset) / QUANTIZE + min_coord_x]++;
          } else {
            freq_coord_x[(p.coord.x) / QUANTIZE + min_coord_x]++;
          }

          if (static_cast<int>(p.coord.y) % QUANTIZE != 0) {
            offset = QUANTIZE - (std::abs((int) p.coord.y) % QUANTIZE);
            if (p.coord.y < 0) offset *= -1;
            freq_coord_y[(p.coord.y + offset) / QUANTIZE + min_coord_y]++;
          } else {
            freq_coord_y[(p.coord.y) / QUANTIZE + min_coord_y]++;
          }
        }
        int max_freq_x = 0;
        for (int i = 0; i < static_cast<int>(freq_coord_x.size()); i++) {
          if (freq_coord_x[max_freq_x] < freq_coord_x[i]) {
            max_freq_x = i;
          }
        }
        max_freq_x -= min_coord_x;

        int max_freq_y = 0;
        for (int i = 0; i < static_cast<int>(freq_coord_y.size()); ++i) {
          if (freq_coord_y[max_freq_y] < freq_coord_y[i]) {
            max_freq_y = i;
          }
        }
        max_freq_y -= min_coord_y;
        int golomb_x = 0, golomb_y = 0;
          std::cout << "cehck1" << std::endl;
        //
        // 頂点復号ベクトルのx成分の頻度
        //
        FILE *fp = fopen((file_path + img_path + csv_file_prefix + "corner_decode_vector_x_freq.csv").c_str(), "w");
        std::ofstream os(file_path + img_path + "gp\\corner_decode_vector_x_freq.txt");
        storeGnuplotFile(file_path + img_path + "gp\\decode_x.gp", "length of x coordinate.", "Frequency",
                         "corner_decode_vector_x_freq.txt");
          std::cout << "cehck2" << std::endl;
        double mean = 0.0;
        std::cout << csv_file_prefix << std::endl;
        for (int i = 0; i < (int) freq_coord_x.size(); i++) {
          std::cout << i << " " << freq_coord_x[i] << std::endl;
          fprintf(fp, "%d,%d\n", i - min_coord_x, freq_coord_x[i]);
          std::cout << "check1" << std::endl;
          os << i - min_coord_x << " " << freq_coord_x[i] << std::endl;
          std::cout << "check2" << std::endl;
          mean += (i - min_coord_x) * freq_coord_x[i];
        }

        fclose(fp);
        os.close();
          std::cout << "cehck3" << std::endl;
        // 平均引いたやつをシフトするもの
        os = std::ofstream(file_path + img_path + "gp\\corner_decode_vector_x_freq_mean.txt");
        storeGnuplotFile(file_path + img_path + "gp\\decode_x_mean.gp", "length of x coordinate", "Frequency",
                         "corner_decode_vector_x_freq_mean.txt");
        mean /= corners.size();
        for (int i = 0; i < (int) freq_coord_x.size(); i++) {
          os << i - min_coord_x - mean << " " << freq_coord_x[i] << std::endl;
        }
        os.close();
          std::cout << "cehck4" << std::endl;
        // max分ずらすグラフ
        os = std::ofstream(file_path + img_path + "gp\\corner_decode_vector_x_freq_max.txt");
        storeGnuplotFile(file_path + img_path + "gp\\decode_x_max.gp", "length of x coordinate", "Frequency",
                         "corner_decode_vector_x_freq_max.txt");
        mean /= corners.size();
        for (int i = 0; i < (int) freq_coord_x.size(); i++) {
          os << i - min_coord_x - max_freq_x << " " << freq_coord_x[i] << std::endl;
          golomb_x += (ozi::getGolombCode(ozi::getGolombParam(0.6), (i - min_coord_x - max_freq_x), ozi::REGION1,
                                          ozi::KTH_GOLOMB,9)) * freq_coord_x[i];
        }
        os.close();
          std::cout << "cehck5" << std::endl;
        //
        // 頂点復号ベクトルのy成分の頻度
        //
        fp = fopen((file_path + img_path + csv_file_prefix + "corner_decode_vector_y_freq.csv").c_str(), "w");
        os = std::ofstream(file_path + img_path + "gp\\corner_decode_vector_y_freq.txt");
        storeGnuplotFile(file_path + img_path + "gp\\decode_y.gp", "length of y coordinate.", "Frequency",
                         "corner_decode_vector_y_freq.txt");
        mean = 0.0;

        for (int i = 0; i < (int) freq_coord_y.size(); i++) {
          fprintf(fp, "%d,%d\n", i - min_coord_y, freq_coord_y[i]);
          os << i - min_coord_y << " " << freq_coord_y[i] << std::endl;
          mean += (i - min_coord_y) * freq_coord_y[i];
        }
        fclose(fp);
        os.close();

        // 平均分ずらすやつ
        os = std::ofstream(file_path + img_path + "gp\\corner_decode_vector_y_freq_mean.txt");
        storeGnuplotFile(file_path + img_path + "gp\\decode_y_mean.gp", "length of y coordinate", "Frequency",
                         "corner_decode_vector_y_freq_mean.txt");
        mean /= corners.size();
        for (int i = 0; i < (int) freq_coord_y.size(); i++) {
          os << i - min_coord_y - mean << " " << freq_coord_y[i] << std::endl;
        }
        os.close();

        // 最大値ずらすやつ
        os = std::ofstream(file_path + img_path + "gp\\corner_decode_vector_y_freq_max.txt");
        storeGnuplotFile(file_path + img_path + "gp\\decode_y_max.gp", "length of y coordinate", "Frequency",
                         "corner_decode_vector_y_freq_max.txt");
        mean /= corners.size();
        for (int i = 0; i < (int) freq_coord_y.size(); i++) {
          os << i - min_coord_y - max_freq_y << " " << freq_coord_y[i] << std::endl;
          golomb_y += (ozi::getGolombCode(ozi::getGolombParam(0.6), (i - min_coord_y - max_freq_y), ozi::REGION1,
                                          ozi::KTH_GOLOMB,9)) * freq_coord_y[i];
        }
        os.close();

        cv::Mat decoded_corner = md.getDecodedCornerImage(coded_coordinate, target, RASTER_SCAN);
        for (const auto &corner : corners) drawPoint(decoded_corner, corner, RED, 3);

        if (HARRIS) {
          cv::imwrite(
                  file_path + img_path + "decoded_corner" + "_cornersize_" + std::to_string(corners.size()) + ".png",
                  decoded_corner);
        } else if (THRESHOLD) {
          cv::imwrite(file_path + img_path + "decoded_corner_threshold_" + std::to_string(threshold) + "_lambda_" +
                      std::to_string(LAMBDA) + ".png", decoded_corner);
        }


        //
        // MVの要素について集計
        //
        std::vector<cv::Point2f> code = md.getPointMotionVectorCode(corners, ref_corners);
        std::cout << "code.size() : " << code.size() << std::endl;

        cv::Mat decoded_mv = md.getDecodedMotionVectorImage(code, corners, target);

//        int golomb_mv_x = 0, golomb_mv_y = 0;

        int min_mv_x = std::numeric_limits<int>::max(), min_mv_y = std::numeric_limits<int>::max();
        int max_mv_x = std::numeric_limits<int>::min(), max_mv_y = std::numeric_limits<int>::min();
        for (const auto &p : code) {
          max_mv_x = std::max(max_mv_x, (int) p.x);
          min_mv_x = std::min(min_mv_x, (int) p.x);
          max_mv_y = std::max(max_mv_y, (int) p.y);
          min_mv_y = std::min(min_mv_y, (int) p.y);
        }
        min_mv_x = abs(min_mv_x);
        min_mv_y = abs(min_mv_y);

        std::vector<int> freq_x(max_mv_x + min_mv_x + 1, 0);
        std::vector<int> freq_y(max_mv_y + min_mv_y + 1, 0);
        for (const auto &p : code) {
          freq_x[p.x + min_mv_x]++;
          freq_y[p.y + min_mv_y]++;
        }

        // mvのxについて集計
        fp = fopen((file_path + img_path + csv_file_prefix + "corner_mv_x_freq.csv").c_str(), "w");
        os = std::ofstream(file_path + img_path + "gp\\corner_mv_x_freq.txt");
        storeGnuplotFile(file_path + img_path + "gp\\corner_mv_x.gp", "length of x coordinate.", "Frequency",
                         "corner_mv_x_freq.txt");
        for (int i = 0; i < (int) freq_x.size(); i++) {
          fprintf(fp, "%d,%d\n", i - min_mv_x, freq_x[i]);
          os << i - min_mv_x << " " << freq_x[i] << std::endl;
          //  golomb_mv_x += ozi::getGolombCode(2, (i - min_mv_x), ozi::REGION1, ozi::GOLOMB, 0);
          /*
          golomb_mv_x +=
                  (ozi::getGolombCode(ozi::getGolombParam(0.24), (i - min_mv_x), 0, ozi::KTH_GOLOMB)) * freq_x[i];
                  */
        }
        fclose(fp);
        os.close();

        // mvのyについて集計
        fp = fopen((file_path + img_path + csv_file_prefix + "corner_mv_y_freq.csv").c_str(), "w");
        os = std::ofstream(file_path + img_path + "gp\\corner_mv_y_freq.txt");
        storeGnuplotFile(file_path + img_path + "gp\\corner_mv_y.gp", "length of y coordinate.", "Frequency",
                         "corner_mv_y_freq.txt");
        for (int i = 0; i < (int) freq_y.size(); i++) {
          fprintf(fp, "%d,%d\n", i - min_mv_y, freq_y[i]);
          os << i - min_mv_y << " " << freq_y[i] << std::endl;
         //   golomb_mv_y += ozi::getGolombCode(2, (i - min_mv_y), ozi::REGION1, ozi::GOLOMB, 0);
          /*
          golomb_mv_y +=
                  (ozi::getGolombCode(ozi::getGolombParam(0.24), (i - min_mv_y), 0, ozi::KTH_GOLOMB)) * freq_y[i];
                  */
        }
        fclose(fp);
        os.close();

        if (HARRIS) {
          cv::imwrite(file_path + img_path + "decoded_mv_corner_size_" + std::to_string(corners.size()) + ".png",
                      decoded_mv);
        } else if (THRESHOLD) {
          cv::imwrite(file_path + img_path + "decoded_mv_threshold_" + std::to_string(threshold) + "_lambda_" +
                      std::to_string(LAMBDA) + ".png", decoded_mv);
        }

        // 三角形の取得
        std::vector<cv::Vec6f> triangles_as_vec6f;
        std::vector<int> leading_edge_list;
        std::vector<cv::Vec4f> edge_list;
        subdiv.getTriangleList(triangles_as_vec6f);

        cv::Mat triangle_target = target.clone();
        cv::Mat triangle_add_target = target.clone();

        std::vector<Triangle> triangles;
        cv::Mat mv_image = target.clone();

        // 頂点とindexを結びつけ
        for (auto t:triangles_as_vec6f) {
          cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);

          // 三角形を描画
          drawTriangle(triangle_target, p1, p2, p3, cv::Scalar(255, 255, 255));
          drawTriangle(mv_image, p1, p2, p3, cv::Scalar(255, 255, 255));
          drawTriangle(triangle_add_target, p1, p2, p3, cv::Scalar(255, 255, 255));

          int i1 = -1, i2 = -1, i3 = -1;
          for (int i = 0; i < (int) corners.size(); i++) {
            if (corners[i] == p1) i1 = i;
            else if (corners[i] == p2) i2 = i;
            else if (corners[i] == p3) i3 = i;
          }

          if (0 <= i1 && 0 <= i2 && 0 <= i3) {
            triangles.emplace_back(i1, i2, i3);
          }
        }


        if (HARRIS) {
          cv::imwrite(file_path + img_path + "triangle_" + out_file[0] + "_corner_size_" +
                      std::to_string(corners.size()) + "_lambda_" + std::to_string(LAMBDA) + ".png", triangle_target);
        } else if (THRESHOLD) {
          cv::imwrite(file_path + img_path + "triangle_" + out_file[0] + "_threshold_" + std::to_string(threshold) +
                      "_lambda_" + std::to_string(LAMBDA) + ".png", triangle_target);
        }

        puts("");

        cv::Mat target_point = target.clone();
        cv::Mat out = cv::Mat::zeros(target.size(), CV_8UC3);

        std::cout << "corners.size():" << corners.size() << std::endl;
        std::cout << "intra col = " << ref_intra.cols << "row = " << ref_intra.rows << std::endl;
        std::vector<cv::Point2f> add_corners;
        int add_count = 0;
        int tmp_mv_x = 0;
        int tmp_mv_y = 0;
        add_corners.clear();
          std::cout << "check point 3" << std::endl;
          PredictedImageResult result = getPredictedImage(ref, refx2, refx4, refx8, target, targetx2, targetx4,
                                                          targetx8, ref_intra, triangles, ref_corners, corners, md,
                                                          add_corners,&add_count, residual_ref,tmp_mv_x,tmp_mv_y,true);
        // 予測画像を得る
            int count = 0;

         // while(add_corners.size() != 0) {
              for(const cv::Point2f p:add_corners) {
                  drawPoint(triangle_add_target, p, GREEN, 4);
              }

              //std::copy(add_corners.begin(),add_corners.end(),std::back_inserter(corners));
              std::copy(add_corners.begin(),add_corners.end(),std::back_inserter(ref_corners));
              for(int i = 0;i < (int)corners.size();i++){
                  std::cout << "corner[" << i <<"] =" << corners[i] << std::endl;
              }
              add_corners.clear();
              if(!INTER_DIV) {
                md = DelaunayTriangulation(Rectangle(0, 0, target.cols, target.rows));
                md.insert(corners);
                md.getTriangleList(triangles_mydelaunay);
                triangles.clear();
                for (auto t:triangles_mydelaunay) {
                  cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
                  int i1 = -1, i2 = -1, i3 = -1;
                  for (int i = 0; i < (int) corners.size(); i++) {
                    if (corners[i] == p1) i1 = i;
                    else if (corners[i] == p2) i2 = i;
                    else if (corners[i] == p3) i3 = i;
                  }

                  if (0 <= i1 && 0 <= i2 && 0 <= i3) {
                    triangles.emplace_back(i1, i2, i3);
                  }
                  drawTriangle_residual(triangle_add_target, p1, p2, p3, BLUE, sobel_target);
                }
              }
              else{
                for(int t = 0;t < (int)triangles.size();t++){
                  Point3Vec triangleVec(corners[triangles[t].p1_idx], corners[triangles[t].p2_idx], corners[triangles[t].p3_idx]);
                  cv::Point2f p1 = triangleVec.p1;
                  cv::Point2f p2 = triangleVec.p2;
                  cv::Point2f p3 = triangleVec.p3;
                  drawTriangle_residual(triangle_add_target, p1, p2, p3, BLUE, sobel_target);
                }
              }
                cv::imwrite(file_path + img_path + "triangle_add_" + out_file[0] + "_count_" + std::to_string(count) +
                            "_lambda_" + std::to_string(LAMBDA) + ".png", triangle_add_target);

/*
              result = getPredictedImage(ref, refx2, refx4, refx8, target, targetx2, targetx4,
                                         targetx8, ref_intra, triangles, ref_corners, corners, md,
                                         add_corners,&add_count, residual_ref,tmp_mv_x,tmp_mv_y,false);
*/
              count++;
         // }

          std::vector<DelaunayTriangulation::PointCode> coded_coordinate_later = md.getPointCoordinateCode(corners, QUEUE);

          // 分布
          // たまに8の倍数からずれる事あるんだけど, それ多分右側と下側の座標が1919と1079なのでだとおもわれ
          // 1画素程度のズレを許容する気持ちで行く
          int min_coord_x_later = std::numeric_limits<int>::max(), min_coord_y_later = std::numeric_limits<int>::max();
          int max_coord_x_later = std::numeric_limits<int>::min(), max_coord_y_later = std::numeric_limits<int>::min();
          std::vector<int> prev_id_count_later(1001, 0);
          for (const auto &p : coded_coordinate_later) {
              prev_id_count_later[p.prev_id + 500]++;
              max_coord_x_later = std::max(max_coord_x_later, (int) p.coord.x);
              min_coord_x_later = std::min(min_coord_x_later, (int) p.coord.x);
              max_coord_y_later = std::max(max_coord_y_later, (int) p.coord.y);
              min_coord_y_later = std::min(min_coord_y_later, (int) p.coord.y);
          }

          int offset_later = QUANTIZE - (std::abs(min_coord_x_later) % QUANTIZE);

          // 8の倍数でない場合は適当にずらす
          if (min_coord_x_later % QUANTIZE != 0) {
              min_coord_x_later = min_coord_x_later < 0 ? min_coord_x_later - offset_later : min_coord_x_later + offset_later;
              std::cout << "offset_later:" << offset_later << std::endl;
          }
          if (max_coord_x_later % QUANTIZE != 0) {
              max_coord_x_later = max_coord_x_later < 0 ? max_coord_x_later - offset_later : max_coord_x_later + offset_later;
              std::cout << "offset_later:" << offset_later << std::endl;
          }
          if (min_coord_y_later % QUANTIZE != 0) {
              min_coord_y_later = min_coord_y_later < 0 ? min_coord_y_later - offset_later : min_coord_y_later + offset_later;
              std::cout << "offset_later:" << offset_later << std::endl;
          }
          if (max_coord_y_later % QUANTIZE != 0) {
              max_coord_y_later = max_coord_y_later < 0 ? max_coord_y_later - offset_later : max_coord_y_later + offset_later;
              std::cout << "offset_later:" << offset_later << std::endl;
          }

          min_coord_x_later = (abs(min_coord_x_later) / QUANTIZE);
          min_coord_y_later = (abs(min_coord_y_later) / QUANTIZE);
          max_coord_x_later = (abs(max_coord_x_later) / QUANTIZE);
          max_coord_y_later = (abs(max_coord_y_later) / QUANTIZE);
          std::vector<int> freq_coord_x_later(max_coord_x_later + min_coord_x_later + 1, 0);
          std::vector<int> freq_coord_y_later(max_coord_y_later + min_coord_y_later + 1, 0);
          std::cout << "freq_coord_x_later.size = " << freq_coord_x_later.size() << std::endl;
          std::cout << "freq_coord_y_later.size = " << freq_coord_y_later.size() << std::endl;
          // 頻度を求める奴
          for (const auto &p : coded_coordinate_later) {
              point_nums += 1.0;
              if (static_cast<int>(p.coord.x) % QUANTIZE != 0) {
                  offset_later = QUANTIZE - (std::abs((int) p.coord.x) % QUANTIZE);
                  if (p.coord.x < 0) offset *= -1;
                  freq_coord_x_later[(p.coord.x + offset_later) / QUANTIZE + min_coord_x_later]++;
              } else {
                  freq_coord_x_later[(p.coord.x) / QUANTIZE + min_coord_x_later]++;
              }

              if (static_cast<int>(p.coord.y) % QUANTIZE != 0) {
                  offset_later = QUANTIZE - (std::abs((int) p.coord.y) % QUANTIZE);
                  if (p.coord.y < 0) offset_later *= -1;
                  freq_coord_y_later[(p.coord.y + offset_later) / QUANTIZE + min_coord_y_later]++;
              } else {
                  freq_coord_y_later[(p.coord.y) / QUANTIZE + min_coord_y_later]++;
              }
          }
          int max_freq_x_later = 0;
          for (int i = 0; i < static_cast<int>(freq_coord_x_later.size()); i++) {
              if (freq_coord_x_later[max_freq_x_later] < freq_coord_x_later[i]) {
                  max_freq_x_later = i;
              }
          }
          max_freq_x_later -= min_coord_x_later;

          int max_freq_y_later = 0;
          for (int i = 0; i < static_cast<int>(freq_coord_y_later.size()); ++i) {
              if (freq_coord_y_later[max_freq_y_later] < freq_coord_y_later[i]) {
                  max_freq_y_later = i;
              }
          }
          max_freq_y_later -= min_coord_y_later;
          golomb_x = 0, golomb_y = 0;
          std::cout << "cehck1" << std::endl;
          //
          // 頂点復号ベクトルのx成分の頻度
          //
          storeGnuplotFile(file_path + img_path + "gp\\decode_x.gp", "length of x coordinate.", "Frequency",
                           "corner_decode_vector_x_freq.txt");
          std::cout << "cehck2" << std::endl;
          mean = 0.0;
          fp = fopen((file_path + img_path + csv_file_prefix + "corner_mv_x_freq.csv").c_str(), "w");
          os = std::ofstream(file_path + img_path + "gp\\corner_mv_x_freq.txt");
          storeGnuplotFile(file_path + img_path + "gp\\corner_mv_x.gp", "length of x coordinate.", "Frequency",
                           "corner_mv_x_freq.txt");
          for (int i = 0; i < (int) freq_coord_x_later.size(); i++) {
              std::cout << i << " " << freq_coord_x_later[i] << std::endl;
              fprintf(fp, "%d,%d\n", i - min_coord_x_later, freq_coord_x_later[i]);
              std::cout << "check1" << std::endl;
              os << i - min_coord_x_later << " " << freq_coord_x_later[i] << std::endl;
              std::cout << "check2" << std::endl;
              mean += (i - min_coord_x_later) * freq_coord_x_later[i];
          }

          fclose(fp);
          os.close();
          std::cout << "cehck3" << std::endl;
          // 平均引いたやつをシフトするもの
          os = std::ofstream(file_path + img_path + "gp\\corner_decode_vector_x_freq_mean.txt");
          storeGnuplotFile(file_path + img_path + "gp\\decode_x_mean.gp", "length of x coordinate", "Frequency",
                           "corner_decode_vector_x_freq_mean.txt");
          mean /= corners.size();
          for (int i = 0; i < (int) freq_coord_x_later.size(); i++) {
              os << i - min_coord_x_later - mean << " " << freq_coord_x_later[i] << std::endl;
          }
          os.close();
          std::cout << "cehck4" << std::endl;
          // max分ずらすグラフ
          os = std::ofstream(file_path + img_path + "gp\\corner_decode_vector_x_freq_max.txt");
          storeGnuplotFile(file_path + img_path + "gp\\decode_x_max.gp", "length of x coordinate", "Frequency",
                           "corner_decode_vector_x_freq_max.txt");
          mean /= corners.size();
          for (int i = 0; i < (int) freq_coord_x_later.size(); i++) {
              os << i - min_coord_x_later - max_freq_x_later << " " << freq_coord_x_later[i] << std::endl;
              golomb_x += (ozi::getGolombCode(ozi::getGolombParam(0.6), (i - min_coord_x_later - max_freq_x_later), ozi::REGION1,
                                              ozi::KTH_GOLOMB,9)) * freq_coord_x_later[i];
          }
          os.close();
          std::cout << "cehck5" << std::endl;
          //
          // 頂点復号ベクトルのy成分の頻度
          //
          fp = fopen((file_path + img_path + csv_file_prefix + "corner_decode_vector_y_freq.csv").c_str(), "w");
          os = std::ofstream(file_path + img_path + "gp\\corner_decode_vector_y_freq.txt");
          storeGnuplotFile(file_path + img_path + "gp\\decode_y.gp", "length of y coordinate.", "Frequency",
                           "corner_decode_vector_y_freq.txt");
          mean = 0.0;

          for (int i = 0; i < (int) freq_coord_y_later.size(); i++) {
              fprintf(fp, "%d,%d\n", i - min_coord_y_later, freq_coord_y_later[i]);
              os << i - min_coord_y_later << " " << freq_coord_y_later[i] << std::endl;
              mean += (i - min_coord_y_later) * freq_coord_y_later[i];
          }
          fclose(fp);
          os.close();

          // 平均分ずらすやつ
          os = std::ofstream(file_path + img_path + "gp\\corner_decode_vector_y_freq_mean.txt");
          storeGnuplotFile(file_path + img_path + "gp\\decode_y_mean.gp", "length of y coordinate", "Frequency",
                           "corner_decode_vector_y_freq_mean.txt");
          mean /= corners.size();
          for (int i = 0; i < (int) freq_coord_y_later.size(); i++) {
              os << i - min_coord_y_later - mean << " " << freq_coord_y_later[i] << std::endl;
          }
          os.close();

          // 最大値ずらすやつ
          os = std::ofstream(file_path + img_path + "gp\\corner_decode_vector_y_freq_max.txt");
          storeGnuplotFile(file_path + img_path + "gp\\decode_y_max.gp", "length of y coordinate", "Frequency",
                           "corner_decode_vector_y_freq_max.txt");
          mean /= corners.size();
          for (int i = 0; i < (int) freq_coord_y_later.size(); i++) {
              os << i - min_coord_y_later - max_freq_y_later << " " << freq_coord_y_later[i] << std::endl;
              golomb_y += (ozi::getGolombCode(ozi::getGolombParam(0.6), (i - min_coord_y_later - max_freq_y_later), ozi::REGION1,
                                              ozi::KTH_GOLOMB,9)) * freq_coord_y_later[i];
          }
          os.close();

          golomb_mv_x += tmp_mv_x;
          golomb_mv_y += tmp_mv_y;

          for (auto t:triangles_mydelaunay) {
              cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
              //drawTriangle_residual(triangle_add_target, p1, p2, p3, BLUE,sobel_target);
          }
          cv::imwrite(file_path + img_path + "triangle_add_" + out_file[0] + "_threshold_" + std::to_string(threshold) +
                      "_lambda_" + std::to_string(LAMBDA) + ".png", triangle_add_target);
              std::cout << "check point 3" << std::endl;
        out = result.out;
        std::cout << "check point 4" << std::endl;
        std::cout << "corners.size():" << corners.size() << std::endl;

        // ===========================================================
        // ログ出力
        // ===========================================================
        puts("======================================================");
          int H = target.rows;
          int W = target.cols;
          for(int crop_W = 8, crop_H = 8;crop_W <= 32;crop_H += 8,crop_W += 8) {

              std::cout << "inner PSNR : "
                        << getPSNR(target, out, cv::Rect(crop_W, crop_H, W - crop_W * 2, H - crop_H * 2)) << " crop " << crop_H
                        << std::endl;
          }
          //int crop_W = 128;
          //int crop_H = 128;
        clock_t end = clock();
        int t = (int) ((double) (end - start) / CLOCKS_PER_SEC);
        std::cout << std::to_string(t / 60) + "m" + std::to_string(t % 60) + "sec" << std::endl;
        std::cout << "freq_block:" << result.freq_block << "(" << result.getBlockMatchingFrequency() << "%)"
                  << std::endl;
        std::cout << "freq_warp:" << result.freq_warp << "(" << result.getWarpingFrequency() << "%)" << std::endl;
        std::cout << "corners.size():" << corners.size() << std::endl;

        // 生成したターゲット画像
        cv::imwrite(out_file_name, out);
        std::cout << "check point 1" << std::endl;
        /*
        // 原画像をクロップしたもの
        if (HARRIS)
          cv::imwrite(file_path + img_path + "crop1_corner_size_" + std::to_string(corners.size()) + ".bmp",
                      target(cv::Rect(crop_H, crop_W,  H- crop_H, W - crop_W)));
        else if (THRESHOLD)
          cv::imwrite(file_path + img_path + "crop1_threshold_" + std::to_string(threshold) + "_lambda_" +
                      std::to_string(LAMBDA) + ".bmp", target(cv::Rect(crop_W, crop_H, W - crop_W * 2, H - crop_H * 2)));
        std::cout << "check point 2" << std::endl;
        // 生成したものをクロップしたもの
        if (HARRIS)
          cv::imwrite(file_path + img_path + "crop2_corner_size_" + std::to_string(corners.size()) + ".bmp",
                      out(cv::Rect(crop_W, crop_H, W - crop_W * 2, H - crop_H * 2)));
        else if (THRESHOLD)
          cv::imwrite(file_path + img_path + "crop2_threshold_" + std::to_string(threshold) + "_lambda_" +
                      std::to_string(LAMBDA) + ".bmp", out(cv::Rect(crop_W, crop_H, W - crop_W * 2, H - crop_H * 2)));
        std::cout << "check point 3" << std::endl;
        // 動きベクトルを出したもの
        if (HARRIS)
          cv::imwrite(file_path + img_path + "mv_image_corner_size_" + std::to_string(corners.size()) + ".png",
                      result.mv_image);
        else if (THRESHOLD)
          cv::imwrite(file_path + img_path + "mv_image_threshold_" + std::to_string(threshold) + "_lambda_" +
                      std::to_string(LAMBDA) + ".png", result.mv_image);
                      */
        cv::Mat residual = cv::Mat::zeros(target.size(), CV_8UC3);
        //out = cv::imread(file_path + img_path + "prediction_HEVC_27.bmp");

        for(int j = 0;j < target.rows;j++){
         for(int i = 0;i < target.cols;i++){
            int y = 4 * abs(R(target,i,j) - R(out,i,j));
            if(y < 0)y = 0;
         else if(y > 255)y = 255;
           R(residual,i,j) = (unsigned char)y;
            G(residual,i,j) = (unsigned char)y;
            B(residual,i,j) = (unsigned char)y;
            }
        }
        md.getTriangleList(triangles_mydelaunay);
          for(const cv::Vec6f t : triangles_mydelaunay){
              cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
              //drawTriangle(corner_reduction, p1, p2, p3, BLUE);
              drawTriangle(residual, p1, p2, p3, RED);
          }
        cv::imwrite(file_path + img_path + "residual.png",residual);
          std::cout << "check point 4" << std::endl;
        double psnr_1;
        printf("%s's PSNR:%f\n", out_file_name.c_str(), (psnr_1 = getPSNR(target, out)));
        std::cout << "check point 5" << std::endl;
        // 四角形を描画した画像を出力
        cv::Point2f p1 = cv::Point2f(150, 100);
        cv::Point2f p2 = cv::Point2f(target.cols - 151, 100);
        cv::Point2f p3 = cv::Point2f(target.cols - 151, target.rows - 101);
        cv::Point2f p4 = cv::Point2f(150, target.rows - 101);
        drawRectangle(out, p1, p2, p3, p4);
        cv::imwrite(file_path + img_path + "rect.png", out);
        std::cout << "check point 6" << std::endl;
        // ログ -------------------------------------------------------------------------------
        fp = fopen("C:\\Users\\takahiro\\CLionProjects\\Research-for-Motion-Compensation\\log.txt", "a");
        time_t tt;
        time(&tt);
        char date[64];
        strftime(date, sizeof(date), "%Y/%m/%d %a %H:%M:%S", localtime(&tt));

        fprintf(fp, (out_file_name + "\n").c_str());
        if (WARP_AVAILABLE) fprintf(fp, "WARPING ON\n");
        if (BM_AVAILABLE) fprintf(fp, "BlockMatching ON\n");
        if (HARRIS) fprintf(fp, "HARRIS CORNER LIMIT MODE\n");
        if (THRESHOLD) fprintf(fp, "THRESHOLD MODE");
        fprintf(fp, ("lambda:" + std::to_string(LAMBDA)).c_str());
        fprintf(fp, "QUANTIZE_STEP:%d\n", QUANTIZE);
        fprintf(fp, "%s\n", date);
        fprintf(fp, "PSNR : %f\n", psnr_1);
        fprintf(fp, "freq_block:%d(%f%%)\n", result.freq_block, result.getBlockMatchingFrequency());
        fprintf(fp, "freq_warp:%d(%f%%)\n", result.freq_warp, result.getWarpingFrequency());
        fprintf(fp, "BlockMatching's PSNR : %f\n", result.getBlockMatchingPatchPSNR());
        fprintf(fp, "Warping's PSNR : %f\n", result.getWarpingPatchPSNR());
//        fprintf(fp, "erase elem size : %d\b", cnt_erased_elem);
        fprintf(fp, (std::to_string(t / 60) + "m" + std::to_string(t % 60) + "sec\n\n").c_str());
        fclose(fp);
        std::cout << "log writed" << std::endl;
        // 符号量たち
        int prev_id_code_amount = 0;
        for (int i = 0; i <= 1000; i++) {
          if (prev_id_count[i] != 0) {
            prev_id_code_amount +=
                    ozi::getGolombCode(ozi::getGolombParam(0.5), i - 500, ozi::REGION1, ozi::KTH_GOLOMB, 3) *
                    prev_id_count[i];
          }
        }
        code_amount << "reference  :" << ref_file_name << std::endl;
        code_amount << "target     :" << target_file_name << std::endl;
        code_amount << "threshold  :" << threshold << std::endl;
        code_amount << "corner size:" << corners.size() << std::endl;
        code_amount << "mode       :" << (THRESHOLD ? "threshold" : "harris") << std::endl;
        code_amount << "coordinate vector ---------------------" << std::endl;
        code_amount << "golomb code(x) : " << golomb_x << std::endl;
        code_amount << "golomb code(y) : " << golomb_y << std::endl;
        code_amount << "motion vector ---------------------" << std::endl;
        code_amount << "golomb code(x) : " << golomb_mv_x << std::endl;
        code_amount << "golomb code(y) : " << golomb_mv_y << std::endl;
        code_amount << "diff vector ---------------------" << std::endl;
        code_amount << "golomb code(x) : " << result.getXbits() << std::endl;
        code_amount << "golomb code(y) : " << result.getYbits() << std::endl;
        code_amount << "technic flag ---------------------" << std::endl;
        code_amount << triangles.size() << std::endl;
        code_amount << "prev_id flag ---------------------" << std::endl;
        code_amount << "golomb code : " << prev_id_code_amount << std::endl << std::endl;
        code_amount << "golomb code full : " << golomb_x + golomb_y + prev_id_code_amount << std::endl << std::endl;


        int cnt = 0;
        for(int j = 0 ; j < target.rows ; j++){
          for(int i = 0 ; i < target.cols ; i++){
            if(R(out, i, j) == 255 && G(out, i, j) == 0 && B(out, i, j) == 0){
              cnt++;
              //std::cout << "(" << i << ", " << j << ")" << std::endl;
            }
          }
        }

        psnr_points_efficience << threshold << " " << psnr_1 << std::endl;

        corner_psnr << corners.size() << " " << psnr_1 << std::endl;
        corner_code_amount << corners.size() << " "
                           << golomb_x + golomb_y + golomb_mv_x + golomb_mv_y + result.getXbits() +
                              result.getYbits() +
                              triangles.size() << std::endl;
        rate_mse << golomb_x + golomb_y + golomb_mv_x + golomb_mv_y + result.getXbits() + result.getYbits() +
                    triangles.size() << " " << getMSE(target, out, cv::Rect(0, 0, target.cols, target.rows))
                 << std::endl;
        rate_psnr << golomb_x + golomb_y + golomb_mv_x + golomb_mv_y + result.getXbits() + result.getYbits() +
                     triangles.size() << " " << psnr_1 << " " << "corner:" << corners.size() << " triangle:"
                  << triangles.size() << " BM:" << result.getBlockMatchingFrequency() << "% Warp:"
                  << result.getWarpingFrequency() << "%" << std::endl;
        rate_psnr_csv << golomb_x + golomb_y + golomb_mv_x + golomb_mv_y + result.getXbits() + result.getYbits() +
                         triangles.size() << "," << psnr_1 << std::endl;

        std::cout << "zahyou = " << golomb_x + golomb_y + prev_id_code_amount<< "ugoki = " << golomb_mv_x + golomb_mv_y << std::endl;
        std::cout << golomb_x + golomb_y + golomb_mv_x + golomb_mv_y + result.getXbits() + result.getYbits() +
                         triangles.size() << " " << psnr_1 << " " << "corner:" << corners.size() << " triangle:"
                         << triangles.size() << " BM:" << result.getBlockMatchingFrequency() << "% Warp:"
                         << result.getWarpingFrequency() << "%" << std::endl;
        threshold += 2.0;

        if (THRESHOLD)
          cv::imwrite(file_path + img_path + "triangle_error_threshold_" + std::to_string(threshold) + ".png",
                      triangle_error_img);
        else if (HARRIS)
          cv::imwrite(file_path + img_path + "triangle_error_corners_" + std::to_string(corners.size()) + ".png",
                      triangle_error_img);
      }
    }
  }

  // 頂点情報をstoreする
  std::vector<int> all(1002, 0);
  for (int i = 0; i <= 1000; i++) all[i] = (count_all_x_coord[i] + count_all_y_coord[i]);

  storeFrequency(graph_file_path + "coord_all.txt", all, 500);

  for (int i = 0; i < (int) all.size(); i++) all[i] = 0;

  // 動きベクトルの統計をstore
  for (int i = 0; i <= 1000; i++) all[i] = (count_all_x_mv[i] + count_all_y_mv[i]);
  storeFrequency(graph_file_path + "mv_all.txt", all, 500);

  for (int i = 0; i < (int) all.size(); i++) all[i] = 0;

  // 平行移動ベクトルの統計をstore
  for (int i = 0; i <= 1000; i++) all[i] = (count_all_diff_x_mv[i] + count_all_diff_y_mv[i]);
  storeFrequency(graph_file_path + "diff_mv.txt", all, 500);

  // prev_idの統計をstore
  for (int i = 0; i <= 1000; i++) all[i] = (count_all_prev_id[i]);
  storeFrequency(graph_file_path + "prev_id.txt", all, 500);
}

/**
 * @fn int addSideCorners(cv::Mat img, std::vector<cv::Point2f> &corners)
 * @brief 外周に頂点を打つ
 * @param[in] img 画像
 * @param[out] corners 点の座標を格納するvector
 */
int addSideCorners(cv::Mat img, std::vector<cv::Point2f> &corners) {
  int point_nums = 0;
  int width = img.cols;
  int height = img.rows;
  int width_points = static_cast<int>(std::sqrt(width / (double)height) * std::sqrt(corners.size()) + 0.5);
  int height_points = static_cast<int>(std::sqrt(height / (double)width) * std::sqrt(corners.size()) + 0.5);

  // 上辺の両端に点を配置する
  corners.emplace_back(0.0, 0.0);
  corners.emplace_back(img.cols - 1, 0.0);
  point_nums += 2;

  // 上辺への点追加 -----------------------------------------------------------------------------------
  std::vector<std::pair<double, cv::Point2f> > pt;

  // 隣り合う画素との誤差を計算し, 降順にソートする
  // 誤差順に座標がほしいので, pair<double, cv::Point2f>にした.
  for (int i = 0, j = 0; i < img.cols - 1; i++) {
    pt.emplace_back(std::make_pair(fabs(MM(img, i, j) - MM(img, i + 1, j)), cv::Point2f(i, j)));
  }
  sort(pt.begin(), pt.end(), [](const pdp &a1, const pdp &a2) { return a1.first < a2.first; });

  std::vector<bool> flag_width((unsigned int) img.cols, 0);
  const int DXX = 500;       // 外周の点の最小許容距離（おそらくX座標）
  const int DYY = 400;       // 外周の点の最小許容距離
  int DX = img.cols / DXX;  // 外周の点の最大取得数（上下）
  int DY = img.rows / DYY;  // 外周の点の最大取得数（左右）

  // 始点と終点からDXX画素にフラグを建てる
  for (int i = 0; i < img.cols; i++) flag_width[i] = (i < DXX || (img.cols - DXX) <= i);

  int cnt = 0;
  int flag = false;
  for (int i = 0; i < DX && !flag; i++) {
    for (int k = 0; k < (int) pt.size(); k++) {
      std::pair<double, cv::Point2f> p = pt[k];
      if (!flag_width[p.second.x]) {
        corners.emplace_back(p.second.x, p.second.y);
        point_nums++;
        cnt++;
        if(width_points == cnt){
          flag = true;
          break;
        }
        for (int j = (int) (p.second.x - DXX); j < (p.second.x + DXX); j++) {
          if (0 <= j && j < img.cols) {
            flag_width[j] = true;
          }
        }
        break;
      }
    }
  }

  for(int i = SIDE_X_MIN ; i < img.cols; i += SIDE_X_MIN){
    corners.emplace_back(i, 0);
  }

  // 下辺に点を追加 -----------------------------------------------------------------------------
  pt.clear();
  // 下辺の両端に点を打つ
  corners.emplace_back(img.cols - 1, img.rows - 1);
  corners.emplace_back(0.0, img.rows - 1);
  point_nums += 2;
  // 隣り合う画素との誤差を計算し, 降順にソートする
  // 誤差順に座標がほしいので, pair<double, cv::Point2f>にした.
  for (int i = img.cols - 2, j = img.rows - 1; i > 0; i--) {
    pt.emplace_back(std::make_pair(fabs(MM(img, i, j) - MM(img, i - 1, j)), cv::Point2f(i, j)));
  }
  sort(pt.begin(), pt.end(), [](const pdp &a1, const pdp &a2) { return a1.first < a2.first; });

  for (int i = 0; i < img.cols; i++) flag_width[i] = i < DXX || (img.cols - DXX) <= i;

  cnt = 0;
  flag = false;
  for (int i = 0; i < DX && !flag; i++) {
    for (int k = 0; k < (int) pt.size(); k++) {
      std::pair<double, cv::Point2f> p = pt[k];
      if (!flag_width[p.second.x]) {
        corners.emplace_back(p.second.x, p.second.y);
        point_nums++;
        cnt++;
        if(cnt == width_points){
          flag = true;
          break;
        }
        for (int j = (int) (p.second.x - DXX); j < (p.second.x + DXX); j++) {
          if (0 <= j && j < img.cols) {
            flag_width[j] = true; // NOLINT
          }
        }
        break;
      }
    }
  }

  for(int i = SIDE_X_MIN ; i < img.cols; i += SIDE_X_MIN){
    corners.emplace_back(i, img.rows - 1);
  }

  // 右辺に点を追加 -----------------------------------------------------------------------------
  pt.clear();

  // 隣り合う画素との誤差を計算し, 降順にソートする
  // 誤差順に座標がほしいので, pair<double, cv::Point2f>にした.
  for (int i = img.cols - 1, j = 1; j < (img.rows - 1); j++) {
    pt.emplace_back(std::make_pair(fabs(MM(img, i, j) - MM(img, i, j + 1)), cv::Point2f(i, j)));
  }
  sort(pt.begin(), pt.end(), [](const pdp &a1, const pdp &a2) { return a1.first < a2.first; });

  std::vector<bool> flag_height((unsigned int) img.rows, 0);

  for (int i = 0; i < img.rows; i++) {
    flag_height[i] = i < DYY || (img.rows - DYY) <= i;
  }

  cnt = 0;
  flag = false;
  for (int i = 0; i < DY && !flag; i++) {
    for (int k = 0; k < (int) pt.size(); k++) {
      std::pair<double, cv::Point2f> p = pt[k];
      if (flag_height[p.second.y] == 0) {
        corners.emplace_back(p.second.x, p.second.y);
        point_nums++;
        cnt++;
        if(cnt == height_points){
          flag = true;
          break;
        }
        for (int j = (int) (p.second.y - DYY); j < (int) p.second.y + DYY; j++) {
          if (0 <= j && j < img.rows) {
            flag_height[j] = true;
          }
        }
        break; // NOLINT
      }
    }
  }

  for(int i = SIDE_Y_MIN ; i < img.rows ; i += SIDE_Y_MIN){
    corners.emplace_back(img.cols - 1, i);
  }

  // 左辺に点を追加 -----------------------------------------------------------------------------
  pt.clear();

  // 隣り合う画素との誤差を計算し, 降順にソートする
  // 誤差順に座標がほしいので, pair<double, cv::Point2f>にした.
  for (int i = 0, j = img.rows - 2; j > 0; j--) {
    pt.emplace_back(std::make_pair(fabs(MM(img, i, j) - MM(img, i, j - 1)), cv::Point2f(i, j)));
  }
  sort(pt.begin(), pt.end(), [](const pdp &a1, const pdp &a2) { return a1.first < a2.first; });

  for (int i = 0; i < img.rows; i++) {
    flag_height[i] = i < DYY || (img.rows - DYY) <= i;
  }

  flag = false;
  cnt = 0;
  for (int i = 0; i < DY && !flag; i++) {
    for (int k = 0; k < (int) pt.size(); k++) {
      std::pair<double, cv::Point2f> p = pt[k];
      if (flag_height[p.second.y] == 0) {
        corners.emplace_back(p.second.x, p.second.y);
        point_nums++;
        cnt++;
        if(cnt == width_points){
          flag = true;
          break;
        }
        for (int j = (int) (p.second.y - DYY); j < p.second.y + DYY; j++) {
          if (0 <= j && j < img.rows) {
            flag_height[j] = true;
          }
        }
        break;
      }
    }
  }

  for(int i = SIDE_Y_MIN ; i < img.rows ; i += SIDE_Y_MIN){
    corners.emplace_back(0, i);
  }

 // for(int i = 0; i < 4; i +=1){
 //   std::cout << "x = " << corners[i].x << " y = " << corners[i].y << std::endl;
 // }

  return point_nums;
}


/*!
 * @fn void cornersQuantization(std::vector<cv::Point2f> &corners);
 * @brief 特徴点の量子化を行う
 * @param[in, out] corners 特徴点
 * @details
 *  入力された特徴点を, Δで量子化する. 今現在Δが決め打ちになっている(Δ=8)
 */



std::vector<cv::Point2f> cornersQuantization(std::vector<cv::Point2f> &corners, const cv::Mat &target) {
  cv::Point2f tmp_corner;
  for (auto &corner : corners) {
    if (corner.x < target.cols - 1) {
      tmp_corner.x = (int) ((corner.x + QUANTIZE / 2.0) / QUANTIZE) * QUANTIZE;
      corner.x = target.cols <= tmp_corner.x ? tmp_corner.x - 1 : tmp_corner.x;
    }

    if (corner.y < target.rows - 1) {
      tmp_corner.y = (int) ((corner.y + QUANTIZE / 2.0) / QUANTIZE) * QUANTIZE;
      corner.y = target.rows <= tmp_corner.y ? tmp_corner.y - 1 : tmp_corner.y;
    } else {

    }
  }

  // ラスタスキャン順にソート
  sort(corners.begin(), corners.end(), [](const cv::Point2f &a1, const cv::Point2f &a2) {
    if (a1.y != a2.y) {
      return a1.y < a2.y;
    } else {
      return a1.x < a2.x;
    }
  });

//  for(int i = 0 ; i < (int)corners.size() ; i++){
//    std::cout << "corners[" << i << "]:" << corners[i] << std::endl;
//  }
  return uniqCoordinate(corners);
}

/**
 * @fn PredictedImageResult getPredictedImage(const cv::Mat &ref, const cv::Mat &refx2, const cv::Mat &refx4, const cv::Mat &refx8,const cv::Mat &target, const cv::Mat &targetx2, const cv::Mat &targetx4, const cv::Mat &targetx8,const cv::Mat &intra,  std::vector<Triangle> &triangles,
                  const std::vector<cv::Point2f> &ref_corners, std::vector<cv::Point2f> &corners, DelaunayTriangulation md,std::vector<cv::Point2f> &add_corners,int *add_count,const cv::Mat& residual_ref,int &tmp_mv_x,int &tmp_mv_y ,bool add_flag)
 * @param ref リファレンス画像
 * @param refx2 参照フレームを2倍縮小
 * @param refx4 4倍縮小
 * @param refx8 8倍縮小
 * @param target ターゲット画像
 * @param targetx2 対象フレームを2倍縮小
 * @param targetx4 4倍縮小
 * @param targetx8 8倍縮小
 * @param intra イントラ符号化した参照フレーム
 * @param triangles 三角形の集合
 * @param ref_corners リファレンス画像上の頂点
 * @param corners ターゲット画像上の頂点
 * @param md 三角パッチ網
 * @param add_corners 頂点追加のためのバッファ(使ってない)
 * @param add_count 頂点を追加した数を格納(使ってない)
 * @param residual_ref 対象フレームと参照フレームとの差分画像
 * @param tmp_mv_x x軸上の動きベクトルの符号量を格納
 * @param tmp_mv_y y軸上の動きベクトルの符号量を格納
 * @param add_flag 頂点追加に制限をかけるためのフラグ(使ってない)
 * @return cv::Mat型の予測画像
 */
PredictedImageResult
getPredictedImage(const cv::Mat &ref, const cv::Mat &refx2, const cv::Mat &refx4, const cv::Mat &refx8,const cv::Mat &target, const cv::Mat &targetx2, const cv::Mat &targetx4, const cv::Mat &targetx8,const cv::Mat &intra,  std::vector<Triangle> &triangles,
                  const std::vector<cv::Point2f> &ref_corners, std::vector<cv::Point2f> &corners, DelaunayTriangulation md,std::vector<cv::Point2f> &add_corners,int *add_count,const cv::Mat& residual_ref,int &tmp_mv_x,int &tmp_mv_y ,bool add_flag) {
    cv::Mat out = cv::Mat::zeros(target.size(), CV_8UC3);
    cv::Mat color = cv::Mat::zeros(target.size(), CV_8UC3);
    cv::Mat expansion_ref = bilinearInterpolation(ref);
    cv::Mat mv_image = target.clone();
    cv::Mat predict_img0 = cv::Mat::zeros(targetx8.size(), CV_8UC3);
    cv::Mat predict_img1 = cv::Mat::zeros(targetx4.size(), CV_8UC3);
    cv::Mat predict_img2 = cv::Mat::zeros(targetx2.size(), CV_8UC3);
    cv::Mat predict_img3 = cv::Mat::zeros(target.size(), CV_8UC3);
    cv::Mat predict_warp = cv::Mat::zeros(target.size(), CV_8UC3);
    cv::Mat predict_para = cv::Mat::zeros(target.size(), CV_8UC3);
    cv::Point2f mv_diff, mv_prev;
    std::vector<cv::Mat> predict_buf;
    //std::vector<std::vector<cv::Point2i>> buffer;//前の差分符号化のバッファ(使ってない)
    std::vector<std::vector<std::pair<std::vector<std::pair<cv::Point2i, int>>, std::pair<bool, bool>>>> mv_basis(
            corners.size());
    std::vector<std::vector<std::tuple<std::vector<std::pair<cv::Point2i, int>>, bool, bool, cv::Point2f, int,cv::Point2f>>> mv_basis_tuple(
            corners.size());//現在の差分符号化に用いるタプル
    std::vector<cv::Point2i> tmp;
    std::ofstream tri_list;
    std::ofstream mv_list = std::ofstream("mv_list.csv");
    int hist_org[201] = {0}, hist_org_x[201] = {0}, hist_org_y[201] = {0};
    int *hist, *hist_x, *hist_y;
    int max_mv_x = 0, min_mv_x = 256, max_mv_y = 0, min_mv_y = 256;
    int basis_mv_tmp_x = 0,basis_mv_tmp_y = 0,sabun_mv_tmp_x = 0,sabun_mv_tmp_y = 0;
    bool para_flag = false;
    int Quant = 4;
    //int count_diff = 0;

    hist = &hist_org[100];
    hist_x = &hist_org_x[100];
    hist_y = &hist_org_y[100];
    tri_list = std::ofstream("tri_list.csv");
    predict_buf.emplace_back(predict_img0);
    predict_buf.emplace_back(predict_img1);
    predict_buf.emplace_back(predict_img2);
    predict_buf.emplace_back(predict_img3);
    std::vector<cv::Mat> predict_buf_dummy = predict_buf;
    /*
    for (int i = 0; i < (int) corners.size(); i++) {
        buffer.emplace_back(tmp);
        buffer[i].emplace_back(corners[i]);
    }
*/
    for (const auto &triangle : triangles) {
        drawPoint(mv_image, corners[triangle.p1_idx], RED, 5);
        drawPoint(mv_image, corners[triangle.p2_idx], RED, 5);
        drawPoint(mv_image, corners[triangle.p3_idx], RED, 5);
        drawTriangle(mv_image, corners[triangle.p1_idx], corners[triangle.p2_idx], corners[triangle.p3_idx], BLUE);
    }

    int numerator = 1;
    int denominator = static_cast<int>(triangles.size());
    int corners_size = corners.size();
    int freq_block = 0;
    int freq_warp = 0;
    int block_matching_pixel_nums = 0;   // ブロックマッチングをおこなった画素の数
    double block_matching_pixel_errors = 0; // ブロックマッチングをおこなった画素の二乗誤差
    int warping_pixel_nums = 0;          // ワーピングをおこなった画素の数
    double warping_pixel_errors = 0;        // ワーピングをおこなった画素の二乗誤差
    //double MSE = 0;
    std::vector<cv::Point2f> diff_vector;
    //bool flag[corners.size()] = {false};
    std::vector<cv::Point2i> ev;
    ev.clear();
    for (int i = 0; i < (int) corners.size(); i++) {
        ev.emplace_back(cv::Point2i(0, 0));
    }
    int cnt = 0;
    tmp_mv_x = 0;
    tmp_mv_y = 0;
//#pragma omp parallel for
    for (int t = 0; t < (int) triangles.size(); t++) { // NOLINT
        tri_list << "triangle[" << t << "], ";
        double sx, sy, lx, ly;
        sx = -1;
        sy = -1;
        lx = -1;
        ly = -1;

        Triangle triangle = triangles[t];
        int p1_idx = triangle.p1_idx;
        int p2_idx = triangle.p2_idx;
        int p3_idx = triangle.p3_idx;


        // 良い方法を選んで貼り付け
        double error_block, error_warp;
        cv::Point2f mv_block;
        Point3Vec triangleVec(corners[triangle.p1_idx], corners[triangle.p2_idx], corners[triangle.p3_idx]);

        // ブロックマッチング
        // block_matching(ref, target, error_block, mv_block, triangleVec, expansion_ref);

        // ワーピング
        std::vector<cv::Point2f> corners_warp;
        Point3Vec prev_corners = Point3Vec(ref_corners[triangle.p1_idx], ref_corners[triangle.p2_idx],
                                           ref_corners[triangle.p3_idx]);
        std::vector<cv::Point2f> add_corner;
        std::vector<cv::Point2f> add_corner_tmp;
        std::vector<Triangle> triangles_tmp;
        add_corner.clear();
        int in_triangle_size;
        std::vector<cv::Point2i> mv = Gauss_Newton2(ref, target, intra, predict_buf, predict_warp, predict_para, color,
                                                    error_warp, triangleVec, prev_corners, tri_list, &para_flag,
                                                    add_corner, add_count, t, residual_ref, in_triangle_size, false,erase_th_global);
        double MSE_prev = error_warp / (double) in_triangle_size;
        std::vector<std::pair<cv::Point2f, double>> add_corner_pair(add_corner.size());
        std::vector<std::tuple<cv::Point2f, double, std::vector<Triangle>>> add_corner_tuple(add_corner.size());
        for (int idx = 0; idx < (int)triangles.size(); idx++) {
            //std::cout << "triangles[" << idx << "] = " << triangles[idx].p1_idx << ", " << triangles[idx].p2_idx << ", " << triangles[idx].p3_idx << std::endl;
        }
        for (int c_idx = 0; c_idx < (int) add_corner.size(); c_idx++) {
            std::cout << "c_idx = " << c_idx << "/ " << add_corner.size() << std::endl;
            int triangle_size_sum_later = 0;
            double MSE_later = 0;
            add_corner_tmp = corners;
            triangles_tmp = triangles;
            add_corner_tmp.emplace_back(add_corner[c_idx]);
            std::vector<bool> dummy(add_corner_tmp.size());
            DelaunayTriangulation md_later(Rectangle(0, 0, target.cols, target.rows));
            std::vector<Triangle> triangles_around;
            if (INTER_DIV) {
                triangles_around = inter_div(triangles_tmp, add_corner_tmp, add_corner[c_idx], t);
            } else {
                md_later.insert(add_corner_tmp);
            }

            std::vector<cv::Vec6f> triangles_mydelaunay;
            md_later.getTriangleList(triangles_mydelaunay);
            //std::cout << "triangle_later_size = " << triangles_mydelaunay.size() << std::endl;

            if (!INTER_DIV) {
                triangles_around = md_later.Get_triangles_around((int) add_corner_tmp.size() - 1, add_corner_tmp,
                                                                 dummy);
            }
            std::cout << "triangles_tmp_size = " << triangles_tmp.size() << std::endl;
            for (int t_idx = 0; t_idx < (int) triangles_around.size(); t_idx++) {
                std::cout << "t_idx = " << t_idx << "/ " << triangles_around.size() << std::endl;
                int triangle_later_size;
                Triangle triangle_later = triangles_around[t_idx];
                Point3Vec triangleVec_later(add_corner_tmp[triangle_later.p1_idx],
                                            add_corner_tmp[triangle_later.p2_idx],
                                            add_corner_tmp[triangle_later.p3_idx]);
                //std::cout <<"triangleVec_later = " << triangleVec_later.p1 << ", " << triangleVec_later.p2 << ", "<< triangleVec_later.p3 << std::endl;
                Gauss_Newton2(ref, target, intra, predict_buf_dummy, predict_warp, predict_para, color, error_warp,
                              triangleVec_later, prev_corners, tri_list, &para_flag, add_corner, add_count, t,
                              residual_ref, triangle_later_size, false,erase_th_global);
                MSE_later += error_warp;
                //Gauss_Newton2(ref,target,intra, predict_buf,predict_warp,predict_para, color, error_warp, triangleVec_later, prev_corners, tri_list,&para_flag,add_corner,add_count,t,residual_ref,triangle_later_size);
                // MSE_later += error_warp;
                triangle_size_sum_later += triangle_later_size;
            }
            add_corner_pair[c_idx].first = add_corner[c_idx];
            add_corner_pair[c_idx].second = MSE_later / (double) triangle_size_sum_later;
            std::get<0>(add_corner_tuple[c_idx]) = add_corner[c_idx];
            std::get<1>(add_corner_tuple[c_idx]) = MSE_later / (double) triangle_size_sum_later;
            std::get<2>(add_corner_tuple[c_idx]) = triangles_tmp;
            //add_corner_pair[c_idx].second = MSE_later / (double)triangle_size_sum_later;
            //std::cout << "in_triangle_size = " << in_triangle_size << std::endl;
            //std::cout << "in_triangle_size_sum_later = " << triangle_size_sum_later << std::endl;
        }
        if (add_corner.size() != 0) {
            bubbleSort(add_corner_pair, add_corner_pair.size());
            bubbleSort(add_corner_tuple, add_corner_tuple.size());
            if ((MSE_prev - std::get<1>(add_corner_tuple[0])) / MSE_prev >= 0) {
                *add_count = *add_count + 1;
                add_corners.emplace_back(std::get<0>(add_corner_pair[0]));
                triangles = std::get<2>(add_corner_tuple[0]);
                corners = add_corner_tmp;
                t += 2;
            }
            std::cout << "MSE_per = " << (MSE_prev - std::get<1>(add_corner_tuple[0])) / MSE_prev << std::endl;
            tri_list << " , MSE_per = ," << (MSE_prev - std::get<1>(add_corner_tuple[0])) / MSE_prev << std::endl;
        }
        std::cout << "MSE_prev = " << MSE_prev << std::endl;

        for (int i = 0; i < (int) add_corner_pair.size(); i++) {
            // add_corners.emplace_back(add_corner_pair[i].first);
            std::cout << "add_corner_pair [" << i << "] = " << add_corner_pair[i].first << " MSE = "
                      << add_corner_pair[i].second << std::endl;
        }

        //mv = Gauss_Golomb(triangle, flag, ev, corners, md, mv,target,targetx8,para_flag);
        //double DC = 0;
        double tmp_x[3], tmp_y[3];
        for (int q = 0; q <= 2; q++) {
            if (max_mv_x < mv[q].x)max_mv_x = mv[q].x;
            if (min_mv_x > mv[q].x)min_mv_x = mv[q].x;
            if (max_mv_y < mv[q].y)max_mv_y = mv[q].y;
            if (min_mv_y > mv[q].y)min_mv_y = mv[q].y;
            tmp_x[q] = mv[q].x;
            tmp_y[q] = mv[q].y;
            tmp_x[q] = (int) (tmp_x[q] * 4);
            tmp_y[q] = (int) (tmp_y[q] * 4);
        }
        for (int q = 1; q <= 2; q++) {
            mv[q].x -= mv[0].x;
            mv[q].y -= mv[0].y;
            if (abs(mv[q].x) <= 100) hist[mv[q].x]++;
            if (abs(mv[q].y) <= 100) hist[mv[q].y]++;
            if (abs(mv[q].x) <= 100) hist_x[mv[q].x]++;
            if (abs(mv[q].y) <= 100) hist_y[mv[q].y]++;
        }
        cv::Point2f mv_diff;
        cv::Point2f mv_diff_tmp;
        tmp_mv_x++;//warp_flagの1bit
/*
      mv_diff.x = mv[0].x + mv[3].x * Quant;
      mv_diff.y = mv[0].y + mv[3].y * Quant;
      mv_diff_tmp = mv_diff - mv_prev;
      mv_prev = mv_diff;
*/
        std::vector<std::pair<cv::Point2i, int>> add_mv_tmp;
        add_mv_tmp.emplace_back(mv[0], triangle.p1_idx);
        if (!para_flag) {
            add_mv_tmp.emplace_back(mv[0] + mv[1], triangle.p2_idx);
            add_mv_tmp.emplace_back(mv[0] + mv[2], triangle.p3_idx);
        } else {
            add_mv_tmp.emplace_back(mv[0], triangle.p2_idx);
            add_mv_tmp.emplace_back(mv[0], triangle.p3_idx);
        }
        std::pair<std::vector<std::pair<cv::Point2i, int>>, std::pair<bool, bool>> add_mv(add_mv_tmp,
                                                                                          std::pair<bool, bool>(false,
                                                                                                                para_flag));
        std::tuple<std::vector<std::pair<cv::Point2i, int>>, bool, bool, cv::Point2f, int,cv::Point2f> add_mv_tuple(add_mv_tmp,
                                                                                                        false,
                                                                                                        para_flag,
                                                                                                        cv::Point2f(0,
                                                                                                                    0),
                                                                                                        0,cv::Point2f(0,0));
        mv_basis[triangle.p1_idx].emplace_back(add_mv);
        mv_basis_tuple[triangle.p1_idx].emplace_back(add_mv_tuple);
/*
      if(!mv_basis[triangle.p1_idx].second){

          mv_diff_tmp = mv[0];
          mv_basis[triangle.p1_idx].first = mv[0];
          mv_basis[triangle.p1_idx].second = true;


          double init_Distance = 10E05;
          double min_Distance = init_Distance;
          int min_num = 0;
          bool flag_around = false;
          std::vector<int> neighbor = md.getNeighborVertexNum(triangle.p1_idx);
          std::vector<cv::Point2f> neighbor_cood = md.getNeighborVertex(triangle.p1_idx);
          for(int k = 0;k < (int)neighbor.size();k++){
              //std::cout << "neighbor[" << k << "] = " << neighbor[k] - 4<< std::endl;
              //std::cout << "neighbor_cood[" << k << "] = " << neighbor_cood[k] << std::endl;
              if(mv_basis[neighbor[k] - 4].second){
                  double Distance = md.getDistance(corners[triangle.p1_idx],neighbor_cood[k]);
                  std::cout << "Distance = " << Distance << std::endl;
                  flag_around = true;
                  if(min_Distance > Distance){
                      min_Distance = Distance;
                      min_num = neighbor[k] - 4;
                  }
              }
          }
          //std::cout << "min_num = " << min_num << std::endl;
          if(flag_around) {
              double min_diff = 10E05;
              int min_i = 0;
              for (int i = 0; i < mv_basis[min_num].first.size(); i++) {
                  cv::Point2f mv_diff_tmp_tmp = mv[0] - mv_basis[min_num].first[i];
                  std::cout << "mv_diff_tmp_tmp [" << i << "] = " << mv_diff_tmp_tmp << std::endl;
                  if (fabs(mv_diff_tmp_tmp.x) + fabs(mv_diff_tmp_tmp.y) < min_diff) {
                      min_diff = fabs(mv_diff_tmp_tmp.x) + fabs(mv_diff_tmp_tmp.y);
                      min_i = i;
                  }
              }
              mv_diff_tmp = mv[0] - mv_basis[min_num].first[min_i];
              mv_basis[triangle.p1_idx].first.emplace_back(mv[0]);
              mv_basis[triangle.p1_idx].second = true;
              //count_diff++;
              std::cout << "count_diff = " << count_diff << std::endl;
              std::cout << "mv_diff_tmp = " << mv_diff_tmp << std::endl;
          }
          else {
              mv_diff_tmp = mv[0];
              mv_basis[triangle.p1_idx].first.emplace_back(mv[0]);
              mv_basis[triangle.p1_idx].second = true;
          }
      } else{
          double min_diff = 10E05;
          int min_i = 0;
          for(int i = 0; i < mv_basis[triangle.p1_idx].first.size();i++) {
              cv::Point2f mv_diff_tmp_tmp = mv[0] - mv_basis[triangle.p1_idx].first[i];
              std::cout << "mv_diff_tmp_tmp [" << i << "] = " << mv_diff_tmp_tmp << std::endl;
              if(fabs(mv_diff_tmp_tmp.x) + fabs(mv_diff_tmp_tmp.y) < min_diff){
                  min_diff = fabs(mv_diff_tmp_tmp.x) + fabs(mv_diff_tmp_tmp.y);
                  min_i = i;
              }
          }
          mv_diff_tmp = mv[0] - mv_basis[triangle.p1_idx].first[min_i];
          mv_basis[triangle.p1_idx].first.emplace_back(mv[0]);
          count_diff++;
          std::cout << "count_diff = " << count_diff << std::endl;
          std::cout << "flag_mv_diff_tmp = " << mv_diff_tmp << std::endl;
      }
      if(!mv_basis[triangle.p2_idx].second){
          mv_basis[triangle.p2_idx].first.emplace_back(mv[0] + mv[1]);
          mv_basis[triangle.p2_idx].second = true;
      }
      if(!mv_basis[triangle.p3_idx].second){
          mv_basis[triangle.p3_idx].first.emplace_back(mv[0] + mv[2]);
          mv_basis[triangle.p3_idx].second = true;

      }
      */
        if (para_flag) {
            cnt++;
            /*
            golomb_mv_x += ozi::getGolombCode(16, (int)mv_diff_tmp.x, ozi::REGION1, ozi::GOLOMB, 0);
            golomb_mv_x += 2;
            golomb_mv_y += ozi::getGolombCode(16, (int)mv_diff_tmp.y, ozi::REGION1, ozi::GOLOMB, 0);
            golomb_mv_y += 2;
             */
            /*
            tmp_mv_x += ozi::getGolombCode(16, (int) mv_diff_tmp.x, ozi::REGION1, ozi::GOLOMB, 0);
            tmp_mv_x += 2;
            tmp_mv_y += ozi::getGolombCode(16, (int) mv_diff_tmp.y, ozi::REGION1, ozi::GOLOMB, 0);
            tmp_mv_y += 2;
             */
        }
            /*
            if((tmp_x[0] == tmp_x[1]) &&(tmp_x[1] == tmp_x[2]) &&
               (tmp_y[0] == tmp_y[1]) &&(tmp_y[1] == tmp_y[2])){
                golomb_mv_x += ozi::getGolombCode(16, mv[0].x, ozi::REGION1, ozi::GOLOMB, 0);
                golomb_mv_x += 2;
                golomb_mv_y += ozi::getGolombCode(16, mv[0].y, ozi::REGION1, ozi::GOLOMB, 0);
                golomb_mv_y += 2;
               // cnt++;
            }*/
        else {
            //golomb_mv_x += 9;
            //golomb_mv_y += 9;
            //mv[0] = mv_diff_tmp;
            for (int q = 1; q <= 2; q++) {
                /*
                if(q == 0)golomb_mv_x += ozi::getGolombCode(16, mv_diff_tmp.x, ozi::REGION1, ozi::GOLOMB, 0);
                else golomb_mv_x += ozi::getGolombCode(16, mv[q].x, ozi::REGION1, ozi::GOLOMB, 0);
                golomb_mv_x += 2;
                if(q == 0)golomb_mv_y += ozi::getGolombCode(16, mv_diff_tmp.y, ozi::REGION1, ozi::GOLOMB, 0);
                else golomb_mv_y += ozi::getGolombCode(16, mv[q].y, ozi::REGION1, ozi::GOLOMB, 0);
                golomb_mv_y += 2;
                 */
                int golomb_para = 32;
                if (q == 0)tmp_mv_x += ozi::getGolombCode(16, mv_diff_tmp.x, ozi::REGION1, ozi::GOLOMB, 0);
                else sabun_mv_tmp_x += ozi::getGolombCode(golomb_para, mv[q].x, ozi::REGION1, ozi::GOLOMB, 0);
                sabun_mv_tmp_x += 1;
                if (q == 0)tmp_mv_y += ozi::getGolombCode(16, mv_diff_tmp.y, ozi::REGION1, ozi::GOLOMB, 0);
                else sabun_mv_tmp_y += ozi::getGolombCode(golomb_para, mv[q].y, ozi::REGION1, ozi::GOLOMB, 0);
                sabun_mv_tmp_y += 1;
                std::cout << "x: " << mv[q].x << " code-length: "
                          << ozi::getGolombCode(golomb_para, mv[q].x, ozi::REGION1, ozi::GOLOMB, 0) << std::endl;
                std::cout << "y: " << mv[q].y << " code-length: "
                          << ozi::getGolombCode(golomb_para, mv[q].y, ozi::REGION1, ozi::GOLOMB, 0) << std::endl;

            }
        }
        /*
        for(const auto p:mv){
            golomb_mv_x += ozi::getGolombCode(7, p.x, ozi::REGION1, ozi::GOLOMB, 0);
            golomb_mv_y += 3;
            golomb_mv_y += ozi::getGolombCode(7, p.y, ozi::REGION1, ozi::GOLOMB, 0);
            golomb_mv_y += 3;
            std::cout << "x: " << p.x << " code-length: " << ozi::getGolombCode(7, p.x, ozi::REGION1, ozi::GOLOMB, 0) << std::endl;
            std::cout << "y: " << p.y << " code-length: " << ozi::getGolombCode(7, p.y, ozi::REGION1, ozi::GOLOMB, 0) << std::endl;
        }
  */
        /*
        buffer[triangle.p1_idx].emplace_back(mv[0]);
        buffer[triangle.p2_idx].emplace_back(mv[1]);
        buffer[triangle.p3_idx].emplace_back(mv[2]);
*/
        //MSE += error_warp;
        //Point3Vec mv_corners = Point3Vec(mv[0],mv[1],mv[2]);
        //corners_warp = warping(ref, target, error_warp, triangleVec, prev_corners);
        error_warp = 0;
        cv::Point2d xp, va, vb, ta, tb, tc;
        ta = triangleVec.p1;
        tb = triangleVec.p2;
        tc = triangleVec.p3;

        cv::Vec4f range = cv::Vec4f(0.0, ref.cols, 0.0, ref.rows);
        // 座標のチェック
        if (check_coordinate(corners[p1_idx], range) && check_coordinate(corners[p2_idx], range) &&
            check_coordinate(corners[p3_idx], range)) {
            lx = sx = corners[p1_idx].x;
            ly = sy = corners[p1_idx].y;
            sx = std::min((int) sx, std::min((int) corners[p2_idx].x, (int) corners[p3_idx].x));
            lx = std::max((int) lx, std::max((int) corners[p2_idx].x, (int) corners[p3_idx].x));
            sy = std::min((int) sy, std::min((int) corners[p2_idx].y, (int) corners[p3_idx].y));
            ly = std::max((int) ly, std::max((int) corners[p2_idx].y, (int) corners[p3_idx].y));
        }

        cv::Point2f diff = getDifferenceVector(triangle, corners, ref_corners, mv_block);
        error_block = error_block + LAMBDA * (ozi::getGolombCode(ozi::getGolombParam(0.5), (int) diff.x, ozi::REGION1,
                                                                 ozi::KTH_GOLOMB, 1) +
                                              ozi::getGolombCode(ozi::getGolombParam(0.5), (int) diff.y, ozi::REGION1,
                                                                 ozi::KTH_GOLOMB, 1));

        //std::cout << "block:" << error_block << " Warping:" << error_warp << std::endl;
/*
    if (error_block < error_warp && BM_AVAILABLE) {
      freq_block++;
//      cv::Point2f diff = getDifferenceVector(triangle, corners, ref_corners, mv_block);
      diff_vector.emplace_back(diff);

      // ブロックマッチング
      for (int j = (int) (round(sy) - 1); j <= round(ly) + 1; j++) {
        for (int i = (int) (round(sx) - 1); i <= round(lx) + 1; i++) {
          xp.x = (double) i;
          xp.y = (double) j;

          // （i, j）がta, tb, tcで構成される三角形の内部なら
          if (isInTriangle(triangleVec, xp)) {
            int w = (int) (mv_block.x + 2 * i);
            int h = (int) (mv_block.y + 2 * j);
            if (w < 0 || expansion_ref.cols <= w || h < 0 || expansion_ref.rows <= h) {
              std::cout << "w:" << w << " h:" << h << std::endl;
              continue;
            }

            R(out, i, j) = R(expansion_ref, w, h);
            G(out, i, j) = G(expansion_ref, w, h);
            B(out, i, j) = B(expansion_ref, w, h);

            if(error_block > 10) {
              R(triangle_error_img, i, j) = 255;
              G(triangle_error_img, i, j) = 0;
              B(triangle_error_img, i, j) = 0;
            }
            block_matching_pixel_errors += (R(out, i, j) - R(target, i, j)) * (R(out, i, j) - R(target, i, j));
            block_matching_pixel_errors += (G(out, i, j) - G(target, i, j)) * (G(out, i, j) - G(target, i, j));
            block_matching_pixel_errors += (B(out, i, j) - B(target, i, j)) * (B(out, i, j) - B(target, i, j));
            block_matching_pixel_nums++;
          }
        }
      }
      line(mv_image, cv::Point2d((ta.x + tb.x + tc.x) / 3.0, (ta.y + tb.y + tc.y) / 3.0),
           cv::Point2d(mv_block.x / 2 + (ta.x + tb.x + tc.x) / 3.0, mv_block.y / 2 + (ta.y + tb.y + tc.y) / 3.0),
           GREEN);
    } *//*else if (WARP_AVAILABLE) {

      freq_warp++;
      // ワーピング 現在のフレーム
      std::cout << "Warping target Frame" << std::endl;
      for (int j = (int) (round(sy) - 1); j <= round(ly) + 1; j++) {
        for (int i = (int) (round(sx) - 1); i <= round(lx) + 1; i++) {
          xp.x = (double) i;
          xp.y = (double) j;
          // (i, j)がta, tb, tcで構成される三角形の内部なら
          if (isInTriangle(triangleVec, xp)) {
            double xx, yy, ee, gg, mmx, mmy, ii, jj;
            unsigned char gg_warp, rr_warp, bb_warp;
            va.x = ta.x - tc.x;
            va.y = ta.y - tc.y;
            vb.x = tb.x - tc.x;
            vb.y = tb.y - tc.y;

            xx = (((double) i - tc.x) * va.y) - (((double) j - tc.y) * va.x);
            xx /= ((va.y * vb.x) - (va.x * vb.y));

            yy = ((vb.x - va.x) * ((double) j - tc.y)) - ((vb.y - va.y) * ((double) i - tc.x));
            yy /= (va.y * vb.x) - (va.x * vb.y);

            gg = yy != 0 ? xx / yy : 0.0;

            ee = yy;
            //for (int i = 0; i < 3; i++) {
            //std::cout << "mv[" << i << "] = " << mv[i].x << ", " << mv[i].y << std::endl;
          //}
            std::vector<cv::Point2f> ret_corners = corners_warp;
            mmx = (ret_corners[1].x - ret_corners[0].x) * gg + ret_corners[0].x;
            mmy = (ret_corners[1].y - ret_corners[0].y) * gg + ret_corners[0].y;

            ii = (mmx - ret_corners[2].x) * ee + ret_corners[2].x;
            jj = (mmy - ret_corners[2].y) * ee + ret_corners[2].y;

            interpolation(expansion_ref, ii, jj, rr_warp, gg_warp, bb_warp);

            R(out, i, j) = rr_warp;
            G(out, i, j) = gg_warp;
            B(out, i, j) = bb_warp;

            if(error_warp > 10) {
              R(triangle_error_img, i, j) = 255;
              G(triangle_error_img, i, j) = 0;
              B(triangle_error_img, i, j) = 0;
            }
            warping_pixel_errors += (R(out, i, j) - R(target, i, j)) * (R(out, i, j) - R(target, i, j));
            warping_pixel_errors += (G(out, i, j) - G(target, i, j)) * (G(out, i, j) - G(target, i, j));
            warping_pixel_errors += (B(out, i, j) - B(target, i, j)) * (B(out, i, j) - B(target, i, j));
            warping_pixel_nums++;
          }
        }
      }
      line(mv_image, ta, corners_warp[0] / 2.0, GREEN);
      line(mv_image, tb, corners_warp[1] / 2.0, GREEN);
      line(mv_image, tc, corners_warp[2] / 2.0, GREEN);
    }*/

        std::cout << numerator++ << "/" << denominator << "\n";
    }
    std::cout << "max_mv_x = " << max_mv_x << "min_mv_x = " << min_mv_x << "max_mv_y = " << max_mv_y << "min_mv_y = "
              << min_mv_y << std::endl;
    for (int i = -100; i <= 100; i++) {
        mv_list << i << "," << hist[i] << "," << i << "," << hist_x[i] << "," << i << "," << hist_y[i] << std::endl;

    }
    std::cout << "check1" << std::endl;
/*
    for (int i = 0; i < corners_size; i++) {
        cv::Point2i prev = buffer[i][1];
        if (buffer[i].size() != 2) {
            for (int j = 1; j < (int) buffer[i].size() - 1; j++) {
                cv::Point2i current = buffer[i][j + 1];
                cv::Point2i diff_mv = current - prev;
                buffer[i][j + 1] = diff_mv;
                prev = current;
            }
        }
    }
    */
    std::vector<std::vector<cv::Point2i>> corded_mv(corners.size());
    std::queue<int> Queue_neighbor;
    bool init_flag = false;
    //const bool prev_flag = true;
    for (int i = 0; i < (int) mv_basis_tuple.size(); i++) {
        for (int j = 0; j < (int) mv_basis_tuple[i].size(); j++) {
            std::get<1>(mv_basis_tuple[i][j]) = true;
            std::get<3>(mv_basis_tuple[i][j]) = std::get<0>(mv_basis_tuple[i][j])[0].first;
            std::get<4>(mv_basis_tuple[i][j]) = 0;
            std::get<5>(mv_basis_tuple[i][j]) = std::get<0>(mv_basis_tuple[i][j])[0].first;
            corded_mv[i].emplace_back(std::get<0>(mv_basis_tuple[i][j])[0].first);
            corded_mv[std::get<0>(mv_basis_tuple[i][j])[1].second].emplace_back(
                    std::get<0>(mv_basis_tuple[i][j])[1].first);
            corded_mv[std::get<0>(mv_basis_tuple[i][j])[2].second].emplace_back(
                    std::get<0>(mv_basis_tuple[i][j])[2].first);
            init_flag = true;
            std::vector<int> neighbor = md.getNeighborVertexNum(std::get<0>(mv_basis[i][j])[1].second);
            for (int k = 0; k < (int) neighbor.size(); k++) {
                Queue_neighbor.push(neighbor[k] - 4);
            }
            break;
        }
        if (init_flag)break;
    }
    while (!Queue_neighbor.empty()) {
        //std::cout << "Queue_size = " << Queue_neighbor.size() << std::endl;
        int current_idx = Queue_neighbor.front();
        //std::cout << "current_idx = " << current_idx << std::endl;
        Queue_neighbor.pop();
        for (int j = 0; j < (int) mv_basis_tuple[current_idx].size(); j++) {
            //std::cout << "j = " << j << std::endl;
            if(j == 0){
                std::vector<int> neighbor = md.getNeighborVertexNum(
                        std::get<0>(mv_basis_tuple[current_idx][j])[0].second);
                for(int k = 0;k < (int)neighbor.size();k++){
                    if(!mv_basis_tuple[neighbor[k] - 4].empty()){
                        if (!std::get<1>(mv_basis_tuple[neighbor[k] - 4][j])) {
                            Queue_neighbor.push(neighbor[k] - 4);
                        }
                    }
                }
            }
            //std::cout << "check1" << std::endl;
            if (!std::get<1>(mv_basis_tuple[current_idx][j])) {
                //std::cout << "check2" << std::endl;
                std::get<1>(mv_basis_tuple[current_idx][j]) = true;
                if (!corded_mv[current_idx].empty()) {
                    //std::cout << "check3" << std::endl;
                    double min_cord = 10E05;
                    double min_prev = 10E05;
                    cv::Point2f min_diff(0, 0);
                    int back_number = 0;
                    for (int i = 0; i < (int) corded_mv[current_idx].size(); i++) {
                        mv_diff = std::get<0>(mv_basis_tuple[current_idx][j])[0].first - corded_mv[current_idx][i];
                        mv_prev = std::get<0>(mv_basis_tuple[current_idx][j])[0].first - corded_mv[current_idx][i];
                        min_prev = fabs(mv_prev.x) + fabs(mv_prev.y);
                        if (fabs(mv_diff.x) + fabs(mv_diff.y) < min_cord) {
                            min_diff = mv_diff;
                            min_cord = fabs(mv_diff.x) + fabs(mv_diff.y);
                            back_number = (int) corded_mv[current_idx].size() - i;
                        }
                    }
                    //std::cout << "check4" << std::endl;
                    mv_diff = std::get<0>(mv_basis_tuple[current_idx][j])[0].first;
                    if (fabs(mv_diff.x) + fabs(mv_diff.y) < min_cord) {
                        min_diff = mv_diff;
                        back_number = 0;
                    }
                    if(fabs(mv_diff.x) + fabs(mv_diff.y) < min_prev){
                        mv_prev = mv_diff;
                        back_number = 0;
                    }
                    //std::cout << "check5" << std::endl;
                    std::get<3>(mv_basis_tuple[current_idx][j]) = min_diff;
                    std::get<4>(mv_basis_tuple[current_idx][j]) = back_number;
                    std::get<5>(mv_basis_tuple[current_idx][j]) = mv_prev;
                    corded_mv[std::get<0>(mv_basis_tuple[current_idx][j])[1].second].emplace_back(
                            std::get<0>(mv_basis_tuple[current_idx][j])[1].first);
                    corded_mv[std::get<0>(mv_basis_tuple[current_idx][j])[2].second].emplace_back(
                            std::get<0>(mv_basis_tuple[current_idx][j])[2].first);
                    //std::cout << "check6" << std::endl;
                } else {
                   // std::cout << "check7" << std::endl;
                    double init_Distance = 10E05;
                    double min_Distance = init_Distance;
                    int min_num = 0;
                    bool flag_arround = false;
                    std::vector<int> neighbor = md.getNeighborVertexNum(
                            std::get<0>(mv_basis_tuple[current_idx][j])[0].second);
                    std::vector<cv::Point2f> neighbor_cood = md.getNeighborVertex(
                            std::get<0>(mv_basis_tuple[current_idx][j])[0].second);
                    //std::cout << "check8" << std::endl;
                    for (int k = 0; k < (int) neighbor.size(); k++) {
                        //std::cout << "check9" << std::endl;
                        if (!corded_mv[neighbor[k] - 4].empty()) {
                           // std::cout << "check10" << std::endl;
                            flag_arround = true;
                            double Distance = md.getDistance(
                                    corners[std::get<0>(mv_basis_tuple[current_idx][j])[0].second], neighbor_cood[k]);
                            //std::cout << "Distance = " << Distance << std::endl;
                            if (min_Distance > Distance) {
                                min_Distance = Distance;
                                min_num = neighbor[k] - 4;
                            }
                        }
                    }
                    if(flag_arround) {
                        //std::cout << "check11" << std::endl;
                        double min_cord = 10E05;
                        double min_prev = 10E05;
                        cv::Point2f min_diff(0, 0);
                        int back_number = 0;
                        for (int i = 0; i < (int) corded_mv[min_num].size(); i++) {
                            mv_diff = std::get<0>(mv_basis_tuple[current_idx][j])[0].first - corded_mv[min_num][i];
                            mv_prev = std::get<0>(mv_basis_tuple[current_idx][j])[0].first - corded_mv[min_num][i];
                            min_prev = fabs(mv_prev.x) + fabs(mv_prev.y);
                            if (fabs(mv_diff.x) + fabs(mv_diff.y) < min_cord) {
                                min_diff = mv_diff;
                                min_cord = fabs(mv_diff.x) + fabs(mv_diff.y);
                                back_number = (int) corded_mv[current_idx].size() - i;
                            }
                        }
                        //std::cout << "check12" << std::endl;
                        mv_diff = std::get<0>(mv_basis_tuple[current_idx][j])[0].first;
                        if (fabs(mv_diff.x) + fabs(mv_diff.y) < min_cord) {
                            min_diff = mv_diff;
                            back_number = 0;
                        }
                        if(fabs(mv_diff.x) + fabs(mv_diff.y) < min_prev){
                            mv_prev = mv_diff;
                            back_number = 0;
                        }
                        //std::cout << "check13" << std::endl;
                        std::get<3>(mv_basis_tuple[current_idx][j]) = min_diff;
                        std::get<4>(mv_basis_tuple[current_idx][j]) = back_number;
                        std::get<5>(mv_basis_tuple[current_idx][j]) = mv_prev;
                        corded_mv[std::get<0>(mv_basis_tuple[current_idx][j])[1].second].emplace_back(
                                std::get<0>(mv_basis_tuple[current_idx][j])[1].first);
                        corded_mv[std::get<0>(mv_basis_tuple[current_idx][j])[2].second].emplace_back(
                                std::get<0>(mv_basis_tuple[current_idx][j])[2].first);
                    }else{
                        std::get<3>(mv_basis_tuple[current_idx][j]) = std::get<0>(mv_basis_tuple[current_idx][j])[0].first;
                        std::get<4>(mv_basis_tuple[current_idx][j]) = 0;
                        std::get<5>(mv_basis_tuple[current_idx][j]) = std::get<0>(mv_basis_tuple[current_idx][j])[0].first;
                        corded_mv[std::get<0>(mv_basis_tuple[current_idx][j])[1].second].emplace_back(
                                std::get<0>(mv_basis_tuple[current_idx][j])[1].first);
                        corded_mv[std::get<0>(mv_basis_tuple[current_idx][j])[2].second].emplace_back(
                                std::get<0>(mv_basis_tuple[current_idx][j])[2].first);
                    }
                   // std::cout << "check14" << std::endl;
                }
            }
        }
    }
    int worth = 0;
for(int i = 0;i < (int)mv_basis_tuple.size();i++) {
    for (int j = 0; j < (int) mv_basis_tuple[i].size(); j++) {
        int golomb_para = 16;
        cv::Point2f mv_diff = std::get<3>(mv_basis_tuple[i][j]);
        cv::Point2f basis = std::get<0>(mv_basis_tuple[i][j])[0].first;
        cv::Point2f prev = std::get<5>(mv_basis_tuple[i][j]);
        tmp_mv_x += 1;
        basis_mv_tmp_x += ozi::getGolombCode(golomb_para, prev.x, ozi::REGION1, ozi::GOLOMB, 0);
        basis_mv_tmp_x += 1;
        basis_mv_tmp_y += ozi::getGolombCode(golomb_para, prev.y, ozi::REGION1, ozi::GOLOMB, 0);
        basis_mv_tmp_y += 1;

        std::cout << "diff x: " << mv_diff.x << " code-length: "
                  << ozi::getGolombCode(golomb_para, mv_diff.x, ozi::REGION1, ozi::GOLOMB, 0) << " basis x: " << basis.x << " code-length: "
                                                                                           << ozi::getGolombCode(golomb_para, basis.x, ozi::REGION1, ozi::GOLOMB, 0) << " prev x: " << prev.x << " code-length: "
                                                                                                                                                                     << ozi::getGolombCode(golomb_para, prev.x, ozi::REGION1, ozi::GOLOMB, 0) << std::endl;
        std::cout << "diff y: " << mv_diff.y << " code-length: "
                  << ozi::getGolombCode(golomb_para, mv_diff.y, ozi::REGION1, ozi::GOLOMB, 0) << " basis y: " << basis.y << " code-length: "
                                                                                                  << ozi::getGolombCode(golomb_para, basis.y, ozi::REGION1, ozi::GOLOMB, 0)<< " prev y: " << prev.y << " code-length: "
                                                                                                  << ozi::getGolombCode(golomb_para, prev.y, ozi::REGION1, ozi::GOLOMB, 0) << std::endl;
       worth += ozi::getGolombCode(golomb_para, prev.x, ozi::REGION1, ozi::GOLOMB, 0) - ozi::getGolombCode(golomb_para, basis.x, ozi::REGION1, ozi::GOLOMB, 0);
       worth += ozi::getGolombCode(golomb_para, prev.y, ozi::REGION1, ozi::GOLOMB, 0) - ozi::getGolombCode(golomb_para, basis.y, ozi::REGION1, ozi::GOLOMB, 0);
    }
}
tmp_mv_x += sabun_mv_tmp_x + basis_mv_tmp_x;
tmp_mv_y += sabun_mv_tmp_y + basis_mv_tmp_y;
std::cout << "worth = " << worth << std::endl;
std::cout << "sabun_mv_tmp = " << sabun_mv_tmp_x + sabun_mv_tmp_y << std::endl;
std::cout << "basis_mv_tmp = " << basis_mv_tmp_x + basis_mv_tmp_y << std::endl;

/*
    for(int i = 0;i < (int)buffer.size();i++){
      std::cout << "buffer[" << i << "]" << std::endl;
        for(int j = 1;j < (int)buffer[i].size();j++){
            golomb_mv_x += ozi::getGolombCode(2, buffer[i][j].x, ozi::REGION1, ozi::GOLOMB, 0);
            golomb_mv_y += 3;
            golomb_mv_y += ozi::getGolombCode(2, buffer[i][j].y, ozi::REGION1, ozi::GOLOMB, 0);
            golomb_mv_y += 3;
            std::cout << "x: " << buffer[i][j].x << " code-length: " << ozi::getGolombCode(2, buffer[i][j].x, ozi::REGION1, ozi::GOLOMB, 0) << std::endl;
            std::cout << "y: " << buffer[i][j].y << " code-length: " << ozi::getGolombCode(2, buffer[i][j].y, ozi::REGION1, ozi::GOLOMB, 0) << std::endl;
        }
    }
*/
    //std::cout << "check point 1" << std::endl;
    for(int k = 0;k < (int)ev.size();k++){
      //std::cout << "ev = (" << ev[k].x <<"," << ev[k].y << std::endl;
    }
    for(int k = 0;k < (int)mv_basis.size();k++){
        for(int l = 0;l < (int)mv_basis[k].size();l++) {
            std::cout << "mv_basis [" << k << "][" << l << "] = " << std::get<0>(mv_basis_tuple[k][l])[0].first << "flag = " << std::get<1>(mv_basis_tuple[k][l])
                      << std::endl;
        }
    }
    std::cout << "cnt = " << cnt << std::endl;
    cv::imwrite("color.png",color);
    cv::imwrite("x1.bmp",predict_buf[0]);
    cv::imwrite("x2.bmp",predict_buf[1]);
    cv::imwrite("x4.bmp",predict_buf[2]);
    cv::imwrite("x8.bmp",predict_buf[3]);
    cv::imwrite("predict_warp.bmp",predict_warp);
    cv::imwrite("predict_para.bmp",predict_para);
    std::cout << "check point 2" << std::endl;
  //double PSNR = 10*log10(255*255/MSE);
  //std::cout << "PSNR = " << PSNR <<std::endl;

  std::vector<int> freq_x(1001, 0), freq_y(1001, 0);
  for (const cv::Point2f &v : diff_vector) {
    freq_x[v.x + 500]++;
    freq_y[v.y + 500]++;

    // これはごロムの符号を出すためのやつ
    count_all_diff_x_mv[v.x + 500]++;
    count_all_diff_y_mv[v.y + 500]++;
  }
    std::cout << "check point 3" << std::endl;
  // これはこの画像の符号料を出すやつ
  int left_x = 0, right_x = 1000;
  int left_y = 0, right_y = 1000;
  while (freq_x[left_x] == 0) left_x++;
  while (freq_x[right_x] == 0) right_x--;
  while (freq_y[left_y] == 0) left_y++;
  while (freq_y[right_y] == 0) right_y--;
    std::cout << "check point 4" << std::endl;
  int x_bits = 0, y_bits = 0;

  for (int i = 0; i <= 1000; i++) {
    if (freq_x[i] != 0) {
      x_bits += ozi::getGolombCode(ozi::getGolombParam(0.5), (i - 500), ozi::REGION1, ozi::KTH_GOLOMB, 1) * freq_x[i];
      std::cout << i - 500 << " " << freq_x[i] << " "
                << ozi::getGolombCode(ozi::getGolombParam(0.5), (i - 500), ozi::REGION1, ozi::KTH_GOLOMB, 1) << " "
                << x_bits << std::endl;
    }

    if (freq_y[i] != 0) {
      y_bits += ozi::getGolombCode(ozi::getGolombParam(0.5), (i - 500), ozi::REGION1, ozi::KTH_GOLOMB, 1) * freq_y[i];
    }
  }

/*
  return PredictedImageResult(out, mv_image, freq_block, freq_warp, block_matching_pixel_nums, warping_pixel_nums,
                                x_bits, y_bits, block_matching_pixel_errors, warping_pixel_errors);
*/
    std::cout << "check point 5" << std::endl;
  return PredictedImageResult(predict_buf[3], mv_image, freq_block, freq_warp, block_matching_pixel_nums, warping_pixel_nums,
                              x_bits, y_bits, block_matching_pixel_errors, warping_pixel_errors);

}

/**
 * @fn std::vector<cv::Point2f> uniqCoordinate(const std::vector<cv::Point2f>& corners)
 * @brief cornersの頂点をユニークにする
 * @param corners 頂点の集合
 * @return cornersに含まれるをユニークにした配列
 */
std::vector<cv::Point2f> uniqCoordinate(const std::vector<cv::Point2f> &corners) {
  cv::Point2f prev;
  std::vector<cv::Point2f> ret;

  prev = corners[0];
  ret.emplace_back(prev);

  for (int i = 1; i < static_cast<int>(corners.size()); i++) {
    if (prev != corners[i]) {
      ret.emplace_back(corners[i]);
    }
    prev = corners[i];
  }

  return ret;
}

/**
 * @fn void storeFrequency(const std::string& file_path, const std::vector<int> freq, int mid)
 * @brief 両側分布にするよ
 * @param file_path
 * @param freq
 * @param mid
 */
void storeFrequency(const std::string &file_path, const std::vector<int> freq, int mid) {
  int left_x, right_x;
  int left_y, right_y;
  int idx = 0;
  while (freq[idx] == 0) idx++;
  left_x = idx;

  idx = 999;
  while (freq[idx] == 0) idx--;
  right_x = idx;

  idx = 0;
  while (freq[idx] == 0) idx++;
  left_y = idx;

  idx = 999;
  while (freq[idx] == 0) idx--;
  right_y = idx;

  std::ofstream os(file_path);
  std::array<double, 1002> all{};
  int left = std::min(left_x, left_y);
  int right = std::max(right_x, right_y);
  double total_point_nums = std::accumulate(freq.begin(), freq.end(), 0);

  for (int i = left; i <= right; i++) {
    all[i] = freq[i] / total_point_nums;
  }
  for (int i = left; i <= right; i++) {
    os << i - 500 << " " << (all[i]) << std::endl;
  }
  os.close();
}

/**
 * @fn cv::Point2f getDifferenceVector(const Triangle& triangle, const std::vector<cv::Point2f>& corners,
                                const std::vector<cv::Point2f>& corners_mv, const cv::Point2f& mv)
 * @brief ブロックマッチングをしたベクトルで送る必要がある差分ベクトルを作る
 * @param triangle 三角パッチ
 * @param corners target上での座標
 * @param corners_mv 頂点のmv
 * @param mv パッチの動きベクトル
 * @return 三角パッチの動きとの差分
 * @details
 *  ブロックマッチングを採用したベクトルは, そのパッチを構成する3点それぞれの動きベクトルの平均を求めて, そいつとの差分を取る
 */
cv::Point2f getDifferenceVector(const Triangle &triangle, const std::vector<cv::Point2f> &corners,
                                const std::vector<cv::Point2f> &corners_mv, const cv::Point2f &mv) {
  cv::Point2f diff;
  int p1, p2, p3;
  p1 = triangle.p1_idx;
  p2 = triangle.p2_idx;
  p3 = triangle.p3_idx;

  cv::Point2f p1_mv = corners_mv[p1] - 2 * corners[p1];
  cv::Point2f p2_mv = corners_mv[p2] - 2 * corners[p2];
  cv::Point2f p3_mv = corners_mv[p3] - 2 * corners[p3];

  // TODO: これ実数じゃね…？
  cv::Point2f ave_mv = (p1_mv + p2_mv + p3_mv) / 3.0;

  diff = ave_mv - mv;
/*
  if (diff.x > 100 || diff.y > 80) {
    std::cout << "p1_mv:" << p1_mv << std::endl;
    std::cout << "p2_mv:" << p2_mv << std::endl;
    std::cout << "p3_mv:" << p3_mv << std::endl;
    std::cout << "mv:" << mv << std::endl;
  }
*/
  diff.x = myRound(diff.x, 1);
  diff.y = myRound(diff.y, 1);
  return diff;
}

