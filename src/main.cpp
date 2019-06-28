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
#include "../includes/Encode.h"
#include "../includes/config.h"
#include "../includes/ME.hpp"
#include "../includes/DelaunayTriangulation.hpp"
#include "../includes/Vector.hpp"
#include "../includes/psnr.h"
#include "../includes/Golomb.hpp"
#include "../includes/TriangleDivision.h"
#include "../includes/Reconstruction.h"
#include "../includes/ImageUtil.h"
#include "../includes/runAdaptive.h"
#include "../includes/main.h"

int golomb_mv_x = 0, golomb_mv_y = 0;
double erase_th_global = 0;

int addSideCorners(cv::Mat img, std::vector<cv::Point2f> &corners);

std::vector<cv::Point2f> cornersQuantization(std::vector<cv::Point2f> &corners, const cv::Mat &target);

PredictedImageResult
getPredictedImage(cv::Mat &ref,cv::Mat &target,cv::Mat &intra, std::vector<Triangle> &triangles, const std::vector<cv::Point2f> &ref_corners,
                  std::vector<cv::Point2f> &corners, DelaunayTriangulation md,std::vector<cv::Point2f> &add_corners,int &add_count,const cv::Mat& residual_ref,int &tmp_mv_x,int &tmp_mv_y,bool add_flag);

std::vector<cv::Point2f> uniqCoordinate(const std::vector<cv::Point2f> &corners);

void storeFrequency(const std::string &file_path, const std::vector<int> freq, int mid);

cv::Point2f getDifferenceVector(const Triangle &triangle, const std::vector<cv::Point2f> &corners,
                                const std::vector<cv::Point2f> &corners_mv, const cv::Point2f &mv);

cv::Mat getReconstructionDivisionImage(cv::Mat image, std::vector<CodingTreeUnit *> ctu);
void run();

// 問題は差分ベクトルどうするの…？って
std::vector<int> count_all_diff_x_mv(1001, 0);
std::vector<int> count_all_diff_y_mv(1001, 0);
cv::Mat triangle_error_img;

#define HARRIS false
#define THRESHOLD true
#define LAMBDA 0.2
#define INTER_DIV true // 頂点追加するかしないか

#define DIVIDE_MODE LEFT_DIVIDE


int qp;
int block_size_x;
int block_size_y;

int main(int argc, char *argv[]){
    // Write test codes below
    // test1();

    // exec ME
//    run();
    runAdaptive();
}

void run() {

    std::cout << "OpenCV_version : " << getVersionOfOpenCV() << std::endl;

    const std::string file_path = getProjectDirectory();
    std::cout << file_path << std::endl;
    FILE *img_list;

    if ((img_list = fopen((file_path + "/list.txt").c_str(), "r")) == NULL) {
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
    while (fgets(buf, sizeof(buf), img_list) != nullptr) {
        if (buf[0] == '#') continue;
        char t_file_name[256], r_file_name[256], o_file_name[256], i_file_path[256], csv_prefix[256],r_intra_file_name[256],target_color_file_name[256],c_file_name[256];
        sscanf(buf, "%s %s %s %s %s %s %s %d %d %d", i_file_path, r_file_name, t_file_name, o_file_name,r_intra_file_name,target_color_file_name,c_file_name, &qp, &block_size_x, &block_size_y);

        std::string img_path = std::string(i_file_path);
        std::string img_directory = file_path + img_path;
        std::string target_file_name = std::string(t_file_name);

        std::string ref_file_name = std::string(r_file_name);
        std::string ref_intra_file_name = std::string(r_intra_file_name);
        std::string corner_file_name = std::string(c_file_name);
        std::string csv_file_prefix = std::string("aaa");

        std::string ref_file_path = file_path + img_path + ref_file_name;
        std::string target_file_path = file_path + img_path + target_file_name;
        std::string ref_intra_file_path = file_path + img_path + ref_intra_file_name;
        std::string target_color_file_path = file_path + img_path + target_color_file_name;

        std::vector<std::string> out_file = splitString(std::string(o_file_name), '.');

        std::cout << "img_path               : " << img_path << std::endl;
        std::cout << "target_file_name       : " << target_file_name << std::endl;
        std::cout << "ref_file_name          : " << ref_file_name << std::endl;
        std::cout << "ref_file_path          : " << ref_file_path << std::endl;
        std::cout << "target_image file path : " << target_file_path << std::endl;
        std::cout << "ref_intra_file_path    : " << ref_intra_file_path << std::endl;
        std::cout << "target_color_file_path : " << target_color_file_path << std::endl;
        std::cout << "ref_gauss file path    : " << ref_file_path << std::endl;

        //RD性能グラフにしたい
        std::ofstream rate_psnr_csv;
        rate_psnr_csv = std::ofstream(img_directory + target_file_name + "rate_psnr_csv.csv");

        // 時間計測
        clock_t start = clock();
        std::cout << "check1" << std::endl;
        // 準備 --------------------------------------------------------------------------------
        // 画像の読み込み
        cv::Mat ref_gauss, ref_gauss_gray;          // 参照フレーム
        cv::Mat ref_image;
        cv::Mat target_image, target_gray;    // 対象フレーム
        cv::Mat refx2,refx4;
        cv::Mat targetx2,targetx4,targetx2_sharp,target_sharp,target_sharp_gray;
        cv::Mat refx8,targetx8;
        cv::Mat targetx4_sharp,targetx8_gray;
        cv::Mat ref_ref;
        cv::Mat target_bi,ref_bi;
        cv::Mat canny,canny_target;

        // QP変化させた参照画像はここで読む
        ref_image = cv::imread(ref_intra_file_path);
        target_image = cv::imread(target_file_path);

        cv::Mat tmp_target, tmp_ref;

//        cv::resize(target_image, tmp_target, cv::Size(1792, 1024));
//
//        target_image = tmp_target.clone();

        std::cout << "width: " << target_image.cols << " height: " << target_image.rows << std::endl;
        int height_mod = target_image.rows % block_size_y;
        int width_mod = target_image.cols % block_size_x;

        // block_sizeの定数倍となるような画像を作る
        cv::Mat crop_target_image = cv::Mat::zeros(target_image.rows - height_mod, target_image.cols - width_mod, CV_8UC3);
        cv::Mat crop_ref_image = cv::Mat::zeros(target_image.rows - height_mod, target_image.cols - width_mod, CV_8UC3);

        cv::resize(target_image, crop_target_image, cv::Size(ref_image.cols - width_mod, ref_image.rows - height_mod));
        cv::resize(ref_image, crop_ref_image, cv::Size(ref_image.cols - width_mod, ref_image.rows - height_mod));

        target_image = crop_target_image.clone();
        ref_image = crop_ref_image.clone();

        std::cout << "target_image.size:" << target_image.cols << " " << target_image.rows << std::endl;

        cv::imwrite(img_directory + "/crop_ref.png", crop_ref_image);
        cv::imwrite(img_directory + "/crop_target.png", crop_target_image);


        TriangleDivision triangle_division(ref_image, target_image);
        int divide_steps = 8;
        triangle_division.initTriangle(block_size_x, block_size_y, divide_steps, LEFT_DIVIDE);
        std::vector<Point3Vec> triangles = triangle_division.getTriangleCoordinateList();

        std::vector<std::pair<Point3Vec, int> > init_triangles = triangle_division.getTriangles();
                std::vector<CodingTreeUnit*> foo(init_triangles.size());
        for(int i = 0 ; i < init_triangles.size() ; i++) {
            foo[i] = new CodingTreeUnit();
            foo[i]->split_cu_flag1 = foo[i]->split_cu_flag2 = false;
            foo[i]->leftNode = foo[i]->rightNode = nullptr;
            foo[i]->triangle_index = i;
        }

        cv::Mat gaussRefImage = cv::imread(ref_file_path);
        cv::Mat spatialMvTestImage;

        cv::Mat new_gauss_output_image = cv::Mat::zeros(gaussRefImage.rows, gaussRefImage.cols, CV_8UC3);

//        TriangleDivision triangle_division(ref_image, target_image);
//        int divide_steps = 8;
//        triangle_division.initTriangle(block_size_x, block_size_y, divide_steps, LEFT_DIVIDE);
//        std::vector<Point3Vec> triangles = triangle_division.getTriangleCoordinateList();
//
//        std::vector<std::pair<Point3Vec, int> > init_triangles = triangle_division.getTriangles();
//                std::vector<CodingTreeUnit*> foo(init_triangles.size());
//        for(int i = 0 ; i < init_triangles.size() ; i++) {
//            foo[i] = new CodingTreeUnit();
//            foo[i]->split_cu_flag1 = foo[i]->split_cu_flag2 = false;
//            foo[i]->leftNode = foo[i]->rightNode = nullptr;
//            foo[i]->triangle_index = i;
//        }
//
//        cv::Mat gaussRefImage = cv::imread(ref_file_path);
//        cv::Mat spatialMvTestImage;
//
//
//
//        cv::Mat new_gauss_output_image = cv::Mat::zeros(gaussRefImage.rows, gaussRefImage.cols, CV_8UC3);
//
//        std::vector<Triangle> tt = triangle_division.getTriangleIndexList();
//        corners = triangle_division.getCorners();
//
//        std::vector<cv::Point2f> tmp_ref_corners(corners.size()), add_corners;
//        int add_count;
//        cv::Mat r_ref = cv::Mat::zeros(target_image.rows, target_image.cols, CV_8UC1);
//        int tmp_mv_x, tmp_mv_y;
//        int p_flag;

//        PredictedImageResult ret = getPredictedImage(gaussRefImage, target_image, ref_image, tt, tmp_ref_corners, corners, triangle_division, add_corners, add_count, r_ref, tmp_mv_x, tmp_mv_y, p_flag);
//        cv::imwrite(img_directory + "/Gauss_Newton2_predicted_image.png", ret.out);
//        std::cout << "PSNR:" << getPSNR(ret.out, target_image) << std::endl;
//        exit(0);
//
    //#pragma omp parallel for
//        for(int i = 0 ; i < init_triangles.size() ; i++) {
        triangle_division.constructPreviousCodingTree(foo, 0);
        for(int i = 0 ; i < 10 ; i++) {
            std::pair<Point3Vec, int> triangle = init_triangles[i];
//            std::cout << "i:" << i << std::endl;
            cv::Point2f p1 = triangle.first.p1;
            cv::Point2f p2 = triangle.first.p2;
            cv::Point2f p3 = triangle.first.p3;
            std::cout << "================== step:" << i << " ================== " << std::endl;
            triangle_division.split(gaussRefImage, foo[i], nullptr, Point3Vec(p1, p2, p3), i, triangle.second, divide_steps);
//            triangle_division.getSpatialTriangleList(triangles.size() - 1);
//            int prev_triangles_max = triangles.size();
//            triangles = triangle_division.getAllTriangleCoordinateList();
//            corners = triangle_division.getCorners();
//            if(prev_triangles_max < triangles.size()) {
//                int draw_triangle_index = triangles.size() - 1;
//                spatialMvTestImage = getReconstructionDivisionImage(gaussRefImage, foo);
//                for(auto& t : triangle_division.getSpatialTriangleList(draw_triangle_index)) {
//                    drawTriangle(spatialMvTestImage, triangles[t].p1, triangles[t].p2, triangles[t].p3, BLUE);
//                }
//                drawTriangle(spatialMvTestImage, triangles[draw_triangle_index].p1, triangles[draw_triangle_index].p2, triangles[draw_triangle_index].p3, RED);
//                cv::imwrite(img_directory + "/spatialTriangle_" + std::to_string(draw_triangle_index) + ".png", spatialMvTestImage);
            }

        getReconstructionDivisionImage(gaussRefImage, foo);
        triangle_division.constructPreviousCodingTree(foo, 0);

        exit(0);
        // 何回再帰的に分割を行うか
        const int division_steps = 1;
        triangle_division.subdivision(cv::imread(ref_file_path), division_steps);
        triangles = triangle_division.getTriangleCoordinateList();
        std::cout << "triangles.size():" << triangles.size() << std::endl;

        corners = triangle_division.getCorners();
        std::cout << "mid: " << corners.size() / 2 << std::endl;
        for(int k = 0 ; k < 10 ; k++){
          cv::Mat triangles_debug = crop_target_image.clone();
          for(const auto& triangle : triangles) {
            drawTriangle(triangles_debug, triangle.p1, triangle.p2, triangle.p3, cv::Scalar(255, 255, 255));
          }
          cv::imwrite(img_directory + "/triangles_step" + std::to_string(division_steps) + ".png", triangles_debug);

          std::vector<Point3Vec> covered_triangles = triangle_division.getIdxCoveredTriangleCoordinateList(corners.size() / 2 + 100 + k);
          for(const auto& triangle : covered_triangles) {
            std::cout << triangle.p1 << " " << triangle.p2 << " " << triangle.p3 << std::endl;
            drawTriangle(triangles_debug, triangle.p1, triangle.p2, triangle.p3, RED);
          }
          drawPoint(triangles_debug, corners[corners.size() / 2 + 100 + k], BLUE, 4);

          cv::imwrite(img_directory + "/triangles_step" + std::to_string(division_steps) + "_" + std::to_string(100 + k) + ".png", triangles_debug);
        }
        exit(0);

        cv::Mat color = cv::Mat::zeros(target_image.size(),CV_8UC3);
        cv::Mat predict_img0 = cv::Mat::zeros(targetx8.size(), CV_8UC3);
        cv::Mat predict_img1 = cv::Mat::zeros(targetx4.size(), CV_8UC3);
        cv::Mat predict_img2 = cv::Mat::zeros(targetx2.size(), CV_8UC3);
        cv::Mat predict_img3 = cv::Mat::zeros(target_image.size(), CV_8UC3);
        cv::Mat predict_warp = cv::Mat::zeros(target_image.size(),CV_8UC3);
        cv::Mat predict_para = cv::Mat::zeros(target_image.size(),CV_8UC3);
        std::ofstream tri_list;

        // デバッグ用に三角パッチごとの座標とPSNRを出していた
        tri_list = std::ofstream("tri_list.csv");

        // ガウスニュートン法の階層化でのみ使用するきれいなiフレーム
        ref_gauss = cv::imread(ref_file_path);

        // ガウスニュートン法で使用する縮小フレーム（移動平均）
        refx2 = half(ref_gauss, 2);
        targetx2 = half(target_image, 2);
        refx4 = half(refx2, 2);
        targetx4 = half(targetx2, 2);
        refx8 = half(refx4, 2);
        targetx8 = half(targetx4, 2);

        // ガウスニュートン法で使用する縮小フレーム（間引き）
        // TODO: デバッグ
        targetx4_sharp = half_sharp(targetx8);
        targetx2_sharp = half_sharp(targetx4_sharp);
        target_sharp = half_sharp(targetx2_sharp);

        // 差分画像（？）
        cv::Mat residual_ref = cv::Mat::zeros(target_image.size(),CV_8UC1);
        cv::Mat Maskx4 = cv::Mat::zeros(targetx4.size(),CV_8UC1);
        cv::Mat Mask = cv::Mat::zeros(target_image.size(),CV_8UC1);


        // GFTTで周りに特徴点をとってほしくないので、8px内側だけ取るように
        int crop_W = 8,crop_H = 8;
        for (int j = crop_H;j < targetx4.rows - crop_H;j++){
            for(int i = crop_W;i < targetx4.cols - crop_W;i++){
                Maskx4.at<unsigned char>(j,i) = 1;
            }
        }
        crop_W = 8,crop_H = 8;
        for (int j = crop_H;j < target_image.rows - crop_H;j++){
            for(int i = crop_W;i < target_image.cols - crop_W;i++){
                Mask.at<unsigned char>(j,i) = 1;
            }
        }

        cv::Mat residual_ref_bi, targetx4_Y;
        cv::Mat target_Y = target_gray;

        cvtColor(ref_gauss, ref_gauss_gray, cv::COLOR_BGR2GRAY);
        cvtColor(target_image, target_gray, cv::COLOR_BGR2GRAY);
        cvtColor(targetx4, targetx4_Y, cv::COLOR_BGR2GRAY);

        // 平滑化フィルタでエッジを減らした
        cv::bilateralFilter(ref_gauss_gray,ref_bi,5,150,150,CV_HAL_BORDER_REFLECT_101);
        cv::bilateralFilter(target_gray,target_bi,5,150,150,CV_HAL_BORDER_REFLECT_101);

        // 参照画像と対象画像の差分画像
        for(int j = 0;j < target_image.rows;j++) {
            for (int i = 0; i < target_image.cols; i++) {
                int y = abs(target_bi.at<unsigned char>(j, i) - ref_bi.at<unsigned char>(j, i));
                if(y < 0)
                    y = 0;
                else if(y > 255)
                    y = 255;
                residual_ref.at<unsigned char>(j,i) = (unsigned char)y;
            }
        }

        // 平滑化フィルタ
        cv::bilateralFilter(residual_ref,residual_ref_bi,5,150,150,CV_HAL_BORDER_REFLECT_101);
        cv::imwrite(img_directory + "residual_ref.bmp",residual_ref);

        cv::Mat residual_refx2 =   half_MONO(residual_ref,2);
        cv::Mat residual_refx4 = half_MONO(residual_refx2,2);

        double high_th = 100;
        double low_th = 0;
        cv::Canny(residual_ref,canny,high_th,low_th);
        high_th = 100;
        low_th = 0;
        cv::Canny(target_Y,canny_target,high_th,low_th);
        cv::imwrite("canny.bmp", canny);
        cv::imwrite("canny_target.bmp", canny_target);
        cv::imwrite("reidal_ref_bi.bmp", residual_ref_bi);

        // ドロネー分割 -------------------------------------------------------------------------
        const int POINT_MAX = 250; // 特徴点の最大個数
        corners.clear();
        std::vector<cv::Point2f> corners_R,corners_G,corners_B,corners_target_Y;
        // 特徴点抽出(GFTTDetector) 差分画像と1/4縮小した対象画像でGFTT
        // image – 8ビットまたは浮動小数点型，シングルチャンネルの入力画像．
        // corners – 検出されたコーナーが出力されるベクトル．
        // maxCorners – 出力されるコーナーの最大数．これより多い数のコーナーが検出された場合，より強いコーナーが出力されます．
        // qualityLevel – 許容される画像コーナーの最低品質を決定します．このパラメータ値を，最良のコーナーを示す測度（ cornerMinEigenVal() で述べた最小固有値や， cornerHarris() で述べた Harris 関数の応答）に乗じます．その掛け合わされた値よりも品質度が低いコーナーは，棄却されます．例えば，コーナーの最高品質度 = 1500， qualityLevel=0.01 である場合，品質度が15より小さいすべてのコーナーが棄却されます．
        // minDistance – 出力されるコーナー間で許容される，最小ユークリッド距離．
        cv::goodFeaturesToTrack(residual_ref, corners_org, POINT_MAX, GFTT_QUAULITY, 24,Mask, 3);
        cv::goodFeaturesToTrack(targetx4_Y, corners_target_Y, POINT_MAX, GFTT_QUAULITY, 16,Maskx4, 3);//8

        for(cv::Point2f &corner : corners_target_Y) corner *= 4;

        // より良い頂点をとるために、差分画像上と1/4にした対象画像上で頂点をとっている
        // 差分画像上での頂点と同じ座標の頂点が縮小した対象画像にある場合は追加しない
        for(auto & corner_target : corners_target_Y){
            bool flag = true;
            for(auto & corners_residual : corners_org){
                if(corner_target.x == corners_residual.x && corner_target.y == corners_residual.y){
                    flag = false;
                }
            }
            if(flag) corners_org.emplace_back(corner_target);
        }
        std::cout << corners_org.size() << std::endl;
        // あまりにも輝度値の変化がなければ頂点を削除
        for(int i = 0;i < (int)corners_org.size();i++){
            if(residual_refx4.at<unsigned char>((int)corners_org[i].y/4.0,(int)corners_org[i].x/4.0) <= 1){
                corners_org.erase(corners_org.begin() + i);
                i--;
            }
        }

        // 外周に点を打つ
        addSideCorners(target_image, corners_org);

        // 頂点の量子化
        corners_org = cornersQuantization(corners_org, target_image);
        puts("Quantized");

        // 頂点の動きをブロックマッチングで求める -----------------------------------------------------------------------
        cv::Mat points = target_image.clone();
        std::pair<std::vector<cv::Point2f>, std::priority_queue<int>> ret_ref_corners = getReferenceImageCoordinates(
                ref_gauss, target_image, corners_org, points);
        ref_corners_org = ret_ref_corners.first;

        corners.clear();
        for (auto & i : corners_org) corners.emplace_back(i);
        ref_corners.clear();
        for (auto & i : ref_corners_org) ref_corners.emplace_back(i);

        {

            double threshold = 17;

            std::cout << "target_image.cols = " << target_image.cols << "target_image.rows = "<< target_image.rows << std::endl;
            // Subdiv2Dの初期化
            cv::Size size = target_image.size();
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

            DelaunayTriangulation md(Rectangle(0, 0, target_image.cols, target_image.rows));
            md.insert(corners);

            std::vector<cv::Vec6f> triangles_mydelaunay;
            md.getTriangleList(triangles_mydelaunay);

            cv::Mat corner_ref = ref_gauss.clone();
            for(const cv::Vec6f& t : triangles_mydelaunay){
                cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
                drawTriangle(corner_ref, p1, p2, p3, BLUE);
            }
            cv::imwrite(img_directory + "corner_ref" + ".png", corner_ref);

            //　頂点削除
            double erase_th_per = 0.6;

            // 密集しているところから消すために、距離でソート
            md.Sort_Coners(corners);

            std::vector<double> sigma_tmp;
            std::vector<Triangle> triangles_t = md.Get_triangles(corners);
            double MSE = 0;
            double triangle_sum = 0;
            // 頂点削除のためのパラメタが欲しいので、1度全パッチ回していた
//#pragma omp parallel for
//            for (int t = 0; t < (int)triangles_t.size(); t++) {
//                cv::Point2f p1(corners[triangles_t[t].p1_idx]), p2(corners[triangles_t[t].p2_idx]), p3(corners[triangles_t[t].p3_idx]);
//                // TODO: 要確認
//                Point3Vec target_corners = Point3Vec(p1, p2, p3);
//                std::vector<cv::Point2f>warping_mv;
//                cv::Point2f parallel_mv;
//                bool parallel_flag;
//                int triangle_size = getTriangleSize(target_corners);
//                std::tie(warping_mv,parallel_mv,parallel_flag) = GaussNewton(ref_image, target_image, ref_gauss, target_corners);
//                std::vector<cv::Point2f> mv;
//                if(parallel_flag){
//                    mv.emplace_back(parallel_mv);
//                    mv.emplace_back(parallel_mv);
//                    mv.emplace_back(parallel_mv);
//                }else{
//                    mv.emplace_back((warping_mv[0]));
//                    mv.emplace_back((warping_mv[1]));
//                    mv.emplace_back((warping_mv[2]));
//                }
//                double MSE_tmp = getSquaredError(ref_image,target_image,target_corners,mv);
//
//                MSE += MSE_tmp;
//                triangle_sum += triangle_size;
//                sigma_tmp.emplace_back(MSE_tmp);
//
//                double RMSE = sqrt(MSE_tmp / triangle_size);
//                std::cout << "t = " << t << "/" << triangles_t.size()  << " RMSE = " << RMSE << std::endl;
//            }
//            double myu = sqrt(MSE / triangle_sum);
//            double sigma = 0;
//            for(const auto e : sigma_tmp){
//                sigma += (sqrt(e) - myu) * (sqrt(e) - myu);
//            }
//            sigma = sqrt(sigma/triangle_sum);
//            std::cout << "myu = "<< myu << "sigma = " << sigma << std::endl;
//
//            // ガウス分布と仮定すると、ここは変曲点（大嘘）
//            double erase_th = (myu + sigma) * (myu + sigma);
//            erase_th_global = erase_th;
        bool erase_flag = false;
         if(erase_flag) {
             for (int q = 0; q < 4; q++) {
                 md.insert(corners);
                 md.getTriangleList(triangles_mydelaunay);
                 md.Sort_Coners(corners);
                 triangles_t = md.Get_triangles(corners);
                 MSE = 0;
                 triangle_sum = 0;

                 // 頂点の数が変化するたびに、変曲点を求めていたが計算量がそれなりに掛かる(全三角パッチに対してやると5分ぐらい)
                 // TODO: 並列化できそうでは？
                 sigma_tmp.clear();
#pragma omp parallel for
                 for (int i = 0; i < (int) triangles_t.size(); i++) {
                     cv::Point2f p1(corners[triangles_t[i].p1_idx]), p2(corners[triangles_t[i].p2_idx]), p3(
                             corners[triangles_t[i].p3_idx]);
                     Point3Vec target_corners = Point3Vec(p1, p2, p3);
                     Point3Vec prev_corners = Point3Vec(p1, p2, p3);
                     std::vector<cv::Point2f> warping_mv;
                     cv::Point2f parallel_mv;
                     bool parallel_flag;
                     int triangle_size;
                     double MSE_tmp;
                     std::tie(warping_mv, parallel_mv, MSE_tmp, triangle_size, parallel_flag) = GaussNewton(ref_image,
                                                                                                            target_image,
                                                                                                            ref_gauss,
                                                                                                            target_corners);
                     std::vector<cv::Point2f> mv;
                     if (parallel_flag) {
                         mv.emplace_back(parallel_mv);
                         mv.emplace_back(parallel_mv);
                         mv.emplace_back(parallel_mv);
                     } else {
                         mv.emplace_back((warping_mv[0]));
                         mv.emplace_back((warping_mv[1]));
                         mv.emplace_back((warping_mv[2]));
                     }
                     MSE += MSE_tmp;
                     triangle_sum += triangle_size;
                     sigma_tmp.emplace_back(MSE_tmp);
                 }
                 double myu = sqrt(MSE / triangle_sum);
                 double sigma = 0;
                 myu = sqrt(MSE / triangle_sum);
                 sigma = 0;
                 for (double i : sigma_tmp) {
                     sigma += (sqrt(i) - myu) * (sqrt(i) - myu);
                 }
                 sigma = sqrt(sigma / triangle_sum);
                 std::cout << "myu = " << myu << "sigma = " << sigma << std::endl;

                 double erase_th = (myu + sigma) * (myu + sigma);
                 erase_th_global = erase_th;
                 std::cout << "erase_th = " << erase_th << std::endl;


                 for (int idx = 0; idx < (int) corners.size(); idx++) {
                     bool erase_flag = false;
                     int erase_count = 0;

                     // 復元できるように、頂点を削除する前の分割形状を保存
                     DelaunayTriangulation md_prev(Rectangle(0, 0, target_image.cols, target_image.rows));
                     md_prev.insert(corners);
                     std::vector<cv::Vec6f> triangles_prev;
                     md_prev.getTriangleList(triangles_prev);

                     // 頂点が減る様子を出力
                     cv::Mat corner_reduction = target_image.clone();
                     for (const cv::Vec6f &t : triangles_prev) {
                         cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
                         drawTriangle(corner_reduction, p1, p2, p3, BLUE);
                     }
                     cv::imwrite(img_directory + "corner_reduction_" + std::to_string(idx) + ".png", corner_reduction);

                     // 外周上に乗っている頂点を抜いたとき、黒い三角形ができる場合は抜かない
                     if (corners[idx].x <= 0.0 || corners[idx].y <= 0.0 || corners[idx].x >= target_image.cols - 1 ||
                         corners[idx].y >= target_image.rows - 1) {
                         // 頂点を抜いた状態で、ドロネー分割をし直す
                         std::vector<cv::Point2f> corners_later(corners);
                         corners_later.erase(corners_later.begin() + idx);
                         DelaunayTriangulation md_later(Rectangle(0, 0, target_image.cols, target_image.rows));
                         md_later.insert(corners_later);
                         std::vector<cv::Vec6f> triangles_tmp;
                         md_later.getTriangleList(triangles_tmp);

                         bool skip_flag = false;
                         md_later.serch_wrong(corners_later, target_image, &skip_flag);
                         if (skip_flag) continue;
                     }

                     std::vector<cv::Point2f> corners_later(corners);

                     // 四隅はそもそも抜かない
                     if ((corners[idx].x == 0.0 && corners[idx].y == 0.0) ||
                         (corners[idx].x == target_image.cols - 1 && corners[idx].y == 0.0)
                         || (corners[idx].x == target_image.cols - 1 && corners[idx].y == target_image.rows - 1) ||
                         (corners[idx].x == 0.0 && corners[idx].y == target_image.rows - 1)) {
                         continue;
                     }

                     std::cout << "idx = " << idx << "/ " << corners.size() << " q = " << q << "/ " << 4 << corners[idx]
                               << std::endl;

                     // idx番目の頂点に隣接している頂点がtrueになった配列が返るアレ
                     std::vector<bool> flag_around = std::vector<bool>(corners.size(), false);
                     std::vector<Triangle> triangles_around = md_prev.Get_triangles_around(idx, corners, flag_around);

                     double MSE_prev = 0, MSE_later = 0;
                     int triangle_size_sum_prev = 0, triangle_size_sum_later = 0;

                     // 周りの頂点だけガウスニュートン法をやる（削除前のMSEを見る）
#pragma omp parallel for
                     for (int i = 0; i < (int) triangles_around.size(); i++) {
                         Point3Vec target_corners(corners[triangles_around[i].p1_idx],
                                                  corners[triangles_around[i].p2_idx],
                                                  corners[triangles_around[i].p3_idx]);
                         Point3Vec prev_corners = Point3Vec(corners[triangles_around[i].p1_idx],
                                                            corners[triangles_around[i].p2_idx],
                                                            corners[triangles_around[i].p3_idx]);

                         std::vector<cv::Point2f> warping_mv;
                         cv::Point2f parallel_mv;
                         bool parallel_flag;
                         int triangle_size;
                         double MSE_tmp;
                         std::tie(warping_mv, parallel_mv, MSE_tmp, triangle_size, parallel_flag) = GaussNewton(
                                 ref_image, target_image, ref_gauss, target_corners);
                         std::vector<cv::Point2f> mv;
                         if (parallel_flag) {
                             mv.emplace_back(parallel_mv);
                             mv.emplace_back(parallel_mv);
                             mv.emplace_back(parallel_mv);
                         } else {
                             mv.emplace_back((warping_mv[0]));
                             mv.emplace_back((warping_mv[1]));
                             mv.emplace_back((warping_mv[2]));
                         }
                         std::cout << "triangle_size = " << triangle_size << "MSE_tmp = " << MSE_tmp << std::endl;
                         MSE_prev += MSE_tmp;
                         triangle_size_sum_prev += triangle_size;
                         if (MSE_tmp / (double) triangle_size >= erase_th) {
                             erase_count++; // 変曲点より後ろにいるやつ(3.40, 3.41式あたりを参照)
                         }
                     }

                     MSE_prev /= triangle_size_sum_prev;
                     std::cout << "MSE_prev = " << MSE_prev << std::endl;

                     corners_later.erase(corners_later.begin() + idx);
                     flag_around.erase(flag_around.begin() + idx);

                     // 削除後のMSEを測る
                     DelaunayTriangulation md_later(Rectangle(0, 0, target_image.cols, target_image.rows));
                     md_later.insert(corners_later);
                     std::vector<Triangle> triangles_later;
                     triangles_later = md_later.Get_triangles_later(md_later, idx, corners_later, flag_around);
#pragma omp parallel for
                     for (int i = 0; i < (int) triangles_later.size(); i++) {
                         Point3Vec target_corners(corners_later[triangles_later[i].p1_idx],
                                                  corners_later[triangles_later[i].p2_idx],
                                                  corners_later[triangles_later[i].p3_idx]);

                         std::vector<cv::Point2f> warping_mv;
                         cv::Point2f parallel_mv;
                         bool parallel_flag;
                         int triangle_size;
                         double MSE_tmp;
                         std::tie(warping_mv, parallel_mv, MSE_tmp, triangle_size, parallel_flag) = GaussNewton(
                                 ref_image, target_image, ref_gauss, target_corners);
                         std::vector<cv::Point2f> mv;
                         if (parallel_flag) {
                             mv.emplace_back(parallel_mv);
                             mv.emplace_back(parallel_mv);
                             mv.emplace_back(parallel_mv);
                         } else {
                             mv.emplace_back((warping_mv[0]));
                             mv.emplace_back((warping_mv[1]));
                             mv.emplace_back((warping_mv[2]));
                         }
                         MSE_later += MSE_tmp;
                         triangle_size_sum_later += triangle_size;
                         std::cout << "triangle_size = " << triangle_size << "MSE_later = " << MSE_tmp << std::endl;
                     }

                     MSE_later /= (double) triangle_size_sum_later;
                     std::cout << "MSE_later = " << MSE_later << std::endl;

                     // 削除前と削除後のRMSEを計算し比較
                     double RMSE_prev = sqrt(MSE_prev);
                     double RMSE_later = sqrt(MSE_later);

                     // サイズ比
                     double S_per = (double) triangle_size_sum_later / (double) triangle_size_sum_prev;
                     std::cout << "RMSE_prev = " << RMSE_prev << " RMSE_later = " << RMSE_later << " RMSE_per = "
                               << (MSE_later - MSE_prev) / MSE_prev << " S_per = "
                               << S_per
                               << " erase_count = " << erase_count << " / " << triangles_around.size()
                               << " erase_per = " << RMSE_later / RMSE_prev << std::endl;
                     std::cout << "MSE x S = " << (RMSE_later - RMSE_prev) * triangle_size_sum_later << std::endl;
                     // 式3.41の条件 & 抜いてもあまり劣化しないところは抜く
                     if ((double) erase_count / triangles_around.size() >= erase_th_per && RMSE_later / RMSE_prev < 1.5)
                         erase_flag = true;

                     // 式3.39(抜いてもあまり劣化しないとき), 式3.42(影響のある三角形の面積の総和をかけて、小さいパッチを抜く）
                     if ((fabs(MSE_prev - MSE_later) / MSE_prev < 0.05) || MSE_prev > MSE_later ||
                         erase_flag || (RMSE_later - RMSE_prev) * triangle_size_sum_later < 1000) {//6 10000
                         std::cout << "erased" << std::endl;
                         corners.erase(corners.begin() + idx);
                         idx--;
                     }
                 }
             }
         }
//
//            // 移動させて変化を見るやつ
//            /*
//            for (int idx = 0; idx < (int) corners.size(); idx++) {
//                double min_distance = md.neighbor_distance(corners, idx);
//                int mv_distance = std::min(8, (int) pow(2, (int) std::log2(sqrt(min_distance) / 2)));
//                std::vector<cv::Point2f> later_corners = corners;
//                for (int idx = 0; idx < (int) corners.size(); idx++) {
//                    if ((corners[idx].x == 0.0 && corners[idx].y == 0.0) ||
//                        (corners[idx].x == target_image.cols - 1 && corners[idx].y == 0.0)
//                        || (corners[idx].x == target_image.cols - 1 && corners[idx].y == target_image.rows - 1) ||
//                        (corners[idx].x == 0.0 && corners[idx].y == target_image.rows - 1)) {
//                        continue;
//                    }
//                    std::vector<std::pair<cv::Point2f, double>> point_pairs;
//                    std::pair<cv::Point2f, double> point_pair;
//                    std::vector<bool> flag_around = std::vector<bool>(corners.size(), false);
//                    for (int direct = 0; direct < 5; direct++) {
//                        if (direct == 1) {
//                            later_corners[idx].x = corners[idx].x + mv_distance;
//                            later_corners[idx].y = corners[idx].y;
//                        } else if (direct == 2) {
//                            later_corners[idx].x = corners[idx].x;
//                            later_corners[idx].y = corners[idx].y + mv_distance;
//                        } else if (direct == 3) {
//                            later_corners[idx].x = corners[idx].x - mv_distance;
//                            later_corners[idx].y = corners[idx].y;
//                        } else if (direct == 4) {
//                            later_corners[idx].x = corners[idx].x;
//                            later_corners[idx].y = corners[idx].y - mv_distance;
//                        } else if (direct == 5) {
//                            later_corners[idx].x = corners[idx].x + mv_distance;
//                            later_corners[idx].y = corners[idx].y + mv_distance;
//                        } else if (direct == 6) {
//                            later_corners[idx].x = corners[idx].x - mv_distance;
//                            later_corners[idx].y = corners[idx].y + mv_distance;
//                        } else if (direct == 7) {
//                            later_corners[idx].x = corners[idx].x - mv_distance;
//                            later_corners[idx].y = corners[idx].y - mv_distance;
//                        } else if (direct == 8) {
//                            later_corners[idx].x = corners[idx].x + mv_distance;
//                            later_corners[idx].y = corners[idx].y - mv_distance;
//                        }
//                        for (int c_idx = 0; c_idx < (int) corners.size(); c_idx++) {
//                            if (later_corners[idx] == corners[c_idx]) {
//                                later_corners[idx] = corners[idx];
//                            }
//                        }
//                        if (later_corners[idx].x < 0)later_corners[idx].x = 0;
//                        else if (later_corners[idx].x > target_image.cols - 1)later_corners[idx].x = target_image.cols;
//                        if (later_corners[idx].y < 0)later_corners[idx].y = 0;
//                        else if (later_corners[idx].y > target_image.rows - 1)later_corners[idx].y = target_image.rows;
//
//                        point_pair.first = later_corners[idx];
//                        DelaunayTriangulation md_later(Rectangle(0, 0, target_image.cols, target_image.rows));
//                        md_later.insert(later_corners);
//                        std::vector<Triangle> triangles_later;
//                        int triangle_size_sum_later = 0;
//                        double MSE_later = 0;
//                        triangles_later = md_later.Get_triangles_around(idx, later_corners, flag_around);
//#pragma omp parallel for
//                        for (int t = 0; t < (int) triangles_later.size(); t++) {
//                            int triangle_size;
//                            Triangle triangle = triangles_later[t];
//                            Point3Vec triangleVec(later_corners[triangle.p1_idx], later_corners[triangle.p2_idx],
//                                                  later_corners[triangle.p3_idx]);
//                            Point3Vec prev_corners = Point3Vec(ref_corners[triangle.p1_idx],
//                                                               ref_corners[triangle.p2_idx],
//                                                               ref_corners[triangle.p3_idx]);
//                            //std::cout << "later_Gauss" << std::endl;
//                            MSE_later += Gauss_Newton(ref_gauss, target_image, ref_image, triangleVec, prev_corners, triangle_size);
//                            triangle_size_sum_later += triangle_size;
//                            //std::cout << "MSE_later = " << MSE_later << std::endl;
//                        }
//                        MSE_later /= triangle_size_sum_later;
//                        point_pair.second = MSE_later;
//                        point_pairs.emplace_back(point_pair);
//                    }
//                }
//            }
//            */
//
//            color = cv::Mat::zeros(target_image.size(),CV_8UC3);
//            std::vector<cv::Point2f> add_corner_dummy;
//            predict_img0 = cv::Mat::zeros(targetx8.size(), CV_8UC3);
//            predict_img1 = cv::Mat::zeros(targetx4.size(), CV_8UC3);
//            predict_img2 = cv::Mat::zeros(targetx2.size(), CV_8UC3);
//            predict_img3 = cv::Mat::zeros(target_image.size(), CV_8UC3);
//            predict_warp = cv::Mat::zeros(target_image.size(),CV_8UC3);
//            predict_para = cv::Mat::zeros(target_image.size(),CV_8UC3);
//            std::vector<cv::Mat> predict_buf;
//            std::vector<std::vector<cv::Point2i>> buffer;
//            std::vector<cv::Point2i> tmp;
//
//            bool para_flag = false;
//            tri_list = std::ofstream("tri_list.csv");
//            predict_buf.emplace_back(predict_img0);
//            predict_buf.emplace_back(predict_img1);
//            predict_buf.emplace_back(predict_img2);
//            predict_buf.emplace_back(predict_img3);
//
//            // エッジ上に頂点をずらす
//            std::vector<cv::Point2f> edge_corners = slide_corner_edge(corners,canny_target,8);
//            std::vector<cv::Point2f> later_corners = corners;
//
//            for(int idx = 0;idx < (int)corners.size();idx++) {
//                // 四隅
//                if (corners[idx].x == 0.0 || corners[idx].y == 0.0 ||
//                    corners[idx].x == target_image.cols - 1 || corners[idx].y == target_image.rows - 1) {
//                    continue;
//                }
//                std::vector<std::pair<cv::Point2f, double>> point_pairs;
//                std::vector<bool> flag_around = std::vector<bool>(corners.size(), false);
//                for(int i = 0;i <= 1;i++) {
//                    if(i == 0)later_corners[idx] = corners[idx];
//                    else if(i == 1)later_corners[idx] = edge_corners[idx];
//
//                    // later_corners[idx]を使ってパッチの残差を見る
//                    std::pair<cv::Point2f, double> point_pair;
//                    point_pair.first = later_corners[idx];
//                    DelaunayTriangulation md_later(Rectangle(0, 0, target_image.cols, target_image.rows));
//                    md_later.insert(later_corners);
//                    std::vector<Triangle> triangles_later;
//                    int triangle_size_sum_later = 0;
//                    double MSE_later = 0;
//                    triangles_later = md_later.Get_triangles_around(idx, later_corners, flag_around);
//#pragma omp parallel for
//                    for (int t = 0; t < (int) triangles_later.size(); t++) {
//                        int triangle_size;
//                        double error_warp;
//                        Triangle triangle = triangles_later[t];
//                        Point3Vec triangleVec(later_corners[triangle.p1_idx], later_corners[triangle.p2_idx],
//                                              later_corners[triangle.p3_idx]);
//                        Point3Vec prev_corners = Point3Vec(ref_corners[triangle.p1_idx], ref_corners[triangle.p2_idx],
//                                                           ref_corners[triangle.p3_idx]);
//
//                        int add_count_dummy = 0;
//                        Gauss_Newton2(ref_gauss, target_image, ref_image, predict_buf, predict_warp, predict_para, color, error_warp,
//                                      triangleVec, prev_corners, tri_list, &para_flag, add_corner_dummy, add_count_dummy,
//                                      t, residual_ref, triangle_size, false, erase_th_global);
//                        MSE_later += error_warp;
//                        triangle_size_sum_later += triangle_size;
//                    }
//                    MSE_later /= triangle_size_sum_later;
//
//                    point_pair.second = MSE_later;
//
//                    if(i == 0)point_pair.second -= 0.5;
//                    point_pairs.emplace_back(point_pair);
//                    std::cout << "idx = " << idx << " / " << corners.size() << "i = " << i << "corners = "
//                              << corners[idx] << "later_corners = " << later_corners[idx] << MSE_later << std::endl;
//                }
//                bubbleSort(point_pairs, point_pairs.size());
//                corners[idx] = point_pairs[0].first;
//            }
//
//            /*
//            later_corners = corners;
//            color = cv::Mat::zeros(target_image.size(),CV_8UC3);
//            int add_count_dummy = 0;
//            para_flag = false;
//
//            for(int idx = 0;idx < (int)corners.size();idx++){
//                if (corners[idx].x == 0.0 || corners[idx].y == 0.0 ||
//                    corners[idx].x == target_image.cols - 1 || corners[idx].y == target_image.rows - 1) {
//                    continue;
//                }
//                double min_distance = md.neighbor_distance(corners,idx);
//                int mv_distance = std::min(4,(int)pow(2,(int)std::log2((min_distance/2))));
//                std::cout << "min_distance = " << sqrt(min_distance) << std::endl;
//                std::cout << "mv_distance = " << mv_distance << std::endl;
//                std::vector<std::pair<cv::Point2f,double>> point_pairs;
//                std::pair<cv::Point2f,double> point_pair;
//                std::vector<bool> flag_around = std::vector<bool>(corners.size(), false);
//                while(mv_distance >= 1) {
//                    int triangle_size_sum_prev;
//                    for (int direct = 0; direct < 9; direct++) {
//                        if(direct == 0){
//                            later_corners[idx].x = corners[idx].x;
//                            later_corners[idx].y = corners[idx].y;
//                        }
//                        else if (direct == 1) {
//                            later_corners[idx].x = corners[idx].x + mv_distance;
//                            later_corners[idx].y = corners[idx].y;
//                        } else if (direct == 2) {
//                            later_corners[idx].x = corners[idx].x;
//                            later_corners[idx].y = corners[idx].y + mv_distance;
//                        } else if (direct == 3) {
//                            later_corners[idx].x = corners[idx].x - mv_distance;
//                            later_corners[idx].y = corners[idx].y;
//                        } else if (direct == 4) {
//                            later_corners[idx].x = corners[idx].x;
//                            later_corners[idx].y = corners[idx].y - mv_distance;
//                        } else if (direct == 5) {
//                            later_corners[idx].x = corners[idx].x + mv_distance;
//                            later_corners[idx].y = corners[idx].y + mv_distance;
//                        } else if (direct == 6) {
//                            later_corners[idx].x = corners[idx].x - mv_distance;
//                            later_corners[idx].y = corners[idx].y + mv_distance;
//                        } else if (direct == 7) {
//                            later_corners[idx].x = corners[idx].x - mv_distance;
//                            later_corners[idx].y = corners[idx].y - mv_distance;
//                        } else if (direct == 8) {
//                            later_corners[idx].x = corners[idx].x + mv_distance;
//                            later_corners[idx].y = corners[idx].y - mv_distance;
//                        }
//                        for(int c_idx = 0;c_idx < (int)corners.size();c_idx++){
//                            if(later_corners[idx] == corners[c_idx]){
//                                later_corners[idx] = corners[idx];
//                            }
//                        }
//                        if (later_corners[idx].x < 0)later_corners[idx].x = 0;
//                        else if (later_corners[idx].x > target_image.cols - 1)later_corners[idx].x = target_image.cols - 1;
//                        if (later_corners[idx].y < 0)later_corners[idx].y = 0;
//                        else if (later_corners[idx].y > target_image.rows - 1)later_corners[idx].y = target_image.rows - 1;
//
//                        point_pair.first = later_corners[idx];
//                        DelaunayTriangulation md_later(Rectangle(0, 0, target_image.cols, target_image.rows));
//                        md_later.insert(later_corners);
//                        std::vector<Triangle> triangles_later;
//                        int triangle_size_sum_later = 0;
//                        double MSE_later = 0;
//                        triangles_later = md_later.Get_triangles_around(idx, later_corners, flag_around);
//#pragma omp parallel for
//                        for (int t = 0; t < (int) triangles_later.size(); t++) {
//                            int triangle_size;
//                            double error_warp;
//                            Triangle triangle = triangles_later[t];
//                            Point3Vec triangleVec(later_corners[triangle.p1_idx], later_corners[triangle.p2_idx],
//                                                  later_corners[triangle.p3_idx]);
//                            Point3Vec prev_corners = Point3Vec(ref_corners[triangle.p1_idx], ref_corners[triangle.p2_idx],
//                                                               ref_corners[triangle.p3_idx]);
//
//                            Gauss_Newton2(ref_gauss,target_image,ref_image, predict_buf,predict_warp,predict_para, color, error_warp, triangleVec, prev_corners, tri_list,&para_flag,add_corner_dummy,&add_count_dummy,t,residual_ref,triangle_size, false, erase_th_global);
//                            MSE_later += error_warp;
//                            triangle_size_sum_later += triangle_size;
//                        }
//                        if(direct == 0)triangle_size_sum_prev = triangle_size_sum_later;
//                        MSE_later /= triangle_size_sum_later;
//
//                        point_pair.second = MSE_later;
//                        if(direct == 0)point_pair.second -= 1;
//                        point_pairs.emplace_back(point_pair);
//                        std::cout << "idx = " << idx << " / " << corners.size() << "direct = " << direct << "corners = "
//                                  << corners[idx] << "later_corners = " << later_corners[idx] << MSE_later << std::endl;
//                    }
//                    bubbleSort(point_pairs, point_pairs.size());
//                    corners[idx] = point_pairs[0].first;
//                    mv_distance /= 2;
//                }
//            }
//*/

//            corners = triangle_division.getCorners();
            if(erase_flag) {
                std::ofstream corner_list_later = std::ofstream("corner_list_" + corner_file_name + "_later.dat");
                for (const cv::Point2f point : corners) {
                    corner_list_later << point.x << " " << point.y << std::endl;
                }
            }
            std::ifstream in_corner_list = std::ifstream("corner_list_" + corner_file_name + "_later.dat");
            std::string str1;
            int point_x,point_y;
            corners.clear();
            while (getline(in_corner_list, str1)) {
                sscanf(str1.data(), "%d %d", &point_x,&point_y);
                corners.emplace_back(cv::Point2f(point_x,point_y));
            }

            std::cout << "corners's size :" << corners.size() << std::endl;
            std::cout << "ref_corners's size :" << ref_corners.size() << std::endl;
            ret_ref_corners = getReferenceImageCoordinates(ref_gauss, target_image, corners, points);

            ref_corners_org = ret_ref_corners.first;
            ref_corners.clear();
            for (auto & i : ref_corners_org) ref_corners.emplace_back(i);

            std::cout << "corners's size :" << corners.size() << std::endl;
            std::cout << "ref_corners's size :" << ref_corners.size() << std::endl;
            std::cout << "corner size(erased):" << corners.size() << std::endl;

            // 減らした点で細分割
            std::cout << rect << std::endl;
            subdiv = cv::Subdiv2D(rect);
            subdiv.insert(corners);

            md = DelaunayTriangulation(Rectangle(0, 0, target_image.cols, target_image.rows));
            md.insert(corners);
            // TODO: 現状これやらないとneighbor_vtxがとれないので許して
            md.getTriangleList(triangles_mydelaunay);

            puts("insert done");

            // 三角網を描画します
            cv::Mat my_triangle = target_image.clone();
            triangle_error_img = target_image.clone();
            for (auto t:triangles_mydelaunay) {
                cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
                drawTriangle(my_triangle, p1, p2, p3, BLUE);
                drawTriangle(triangle_error_img, p1, p2, p3, BLUE);
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
            double mean = 0.0;
            std::cout << csv_file_prefix << std::endl;
            for (int i = 0; i < (int) freq_coord_x.size(); i++) {
                fprintf(fp, "%d,%d\n", i - min_coord_x, freq_coord_x[i]);
                os << i - min_coord_x << " " << freq_coord_x[i] << std::endl;
                mean += (i - min_coord_x) * freq_coord_x[i];
            }

            fclose(fp);
            os.close();
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

            cv::Mat decoded_corner = md.getDecodedCornerImage(coded_coordinate, target_image, RASTER_SCAN);
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

            cv::Mat decoded_mv = md.getDecodedMotionVectorImage(code, corners, target_image);

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

            cv::Mat triangle_target = target_image.clone();
            cv::Mat mv_image = target_image.clone();
//            for(auto t : triangles) {
//                // 三角形を描画
//                drawTriangle(triangle_target, t.p1, t.p2, t.p3, cv::Scalar(255, 255, 255));
//                drawTriangle(mv_image, t.p1, t.p2, t.p3, cv::Scalar(255, 255, 255));
//            }

            if(DIVIDE_MODE == LEFT_DIVIDE) {
                cv::imwrite(file_path + img_path + "triangle_" + out_file[0] + "_" + std::to_string(block_size_x) + "_" + std::to_string(block_size_y) + "_LEFT.png", triangle_target);
            }else{
                cv::imwrite(file_path + img_path + "triangle_" + out_file[0] + "_" + std::to_string(block_size_x) + "_" + std::to_string(block_size_y) + "_RIGHT.png", triangle_target);
            }

            std::vector<Triangle> triangles;

            // 頂点とindexを結びつけ
            for (auto t:triangles_as_vec6f) {
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
            }



            puts("");

            cv::Mat out = cv::Mat::zeros(target_image.size(), CV_8UC3);

            std::cout << "corners.size():" << corners.size() << std::endl;
            std::cout << "intra col = " << ref_image.cols << "row = " << ref_image.rows << std::endl;
            std::vector<cv::Point2f> add_corners;
            int add_count = 0;
            int tmp_mv_x = 0;
            int tmp_mv_y = 0;
            add_corners.clear();
            std::cout << "check point 3" << std::endl;

//            triangles = triangle_division.getTriangleIndexList();

            PredictedImageResult result = getPredictedImage(ref_gauss, target_image, ref_image, triangles, ref_corners, corners, md,
                                                            add_corners, add_count, residual_ref,tmp_mv_x,tmp_mv_y,true);
            // 予測画像を得る

            // add_cornersには予測精度が悪いところへの追加の頂点が入っているが、現状これには何も入れていないので実質追加なし
            std::copy(add_corners.begin(),add_corners.end(),std::back_inserter(ref_corners));
            for(int i = 0;i < (int)corners.size();i++){
                std::cout << "corner[" << i <<"] =" << corners[i] << std::endl;
            }
            add_corners.clear();
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
                os << i - min_coord_x_later << " " << freq_coord_x_later[i] << std::endl;
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
            }

            std::cout << "check point 3" << std::endl;
            out = result.out;
            std::cout << "check point 4" << std::endl;
            std::cout << "corners.size():" << corners.size() << std::endl;

            // ===========================================================
            // ログ出力
            // ===========================================================
            puts("======================================================");
            int H = target_image.rows;
            int W = target_image.cols;
            for(int crop_W = 8, crop_H = 8;crop_W <= 32;crop_H += 8,crop_W += 8) {

                std::cout << "inner PSNR : "
                          << getPSNR(target_image, out, cv::Rect(crop_W, crop_H, W - crop_W * 2, H - crop_H * 2)) << " crop " << crop_H
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
            std::vector<std::string> split_output_name = splitString(std::string(o_file_name), '.');
            std::string outFileName = split_output_name[0];
            std::string extension = split_output_name[1];

            std::string outFilePath = img_directory + outFileName + "_" + std::to_string(block_size_x) + "_" + std::to_string(block_size_y) +
                          (DIVIDE_MODE == LEFT_DIVIDE ? "_LEFT." : "_RIGHT.") + extension;

            std::cout << "outFilePath:" << outFilePath << std::endl;
            cv::imwrite(outFilePath, out);
            std::cout << "check point 1" << std::endl;

            cv::Mat residual = cv::Mat::zeros(target_image.size(), CV_8UC3);

            for(int j = 0;j < target_image.rows;j++){
                for(int i = 0;i < target_image.cols;i++){
                    int y = 4 * abs(R(target_image,i,j) - R(out,i,j));
                    if(y < 0)y = 0;
                    else if(y > 255)y = 255;
                    R(residual,i,j) = (unsigned char)y;
                    G(residual,i,j) = (unsigned char)y;
                    B(residual,i,j) = (unsigned char)y;
                }
            }
            md.getTriangleList(triangles_mydelaunay);
//            for(const Point3Vec &t : triangle_division.getTriangleCoordinateList()){
//                drawTriangle(residual, t.p1, t.p2, t.p3, RED);
//            }
            cv::imwrite(file_path + img_path + "residual.png",residual);
            std::cout << "check point 4" << std::endl;
            double psnr_1;
            printf("%s's PSNR:%f\n", outFilePath.c_str(), (psnr_1 = getPSNR(target_image, out)));
            std::cout << "check point 5" << std::endl;
            // 四角形を描画した画像を出力
            cv::Point2f p1 = cv::Point2f(150, 100);
            cv::Point2f p2 = cv::Point2f(target_image.cols - 151, 100);
            cv::Point2f p3 = cv::Point2f(target_image.cols - 151, target_image.rows - 101);
            cv::Point2f p4 = cv::Point2f(150, target_image.rows - 101);
            drawRectangle(out, p1, p2, p3, p4);
            cv::imwrite(file_path + img_path + "rect.png", out);
            std::cout << "check point 6" << std::endl;
            // ログ -------------------------------------------------------------------------------
            std::string logPath = getProjectDirectory() + "/log.txt";
            fp = fopen(logPath.c_str(), "a");
            time_t tt;
            time(&tt);
            char date[64];
            strftime(date, sizeof(date), "%Y/%m/%d %a %H:%M:%S", localtime(&tt));

            fprintf(fp, (outFilePath + "\n").c_str());
            // 符号量たち
            int prev_id_code_amount = 0;
            for (int i = 0; i <= 1000; i++) {
                if (prev_id_count[i] != 0) {
                    prev_id_code_amount +=
                            ozi::getGolombCode(ozi::getGolombParam(0.5), i - 500, ozi::REGION1, ozi::KTH_GOLOMB, 3) *
                            prev_id_count[i];
                }
            }
            fprintf(fp, ("lambda:" + std::to_string(LAMBDA)).c_str());
            fprintf(fp, "QUANTIZE_STEP:%d\n", QUANTIZE);
            fprintf(fp, "%s\n", date);
            fprintf(fp, "PSNR : %f\n", psnr_1);
            fprintf(fp, "code amount : %d[bit]\n", golomb_x + golomb_y + prev_id_code_amount + golomb_mv_x + golomb_mv_y);
            fprintf(fp, "freq_block:%d(%f%%)\n", result.freq_block, result.getBlockMatchingFrequency());
            fprintf(fp, "freq_warp:%d(%f%%)\n", result.freq_warp, result.getWarpingFrequency());
            fprintf(fp, "BlockMatching's PSNR : %f\n", result.getBlockMatchingPatchPSNR());
            fprintf(fp, "Warping's PSNR : %f\n", result.getWarpingPatchPSNR());
            fprintf(fp, (std::to_string(t / 60) + "m" + std::to_string(t % 60) + "sec\n").c_str());
            fprintf(fp, "reference        : %s\n", ref_file_name.c_str());
            fprintf(fp, "target_image     : %s\n", target_file_name.c_str());
            fprintf(fp, "threshold        : %d\n", threshold);
            fprintf(fp, "corner size      : %d\n", corners.size());
            fprintf(fp, "triangles.size() : %d\n", triangles.size());
            fprintf(fp, "block_size_x     : %d\n", block_size_x);
            fprintf(fp, "block_size_y     : %d\n", block_size_y);
            fprintf(fp, "qp               : %d\n", qp);
            fprintf(fp, "coordinate vector ---------------------\n");
            fprintf(fp, "golomb code(x)   : %d\n", golomb_x);
            fprintf(fp, "golomb code(y)   : %d\n", golomb_y);
            fprintf(fp, "motion vector ---------------------\n");
            fprintf(fp, "golomb code(x)   : %d\n", golomb_mv_x);
            fprintf(fp, "golomb code(y)   : %d\n", golomb_mv_y);
            fprintf(fp, "diff vector ---------------------\n");
            fprintf(fp, "golomb code(x)   : %d\n", result.getXbits());
            fprintf(fp, "golomb code(y)   : %d\n", result.getYbits());
            fprintf(fp, "prev_id flag ---------------------\n");
            fprintf(fp, "golomb code      : %d\n",  prev_id_code_amount);
            fprintf(fp, "golomb code full : %d\n\n", golomb_x + golomb_y + prev_id_code_amount);
            fclose(fp);
            std::cout << "log written" << std::endl;


            int cnt = 0;
            for(int j = 0 ; j < target_image.rows ; j++){
                for(int i = 0 ; i < target_image.cols ; i++){
                    if(R(out, i, j) == 255 && G(out, i, j) == 0 && B(out, i, j) == 0){
                        cnt++;
                    }
                }
            }

            rate_psnr_csv << golomb_x + golomb_y + golomb_mv_x + golomb_mv_y + result.getXbits() + result.getYbits() +
                             triangles.size() << "," << psnr_1 << std::endl;

            std::cout << "zahyou = " << golomb_x + golomb_y + prev_id_code_amount<< "ugoki = " << golomb_mv_x + golomb_mv_y << std::endl;
            std::cout << golomb_x + golomb_y + golomb_mv_x + golomb_mv_y + result.getXbits() + result.getYbits() +
                         triangles.size() << " " << psnr_1 << " " << "corner:" << corners.size() << " triangle:"
                      << triangles.size() << " BM:" << result.getBlockMatchingFrequency() << "% Warp:"
                      << result.getWarpingFrequency() << "%" << std::endl;
            threshold += 2.0;

            std::cout << "PSNR: " << psnr_1 << std::endl;
            std::cout << "code amount: " << golomb_x + golomb_y + prev_id_code_amount + golomb_mv_x + golomb_mv_y << "[bit]" << std::endl;

            if (THRESHOLD)
                cv::imwrite(file_path + img_path + "triangle_error_threshold_" + std::to_string(threshold) + ".png",
                            triangle_error_img);
            else if (HARRIS)
                cv::imwrite(file_path + img_path + "triangle_error_corners_" + std::to_string(corners.size()) + ".png",
                            triangle_error_img);

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
#pragma clang diagnostic pop

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

#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"
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
getPredictedImage(cv::Mat &ref,cv::Mat &target,cv::Mat &intra,  std::vector<Triangle> &triangles,
                  const std::vector<cv::Point2f> &ref_corners, std::vector<cv::Point2f> &corners, DelaunayTriangulation md,std::vector<cv::Point2f> &add_corners,int &add_count,const cv::Mat& residual_ref,int &tmp_mv_x,int &tmp_mv_y ,bool add_flag) {
    cv::Mat out = cv::Mat::zeros(target.size(), CV_8UC3);
    cv::Mat color = cv::Mat::zeros(target.size(), CV_8UC3);
    cv::Mat expansion_ref = bilinearInterpolation(ref);
    cv::Mat predict_img0 = cv::Mat::zeros(target.size()/8, CV_8UC3);//TODO .size()に除算オペレーター使えるのか確認
    cv::Mat predict_img1 = cv::Mat::zeros(target.size()/4, CV_8UC3);
    cv::Mat predict_img2 = cv::Mat::zeros(target.size()/2, CV_8UC3);
    cv::Mat predict_img3 = cv::Mat::zeros(target.size(), CV_8UC3);
    cv::Mat predict_warp = cv::Mat::zeros(target.size(), CV_8UC3);
    cv::Mat predict_para = cv::Mat::zeros(target.size(), CV_8UC3);
    cv::Point2f mv_diff, mv_prev;
    std::vector<std::vector<std::pair<std::vector<std::pair<cv::Point2i, int>>, std::pair<bool, bool>>>> mv_basis(
            corners.size());
    std::vector<std::vector<std::tuple<std::vector<std::pair<cv::Point2i, int>>, bool, bool, cv::Point2f, int,cv::Point2f>>> mv_basis_tuple(
            corners.size());//現在の差分符号化に用いるタプル
    std::vector<cv::Point2i> tmp;
    std::ofstream tri_list;
    std::ofstream mv_list = std::ofstream("mv_list.csv");
    int basis_mv_tmp_x = 0,basis_mv_tmp_y = 0,sabun_mv_tmp_x = 0,sabun_mv_tmp_y = 0;

    tri_list = std::ofstream("tri_list.csv");

    std::vector<cv::Mat> predict_buf;
    predict_buf
    .emplace_back(predict_img0);
    predict_buf.emplace_back(predict_img1);
    predict_buf.emplace_back(predict_img2);
    predict_buf.emplace_back(predict_img3);
    std::vector<cv::Mat> predict_buf_dummy = predict_buf;

    cv::Mat mv_image = target.clone();
    for (const auto &triangle : triangles) {
        drawPoint(mv_image, corners[triangle.p1_idx], RED, 5);
        drawPoint(mv_image, corners[triangle.p2_idx], RED, 5);
        drawPoint(mv_image, corners[triangle.p3_idx], RED, 5);
        drawTriangle(mv_image, corners[triangle.p1_idx], corners[triangle.p2_idx], corners[triangle.p3_idx], BLUE);
    }

    int numerator = 1;
    int denominator = static_cast<int>(triangles.size());
    int freq_block = 0;
    int freq_warp = 0;
    int block_matching_pixel_nums = 0;      // ブロックマッチングをおこなった画素の数
    double block_matching_pixel_errors = 0; // ブロックマッチングをおこなった画素の二乗誤差
    int warping_pixel_nums = 0;             // ワーピングをおこなった画素の数
    double warping_pixel_errors = 0;        // ワーピングをおこなった画素の二乗誤差
    std::vector<cv::Point2f> diff_vector;
    std::vector<cv::Point2i> ev;
    ev.clear();
    for (int i = 0; i < (int) corners.size(); i++) {
        ev.emplace_back(cv::Point2i(0, 0));
    }
    int cnt = 0;
    tmp_mv_x = 0;
    tmp_mv_y = 0;

    // 平行移動とワーピングを塗り分ける用
    cv::Mat parallel_flag_img_color = target.clone();

#pragma omp parallel for
    // 基準ベクトル以外のベ   クトルを符号化
    for (int t = 0; t < (int)triangles.size(); t++) { // NOLINT

//        tri_list << "triangle[" << t << "], ";

        Triangle triangle = triangles[t];

//        std::cout << triangle.p1_idx << " " << triangle.p2_idx << " " << triangle.p3_idx << std::endl;
//        std::cout << corners[triangle.p1_idx] << " " << corners[triangle.p2_idx] << " " << corners[triangle.p3_idx] << std::endl;

        // 良い方法を選んで貼り付け
        Point3Vec triangleVec(corners[triangle.p1_idx], corners[triangle.p2_idx], corners[triangle.p3_idx]);

        // ガウスニュートン法による変形
        Point3Vec prev_corners = Point3Vec(ref_corners[triangle.p1_idx], ref_corners[triangle.p2_idx],
                                           ref_corners[triangle.p3_idx]);
        std::vector<cv::Point2f> add_corner;
        std::vector<cv::Point2f> add_corner_tmp;
        std::vector<Triangle> triangles_tmp;



        // para_flag == trueなら平行移動
        // add_corner: 追加候補の頂点
//        Gauss_Newton(ref, target, intra, triangleVec, prev_corners, in_triangle_size);
//        std::vector<cv::Point2i> mv;
        cv::Point2f parallel_mv;
        std::vector<cv::Point2f> warping_mv;
        bool parallel_flag;
        int triangle_size;
        double MSE_tmp;
        std::tie(warping_mv, parallel_mv, MSE_tmp, triangle_size, parallel_flag) = GaussNewton(
                intra, target, ref, triangleVec);
        std::vector<cv::Point2f> mv;
        if (parallel_flag) {
            mv.emplace_back(parallel_mv);
            mv.emplace_back(parallel_mv);
            mv.emplace_back(parallel_mv);
        } else {
            mv.emplace_back((warping_mv[0]));
            mv.emplace_back((warping_mv[1]));
            mv.emplace_back((warping_mv[2]));
        }
        getPredictedImage(intra,target,out,triangleVec,mv,parallel_flag);
        cv::Point2f a(mv[0].x, mv[0].y), b(mv[1].x, mv[1].y), c(mv[2].x, mv[2].y);

        std::cout <<     a << " " << b << " " << c << std::endl;
        double MSE_prev = MSE_tmp / (double) triangle_size;
//        continue;

        // 並行移動のみ塗る
        if(parallel_flag) {
            double sx = std::min({(int)triangleVec.p1.x, (int)triangleVec.p2.x, (int)triangleVec.p3.x});
            double lx = std::max({(int)triangleVec.p1.x, (int)triangleVec.p2.x, (int)triangleVec.p3.x});
            double sy = std::min({(int)triangleVec.p1.y, (int)triangleVec.p2.y, (int)triangleVec.p3.y});
            double ly = std::max({(int)triangleVec.p1.y, (int)triangleVec.p2.y, (int)triangleVec.p3.y});

            for (int y = (int) (round(sy) - 1); y <= round(ly) + 1; y++) {
                for (int x= (int) (round(sx) - 1); x <= round(lx) + 1; x++) {
                    cv::Point2f xp(x, y);

                    if(isInTriangle(triangleVec, xp)) {
                        R(parallel_flag_img_color, x, y) = 255;
                        G(parallel_flag_img_color, x, y) = 0;
                        B(parallel_flag_img_color, x, y) = 0;
                    }

                }
            }
        }

        // <頂点座標, 残差>
        std::vector<std::pair<cv::Point2f, double>> add_corner_pair(add_corner.size());
        // <座標, 残差, 三角パッチ>
        std::vector<std::tuple<cv::Point2f, double, std::vector<Triangle>>> add_corner_tuple(add_corner.size());

        // 追加候補の頂点を入れることで予測精度が向上するかどうかを検証
        for (int c_idx = 0; c_idx < (int) add_corner.size(); c_idx++) {
            add_corner_tmp = corners;
            triangles_tmp = triangles;
            add_corner_tmp.emplace_back(add_corner[c_idx]);

            DelaunayTriangulation md_later(Rectangle(0, 0, target.cols, target.rows));
            std::vector<Triangle> triangles_around;

            std::vector<cv::Vec6f> triangles_mydelaunay;
            std::vector<bool> dummy(add_corner_tmp.size());

            if (INTER_DIV) {
                triangles_around = inter_div(triangles_tmp, add_corner_tmp, add_corner[c_idx], t);
            } else {
                md_later.insert(add_corner_tmp);
                md_later.getTriangleList(triangles_mydelaunay);
                triangles_around = md_later.Get_triangles_around((int) add_corner_tmp.size() - 1, add_corner_tmp,
                                                                 dummy);
            }

            double MSE_later = 0.0;
            int triangle_size_sum_later = 0;
            for (int t_idx = 0; t_idx < (int) triangles_around.size(); t_idx++) {
//                std::cout << "t_idx = " << t_idx << "/ " << triangles_around.size() << std::endl;
                Triangle triangle_later = triangles_around[t_idx];
                Point3Vec triangleVec_later(add_corner_tmp[triangle_later.p1_idx],
                                            add_corner_tmp[triangle_later.p2_idx],
                                            add_corner_tmp[triangle_later.p3_idx]);

                int triangle_later_size;
                // add_countは足した頂点の数らしい(ループで共有)
                Gauss_Newton2(ref, target, intra, predict_buf_dummy, predict_warp, predict_para, MSE_tmp,
                         triangleVec_later, prev_corners, &parallel_flag, t,
                              residual_ref, triangle_later_size,erase_th_global);
                MSE_later += MSE_tmp;
                triangle_size_sum_later += triangle_later_size;
            }
            // 加える可能性のある頂点の座標
            add_corner_pair[c_idx].first = add_corner[c_idx];
            // 単位面積当たりの残差
            add_corner_pair[c_idx].second = MSE_later / (double) triangle_size_sum_later;

            std::get<0>(add_corner_tuple[c_idx]) = add_corner[c_idx];
            std::get<1>(add_corner_tuple[c_idx]) = MSE_later / (double) triangle_size_sum_later;
            std::get<2>(add_corner_tuple[c_idx]) = triangles_tmp;
        }

        if (!add_corner.empty()) {
            // 残差でソート
            bubbleSort(add_corner_pair, add_corner_pair.size());
            bubbleSort(add_corner_tuple, add_corner_tuple.size());
            if ((MSE_prev - std::get<1>(add_corner_tuple[0])) / MSE_prev >= 0) {
                add_count = add_count + 1;
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
            std::cout << "add_corner_pair [" << i << "] = " << add_corner_pair[i].first << " MSE = "
                      << add_corner_pair[i].second << std::endl;
        }

        // 差分ベクトル
        for (int q = 1; q <= 2; q++) {
            mv[q].x -= mv[0].x;
            mv[q].y -= mv[0].y;
        }
        tmp_mv_x++; //warp_flagの1bit

        // もともとの頂点それぞれの動きベクトル
        std::vector<std::pair<cv::Point2i, int>> add_mv_original;
        add_mv_original.emplace_back(mv[0], triangle.p1_idx);

        // para_flag: ガウスニュートン法が返すやつ
        // 平行移動のほうが良い場合はpara_flag=true
        if (!parallel_flag) {
            add_mv_original.emplace_back(mv[0] + mv[1], triangle.p2_idx);
            add_mv_original.emplace_back(mv[0] + mv[2], triangle.p3_idx);
        } else {
            add_mv_original.emplace_back(mv[0], triangle.p2_idx);
            add_mv_original.emplace_back(mv[0], triangle.p3_idx);
        }

        //pair<3つの動きベクトル,頂点番号>, 基準ベクトルが存在するか表すフラグ, 平行移動を表すフラグ,実際に符号化する基準ベクトル(差分をとったもの),スタックを遡る個数,スタックの一番上の参照するベクトル
        std::tuple<std::vector<std::pair<cv::Point2i, int>>, bool, bool, cv::Point2f, int, cv::Point2f> add_mv_tuple(add_mv_original,
                                                                                                                    false,
                                                                                                                    parallel_flag,
                                                                                                                    cv::Point2f(0,
                                                                                                                                0),
                                                                                                                    0,cv::Point2f(0,0));

        mv_basis_tuple[triangle.p1_idx].emplace_back(add_mv_tuple);

        if (parallel_flag) {
            cnt++;
        } else {
            for (int q = 1; q <= 2; q++) {
                // see: https://ja.wikipedia.org/wiki/%E3%82%B4%E3%83%AD%E3%83%A0%E7%AC%A6%E5%8F%B7
                // これでいうところのmがgolomb_parameter
                int golomb_parameter = 32;
                // x要素の符号化
                sabun_mv_tmp_x += ozi::getGolombCode(golomb_parameter, mv[q].x, ozi::REGION1, ozi::GOLOMB, 0);
                sabun_mv_tmp_x += 1;

                // y要素の符号化
                sabun_mv_tmp_y += ozi::getGolombCode(golomb_parameter, mv[q].y, ozi::REGION1, ozi::GOLOMB, 0);
                sabun_mv_tmp_y += 1;

                std::cout << "x: " << mv[q].x << " code-length: "
                          << ozi::getGolombCode(golomb_parameter, mv[q].x, ozi::REGION1, ozi::GOLOMB, 0) << std::endl;
                std::cout << "y: " << mv[q].y << " code-length: "
                          << ozi::getGolombCode(golomb_parameter, mv[q].y, ozi::REGION1, ozi::GOLOMB, 0) << std::endl;

            }
        }

        MSE_tmp = 0;
        cv::Point2d xp, va, vb, ta, tb, tc;
        ta = triangleVec.p1;
        tb = triangleVec.p2;
        tc = triangleVec.p3;

        std::cout << numerator++ << "/" << denominator << "\n";
    }
    std::cout << "check" << std::endl;
    // 基準ベクトルの符号化
    std::vector<std::vector<cv::Point2i>> corded_mv(corners.size());
    std::queue<int> Queue_neighbor;
    bool init_flag = false;

    // 初期化のようなもの（1番最初に符号化するための頂点を見つける）
//        int start_idx = 0;
//        while(mv_basis_tuple[start_idx].empty()) start_idx++;

    for (int i = 0; i < (int) mv_basis_tuple.size(); i++) { // i: 三角パッチの番号
        for (int j = 0; j < (int) mv_basis_tuple[i].size(); j++) { // j: スタックのようなものを辿る
            std::get<1>(mv_basis_tuple[i][j]) = true; // 符号化済みにする
            std::get<3>(mv_basis_tuple[i][j]) = std::get<0>(mv_basis_tuple[i][j])[0].first; // 基準ベクトルの差分ベクトル（実際に符号化するもの）
            std::get<4>(mv_basis_tuple[i][j]) = 0; // 遡る個数
            std::get<5>(mv_basis_tuple[i][j]) = std::get<0>(mv_basis_tuple[i][j])[0].first; // スタックの1番上

            int pt1_idx = i;
            int pt2_idx = std::get<0>(mv_basis_tuple[i][j])[1].second;
            int pt3_idx = std::get<0>(mv_basis_tuple[i][j])[2].second;
            corded_mv[pt1_idx].emplace_back(std::get<0>(mv_basis_tuple[i][j])[0].first);
//            corded_mv[pt2_idx].emplace_back(std::get<0>(mv_basis_tuple[i][j])[1].first);
//            corded_mv[pt3_idx].emplace_back(std::get<0>(mv_basis_tuple[i][j])[2].first);

            init_flag = true;
            // TODO: bug fix
            std::vector<int> neighbor = md.getNeighborVertexNum(pt1_idx);
//            std::vector<int> neighbor = md.getNeighborVertexNum(std::get<0>(mv_basis[i][j])[1].second);
            for (int k = 0; k < (int) neighbor.size(); k++) {
                Queue_neighbor.push(neighbor[k]);
            }
            break;
        }
        if (init_flag)break;
    }
    std::cout << "check1" << std::endl;

    std::vector<bool> inqueue_flag(corners.size(), false);
    while (!Queue_neighbor.empty()) {
        // 探索用の変数
        int current_idx = Queue_neighbor.front();
        Queue_neighbor.pop();
        // 取り出した頂点の隣接を入れる
        if (!(int)mv_basis_tuple[current_idx].empty()) {
            std::vector<int> neighbor = md.getNeighborVertexNum(current_idx);

            for(int k = 0;k < (int)neighbor.size();k++){
                if(!mv_basis_tuple[neighbor[k]].empty()){
                    bool f = false;
                    for(int j = 0 ; j < (int)(mv_basis_tuple[neighbor[k]]).size() ; j++) {
                        if(!std::get<1>(mv_basis_tuple[neighbor[k]][j])) f = true;
                    }

                    if (f && !inqueue_flag[k]) {
                        Queue_neighbor.push(neighbor[k]);
                        inqueue_flag[neighbor[k]] = true;
                    }
                }
            }
        }
        std::cout << "check2" << std::endl;

        // その頂点に設定されている基準ベクトルをすべて符号化する
        for (int j = 0; j < (int) mv_basis_tuple[current_idx].size(); j++) {
            bool is_coded = std::get<1>(mv_basis_tuple[current_idx][j]);
            if (!is_coded) {
                std::get<1>(mv_basis_tuple[current_idx][j]) = true; // 符号化済みに変更
                // その頂点にいくつか動きベクトルがある場合
                if (!corded_mv[current_idx].empty()) {
                    double min_cord = 10E05;
                    double min_prev = 10E05;
                    cv::Point2f min_diff(0, 0);
                    int back_number = 0;
                    // その頂点
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
                    mv_diff = std::get<0>(mv_basis_tuple[current_idx][j])[0].first;
                    if (fabs(mv_diff.x) + fabs(mv_diff.y) < min_cord) {
                        min_diff = mv_diff;
                        back_number = 0;
                    }
                    if(fabs(mv_diff.x) + fabs(mv_diff.y) < min_prev){
                        mv_prev = mv_diff;
                        back_number = 0;
                    }
                    std::get<3>(mv_basis_tuple[current_idx][j]) = min_diff;
                    std::get<4>(mv_basis_tuple[current_idx][j]) = back_number;
                    std::get<5>(mv_basis_tuple[current_idx][j]) = mv_prev;
                    // TODO: 基準ベクトル以外を使っていいのか？
//                    corded_mv[std::get<0>(mv_basis_tuple[current_idx][j])[1].second].emplace_back(
//                            std::get<0>(mv_basis_tuple[current_idx][j])[1].first);
//                    corded_mv[std::get<0>(mv_basis_tuple[current_idx][j])[2].second].emplace_back(
//                            std::get<0>(mv_basis_tuple[current_idx][j])[2].first);
                } else {
                    double init_Distance = 10E05;
                    double min_Distance = init_Distance;
                    int min_num = 0;
                    bool flag_arround = false;
                    std::vector<int> neighbor = md.getNeighborVertexNum(
                            std::get<0>(mv_basis_tuple[current_idx][j])[0].second);
                    std::vector<cv::Point2f> neighbor_cood = md.getNeighborVertex(
                            std::get<0>(mv_basis_tuple[current_idx][j])[0].second);
                    for (int k = 0; k < (int) neighbor.size(); k++) {
                        if (!corded_mv[neighbor[k]].empty()) {
                            flag_arround = true;
                            double Distance = md.getDistance(
                                    corners[std::get<0>(mv_basis_tuple[current_idx][j])[0].second], neighbor_cood[k]);
                            if (min_Distance > Distance) {
                                min_Distance = Distance;
                                min_num = neighbor[k];
                            }
                        }
                    }
                    if(flag_arround) {
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
                        mv_diff = std::get<0>(mv_basis_tuple[current_idx][j])[0].first;
                        if (fabs(mv_diff.x) + fabs(mv_diff.y) < min_cord) {
                            min_diff = mv_diff;
                            back_number = 0;
                        }
                        if(fabs(mv_diff.x) + fabs(mv_diff.y) < min_prev){
                            mv_prev = mv_diff;
                            back_number = 0;
                        }
                        std::get<3>(mv_basis_tuple[current_idx][j]) = min_diff;
                        std::get<4>(mv_basis_tuple[current_idx][j]) = back_number;
                        std::get<5>(mv_basis_tuple[current_idx][j]) = mv_prev;
//                        corded_mv[std::get<0>(mv_basis_tuple[current_idx][j])[1].second].emplace_back(
//                                std::get<0>(mv_basis_tuple[current_idx][j])[1].first);
//                        corded_mv[std::get<0>(mv_basis_tuple[current_idx][j])[2].second].emplace_back(
//                                std::get<0>(mv_basis_tuple[current_idx][j])[2].first);
                    }else{
                        std::get<3>(mv_basis_tuple[current_idx][j]) = std::get<0>(mv_basis_tuple[current_idx][j])[0].first;
                        std::get<4>(mv_basis_tuple[current_idx][j]) = 0;
                        std::get<5>(mv_basis_tuple[current_idx][j]) = std::get<0>(mv_basis_tuple[current_idx][j])[0].first;
//                        corded_mv[std::get<0>(mv_basis_tuple[current_idx][j])[1].second].emplace_back(
//                                std::get<0>(mv_basis_tuple[current_idx][j])[1].first);
//                        corded_mv[std::get<0>(mv_basis_tuple[current_idx][j])[2].second].emplace_back(
//                                std::get<0>(mv_basis_tuple[current_idx][j])[2].first);
                    }
                }
            }
        }
    }
    std::cout << "chek3" << std::endl;
    cv::imwrite(getProjectDirectory() + "/img/minato/flag_img_" + std::to_string(block_size_x) + "_" + std::to_string(block_size_y) + "_QP" + std::to_string(qp) + "_RIGHT.png", parallel_flag_img_color);

    int worth = 0;
    for(int i = 0;i < (int)mv_basis_tuple.size();i++) {
        for (int j = 0; j < (int) mv_basis_tuple[i].size(); j++) {
            int golomb_para = 16;

            cv::Point2f prev = std::get<5>(mv_basis_tuple[i][j]);
            tmp_mv_x += 1;
            basis_mv_tmp_x += ozi::getGolombCode(golomb_para, prev.x, ozi::REGION1, ozi::GOLOMB, 0);
            basis_mv_tmp_x += 1;
            basis_mv_tmp_y += ozi::getGolombCode(golomb_para, prev.y, ozi::REGION1, ozi::GOLOMB, 0);
            basis_mv_tmp_y += 1;

            // for debug
            cv::Point2f mv_diff = std::get<3>(mv_basis_tuple[i][j]); // 遡って一番符号量が少ないときのベクトル
            cv::Point2f basis = std::get<0>(mv_basis_tuple[i][j])[0].first; // 基準ベクトル（差分もとっていないそのままのもの）
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

    std::cout << "check point 5" << std::endl;
    return PredictedImageResult(out, mv_image, freq_block, freq_warp, block_matching_pixel_nums, warping_pixel_nums,
                                x_bits, y_bits, block_matching_pixel_errors, warping_pixel_errors);

}
#pragma clang diagnostic pop

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

cv::Mat getReconstructionDivisionImage(cv::Mat image, std::vector<CodingTreeUnit *> ctu) {
    Reconstruction rec(image);
    rec.init(128, 128, LEFT_DIVIDE);
    puts("");
    rec.reconstructionTriangle(ctu);
    std::vector<Point3Vec> hoge = rec.getTriangleCoordinateList();

    cv::Mat reconstructedImage = cv::imread(getProjectDirectory() + "/img/minato/minato_000413_limit.bmp ");
    for(const auto foo : hoge) {
        drawTriangle(reconstructedImage, foo.p1, foo.p2, foo.p3, cv::Scalar(255, 255, 255));
    }
    cv::imwrite(getProjectDirectory() + "/img/minato/reconstruction.png", reconstructedImage);

    return reconstructedImage;
}