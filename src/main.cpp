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
#include <cassert>
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
#include "../includes/Utils.h"
#include "../includes/tests.h"

void run(std::string config_path);

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
int division_steps;

int main(int argc, char *argv[]){
    // Write test codes below

    std::string config_path = std::string(argv[1]);
    run(config_path);
}
std::string out_file_suffix = "_debug";

void run(std::string config_path) {

    std::cout << "OpenCV_version : " << getVersionOfOpenCV() << std::endl;

    const std::string project_directory_path = getProjectDirectory(OS);
    FILE *img_list;

    std::string list_path = (OS == "Win" ? replaceBackslash(project_directory_path + config_path) :
                             project_directory_path + config_path);
    if ((img_list = fopen((list_path).c_str(), "r")) == NULL) {
        std::cerr << "Error : Can not open file" << std::endl;
        exit(1);
    }

    char buf[512];

    std::vector<cv::Point2f> corners, corners_org;
    std::vector<cv::Point2f> ref_corners, ref_corners_org;

    // 全画像分ループ
    while (fgets(buf, sizeof(buf), img_list) != nullptr) {
        if (buf[0] == '#') continue;
        char t_file_name[256], r_file_name[256], o_file_name[256], i_file_path[256], csv_prefix[256], r_intra_file_name[256], target_color_file_name[256], c_file_name[256];
        sscanf(buf, "%s %s %s %s %s %s %s %d %d %d %d", i_file_path, r_file_name, t_file_name, o_file_name,
               r_intra_file_name, target_color_file_name, c_file_name, &qp, &block_size_x, &block_size_y,
               &division_steps);

        std::string img_path = ((OS == "Win") ? replaceBackslash(std::string(i_file_path)) : std::string(i_file_path));
        std::string img_directory = project_directory_path + img_path;
        std::string target_file_name = std::string(t_file_name);

        std::string ref_file_name = std::string(r_file_name);
        std::string ref_intra_file_name = std::string(r_intra_file_name);
        std::string corner_file_name = std::string(c_file_name);
        std::string csv_file_prefix = std::string("aaa");

        std::string ref_file_path = project_directory_path + img_path + ref_file_name;
        std::string target_file_path = project_directory_path + img_path + target_file_name;
        std::string ref_intra_file_path = project_directory_path + img_path + ref_intra_file_name;
        std::string target_color_file_path = project_directory_path + img_path + target_color_file_name;

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
        cv::Mat refx2, refx4;
        cv::Mat targetx2, targetx4, targetx2_sharp, target_sharp, target_sharp_gray;
        cv::Mat refx8, targetx8;
        cv::Mat targetx4_sharp, targetx8_gray;
        cv::Mat ref_ref;
        cv::Mat target_bi, ref_bi;
        cv::Mat canny, canny_target;

        // QP変化させた参照画像はここで読む
        ref_image = cv::imread(ref_intra_file_path);
        target_image = cv::imread(target_file_path);

        cv::Mat tmp_target, tmp_ref;

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

        cv::Mat gaussRefImage = cv::imread(ref_file_path);
        TriangleDivision triangle_division(ref_image, target_image, gaussRefImage);

        triangle_division.initTriangle(block_size_x, block_size_y, division_steps, qp, LEFT_DIVIDE);
        std::vector<Point3Vec> triangles = triangle_division.getTriangleCoordinateList();

        std::vector<std::pair<Point3Vec, int> > init_triangles = triangle_division.getTriangles();
        std::cout << init_triangles.size() << std::endl;
        std::vector<CodingTreeUnit *> foo(init_triangles.size());
        for (int i = 0; i < init_triangles.size(); i++) {
            foo[i] = new CodingTreeUnit();
            foo[i]->split_cu_flag = false;
            foo[i]->leftNode = foo[i]->rightNode = nullptr;
            foo[i]->triangle_index = i;
        }

        cv::Mat spatialMvTestImage;
        cv::Mat new_gauss_output_image = cv::Mat::zeros(gaussRefImage.rows, gaussRefImage.cols, CV_8UC3);

        std::vector<Triangle> tt = triangle_division.getTriangleIndexList();
        corners = triangle_division.getCorners();

        std::vector<cv::Point2f> tmp_ref_corners(corners.size()), add_corners;

        cv::Mat r_ref = cv::Mat::zeros(target_image.rows, target_image.cols, CV_8UC1);

        std::vector<std::vector<cv::Mat>> ref_images, target_images;

        ref_images = getRefImages(ref_image, gaussRefImage);
        target_images = getTargetImages(target_image);

        std::vector<std::vector<std::vector<unsigned char **>>> expand_images;

        int expand = 500;
        expand_images = getExpandImages(ref_images, target_images, expand);

        triangle_division.constructPreviousCodingTree(foo, 0);

        std::vector<std::vector<std::vector<int>>> diagonal_line_area_flag(init_triangles.size(), std::vector< std::vector<int> >(block_size_x, std::vector<int>(block_size_y, -1)) );

        for (int i = 0; i < init_triangles.size(); i++) {
//            std::vector<std::vector<int>> diagonal_line_area_flag(block_size_x, std::vector<int>(block_size_y, 0)); // 斜め線でどちらを取るか表すフラグ flag[x][y]
            if(i % 2 == 0){
                bool flag = false;
                for (int x = 0; x < block_size_x; x++) {
                    // diagonal line
                    diagonal_line_area_flag[i/2][x][block_size_y - x - 1] = (flag ? i : i + 1);
                    flag = !flag;
                }
            }

            std::pair<Point3Vec, int> triangle = init_triangles[i];
            cv::Point2f p1 = triangle.first.p1;
            cv::Point2f p2 = triangle.first.p2;
            cv::Point2f p3 = triangle.first.p3;
            std::cout << "================== step:" << i << " ================== " << std::endl;
            triangle_division.split(expand_images, foo[i], nullptr, Point3Vec(p1, p2, p3), i, triangle.second, division_steps, diagonal_line_area_flag[i/2]);
        }
        std::cout << "split finished" << std::endl;

        // TODO: ログだすやつ書く
        // ===========================================================
        // ログ出力
        // ===========================================================
        // TODO: init処理を書き直さないといけない
        getReconstructionDivisionImage(gaussRefImage, foo, block_size_x, block_size_y);
        cv::Mat p_image = triangle_division.getPredictedImageFromCtu(foo, diagonal_line_area_flag);
        cv::Mat color = triangle_division.getPredictedColorImageFromCtu(foo, diagonal_line_area_flag, getPSNR(target_image, p_image));

        int code_length = triangle_division.getCtuCodeLength(foo);
        std::cout << "qp:" << qp << " divide:" << division_steps << std::endl;
        std::cout << "PSNR:" << getPSNR(target_image, p_image) << " code_length:" << code_length << std::endl;
        std::cout << img_directory + "p_mv_image_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png" << std::endl;
        cv::imwrite(img_directory + "p_image_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png", p_image);
        cv::imwrite( img_directory + "p_residual_image_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png", getResidualImage(target_image, p_image, 4));
        cv::imwrite(img_directory + "p_mv_image_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png", triangle_division.getMvImage(foo));
        cv::imwrite(img_directory + "p_color_image_"  + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png", color);

    }
}
