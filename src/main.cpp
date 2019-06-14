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
#include "../includes/Utils.h"


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

std::vector<cv::Point2f> uniqCoordinate(const std::vector<cv::Point2f> &corners);


cv::Mat getReconstructionDivisionImage(cv::Mat image, std::vector<CodingTreeUnit *> ctu);
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

#pragma clang diagnostic push
#pragma ide diagnostic ignored "hicpp-signed-bitwise"

void storeResidualImage(){
//    cv::imwrite(getProjectDirectory(OS) + "/img/minato/p_residual_image_22_divide_5_billinear.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/p_image_22_divide_5_billinear.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp"),2));
//    cv::imwrite(getProjectDirectory(OS) + "/img/minato/p_residual_image_27_divide_5_billinear.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/p_image_27_divide_5_billinear.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp"),2));
//    cv::imwrite(getProjectDirectory(OS) + "/img/minato/p_residual_image_32_divide_5_billinear.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/p_image_32_divide_5_billinear.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp"),2));
//    cv::imwrite(getProjectDirectory(OS) + "/img/minato/p_residual_image_37_divide_5_billinear.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/p_image_37_divide_5_billinear.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp"),2));
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/residual_HM_qp22_P.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/minato_qp22_filter_off.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp"),4));
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/residual_HM_qp27_P.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/minato_qp27_filter_off.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp"),4));
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/residual_HM_qp32_P.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/minato_qp32_filter_off.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp"),4));
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/residual_HM_qp37_P.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/minato_qp37_filter_off.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp"),4));

    cv::imwrite(getProjectDirectory(OS) + "/img/minato/p_residual_image_22_divide_5_bicubic.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/p_image_22_divide_5_bicubic.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp"),4));
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/p_residual_image_27_divide_5_bicubic.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/p_image_27_divide_5_bicubic.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp"),4));
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/p_residual_image_32_divide_5_bicubic.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/p_image_32_divide_5_bicubic.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp"),4));
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/p_residual_image_37_divide_5_bicubic.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/p_image_37_divide_5_bicubic.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp"),4));


    cv::imwrite(getProjectDirectory(OS) + "/img/minato/residual_residual_qp22_P.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/residual_HM_qp22_P.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/p_residual_image_22_divide_5.png")));
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/residual_residual_qp27_P.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/residual_HM_qp27_P.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/p_residual_image_27_divide_5.png")));
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/residual_residual_qp32_P.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/residual_HM_qp32_P.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/p_residual_image_32_divide_5.png")));
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/residual_residual_qp37_P.png", getResidualImage(cv::imread(getProjectDirectory(OS)+ "/img/minato/residual_HM_qp37_P.png"), cv::imread(getProjectDirectory(OS) + "/img/minato/p_residual_image_37_divide_5.png")));
    exit(0);
}

void test1(){
    cv::Mat gaussRefImage = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_limit_2_I22.bmp");
    cv::Mat ref_image     = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_limit_2_I22.bmp");
    cv::Mat target_image  = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp");

    std::vector<std::vector<cv::Mat>> ref_images, target_images;

    ref_images = getRefImages(ref_image, gaussRefImage);
    target_images = getTargetImages(target_image);

    EXPAND_ARRAY_TYPE expand_images = getExpandImages(ref_images, target_images, 500);

    cv::Point2f p1(0.0, 0.0), p2(64, 0), p3(128, 0), p4(0, 64), p5(64, 64), p6(0, 128);

    // 一番大きな三角形
    double error1;
    int area;
    std::tie(std::ignore, std::ignore, error1, area, std::ignore) = GaussNewton(ref_images, target_images, expand_images, Point3Vec(p1, p3, p6));
    error1 /= area;

    // 1分割
    double error2 = 0.0, error_tmp = 0.0;
    double area2 = 0.0, area2_tmp;
    cv::Point2f mv;
    std::tie(std::ignore, mv, error_tmp, area2_tmp, std::ignore) = GaussNewton(ref_images, target_images, expand_images, Point3Vec(p1, p3, p5));
    error2 += (error_tmp);
    area2 += area2_tmp;
    std::tie(std::ignore, mv, error_tmp, area2_tmp, std::ignore) = GaussNewton(ref_images, target_images, expand_images, Point3Vec(p1, p5, p6));
    error2 += (error_tmp);
    area2 += area2_tmp;
    error2 /= area2;

    double error3 = 0.0;
    double area3 = 0.0, area3_tmp;
    std::tie(std::ignore, std::ignore, error_tmp, area3_tmp, std::ignore) = GaussNewton(ref_images, target_images, expand_images, Point3Vec(p1, p2, p5));
    error3 += (error_tmp);
    area3 += area3_tmp;
    std::tie(std::ignore, std::ignore, error_tmp, area3_tmp, std::ignore) = GaussNewton(ref_images, target_images, expand_images, Point3Vec(p2, p3, p5));
    error3 += (error_tmp);
    area3 += area3_tmp;
    std::tie(std::ignore, std::ignore, error_tmp, area3_tmp, std::ignore) = GaussNewton(ref_images, target_images, expand_images, Point3Vec(p1, p4, p5));
    error3 += (error_tmp);
    area3 += area3_tmp;
    std::tie(std::ignore, std::ignore, error_tmp, area3_tmp, std::ignore) = GaussNewton(ref_images, target_images, expand_images, Point3Vec(p4, p5, p6));
    error3 += (error_tmp);
    area3 += area3_tmp;

    error3 /= area3;

    freeExpandImages(expand_images, 500, 2, 4, ref_image.rows, ref_image.cols);

    std::cout << "error1: " << error1 << " error2:" << error2 << " error3:" << error3 << std::endl;
}

int main(int argc, char *argv[]){
    // Write test codes below
     test1();
//    storeResidualImage();
//    std::cout << getPSNR(cv::imread(getProjectDirectory(OS)+ std::string(argv[1])), cv::imread(getProjectDirectory(OS) + std::string(argv[2]))) << std::endl;
//    exit(0);
    std::string config_path = std::string(argv[1]);
    // exec ME
    run(config_path);
//    storeResidualImage();

}
std::string out_file_suffix = "";

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

//        cv::resize(target_image, tmp_target, cv::Size(1792, 1024));
//
//        target_image = tmp_target.clone();

        std::cout << "width: " << target_image.cols << " height: " << target_image.rows << std::endl;
        int height_mod = target_image.rows % block_size_y;
        int width_mod = target_image.cols % block_size_x;

        // block_sizeの定数倍となるような画像を作る
        cv::Mat crop_target_image = cv::Mat::zeros(target_image.rows - height_mod, target_image.cols - width_mod,
                                                   CV_8UC3);
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

        triangle_division.initTriangle(block_size_x, block_size_y, division_steps, LEFT_DIVIDE);
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
        int add_count;
        cv::Mat r_ref = cv::Mat::zeros(target_image.rows, target_image.cols, CV_8UC1);
        int tmp_mv_x, tmp_mv_y;
        int p_flag;

        std::vector<std::vector<cv::Mat>> ref_images, target_images;
        // 参照画像のフィルタ処理（１）
        std::vector<cv::Mat> ref1_levels;
        cv::Mat ref_level_1, ref_level_2, ref_level_3, ref_level_4;
        ref_level_1 = gaussRefImage;
        ref_level_2 = half(ref_level_1, 2);
        ref_level_3 = half(ref_level_2, 2);
        ref_level_4 = half(ref_level_3, 2);
        ref1_levels.emplace_back(ref_level_4);
        ref1_levels.emplace_back(ref_level_3);
        ref1_levels.emplace_back(ref_level_2);
        ref1_levels.emplace_back(ref_image);

        // 対象画像のフィルタ処理（１）
        std::vector<cv::Mat> target1_levels;
        cv::Mat target_level_1, target_level_2, target_level_3, target_level_4;
        target_level_1 = target_image;
        target_level_2 = half(target_level_1, 2);
        target_level_3 = half(target_level_2, 2);
        target_level_4 = half(target_level_3, 2);
        target1_levels.emplace_back(target_level_4);
        target1_levels.emplace_back(target_level_3);
        target1_levels.emplace_back(target_level_2);
        target1_levels.emplace_back(target_level_1);

        // 参照画像のフィルタ処理（２）
        std::vector<cv::Mat> ref2_levels;
        cv::Mat ref2_level_1 = gaussRefImage;
        cv::Mat ref2_level_2 = half(ref2_level_1, 2);
        cv::Mat ref2_level_3 = half(ref2_level_2, 1);
        cv::Mat ref2_level_4 = half(ref2_level_3, 1);
        ref2_levels.emplace_back(ref2_level_4);
        ref2_levels.emplace_back(ref2_level_3);
        ref2_levels.emplace_back(ref2_level_2);
        ref2_levels.emplace_back(ref_image);

        // 対象画像のフィルタ処理（２）
        std::vector<cv::Mat> target2_levels;
        cv::Mat target2_level_1 = target_image;
        cv::Mat target2_level_2 = half(target2_level_1, 2);
        cv::Mat target2_level_3 = half(target2_level_2, 1);
        cv::Mat target2_level_4 = half(target2_level_3, 1);
        target2_levels.emplace_back(target2_level_4);
        target2_levels.emplace_back(target2_level_3);
        target2_levels.emplace_back(target2_level_2);
        target2_levels.emplace_back(target2_level_1);

        ref_images.emplace_back(ref1_levels);
        ref_images.emplace_back(ref2_levels);
        target_images.emplace_back(target1_levels);
        target_images.emplace_back(target2_levels);

        std::vector<std::vector<std::vector<unsigned char **>>> expand_images;
        expand_images.resize(ref_images.size());
        for (int filter_num = 0; filter_num < static_cast<int>(ref_images.size()); filter_num++) {
            expand_images[filter_num].resize(ref_images[filter_num].size());
            for (int step = 0; step < static_cast<int>(ref_images[filter_num].size()); step++) {
                expand_images[filter_num][step].resize(4);
            }
        }

        unsigned char **current_target_expand, **current_target_org_expand; //画像の周りに500ピクセルだけ黒の領域を設ける(念のため)
        unsigned char **current_ref_expand, **current_ref_org_expand;       //f_expandと同様

        int expand = 500;
        expand_images = getExpandImages(ref_images, target_images, expand);

        triangle_division.constructPreviousCodingTree(foo, 0);
        for (int i = 0; i < init_triangles.size(); i++) {
            std::pair<Point3Vec, int> triangle = init_triangles[i];
            cv::Point2f p1 = triangle.first.p1;
            cv::Point2f p2 = triangle.first.p2;
            cv::Point2f p3 = triangle.first.p3;
            std::cout << "================== step:" << i << " ================== " << std::endl;
            triangle_division.split(expand_images, foo[i], nullptr, Point3Vec(p1, p2, p3), i, triangle.second,
                                    division_steps);
        }
        std::cout << "split finished" << std::endl;
        getReconstructionDivisionImage(gaussRefImage, foo);
        cv::Mat p_image = triangle_division.getPredictedImageFromCtu(foo);
        int code_length = triangle_division.getCtuCodeLength(foo);
        std::cout << "qp:" << qp << " divide:" << division_steps << std::endl;
        std::cout << "PSNR:" << getPSNR(target_image, p_image) << " code_length:" << code_length << std::endl;
        std::cout << img_directory + "p_mv_image_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) +
                     out_file_suffix + ".png" << std::endl;
        cv::imwrite(img_directory + "p_image_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) +
                    out_file_suffix + ".png", p_image);
        cv::imwrite(
                img_directory + "p_residual_image_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) +
                out_file_suffix + ".png", getResidualImage(target_image, p_image, 4));
        cv::imwrite(img_directory + "p_mv_image_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) +
                    out_file_suffix + ".png", triangle_division.getMvImage(foo));


        // TODO: ログだすやつ書く
        // ===========================================================
        // ログ出力
        // ===========================================================

    }
}

/**
 * @fn cv::Mat getReconstructionDivisionImage(cv::Mat image, std::vector<CodingTreeUnit *> ctu)
 * @brief CodingTreeをもらって、三角形を書いた画像を返す
 * @param image 下地
 * @param ctu CodingTree
 * @return 画像
 */
cv::Mat getReconstructionDivisionImage(cv::Mat image, std::vector<CodingTreeUnit *> ctu) {
    Reconstruction rec(image);
    rec.init(block_size_x, block_size_y, LEFT_DIVIDE);
    puts("");
    rec.reconstructionTriangle(ctu);
    std::vector<Point3Vec> hoge = rec.getTriangleCoordinateList();

    cv::Mat reconstructedImage = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp");
    for(const auto foo : hoge) {
        drawTriangle(reconstructedImage, foo.p1, foo.p2, foo.p3, cv::Scalar(255, 255, 255));
    }
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/reconstruction_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png", reconstructedImage);

    return reconstructedImage;
}