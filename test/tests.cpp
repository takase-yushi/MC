//
// Created by kasph on 2019/06/16.
//

#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include <fstream>
#include <random>

#include "../includes/tests.h"
#include "../includes/Utils.h"
#include "../includes/ImageUtil.h"
#include "../includes/ME.hpp"
#include "../includes/TriangleDivision.h"
#include "../includes/Reconstruction.h"
#include "../includes/psnr.h"
#include "../includes/ConfigUtil.h"

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

void test2(){
    cv::Mat gauss_ref_image = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_limit_2_I22.bmp");
    cv::Mat ref_image       = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_limit_2_I22.bmp");
    cv::Mat target_image    = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp");

    TriangleDivision triangle_division(ref_image, target_image, gauss_ref_image);

    // initする
    triangle_division.initTriangle(128, 128, 5, LEFT_DIVIDE);
    std::vector<Point3Vec> triangles = triangle_division.getTriangleCoordinateList();

    std::vector<std::pair<Point3Vec, int> > init_triangles = triangle_division.getTriangles();

    cv::Mat out = target_image.clone();
    for(auto triangle : init_triangles) {
        drawTriangle(out, triangle.first.p1, triangle.first.p2, triangle.first.p3, cv::Scalar(255, 255, 255));
    }

    cv::imwrite(getProjectDirectory(OS) + "/img/minato/init_triangle_127.png", out);
}

void test3(){
    // これ、最適化とかかけた環境下だと多分assertそのものが消されている説があるので
    assert(isPointOnTheLine(cv::Point2f(0, 0), cv::Point2f(10, 10), cv::Point2f(5, 5)));
    assert(isPointOnTheLine(cv::Point2f(0, 0), cv::Point2f(10, 10), cv::Point2f(5, 7)));

    // このコードに引っかかるとコケてる
    if(isPointOnTheLine(cv::Point2f(0, 0), cv::Point2f(10, 10), cv::Point2f(5, 5)) == false){
        std::cout << "TestCase1 failed" << std::endl;
        exit(1);
    }

    if(isPointOnTheLine(cv::Point2f(0, 0), cv::Point2f(10, 10), cv::Point2f(5, 7)) == true){
        std::cout << "TestCase2 failed" << std::endl;
        exit(1);
    }

    if(isPointOnTheLine(cv::Point2f(64, 64), cv::Point2f(64, 128), cv::Point2f(64,96)) == false) {
        std::cout << "TestCase3 failed" << std::endl;
        exit(1);
    }
}

void test4(){
    cv::Mat gauss_ref_image = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_limit_2_I22.bmp");
    cv::Mat ref_image       = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_limit_2_I22.bmp");
    cv::Mat target_image    = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp");

    TriangleDivision triangle_division(ref_image, target_image, gauss_ref_image);

    // initする
    triangle_division.initTriangle(128, 128, 5, LEFT_DIVIDE);
    std::vector<Point3Vec> triangles = triangle_division.getTriangleCoordinateList();

    std::vector<std::pair<Point3Vec, int> > init_triangles = triangle_division.getTriangles();

    cv::Mat out = target_image.clone();
    for(auto triangle : init_triangles) {
        drawTriangle(out, triangle.first.p1, triangle.first.p2, triangle.first.p3, cv::Scalar(255, 255, 255));
    }

    cv::imwrite(getProjectDirectory(OS) + "/img/minato/init_triangle_127.png", out);
}

/**
 * @fn void test5()
 * @brief 斜め線の判定テスト
 */
void test5(){
    cv::Mat gauss_ref_image = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_limit_2_I22.bmp");
    cv::Mat ref_image       = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_limit_2_I22.bmp");
    cv::Mat target_image    = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp");

    TriangleDivision triangle_division(ref_image, target_image, gauss_ref_image);

    int divide_steps = 8;
    // initする
    triangle_division.initTriangle(128, 128, divide_steps, LEFT_DIVIDE);
    std::vector<Point3Vec> triangles = triangle_division.getTriangleCoordinateList();

    std::vector<std::pair<Point3Vec, int>> init_triangles = triangle_division.getTriangles();

    // 色塗り分け画像

    std::vector<std::vector<cv::Mat>> ref_images, target_images;
    ref_images = getRefImages(ref_image, gauss_ref_image);
    target_images = getTargetImages(target_image);
    std::vector<std::vector<std::vector<unsigned char **>>> expand_images = getExpandImages(ref_images, target_images, 500);

    std::vector<CodingTreeUnit *> foo(init_triangles.size());
    for (int i = 0; i < init_triangles.size(); i++) {
        foo[i] = new CodingTreeUnit();
        foo[i]->split_cu_flag = false;
        foo[i]->node1 = foo[i]->node2 = foo[i]->node3 = foo[i]->node4 = nullptr;
        foo[i]->triangle_index = i;
    }
    // std::vector<std::vector<std::vector<int>>> area_flag_cache(init_triangles.size(), std::vector< std::vector<int> >(128, std::vector<int>(128)) );
    std::vector<std::vector<std::vector<int>>> diagonal_line_area_flag(init_triangles.size() / 2, std::vector< std::vector<int> >(128, std::vector<int>(128, -1)) ); // 斜め線でどちらを取るか表すフラグ flag[x][y]

    for(int i = 0 ; i < init_triangles.size() ; i++) {

        if (i % 2 == 0) {
            bool flag = false;
            for (int x = 0; x < 128; x++) {
                // diagonal line
                diagonal_line_area_flag[i/2][x][128 - x - 1] = (flag ? i : i + 1);
                flag = !flag;
            }

            diagonal_line_area_flag[i/2][0][0] = 0;
            diagonal_line_area_flag[i/2][127][0] = 0;
            diagonal_line_area_flag[i/2][0][127] = i + 1;
            diagonal_line_area_flag[i/2][127][127] = i + 1;
        }

        std::pair<Point3Vec, int> triangle = init_triangles[i];

        cv::Point2f p1 = triangle.first.p1;
        cv::Point2f p2 = triangle.first.p2;
        cv::Point2f p3 = triangle.first.p3;
        std::cout << "================== step:" << i << " ================== " << std::endl;

        triangle_division.split(expand_images, foo[i], nullptr, Point3Vec(p1, p2, p3), i, triangle.second, divide_steps, diagonal_line_area_flag[i/2]);
    }

    // ***************
    // テスト画像出力
    // ***************
    triangles = triangle_division.getTriangleCoordinateList();
    cv::Mat out = triangle_division.getPredictedDiagonalImageFromCtu(foo, diagonal_line_area_flag);

//    cv::Mat p_image = triangle_division.getPredictedImageFromCtu(foo);

    cv::imwrite(getProjectDirectory(OS) + "/img/minato/diagonal_test.png", out);
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/diagonal_reconstruction.png", getReconstructionDivisionImage(gauss_ref_image, foo, 128, 128));

}

void test6(){
    cv::Mat ref_image = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_limit_2_I22.bmp");
    unsigned char **img1 = getExpansionImage(ref_image, 4, 16);

    cv::Mat out = cv::Mat::zeros(ref_image.rows * 4 + 2 * 4 * 16, ref_image.cols * 4 + 2 * 4 * 16, CV_8UC3);
    for(int y = -16 * 4 ; y < 4 * ref_image.rows + 4 * 16 ; y++){
        for(int x = -16 * 4 ; x < 4 * ref_image.cols + 4 * 16 ; x++){
            R(out, x + 16 * 4, y + 16 * 4) = img1[x][y];
            G(out, x + 16 * 4, y + 16 * 4) = img1[x][y];
            B(out, x + 16 * 4, y + 16 * 4) = img1[x][y];
        }
    }

    cv::imwrite(getProjectDirectory(OS) + "/img/minato/out.png", out);
}

void test7(){
    cv::Mat ref_image = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_limit_2_I22.bmp");
    unsigned char **img1 = getExpansionHEVCImage(ref_image, 4, 16);


    cv::Mat out = cv::Mat::zeros(ref_image.rows * 4 + 2 * 4 * 16, ref_image.cols * 4 + 2 * 4 * 16, CV_8UC3);
    for(int y = -16 * 4 ; y < 4 * ref_image.rows + 4 * 16 ; y++){
        for(int x = -16 * 4 ; x < 4 * ref_image.cols + 4 * 16 ; x++){
            if(x % 4 == 0 && y % 4 == 0){
                R(out, x + 16 * 4, y + 16 * 4) = img1[x][y];
                G(out, x + 16 * 4, y + 16 * 4) = img1[x][y];
                B(out, x + 16 * 4, y + 16 * 4) = img1[x][y];
            }else{
                if(x == 3823 && y == 928) {
                    std::cout << img1[x][y] << std::endl;
                }
                int val = (img1[x][y] + 32)  / 64;
                val = (val > 255 ? 255 : (val < 0 ? 0 : val));
                R(out, x + 16 * 4, y + 16 * 4) = val;
                G(out, x + 16 * 4, y + 16 * 4) = val;
                B(out, x + 16 * 4, y + 16 * 4) = val;
            }

        }
    }

    cv::imwrite(getProjectDirectory(OS) + "/img/minato/out_hevc_filter.png", out);
}

void testFilter(){
    cv::Mat ref_image = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_limit_2_I22.bmp");
    cv::Mat out;
    cv::resize(ref_image, out, cv::Size(), 0.25, 0.25);

    cv::Mat bilinear = getExpansionMatImage(out, 4, 0, IP_MODE::BILINEAR);
    cv::Mat bicubic = getExpansionMatImage(out, 4, 0, IP_MODE::BICUBIC);
    unsigned char **hevc_ip = getExpansionHEVCImage(out, 4, 0);

    cv::Mat hevc = cv::Mat::zeros(ref_image.size(), CV_8UC3);
    for(int y = 0 ; y < hevc.rows ; y++) {
        for (int x = 0; x < hevc.cols; x++) {
            int val = hevc_ip[x][y];
            val = (val > 255 ? 255 : (val < 0 ? 0 : val));
            R(hevc, x, y) = val;
            G(hevc, x, y) = val;
            B(hevc, x, y) = val;
        }
    }

    std::cout << "Bilinear:" << getPSNR(bilinear, ref_image) << std::endl;
    std::cout << "Bicubic :" << getPSNR(bicubic, ref_image) << std::endl;
    std::cout << "HEVC    :" << getPSNR(hevc, ref_image) << std::endl;
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/resize_bilinear.png", bilinear);
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/resize_bicubic.png", bicubic);
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/resize_hevc.png", hevc);
}

void test4xHEVCImage(){
    cv::Mat ref_image = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_limit_2_I22.bmp");

    int k = 4;
    int expansion_size = 16;
    unsigned char **hevc_ip = getExpansionHEVCImage(ref_image, k, expansion_size);

    cv::Mat hevc = cv::Mat::zeros(k * (ref_image.rows + 2 * expansion_size), k * (ref_image.cols + 2 * expansion_size), CV_8UC3);

    for(int y = 0 ; y < k * (ref_image.rows + 2 * expansion_size) ; y++){
        for(int x = 0 ; x < k * (ref_image.cols + 2 * expansion_size); x++){
            R(hevc, x, y) = hevc_ip[x - k * expansion_size][y - k * expansion_size];
            G(hevc, x, y) = hevc_ip[x - k * expansion_size][y - k * expansion_size];
            B(hevc, x, y) = hevc_ip[x - k * expansion_size][y - k * expansion_size];
        }
    }

    cv::imwrite(getProjectDirectory(OS) + "/img/minato/test_4x_hevc.png", hevc);
}

void testRoundVecQuarter(){
    cv::Point2f a(-1.6, 1.1);
    cv::Point2f b(1.9, 1.1);
    cv::Point2f c(-1.6, -100.4);

    std::cout << roundVecQuarter(a) << std::endl;
    std::cout << roundVecQuarter(b) << std::endl;
    std::cout << roundVecQuarter(c) << std::endl;
}

void testHevcPSNR(){
    cv::Mat qp22 = cv::imread(getProjectDirectory(OS) + "/img/minato/HM_22.png");
    cv::Mat qp27 = cv::imread(getProjectDirectory(OS) + "/img/minato/HM_27.png");
    cv::Mat qp32 = cv::imread(getProjectDirectory(OS) + "/img/minato/HM_32.png");
    cv::Mat qp37 = cv::imread(getProjectDirectory(OS) + "/img/minato/HM_37.png");

    cv::Mat target = cv::imread(getProjectDirectory(OS) + "/img/minato/minato_000413_limit.bmp");

    std::cout << "QP22:" << getPSNR(target, qp22) << std::endl;
    std::cout << "QP27:" << getPSNR(target, qp27) << std::endl;
    std::cout << "QP32:" << getPSNR(target, qp32) << std::endl;
    std::cout << "QP37:" << getPSNR(target, qp37) << std::endl;
}

std::vector<std::string> split(std::string str, char del) {
    int first = 0;
    int last = str.find_first_of(del);

    std::vector<std::string> result;

    while (first < str.size()) {
        std::string subStr(str, first, last - first);

        result.push_back(subStr);

        first = last + 1;
        last = str.find_first_of(del, first);

        if (last == std::string::npos) {
            last = str.size();
        }
    }

    return result;
}

void getDiff_vector(){

    const std::string img_directory = getProjectDirectory(OS) + "\\img\\minato\\";

    //ファイルへのpath
    std::string quarter_mv_name = img_directory + "quarter.txt";
    std::string gauss_mv_name = img_directory + "mv_list.txt";

    std::ifstream quarter_mv;
    std::ifstream gauss_mv;

    quarter_mv.open(quarter_mv_name, std::ios::in);
    gauss_mv.open(gauss_mv_name, std::ios::in);

    //差分をファイルに格納
    std::ofstream diff_list;
    std::string diff_list_name = img_directory + "diff_list.txt";
    diff_list.open(diff_list_name, std::ios::out);

    if(quarter_mv.fail()){
        std::cerr << "Failed to open quarter.txt" <<std::endl;
    }
    if(gauss_mv.fail()){
        std::cerr << "Failed to open gauss_mv.txt" <<std::endl;
    }

    char del = ' ';
    int n, diff = 0, diff_y[200], diff_x[200];
    std::string str1, str2;
    std::string a[20], b[3];
    cv::Point2f q, g;

    for(int i = 0 ; i < 200 ; i++){
        diff_x[i] = diff_y[i] = 0;
    }

    for(int i = 0 ; i < 61440 ; i++){
        std::getline(gauss_mv, str1);
        std::getline(quarter_mv, str2);

        n = 0;
        for (const auto substr : split(str1, del)) {
            a[n] = substr;
            n++;
        }
        n = 0;
        for (const auto substr : split(str2, del)) {
            b[n] = substr;
            n++;
        }
        g.x = stod(a[2]); g.y = stod(a[3]);
        q.x = stod(b[1]); q.y = stod(b[2]);

        if(diff < g.x) diff = g.x;
    }
}

void draw_HEVC_MergeMode(std::string p_image_name, std::string result_txt_name, int block_num) {

    const std::string img_directory = getProjectDirectory(OS) + "\\img\\minato\\";

    std::string p_image_path = img_directory + p_image_name;
    std::string result_path  = img_directory + result_txt_name;

    //x,   y,   x,   y,  merge
    int block_info[block_num][5];
    std::ifstream result;
    cv::Mat p_image;

    result.open(result_path, std::ios::in);
    p_image = cv::imread(p_image_path);

    if(result.fail()){
        std::cerr << "Failed to open result.txt" <<std::endl;
        exit(1);
    }

    char del = ' ';
    int n;
    std::string info;

    for(int i = 0 ; i < block_num ; i++) {
        std::getline(result, info);

        n = 0;
        for (const auto subinfo : split(info, del)) {
            block_info[i][n] = stoi(subinfo);
            n++;
        }
        std::cout << "x : " << block_info[i][0] << ", y : " << block_info[i][1] <<
                   ", x : " << block_info[i][2] << ", y : " << block_info[i][3] << ", merge : " << block_info[i][4] << std::endl;
    }


}

void draw_mv(){

    const std::string img_directory = getProjectDirectory(OS) + "\\img\\minato\\";

    //ファイルへのpath
    std::string quarter_mv_name = img_directory + "quarter.txt";
    std::string gauss_mv_name = img_directory + "gauss.txt";
    std::string fullquarter_mv_name = img_directory + "fullquarter.txt";
    std::string mc_image_name = img_directory + "block_matching_quarter_gauss_MC.bmp";

    std::ifstream quarter_mv;
    std::ifstream gauss_mv;
    std::ifstream fullquarter_mv;

    quarter_mv.open(quarter_mv_name, std::ios::in);
    gauss_mv.open(gauss_mv_name, std::ios::in);
    fullquarter_mv.open(fullquarter_mv_name, std::ios::in);

    if(quarter_mv.fail()){
        std::cerr << "Failed to open quarter.txt" <<std::endl;
    }
    if(gauss_mv.fail()){
        std::cerr << "Failed to open gauss_mv.txt" <<std::endl;
    }
    if(fullquarter_mv.fail()){
        std::cerr << "Failed to open fullquarter.txt" <<std::endl;
    }

    //MVを書く画像の読み込み
    cv::Mat mc_image;
    mc_image = cv::imread(mc_image_name);

    char del = ' ';
    std::string str1, str2, str3;
    std::string a[3], b[3], c[3];
    int n;
    int nw, nh;
    int tsize = 8;
    cv::Point2f fq1, fq2, q1, q2, g1, g2, cog1, cog2;
    nw = 1920 / tsize;  nh = 1024 / tsize;

    //描画するブロックが8*8になるようにしている
    //1ブロックおきのブロックを描画
    for(int i = 1 ; i < nw - 1 ; i += 2){
        q1.x = tsize * i ; q1.y = 0;
        q2.x = q1.x; q2.y = 1023;
        g1.x = q1.x + 7; g1.y = 0;
        g2.x = g1.x; g2.y = 1023;
        cv::line(mc_image, q1, q2, CV_RGB(255, 255, 255));
        cv::line(mc_image, g1, g2, CV_RGB(255, 255, 255));
    }
    for(int j = 1 ; j < nh - 1 ; j += 2){
        q1.x = 0; q1.y = tsize * j - 1;
        q2.x = 1919; q2.y = q1.y;
        g1.x = 0; g1.y = q1.y + 9;
        g2.x = 1919; g2.y = g1.y;
        cv::line(mc_image, q1, q2, CV_RGB(255, 255, 255));
        cv::line(mc_image, g1, g2, CV_RGB(255, 255, 255));
    }

    //8*8のブロックごとにまわす。
    for(int j = 0 ; j < nh ; j ++){
        for(int i = 0 ; i < nw; i ++) {
            //三角形2このMVを読み込む
            std::getline(fullquarter_mv, str1);
            std::getline(gauss_mv, str2);
            std::getline(quarter_mv, str3);
            n = 0;
            for (const auto substr : split(str1, del)) {
                a[n] = substr;
                n++;
            }
            n = 0;
            for (const auto substr : split(str2, del)) {
                b[n] = substr;
                n++;
            }
            n = 0;
            for (const auto substr : split(str3, del)) {
                c[n] = substr;
                n++;
            }
            fq1.x = stod(a[1]); fq1.y = stod(a[2]);
            g1.x  = stod(b[1]); g1.y  = stod(b[2]);
            q1.x  = stod(c[1]); q1.y  = stod(c[2]);
            std::getline(fullquarter_mv, str1);
            std::getline(gauss_mv, str2);
            std::getline(quarter_mv, str3);
            n = 0;
            for (const auto substr : split(str1, del)) {
                a[n] = substr;
                n++;
            }
            n = 0;
            for (const auto substr : split(str2, del)) {
                b[n] = substr;
                n++;
            }
            n = 0;
            for (const auto substr : split(str3, del)) {
                c[n] = substr;
                n++;
            }
            fq2.x = stod(a[1]); fq2.y = stod(a[2]);
            g2.x  = stod(b[1]); g2.y  = stod(b[2]);
            q2.x  = stod(c[1]); q2.y  = stod(c[2]);

            if(i % 2 == 0 && j % 2 == 1) {
                //MVの描画
                //それぞれの三角形の重心点を計算。
                cog1.x = tsize * i + 2;
                cog1.y = tsize * j + 2;
                cog2.x = tsize * i + 5;
                cog2.y = tsize * j + 5;
                //1つめの三角形へ描画
                cv::line(mc_image, cog1, cog1 + fq1, CV_RGB(255, 0, 0));
                cv::line(mc_image, cog1, cog1 + q1, CV_RGB(0, 0, 255));
                cv::line(mc_image, cog1, cog1 + g1, CV_RGB(0, 255, 0));
                //2つめの三角形へ描画
                cv::line(mc_image, cog2, cog2 + fq2, CV_RGB(255, 0, 0));
                cv::line(mc_image, cog2, cog2 + q2, CV_RGB(0, 0, 255));
                cv::line(mc_image, cog2, cog2 + g2, CV_RGB(0, 255, 0));
            }
        }
    }
    cv::imwrite(img_directory + "full_pel_gauss_mv.bmp", mc_image);
}

void getDiff_image(){

    const std::string img_directory = getProjectDirectory(OS) + "\\img\\minato\\";

    std::cout << "img_directory:" << img_directory << std::endl;

    //path
    std::string image1_path = img_directory + "minato_mirai_000413.png";
    std::string image2_path = img_directory + "HM_minato_mirai_nointra_27.png";

    cv::Mat image1;
    cv::Mat image2;
//    cv::Mat image2_tmp = cv::Mat::zeros(image1.size(), CV_8UC3);
    //read
    image1 = cv::imread(image1_path);
    image2 = cv::imread(image2_path);

    cv::Mat diff = cv::Mat::zeros(image1.size(),CV_8UC3);

    //差分
    for(int j = 0 ;j < image1.rows ;j++){
        for(int i = 0 ;i < image1.cols ;i++){
            int y =  abs(R(image1,i,j) - R(image2,i,j));
            if(y < 0) y = 0;
            else if(y > 255) y = 255;
//            int r = ((R(image1,i,j) + R(image2,i,j)) / 2 + 0.5);
//            int g = ((G(image1,i,j) + G(image2,i,j)) / 2 + 0.5);
//            int b = ((B(image1,i,j) + B(image2,i,j)) / 2 + 0.5);
            R(diff,i,j) = (unsigned char)y;
            G(diff,i,j) = (unsigned char)y;
            B(diff,i,j) = (unsigned char)y;
        }
    }
    std::cout << getPSNR(image1, image2) << std::endl;
//    cv::imwrite(img_directory + "diff.bmp", diff);

}



unsigned char filter(cv::Mat target_image, int x, int y){
    double tmp = 0;

    if(0 <= y - 1 && y + 1 < target_image.rows && 0 <= x - 1 && x + 1 < target_image.cols) {
        tmp = (0.0625 * M(target_image, x - 1, y - 1) + 0.125 * M(target_image, x, y - 1) + 0.0625 * M(target_image, x + 1, y - 1)
               + 0.125 * M(target_image, x - 1, y) + 0.25 * M(target_image, x, y) + 0.125 * M(target_image, x + 1, y)
               + 0.0625 * M(target_image, x - 1, y + 1) + 0.125 * M(target_image, x, y + 1) + 0.0625 * M(target_image, x + 1, y + 1));
    }
    else{
        tmp = M(target_image, x, y);
    }

    if(tmp > 255){
        return 255;
    }
    else if(tmp < 0){
        return 0;
    }
    else return (unsigned char)(tmp + 0.5);
}

unsigned char LPF_5_5(cv::Mat target_image, int x, int y)
{
//    static double h[][5] = {{0.009116, 0.029507, 0.030166, 0.029507, 0.009116},
//                            {0.029507, 0.095510, 0.097643, 0.095510, 0.029507},
//                            {0.030166, 0.097643, 0.099824, 0.097643, 0.030166},
//                            {0.029507, 0.095510, 0.097643, 0.095510, 0.029507},
//                            {0.009116, 0.029507, 0.030166, 0.029507, 0.009116}};
//    static int first = 1;
    double tmp;

//    if (first) {
//        tmp = 0.0;
//        for(int j = -2 ; j <= 2 ; j++) {
//            for (int i = -2; i <= 2; i++) {
//                tmp += h[i + 2][j + 2];
//            }
//        }
//        for(int j = -2 ; j <= 2 ; j++) {
//            for (int i = -2; i <= 2; i++) {
//                h[i + 2][j + 2] /= tmp;
//            }
//        }
//        first = 0;
//    }
    static double h[][5] = {{0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625},
                            {0.015625  , 0.0625  , 0.09375  , 0.0625  , 0.015625  },
                            {0.0234375 , 0.09375 , 0.140625 , 0.09375 , 0.0234375 },
                            {0.015625  , 0.0625  , 0.09375  , 0.0625  , 0.015625  },
                            {0.00390625, 0.015625, 0.0234375, 0.015625, 0.00390625}};
    tmp = 0.0;
    if(0 <= y - 2 && y + 2 < target_image.rows && 0 <= x - 2 && x + 2 < target_image.cols) {
        for(int j = -2 ; j <= 2 ; j++){
            for(int i = -2 ; i <= 2 ; i++){
                tmp += h[i + 2][j + 2] * M(target_image, x - i, y - j);
            }
        }
    }
    else{
        tmp = M(target_image, x, y);
    }

//    std::cout << tmp << std::endl;
//    exit(0);
    if(tmp > 255){
        return 255;
    }
    else if(tmp < 0){
        return 0;
    }
    else return (unsigned char)(tmp + 0.5);
}

unsigned char limit(unsigned char tmp)
{
    if(tmp > 255){
        return 255;
    }
    else if(tmp < 0){
        return 0;
    }
    else return (unsigned char)(tmp + 0.5);
}

void filterTest(){

    const std::string img_directory = getProjectDirectory(OS) + "\\img\\minato\\";

    std::cout << "img_directory:" << img_directory << std::endl;

    //path
    std::string target_image_path = img_directory + "minato_limit_2_I22.bmp";

    cv::Mat target_image;
    //read
    target_image = cv::imread(target_image_path);

    int scaled_expand_size = 4;

    cv::Mat expansion_image;
//    unsigned int **expansion_image;
    unsigned char **scaled_target_image;
    unsigned char **scaled_half_target_image;
    cv::Mat scaled_target_image_mat = cv::Mat::zeros(target_image.rows / 4, target_image.cols / 4, CV_8UC3);   //DCTFilterにわたすやつ
    cv::Mat scaled_half_target_image_mat = cv::Mat::zeros(target_image.rows / 2, target_image.cols / 2, CV_8UC3);
    //LPR_5_5を通すための拡張mat
//    int expand = 4;
//    cv::Mat target_expand_image = cv::Mat::zeros(target_image.rows + expand * 2, target_image.cols + expand * 2, CV_8UC3); //1/2縮小に使う画像
//    cv::Mat half_target_expand_image = cv::Mat::zeros(target_image.rows / 2 + expand / 2 * 2, target_image.cols / 2 + expand / 2 * 2, CV_8UC3); //さらに1/2するのに使う画像

    //malloc
    if((scaled_target_image = (unsigned char **)malloc(sizeof(unsigned char *) * (target_image.cols / 4 + 2 * scaled_expand_size))) == NULL) {
        fprintf(stderr, "malloc error");
        exit(1);
    }
    scaled_target_image += scaled_expand_size;
    if((scaled_half_target_image = (unsigned char **)malloc(sizeof(unsigned char *) * target_image.cols / 2)) == NULL) {
        fprintf(stderr, "malloc error");
        exit(1);
    }
    for(int x = -scaled_expand_size ; x < target_image.cols / 4 + scaled_expand_size ; x++) {
        scaled_target_image[x] = (unsigned char *)malloc(sizeof(unsigned char) * (target_image.rows / 4 + 2 * scaled_expand_size));
        scaled_target_image[x] += scaled_expand_size;
    }
    for(int x = 0 ; x < target_image.cols / 2 ; x++) {
        scaled_half_target_image[x] = (unsigned char *)malloc(sizeof(unsigned char) * (target_image.rows / 2));
    }

//    //filterに通すために拡張
//    for(int y = 0 ; y < target_image.rows ; y++) {
//        for (int x = 0; x < target_image.cols; x++) {
//            R(target_expand_image, x + expand, y + expand) = M(target_image, x, y);
//            G(target_expand_image, x + expand, y + expand) = R(target_expand_image, x + expand, y + expand);
//            B(target_expand_image, x + expand, y + expand) = R(target_expand_image, x + expand, y + expand);
//        }
//    }
//    //左右拡張
//    for(int y = 0 ; y < target_image.rows ; y++){
//        for(int x = -expand ; x < 0 ; x++) {
//            R(target_expand_image, x + expand, y + expand) = R(target_expand_image, expand, y + expand);
//            G(target_expand_image, x + expand, y + expand) = R(target_expand_image, x + expand, y + expand);
//            B(target_expand_image, x + expand, y + expand) = R(target_expand_image, x + expand, y + expand);
//            R(target_expand_image, target_image.cols + 2 * expand + x, y + expand) = R(target_expand_image, target_image.cols - 1 + expand, y + expand);
//            G(target_expand_image, target_image.cols + 2 * expand + x, y + expand) = R(target_expand_image, target_image.cols + 2 * expand + x, y + expand);
//            B(target_expand_image, target_image.cols + 2 * expand + x, y + expand) = R(target_expand_image, target_image.cols + 2 * expand + x, y + expand);
//        }
//    }
//    //上下拡張
//    for (int y = -expand ; y < 0 ; y++) {
//        for (int x = -expand ; x < target_image.cols + expand; x++) {
//            R(target_expand_image, x + expand, y + expand) = R(target_expand_image, x + expand, expand);
//            G(target_expand_image, x + expand, y + expand) = R(target_expand_image, x + expand, y + expand);
//            B(target_expand_image, x + expand, y + expand) = R(target_expand_image, x + expand, y + expand);
//            R(target_expand_image, x + expand , target_image.rows + 2 * expand + y) = R(target_expand_image, x + expand , target_image.rows - 1 + expand);
//            G(target_expand_image, x + expand , target_image.rows + 2 * expand + y) = R(target_expand_image, x + expand , target_image.rows + 2 * expand + y);
//            B(target_expand_image, x + expand , target_image.rows + 2 * expand + y) = R(target_expand_image, x + expand , target_image.rows + 2 * expand + y);
//        }
//    }
//    cv::imwrite(img_directory + "target_expand_image.bmp", target_expand_image);

//    //1/2に縮小
//    for(int y = 1024 ; y < target_expand_image.rows - expand / 2 ; y += 2) {
//        for (int x = 1924; x < target_expand_image.cols - expand / 2 ; x += 2) {
////            R(half_target_expand_image, x / 2, y / 2) = R(target_expand_image, x + expand / 2, y + expand / 2);
//            R(half_target_expand_image, x / 2, y / 2) = LPF_5_5(target_expand_image, x + expand / 2, y + expand / 2);
////            G(half_target_expand_image, x / 2, y / 2) = R(half_target_expand_image, x / 2, y / 2);
////            B(half_target_expand_image, x / 2, y / 2) = R(half_target_expand_image, x / 2, y / 2);
//        }
//    }
//    cv::imwrite(img_directory + "Half_scaled_target_expand_image.bmp", half_target_expand_image);
//    exit(0);

//    //さらに1/2に縮小(1/4に縮小)
//    for(int y = 0 ; y < half_target_expand_image.rows ; y += 2){
//        for(int x = 0 ; x < half_target_expand_image.cols ; x += 2){
//            scaled_target_image[x / 2][y / 2] = LPF_5_5(half_target_expand_image, x + 2, y + 2);
//            R(scaled_target_image_mat, x / 2, y / 2) = scaled_target_image[x / 2][y / 2];
//            G(scaled_target_image_mat, x / 2, y / 2) = scaled_target_image[x / 2][y / 2];
//            B(scaled_target_image_mat, x / 2, y / 2) = scaled_target_image[x / 2][y / 2];
//        }
//    }
//    cv::imwrite(img_directory + "quarter_scaled_target_image.bmp", scaled_target_image_mat);
    //1/2に縮小
    for(int y = 0 ; y < target_image.rows ; y += 2){
        for(int x = 0 ; x < target_image.cols ; x += 2){
            scaled_half_target_image[x / 2][y / 2] = LPF_5_5(target_image, x, y);
            R(scaled_half_target_image_mat, x / 2, y / 2) = scaled_half_target_image[x / 2][y / 2];
            G(scaled_half_target_image_mat, x / 2, y / 2) = scaled_half_target_image[x / 2][y / 2];
            B(scaled_half_target_image_mat, x / 2, y / 2) = scaled_half_target_image[x / 2][y / 2];
        }
    }
    cv::imwrite(img_directory + "Half_scaled_target_image.bmp", scaled_half_target_image_mat);
    //さらに1/2に縮小(1/4に縮小)
    for(int y = 0 ; y < scaled_half_target_image_mat.rows ; y += 2){
        for(int x = 0 ; x < scaled_half_target_image_mat.cols ; x += 2){
            scaled_target_image[x / 2][y / 2] = LPF_5_5(scaled_half_target_image_mat, x, y);
            R(scaled_target_image_mat, x / 2, y / 2) = scaled_target_image[x / 2][y / 2];
            G(scaled_target_image_mat, x / 2, y / 2) = scaled_target_image[x / 2][y / 2];
            B(scaled_target_image_mat, x / 2, y / 2) = scaled_target_image[x / 2][y / 2];
        }
    }
    cv::imwrite(img_directory + "quarter_scaled_target_image.bmp", scaled_target_image_mat);

    //img_ipに渡す画像を左右拡張
    for(int y = 0 ; y < target_image.rows / 4 ; y++){
        for(int x = -scaled_expand_size ; x < 0 ; x++) {
            scaled_target_image[x][y] = scaled_target_image[0][y];
            scaled_target_image[target_image.cols / 4 + scaled_expand_size + x][y] = scaled_target_image[target_image.cols / 4 -1][y];
        }
    }
    //img_ipに渡す画像を上下拡張
    for (int y = -scaled_expand_size ; y < 0 ; y++) {
        for (int x = -scaled_expand_size ; x < target_image.cols / 4 + scaled_expand_size; x++) {
            scaled_target_image[x][y] = scaled_target_image[x][0];
            scaled_target_image[x][target_image.rows / 4 + scaled_expand_size  + y] = scaled_target_image[x][target_image.rows / 4 - 1];
        }
    }

    int k = 4;
    int expand = 0;
    cv::Mat bilinear_image = cv::Mat::zeros(target_image.rows, target_image.cols, CV_8UC3);
    cv::Mat bicubic_image = cv::Mat::zeros(target_image.rows, target_image.cols, CV_8UC3);
    cv::Mat dctfilter_image = cv::Mat::zeros(target_image.rows + 2 * k * expand, target_image.cols + 2 * k * expand, CV_8UC3);
    //補間
//    expansion_image = getDCTFilterUnIntImage(scaled_target_image_mat, expand);
    expansion_image = getExpansionMatHEVCImage(scaled_target_image_mat, k, expand);

    for(int y = 0 ; y < target_image.rows ; y++) {
        for (int x = 0; x < target_image.cols; x++) {
            R(bilinear_image, x, y) = img_ip(scaled_target_image, cv::Rect(-scaled_expand_size, -scaled_expand_size, target_image.cols / 4 + 2 * scaled_expand_size, target_image.rows / 4 + 2 * scaled_expand_size), x / (double)4, y / (double)4, 1);
            G(bilinear_image, x, y) = R(bilinear_image, x, y);
            B(bilinear_image, x, y) = R(bilinear_image, x, y);
            R(bicubic_image, x, y) = img_ip(scaled_target_image, cv::Rect(-scaled_expand_size, -scaled_expand_size, target_image.cols / 4 + 2 * scaled_expand_size, target_image.rows / 4 + 2 * scaled_expand_size), x / (double)4, y / (double)4, 2);
            G(bicubic_image, x, y) = R(bicubic_image, x, y);
            B(bicubic_image, x, y) = R(bicubic_image, x, y);
//            R(dctfilter_image, x, y) = limit(expansion_image[x][y]);
//            G(dctfilter_image, x, y) = limit(expansion_image[x][y]);
//            B(dctfilter_image, x, y) = limit(expansion_image[x][y]);
//            R(dctfilter_image, x, y) = limit(R(expansion_image, x, y));
//            G(dctfilter_image, x, y) = limit(R(expansion_image, x, y));
//            B(dctfilter_image, x, y) = limit(R(expansion_image, x, y));
        }
    }
    for(int y = 0 ; y < target_image.rows + 2 * k * expand ; y++) {
        for (int x = 0 ; x < target_image.cols + 2 * k * expand ; x++) {
//            R(dctfilter_image, x, y) = limit(expansion_image[x - k * expand][y - k * expand]);
//            G(dctfilter_image, x, y) = limit(expansion_image[x - k * expand][y - k * expand]);
//            B(dctfilter_image, x, y) = limit(expansion_image[x - k * expand][y - k * expand]);
            R(dctfilter_image, x, y) = limit(R(expansion_image, x, y));
            G(dctfilter_image, x, y) = limit(R(expansion_image, x, y));
            B(dctfilter_image, x, y) = limit(R(expansion_image, x, y));
        }
    }

    for(int y = 0 ; y < scaled_target_image_mat.rows ; y++) {
        for (int x = 0; x < scaled_target_image_mat.cols; x++) {
            if(R(scaled_target_image_mat, x, y) != R(dctfilter_image, 4 * x, 4 * y)){
                std::cout << std::to_string(4 * x) << ", " << std::to_string(4 * y) << "is not equal" << std::endl;
                std::cout << "target_image    : " << std::to_string(R(scaled_target_image_mat, x, y)) << std::endl;
                std::cout << "dctfilter_image : " << std::to_string(R(dctfilter_image, 4 * x, 4 * y)) << std::endl;
            }
//            std::cout << "target_image    : " << std::to_string(R(scaled_target_image_mat, x, y)) << std::endl;
//            std::cout << "dctfilter_image : " << std::to_string(R(dctfilter_image, 4 * x, 4 * y)) << std::endl;
        }
    }

    std::cout << "bilinear   PSNR:" << getPSNR(target_image, bilinear_image) << std::endl;
    std::cout << "bicubic    PSNR:" << getPSNR(target_image, bicubic_image) << std::endl;
    std::cout << "dctfilter PSNR:" << getPSNR(target_image, dctfilter_image) << std::endl;
    cv::imwrite(img_directory + "quarter_scaled_target_bilinear_image.bmp", bilinear_image);
    cv::imwrite(img_directory + "quarter_scaled_target_bicubic_image.bmp", bicubic_image);
    cv::imwrite(img_directory + "quarter_scaled_target_dctfilter_image.bmp", dctfilter_image);
}

void test_config_file() {
    std::vector<Config> tasks = readTasks();

    for(auto& task : tasks){
        std::cout << task.getImgDirectory() << std::endl;
        std::cout << task.getGaussRefImage() << std::endl;
        std::cout << task.getRefImage() << std::endl;
        std::cout << task.getTargetImage() << std::endl;

        std::cout << task.getQp() << std::endl;
        std::cout << task.getCtuWidth() << std::endl;
        std::cout << task.getCtuHeight() << std::endl;
        std::cout << task.getDivisionStep() << std::endl;
    }

}

/**
 *
 */
void testPredMv() {

    std::vector<cv::Point2f> ref_triangle_coordinate;
    std::vector<cv::Point2f> target_triangle_coordinate;
    std::vector<cv::Point2f> ref_mvs;

    ref_triangle_coordinate.emplace_back(127, 0);
    ref_triangle_coordinate.emplace_back(0, 127);
    ref_triangle_coordinate.emplace_back(127, 127);

    target_triangle_coordinate.emplace_back(128, 0);
    target_triangle_coordinate.emplace_back(255, 0);
    target_triangle_coordinate.emplace_back(128, 127);

    ref_mvs.emplace_back(-10.75, 3.5);
    ref_mvs.emplace_back(-11, 4);
    ref_mvs.emplace_back(-10.5, 4.25);

    std::vector<cv::Point2f> mvs = getPredictedWarpingMv(ref_triangle_coordinate, ref_mvs, target_triangle_coordinate);

    std::cout << "Warping test ==================================" << std::endl;
    for(const auto& mv : mvs){
        std::cout << mv << std::endl;
    }

    std::cout << "Parallel test =================================" << std::endl;
    target_triangle_coordinate.clear();
    cv::Point2f center_of_gravity((128.0 + 255.0 + 128.0) / 3.0, (127.0) / 3.0);
    target_triangle_coordinate.emplace_back(center_of_gravity);

    mvs = getPredictedWarpingMv(ref_triangle_coordinate, ref_mvs, target_triangle_coordinate);
    for(const auto& mv : mvs){
        std::cout << mv << std::endl;
    }

}
