//
// Created by kasph on 2019/06/16.
//

#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <iostream>
#include <random>

#include "../includes/tests.h"
#include "../includes/Utils.h"
#include "../includes/ImageUtil.h"
#include "../includes/ME.hpp"
#include "../includes/TriangleDivision.h"
#include "../includes/Reconstruction.h"
#include "../includes/psnr.h"

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
        foo[i]->leftNode = foo[i]->rightNode = nullptr;
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
    unsigned int **img1 = getExpansionHEVCImage(ref_image, 4, 16);


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
    unsigned int **hevc_ip = getExpansionHEVCImage(out, 4, 0);

    cv::Mat hevc = cv::Mat::zeros(ref_image.size(), CV_8UC3);
    for(int y = 0 ; y < hevc.rows ; y++) {
        for (int x = 0; x < hevc.cols; x++) {
            if(x % 4 == 0 && y % 4 == 0){
                R(hevc, x, y) = hevc_ip[x][y];
                G(hevc, x, y) = hevc_ip[x][y];
                B(hevc, x, y) = hevc_ip[x][y];
            }else{
                int val = (hevc_ip[x][y] + 32)  / 64;
                val = (val > 255 ? 255 : (val < 0 ? 0 : val));
                R(hevc, x, y) = val;
                G(hevc, x, y) = val;
                B(hevc, x, y) = val;
            }
        }
    }

    std::cout << "Bilinear:" << getPSNR(bilinear, ref_image) << std::endl;
    std::cout << "Bicubic :" << getPSNR(bicubic, ref_image) << std::endl;
    std::cout << "HEVC    :" << getPSNR(hevc, ref_image) << std::endl;
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/resize_bilinear.png", bilinear);
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/resize_bicubic.png", bicubic);
    cv::imwrite(getProjectDirectory(OS) + "/img/minato/resize_hevcpng.png", hevc);
}