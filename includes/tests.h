//
// Created by kasph on 2019/06/16.
//

#ifndef ENCODER_TESTS_H
#define ENCODER_TESTS_H

//
// Created by kasph on 2019/06/16.
//

#include <opencv2/core.hpp>
#include <opencv/cv.hpp>
#include <iostream>

#include "../includes/tests.h"
#include "../includes/Utils.h"
#include "../includes/ImageUtil.h"
#include "../includes/ME.hpp"
#include "../includes/TriangleDivision.h"

void storeResidualImage();
void test2();
void test3();
void test4();
void test5();
void test6();
void test7();
void testFilter();
void test4xHEVCImage();
void testRoundVecQuarter();
void testHevcPSNR();

std::vector<std::string> split(std::string str, char del);
void getDiff_vector();
void draw_HEVC_MergeMode(std::string p_image_name, std::string result_txt_name, int block_num);
void draw_mv();
void getDiff_image();
void filterTest();
void test();
void test_config_file();

void testPredMv();
#endif //ENCODER_TESTS_H
