#include <utility>

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
#include "../includes/Analyzer.h"
#include "../includes/Encode.h"
#include "../includes/ME.hpp"
#include "../includes/Vector.hpp"
#include "../includes/psnr.h"
#include "../includes/TriangleDivision.h"
#include "../includes/Reconstruction.h"
#include "../includes/ImageUtil.h"
#include "../includes/Utils.h"
#include "../includes/tests.h"
#include "../includes/Decoder.h"
#include "../includes/ConfigUtil.h"
#include "../includes/MELog.h"

void run(std::string config_name);
void tests();

#define HARRIS false
#define THRESHOLD true
#define LAMBDA 0.2
#define INTER_DIV true // 頂点追加するかしないか

#define DIVIDE_MODE RIGHT_DIVIDE

int qp;
int qp_offset;
int block_size_x;
int block_size_y;
int division_steps;
double injected_lambda;
bool lambda_inject_flag;

std::string out_file_suffix = "_rd_";

std::vector<std::vector<double>> freq_newton_warping, freq_newton_translation;
std::vector<std::vector<std::vector<cv::Point2f>>> mv_newton_translation, coordinate_newton_translation1, coordinate_newton_translation2, coordinate_newton_translation3;
std::vector<std::vector<std::vector<cv::Point2f>>> coordinate_newton_warping1, coordinate_newton_warping2, coordinate_newton_warping3;
std::vector<std::vector<std::vector<std::vector<cv::Point2f>>>> mv_newton_warping;
std::vector<std::vector<std::vector<double>>> slow_newton_warping, slow_newton_translation;

std::vector<MELog> ME_log_translation_0;
std::vector<MELog> ME_log_translation_1;
std::vector<MELog> ME_log_warping_0;
std::vector<MELog> ME_log_warping_1;

std::vector<int> pells;
std::vector<double> residuals;

std::vector<int> divide_flags;

std::vector<CodingTreeUnit *> ctus;

void storeNewtonLogs(std::string logDirectoryPath);

int main(int argc, char *argv[]){
    // Write test codes below

#if TEST_MODE
    tests();
#else

    std::string config_name;
    if(argc == 1) {
        config_name = "config.json";
    }else{
        config_name = std::string(argv[1]);
    }

    run(config_name);
#endif

}

void tests(){
    test_config_file();
    testPredMv();
}

void run(std::string config_name) {

    std::cout << "OpenCV_version : " << getVersionOfOpenCV() << std::endl;

    const std::string project_directory_path = getProjectDirectory(OS);

    std::vector<cv::Point2f> corners, corners_org;
    std::vector<cv::Point2f> ref_corners, ref_corners_org;

    std::string config_file_name = config_name.substr(config_name.rfind("/")+ 1, config_name.size());
    config_file_name = config_file_name.substr(0, config_file_name.rfind("."));

    // 各タスクの情報が入ったvector
    std::vector<Config> tasks = readTasks(std::move(config_name));

    std::ofstream ofs;

    ofs.open(getProjectDirectory(OS) + tasks[0].getLogDirectory() + "/" + "code_amount" + out_file_suffix + config_file_name + "_" + getCurrentTimestamp() + ".csv");
    
    std::map<int, std::vector<std::vector<cv::Mat>>> ref_images_with_qp, target_images_with_qp;
    std::map<int, EXPAND_ARRAY_TYPE> expand_images_with_qp;

    int previous_qp = -1;

    // 全画像分ループ
    for(const auto& task : tasks){

        if(!task.isEnable()) continue;

        std::string img_path                    = ((OS == "Win") ? replaceBackslash(task.getImgDirectory()) : std::string(task.getImgDirectory()));
        std::string img_directory               = project_directory_path + img_path;
        std::string log_directory               = project_directory_path + task.getLogDirectory();
        const std::string& target_file_name     = task.getTargetImage();

        const std::string& ref_file_name        = task.getGaussRefImage();
        const std::string& ref_intra_file_name  = task.getRefImage();

        std::string ref_file_path               = project_directory_path + img_path + ref_file_name;
        std::string target_file_path            = project_directory_path + img_path + target_file_name;
        std::string ref_intra_file_path         = project_directory_path + img_path + ref_intra_file_name;

        block_size_x                            = task.getCtuWidth();
        block_size_y                            = task.getCtuHeight();
        qp                                      = task.getQp();
        qp_offset                               = task.getQpOffset();
        division_steps                          = task.getDivisionStep();

        lambda_inject_flag                      = task.isLambdaEnable();
        injected_lambda                         = task.getLambda() * task.getLambda();

        std::cout << "img_directory          : " << img_directory << std::endl;
        std::cout << "log_directory          : " << log_directory << std::endl;
        std::cout << "target_file_name       : " << target_file_name << std::endl;
        std::cout << "ref_file_name          : " << ref_file_name << std::endl;
        std::cout << "ref_file_path          : " << ref_file_path << std::endl;
        std::cout << "target_image file path : " << target_file_path << std::endl;
        std::cout << "ref_intra_file_path    : " << ref_intra_file_path << std::endl;
        std::cout << "ref_gauss file path    : " << ref_file_path << std::endl;
        std::cout << "QP                     : " << qp << std::endl;
        std::cout << "QP + offset            : " << qp + qp_offset << std::endl;
        std::cout << "CTU_WIDTH              : " << block_size_x << std::endl;
        std::cout << "CTU_HEIGHT             : " << block_size_y << std::endl;
        std::cout << "lambda_inject_flag     : " << lambda_inject_flag << std::endl;
        std::cout << "injected lambda        : " << injected_lambda << std::endl;

        if(previous_qp == -1) previous_qp = qp;

        // オフセットを足して計測する
        qp = qp + qp_offset;

        out_file_suffix = "_lambda_" + std::to_string(getLambdaPred(qp)) + "_";

        // 時間計測
        clock_t start = clock();

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

        cv::Mat r_ref = cv::Mat::zeros(target_image.rows, target_image.cols, CV_8UC1);

        std::vector<std::vector<cv::Mat>> ref_images, target_images;

        if(ref_images_with_qp.count(qp) == 0) {
            ref_images = getRefImages(ref_image, gaussRefImage);
            ref_images_with_qp[qp] = ref_images;
        }else{
            ref_images = ref_images_with_qp[qp];
        }

        if(target_images_with_qp.count(qp) == 0) {
            target_images = getTargetImages(target_image);
            target_images_with_qp[qp] = target_images;
        }else{
            target_images = target_images_with_qp[qp];
        }

        std::vector<std::vector<std::vector<unsigned char *>>> expand_images;
        int expand = SEARCH_RANGE;
        if(expand_images_with_qp.count(qp) == 0) {
            expand_images = getExpandImages(ref_images, target_images, expand);
            expand_images_with_qp[qp] = expand_images;
        }else{
            expand_images = expand_images_with_qp[qp];
        }

        if(qp != previous_qp){
            std::cout << "--------------------- free ---------------------" << std::endl;

            freeHEVCExpandImage(expand_images_with_qp[previous_qp], 22, 2, 4, 1920, 1024);
            expand_images_with_qp.erase(previous_qp);
            for(int i = 0 ; i < target_images_with_qp[previous_qp].size() ; i++){
                for(int j = 0 ; j < target_images_with_qp[previous_qp][i].size() ; j++){
                    target_images_with_qp[previous_qp][i][j].release();
                }
            }
            target_images_with_qp.erase(previous_qp);
            for(int i = 0 ; i < ref_images_with_qp[previous_qp].size() ; i++){
                for(int j = 0 ; j < ref_images_with_qp[previous_qp][i].size() ; j++){
                    ref_images_with_qp[previous_qp][i][j].release();
                }
            }
            ref_images_with_qp.erase(previous_qp);

        }

        triangle_division.initTriangle(block_size_x, block_size_y, division_steps, qp, expand_images, DIVIDE_MODE);
        std::vector<Point3Vec> triangles = triangle_division.getTriangleCoordinateList();

        std::vector<std::pair<Point3Vec, int> > init_triangles = triangle_division.getTriangles();

        ctus.resize(init_triangles.size());
        for (int i = 0; i < init_triangles.size(); i++) {
            ctus[i] = new CodingTreeUnit();
            ctus[i]->split_cu_flag = false;
            ctus[i]->node1 = ctus[i]->node2 = ctus[i]->node3 = ctus[i]->node4 = nullptr;
            ctus[i]->triangle_index = i;
            ctus[i]->mvds.resize(3);
            ctus[i]->x_greater_0_flag.resize(3);
            ctus[i]->x_greater_1_flag.resize(3);
            ctus[i]->y_greater_0_flag.resize(3);
            ctus[i]->y_greater_1_flag.resize(3);
            ctus[i]->x_sign_flag.resize(3);
            ctus[i]->y_sign_flag.resize(3);

            ctus[i]->top_left_x     = std::min({init_triangles[i].first.p1.x, init_triangles[i].first.p2.x, init_triangles[i].first.p3.x});
            ctus[i]->top_left_y     = std::min({init_triangles[i].first.p1.y, init_triangles[i].first.p2.y, init_triangles[i].first.p3.y});
            ctus[i]->bottom_right_x = std::max({init_triangles[i].first.p1.x, init_triangles[i].first.p2.x, init_triangles[i].first.p3.x});
            ctus[i]->bottom_right_y = std::max({init_triangles[i].first.p1.y, init_triangles[i].first.p2.y, init_triangles[i].first.p3.y});
        }

        cv::Mat spatialMvTestImage;
        cv::Mat new_gauss_output_image = cv::Mat::zeros(gaussRefImage.rows, gaussRefImage.cols, CV_8UC3);

        std::vector<Triangle> tt = triangle_division.getTriangleIndexList();
        corners = triangle_division.getCorners();

        std::vector<cv::Point2f> tmp_ref_corners(corners.size()), add_corners;

        triangle_division.constructPreviousCodingTree(ctus, 0);

        std::vector<std::vector<std::vector<int>>> diagonal_line_area_flag(init_triangles.size(), std::vector< std::vector<int> >(block_size_x, std::vector<int>(block_size_y, -1)) );

        freq_newton_warping.resize(2);
        freq_newton_translation.resize(2);
        freq_newton_warping[0].resize(21);
        freq_newton_warping[1].resize(21);
        freq_newton_translation[0].resize(21);
        freq_newton_translation[1].resize(21);
        slow_newton_translation.resize(2);
        slow_newton_warping.resize(2);
        mv_newton_translation.resize(2);
        mv_newton_warping.resize(2);
        coordinate_newton_translation1.resize(2);
        coordinate_newton_translation2.resize(2);
        coordinate_newton_translation3.resize(2);
        coordinate_newton_warping1.resize(2);
        coordinate_newton_warping2.resize(2);
        coordinate_newton_warping3.resize(2);
        for(int i = 0 ; i < freq_newton_warping.size() ; i++) {
            for(int j = 0 ; j < freq_newton_warping[i].size() ; j++){
                freq_newton_warping[i][j]     = 0;
                freq_newton_translation[i][j] = 0;
            }
        }

        pells.resize(5);
        residuals.resize(5);

        for (int i = 0; i < init_triangles.size(); i++) {
            if(i % 2 == 0){
                bool flag = false;
                for (int x = 0; x < block_size_x; x++) {
                    // diagonal line
                    if(divide_flags[(i/2)] == LEFT_DIVIDE) {
                        diagonal_line_area_flag[i/2][x][block_size_y - x - 1] = (flag ? i : i + 1);
                        flag = !flag;
                    }else{
                        diagonal_line_area_flag[i/2][x][x] = (flag ? i : i + 1);
                        flag = !flag;
                    }

                }
            }

            std::pair<Point3Vec, int> triangle = init_triangles[i];
            cv::Point2f p1 = triangle.first.p1;
            cv::Point2f p2 = triangle.first.p2;
            cv::Point2f p3 = triangle.first.p3;
            std::cout << "================== step:" << i << " ================== " << std::endl;
            triangle_division.split(expand_images, ctus[i], nullptr, Point3Vec(p1, p2, p3), i, triangle.second, division_steps, diagonal_line_area_flag[i/2]);
        }
        std::cout << "split finished" << std::endl;

        // TODO: ログだすやつ書く
        // ===========================================================
        // ログ出力
        // ===========================================================
        cv::Rect rect(0, 0, target_image.cols, target_image.rows);
        cv::Mat recon = getReconstructionDivisionImage(gaussRefImage, ctus, block_size_x, block_size_y);
        cv::Mat p_image = triangle_division.getPredictedImageFromCtu(ctus, diagonal_line_area_flag);
        cv::Mat color = triangle_division.getPredictedColorImageFromCtu(ctus, diagonal_line_area_flag, getPSNR(target_image, p_image, rect));
        cv::Mat merge_color = triangle_division.getMergeModeColorImageFromCtu(ctus, diagonal_line_area_flag);

        int code_length = triangle_division.getCtuCodeLength(ctus);
        std::string log_file_suffix = out_file_suffix + std::to_string(qp) + "_" + getCurrentTimestamp();
        std::cout << "qp:" << qp << " divide:" << division_steps << std::endl;
        std::cout << "PSNR:" << getPSNR(target_image, p_image, rect) << " code_length:" << code_length << std::endl;
        std::cout << log_directory + "/log" + log_file_suffix + "/p_mv_image_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png" << std::endl;

        time_t end = clock();

        const double time = static_cast<double>(end - start) / CLOCKS_PER_SEC;
        printf("time %d[m]%d[sec]\n", (int)time/60, (int)time%60);
//        Decoder decoder(ref_image, target_image);
//        decoder.initTriangle(block_size_x, block_size_y, division_steps, qp, LEFT_DIVIDE);
//        decoder.reconstructionTriangle(ctus);
//        cv::imwrite(img_directory + "/p_recon_decoder_test.png", decoder.getReconstructionTriangleImage());
//        cv::imwrite(img_directory + "/p_recon_mode_image_test.png", decoder.getModeImage(ctus, diagonal_line_area_flag));
//
//        cv::imwrite(img_directory + "/p_mv_image_test.png", decoder.getMvImage(color));

#if STORE_DISTRIBUTION_LOG
#if GAUSS_NEWTON_TRANSLATION_ONLY
        Analyzer analyzer(log_file_suffix);
        #if STORE_MVD_DISTRIBUTION_LOG
        analyzer.storeDistributionOfMv(ctus, log_directory);
        analyzer.storeMarkdownFile(getPSNR(target_image, p_image) , log_directory);
        #endif
        analyzer.storeCsvFileWithStream(ofs, getPSNR(target_image, p_image), time);
#if STORE_MERGE_LOG
        analyzer.storeMergeMvLog(ctus, log_directory + "/log" + log_file_suffix + "/merge_log_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".txt");
#endif
#else
        Analyzer analyzer(ctus, log_directory, log_file_suffix, target_image, p_image, pells, residuals, block_size_x, block_size_y);
#if STORE_MVD_DISTRIBUTION_LOG
        analyzer.storeDistributionOfMv();
        analyzer.storeMarkdownFile(getPSNR(target_image, p_image) , log_directory);
#endif
        analyzer.storeLog();
        analyzer.storeCsvFileWithStream(ofs, getPSNR(target_image, p_image, rect), time); // WARNING: こいつはstoreDistributionOfMv以降で呼ばないといけない
#if STORE_MERGE_LOG
        analyzer.storeMergeMvLog(ctus, log_directory + "/log" + log_file_suffix + "/merge_log_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".txt");
#endif

#endif
#endif

#if STORE_IMG_LOG
            cv::imwrite( log_directory + "/log" + log_file_suffix + "/p_image_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png", p_image);
            cv::imwrite( log_directory + "/log" + log_file_suffix + "/p_residual_image_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png", getResidualImage(target_image, p_image, 4));
            cv::imwrite( log_directory + "/log" + log_file_suffix + "/p_mv_image_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png", triangle_division.getMvImage(ctus));
            cv::imwrite( log_directory + "/log" + log_file_suffix + "/p_mode_image_"  + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png", color);
            cv::imwrite( log_directory + "/log" + log_file_suffix + "/p_patch_image_"  + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png", recon);
            cv::imwrite( log_directory + "/log" + log_file_suffix + "/p_merge_image_"  + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".png", merge_color);
#endif

        for(int i = 0 ; i < ctus.size() ; i++) {
            delete ctus[i];
        }
        std::vector<CodingTreeUnit *>().swap(ctus);

        previous_qp = qp;

#if STORE_NEWTON_LOG
        storeNewtonLogs(getProjectDirectory(OS) + tasks[0].getLogDirectory());
#endif
    }
    ofs.close();
}

void storeNewtonLogs(std::string logDirectoryPath){
//    std::ofstream ofs_newton_0;
//    ofs_newton_0.open(logDirectoryPath + "/Newton_freq_ref_0_" + getCurrentTimestamp() + "_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".csv");

    /**
     *
     * イテレーション回数の頻度をCSVに吐き出す
     *
     */
//    ofs_newton_0 << "translation" << std::endl;
//    for(int i = 1 ; i < freq_newton_translation[0].size() ; i++) {
//        ofs_newton_0 << i << "," << freq_newton_translation[0][i] << std::endl;
//    }
//    ofs_newton_0 << "warping" << std::endl;
//    for(int i = 1 ; i < freq_newton_warping[0].size() ; i++) {
//        ofs_newton_0 << i << "," << freq_newton_warping[0][i] << std::endl;
//    }
//
//    ofs_newton_0.close();
//
//    std::ofstream ofs_newton_1;
//    ofs_newton_1.open(logDirectoryPath + "/Newton_freq_ref_1_" + getCurrentTimestamp() + "_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".csv");
//
//    ofs_newton_1 << "translation" << std::endl;
//    for(int i = 1 ; i < freq_newton_translation[1].size() ; i++) {
//        ofs_newton_1 << i << "," << freq_newton_translation[1][i] << std::endl;
//    }
//    ofs_newton_1 << "warping" << std::endl;
//    for(int i = 1 ; i < freq_newton_warping[1].size() ; i++) {
//        ofs_newton_1 << i << "," << freq_newton_warping[1][i] << std::endl;
//    }
//
//    ofs_newton_1.close();

    /**
     *
     * 残差減少の過程をCSVファイルに書き出す
     *
     */
    std::ofstream ofs_newton2_0;
    ofs_newton2_0.open(logDirectoryPath + "/Slowlog_ref_0_" + getCurrentTimestamp() + "_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".csv");

    ofs_newton2_0 << "translation" << std::endl;
    for(auto & m : ME_log_translation_0){
        ofs_newton2_0 << "Initial Vector," << m.residual[0] << "," << m.mv_newton_translation[0] << std::endl;
        for(int j = 1 ; j < (int)m.residual.size() ; j++){
            ofs_newton2_0 << j << "," << m.residual[j] << "," << m.mv_newton_translation[j] << "," << m.coordinate_after_move1[j] << "," << m.coordinate_after_move2[j] << "," << m.coordinate_after_move3[j] << std::endl;
        }
        ofs_newton2_0 << std::endl;
    }

    ofs_newton2_0 << "warping" << std::endl;
    for(auto & m : ME_log_warping_0){
        ofs_newton2_0 << "Initial Vector," << m.residual[0] << "," << m.mv_newton_warping[0][0] << "," << m.mv_newton_warping[0][1] << "," << m.mv_newton_warping[0][2] << std::endl;
        for(int j = 1 ; j < (int)m.residual.size() ; j++){
            ofs_newton2_0 << j << "," << m.residual[j] << "," << m.mv_newton_warping[j][0] << "," << m.mv_newton_warping[j][1] << "," << m.mv_newton_warping[j][2] << "," << m.coordinate_after_move1[j] << "," << m.coordinate_after_move2[j] << "," << m.coordinate_after_move3[j] << std::endl;
        }
        ofs_newton2_0 << std::endl;
    }

    ofs_newton2_0.close();

    std::ofstream ofs_newton2_1;
    ofs_newton2_1.open(logDirectoryPath + "/Slowlog_ref_1_" + getCurrentTimestamp() + "_" + std::to_string(qp) + "_divide_" + std::to_string(division_steps) + out_file_suffix + ".csv");
    ofs_newton2_1 << "translation" << std::endl;
    for(auto & m : ME_log_translation_0){
        ofs_newton2_1 << "Initial Vector," << m.residual[0] << "," << m.mv_newton_translation[0] << std::endl;
        for(int j = 1 ; j < (int)m.residual.size() ; j++){
            ofs_newton2_1 << j << "," << m.residual[j] << "," << m.mv_newton_translation[j] << "," << m.coordinate_after_move1[j] << "," << m.coordinate_after_move2[j] << "," << m.coordinate_after_move3[j] << std::endl;
        }
        ofs_newton2_1 << std::endl;
    }

    ofs_newton2_1 << "warping" << std::endl;
    for(auto & m : ME_log_warping_0){
        ofs_newton2_1 << "Initial Vector," << m.residual[0] << "," << m.mv_newton_warping[0][0] << "," << m.mv_newton_warping[0][1] << "," << m.mv_newton_warping[0][2] << std::endl;
        for(int j = 1 ; j < (int)m.residual.size() ; j++){
            ofs_newton2_1 << j << "," << m.residual[j] << "," << m.mv_newton_warping[j][0] << "," << m.mv_newton_warping[j][1] << "," << m.mv_newton_warping[j][2] << "," << m.coordinate_after_move1[j] << "," << m.coordinate_after_move2[j] << "," << m.coordinate_after_move3[j] << std::endl;
        }
        ofs_newton2_1 << std::endl;
    }

    ofs_newton2_1.close();

}