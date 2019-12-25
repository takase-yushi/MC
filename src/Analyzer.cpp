//
// Created by Kamiya Keisuke on 2019-07-18.
//

#include "../includes/Analyzer.h"
#include "../includes/Utils.h"
#include "../includes/Encode.h"
#include "../includes/psnr.h"
#include <cstdio>
#include <iostream>
#include <sys/stat.h>
#include <fstream>

/**
 * @fn void Analyzer::storeDistributionOfMv()
 * @brief MVの分布を見る
 * @param ctus
 */
void Analyzer::storeDistributionOfMv() {
    FILE *fp = std::fopen((log_path + "/mvd_distribution" + file_suffix + ".csv").c_str(), "w");
    for(auto x : mvd_counter){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((log_path + "/mvd_distribution_x" + file_suffix + ".csv").c_str(), "w");
    for(auto x : mvd_counter_x){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((log_path + "/mvd_distribution_y" + file_suffix + ".csv").c_str(), "w");
    for(auto x : mvd_counter_y){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((log_path + "/MV_distribution" + file_suffix + ".csv").c_str(), "w");
    for(auto x : MV_counter){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((log_path + "/mvd_greater_0_flag_distribution" + file_suffix + ".csv").c_str(), "w");
    for(auto x : greater_0_flag_counter){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

    fp = std::fopen((log_path + "/mvd_greater_1_flag_distribution" + file_suffix + ".csv").c_str(), "w");
    for(auto x : greater_1_flag_counter){
        fprintf(fp, "%d,%d\n", x.first, x.second);
    }
    fclose(fp);

}


/**
 *
 * @param ctu
 */
void Analyzer::collectResults(CodingTreeUnit *ctu) {
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr){
        code_sum += (ctu->code_length);
        patch_num++;

        if(ctu->method == MV_CODE_METHOD::MERGE2) {
            int x_ = (int)abs(((ctu->mv1).x * 4));
            int y_ = (int)abs(((ctu->mv1).y * 4));
            MV_counter[x_]++;
            MV_counter[y_]++;
            int x = (ctu->mvds_x)[0];
            mvd_counter_x[x]++;
            int y = (ctu->mvds_y)[0];
            mvd_counter_y[y]++;

            mvd_counter[x]++;
            mvd_counter[y]++;

            greater_0_flag_counter[(int)(ctu->flags_code_sum.getXGreater0Flag()[0])]++;
            greater_0_flag_counter[(int)(ctu->flags_code_sum.getYGreater0Flag()[0])]++;

            if(ctu->flags_code_sum.getXGreater0Flag()[0]) {
                greater_1_flag_counter[(int)(ctu->flags_code_sum.getXGreater1Flag()[0])]++;
            }
            if(ctu->flags_code_sum.getYGreater0Flag()[0]) {
                greater_1_flag_counter[(int)(ctu->flags_code_sum.getYGreater1Flag()[0])]++;
            }
            merge2_counter++;
            greater_0_flag_sum += ctu->flags_code_sum.getGreater0FlagCodeLength();
            greater_1_flag_sum += ctu->flags_code_sum.getGreaterThanOneCodeLength();
            sign_flag_sum += ctu->flags_code_sum.getSignFlagCodeLength();
            mvd_code_sum += ctu->flags_code_sum.getMvdCodeLength();

            affine_new_merge++;
        } else if(ctu->method != MV_CODE_METHOD::MERGE){
            if(ctu->translation_flag){
                int x_ = (int)abs(((ctu->mv1).x * 4));
                int y_ = (int)abs(((ctu->mv1).y * 4));
                MV_counter[x_]++;
                MV_counter[y_]++;
                int x = (ctu->mvds_x)[0];
                mvd_counter_x[x]++;
                int y = (ctu->mvds_y)[0];
                mvd_counter_y[y]++;

                mvd_counter[x]++;
                mvd_counter[y]++;

                greater_0_flag_counter[(int)(ctu->flags_code_sum.getXGreater0Flag()[0])]++;
                greater_0_flag_counter[(int)(ctu->flags_code_sum.getYGreater0Flag()[0])]++;

                if(ctu->flags_code_sum.getXGreater0Flag()[0]) {
                    greater_1_flag_counter[(int)(ctu->flags_code_sum.getXGreater1Flag()[0])]++;
                }
                if(ctu->flags_code_sum.getYGreater0Flag()[0]) {
                    greater_1_flag_counter[(int)(ctu->flags_code_sum.getYGreater1Flag()[0])]++;
                }

                mvd_translation_code_sum += ctu->flags_code_sum.getMvdCodeLength();

                translation_diff++;
            }else{
                for(int i = 0 ; i < 3 ; i++) {
                    int x = (ctu->mvds_x)[i];
                    mvd_counter_x[x]++;
                    int y = (ctu->mvds_y)[i];
                    mvd_counter_y[y]++;
                    mvd_counter[x]++;
                    mvd_counter[y]++;

                    if(ctu->flags_code_sum.getXGreater0Flag()[i]) {
                        greater_1_flag_counter[(int)(ctu->flags_code_sum.getXGreater1Flag()[i])]++;
                    }
                    if(ctu->flags_code_sum.getYGreater0Flag()[i]) {
                        greater_1_flag_counter[(int)(ctu->flags_code_sum.getYGreater1Flag()[i])]++;
                    }

                    greater_0_flag_counter[(int)(ctu->flags_code_sum.getXGreater0Flag()[i])]++;
                    greater_0_flag_counter[(int)(ctu->flags_code_sum.getYGreater0Flag()[i])]++;
                }
                mvd_affine_code_sum += ctu->flags_code_sum.getMvdCodeLength();
                affine_diff++;
            }

            greater_0_flag_sum += ctu->flags_code_sum.getGreater0FlagCodeLength();
            greater_1_flag_sum += ctu->flags_code_sum.getGreaterThanOneCodeLength();
            sign_flag_sum += ctu->flags_code_sum.getSignFlagCodeLength();
            mvd_code_sum += ctu->flags_code_sum.getMvdCodeLength();
            differential_counter++;

            merge_flag_counter[0]++;
            intra_flag_counter[0]++;
        }else if(ctu->method == MV_CODE_METHOD::MERGE){
            merge_counter++;
            merge_flag_counter[1]++;
            intra_flag_counter[0]++;

            if(ctu->translation_flag){
                translation_merge++;
            }else{
                affine_merge++;
            }
        }else if(ctu->method == MV_CODE_METHOD::INTRA){
            intra_counter++;
            merge_flag_counter[0]++;
            intra_flag_counter[1]++;
        }

        if(ctu->translation_flag) translation_patch_num++;
        else affine_patch_num++;

        return;
    }

    if(ctu->node1 != nullptr) collectResults(ctu->node1);
    if(ctu->node2 != nullptr) collectResults(ctu->node2);
    if(ctu->node3 != nullptr) collectResults(ctu->node3);
    if(ctu->node4 != nullptr) collectResults(ctu->node4);
}

/**
 * @fn void Analyzer::storeMarkdownFile(double psnr, std::string log_path)
 * @brief Markdownとして結果を書き出す
 * @param psnr PSNR
 * @param log_path ログのパス
 */
void Analyzer::storeMarkdownFile(double psnr, std::string log_path) {
    log_path = log_path + "/log" + file_suffix;

    extern int qp;
    FILE *fp = fopen((log_path + "/result.md").c_str(), "w");
    fprintf(fp, "|%d|%f|%d|%f|%d|\n", qp, getLambdaPred(qp, 1.0), code_sum, psnr, affine_patch_num + translation_patch_num);
    fclose(fp);
}

/**
 * @fn void Analyzer::storeCsvFileWithStream(std::ofstream &ofs, double psnr)
 * @breif 外部からOutputStreamを受け取って，そこにCSV形式で書き出す
 * @param ofs OutputStream
 * @param psnr PSNR値
 */
void Analyzer::storeCsvFileWithStream(std::ofstream &ofs, double psnr, double time) {
    extern int qp;
    int tmp_code_sum = code_sum - (int)ceil(greater_0_flag_sum * (1.0-getEntropy({greater_0_flag_counter[0], greater_0_flag_counter[1]})));
    std::cout << (int)ceil(greater_0_flag_sum * (1.0 - getEntropy({greater_0_flag_counter[0], greater_0_flag_counter[1]})))<< std::endl;
    std::cout << "tmp_code_sum:" << tmp_code_sum << std::endl;
    tmp_code_sum = tmp_code_sum - (int)ceil(greater_1_flag_sum * (1.0 - getEntropy({greater_1_flag_counter[0], greater_1_flag_counter[1]})));

#if MERGE_MODE
    std::cout << (int)ceil(patch_num * (1.0 - getEntropy({merge_flag_counter[0], merge_flag_counter[1]}))) << std::endl;
    std::cout << "tmp_code_sum:" << tmp_code_sum << std::endl;
    tmp_code_sum = tmp_code_sum - (int)ceil(patch_num * (1.0 - getEntropy({merge_flag_counter[0], merge_flag_counter[1]})));
#endif

#if INTRA_MODE
    std::cout << "tmp_code_sum:" << tmp_code_sum << std::endl;
    if(INTRA_MODE) tmp_code_sum = tmp_code_sum - (int)ceil(intra_counter * getEntropy({intra_flag_counter[0], intra_flag_counter[1]}));
#endif

    ofs << qp << "," << getLambdaPred(qp, 1.0) << "," << code_sum << "," << tmp_code_sum << "," << psnr << "," << patch_num << "," << differential_counter << "," << merge_counter << "," << merge2_counter << "," << intra_counter << "," << time << std::endl;
}

void Analyzer::storeMergeMvLog(std::vector<CodingTreeUnit*> ctus, std::string log_path) {
    std::ofstream ofs;
    ofs.open(log_path);

    for(const auto ctu : ctus) {
        storeMergeMvLog(ctu, ofs);
    }

    ofs << std::endl;
    ofs << "max_diff(x):" << max_merge_mv_diff_x << " max_diff(y):" << max_merge_mv_diff_y << std::endl;

    ofs.close();
}

void Analyzer::storeMergeMvLog(CodingTreeUnit *ctu, std::ofstream &ofs) {
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        if(ctu->method != MV_CODE_METHOD::MERGE) return;

        if(ctu->translation_flag) {
            ofs << "merge(transaltion)" << std::endl;
            ofs << "original:" << ctu->original_mv1 << " merged_mv:" << ctu->mv1 << std::endl;
            ofs << std::endl;

            max_merge_mv_diff_x = std::max(max_merge_mv_diff_x, std::fabs(ctu->original_mv1.x - ctu->mv1.x));
            max_merge_mv_diff_y = std::max(max_merge_mv_diff_y, std::fabs(ctu->original_mv1.y - ctu->mv1.y));
        } else {
            ofs << "merge(warping)" << std::endl;
            ofs << "original(1):" << ctu->original_mv1 << " merged_mv(1):" << ctu->mv1 << std::endl;
            ofs << "original(2):" << ctu->original_mv2 << " merged_mv(2):" << ctu->mv2 << std::endl;
            ofs << "original(3):" << ctu->original_mv3 << " merged_mv(3):" << ctu->mv3 << std::endl;
            ofs << std::endl;

            max_merge_mv_diff_x = std::max(max_merge_mv_diff_x, std::fabs(ctu->original_mv1.x - ctu->mv1.x));
            max_merge_mv_diff_x = std::max(max_merge_mv_diff_x, std::fabs(ctu->original_mv2.x - ctu->mv2.x));
            max_merge_mv_diff_x = std::max(max_merge_mv_diff_x, std::fabs(ctu->original_mv3.x - ctu->mv3.x));
            max_merge_mv_diff_y = std::max(max_merge_mv_diff_y, std::fabs(ctu->original_mv1.y - ctu->mv1.y));
            max_merge_mv_diff_y = std::max(max_merge_mv_diff_y, std::fabs(ctu->original_mv2.y - ctu->mv2.y));
            max_merge_mv_diff_y = std::max(max_merge_mv_diff_y, std::fabs(ctu->original_mv3.y - ctu->mv3.y));
        }
        return;
    }

    storeMergeMvLog(ctu->node1, ofs);
    storeMergeMvLog(ctu->node2, ofs);
    storeMergeMvLog(ctu->node3, ofs);
    storeMergeMvLog(ctu->node4, ofs);
}

/**
 * @fn int Analyzer::getEntropyCodingCode()
 * @brief 出現確率をもとにエントロピーを計算し，擬似的な圧縮を施した符号量を求める
 * @return int 符号量
 * @attention これを使う前にstoreDistributionOfMvを呼ぶこと
 * @deprecated これを使う前にstoreDistributionOfMvを呼ぶこと
 */
int Analyzer::getEntropyCodingCode() {
    extern int qp;
    int tmp_code_sum = code_sum - (int)ceil(greater_0_flag_sum * (1.0-getEntropy({greater_0_flag_counter[0], greater_0_flag_counter[1]})));
    std::cout << (int)ceil(greater_0_flag_sum * (1.0 - getEntropy({greater_0_flag_counter[0], greater_0_flag_counter[1]})))<< std::endl;
    std::cout << "tmp_code_sum:" << tmp_code_sum << std::endl;
    tmp_code_sum = tmp_code_sum - (int)ceil(greater_1_flag_sum * (1.0 - getEntropy({greater_1_flag_counter[0], greater_1_flag_counter[1]})));

#if MERGE_MODE
    std::cout << (int)ceil(patch_num * (1.0 - getEntropy({merge_flag_counter[0], merge_flag_counter[1]}))) << std::endl;
    std::cout << "tmp_code_sum:" << tmp_code_sum << std::endl;
    tmp_code_sum = tmp_code_sum - (int)ceil(patch_num * (1.0 - getEntropy({merge_flag_counter[0], merge_flag_counter[1]})));
#endif

#if INTRA_MODE
    std::cout << "tmp_code_sum:" << tmp_code_sum << std::endl;
    if(INTRA_MODE) tmp_code_sum = tmp_code_sum - (int)ceil(intra_counter * getEntropy({intra_flag_counter[0], intra_flag_counter[1]}));
#endif

    return tmp_code_sum;
}

Analyzer::Analyzer(std::vector<CodingTreeUnit *> ctus, std::string _log_path, const std::string &fileSuffix, cv::Mat targetImage, cv::Mat pImage, std::vector<int> _pells, std::vector<double> _residuals) {
    greater_0_flag_sum = greater_1_flag_sum = sign_flag_sum = mvd_code_sum = affine_patch_num = translation_patch_num = 0;
    mvd_affine_code_sum = mvd_translation_code_sum = 0;
    merge_counter = differential_counter = 0;
    code_sum = 0;
    intra_counter = 0;
    patch_num = 0;
    max_merge_mv_diff_x = 0;
    max_merge_mv_diff_y = 0;
    merge2_counter = 0;
    translation_diff = translation_merge = 0;
    affine_diff = affine_merge = affine_new_merge = 0;
    pells = _pells;
    residuals = _residuals;
    log_path = _log_path;
    file_suffix = fileSuffix;
    target_image = targetImage;
    p_image = pImage;

    for(auto ctu : ctus){
        collectResults(ctu);
    }

    log_path = log_path + "/log" + file_suffix;
    mkdir((log_path).c_str());
}

/**
 * @fn void Analyzer::storeLog()
 * @brief ログを吐き出す
 */
void Analyzer::storeLog() {
    FILE *fp = std::fopen((log_path + "/mvd_result" + file_suffix + ".txt").c_str(), "w");
    cv::Rect rect(0, 0, target_image.cols, target_image.rows);

    fprintf(fp, "summary =======================================================\n");
    fprintf(fp, "code_sum                              :%d\n", code_sum);
    fprintf(fp, "code_sum(entropy coding)              :%d\n", getEntropyCodingCode());
    fprintf(fp, "PSNR[dB]                              :%.2f[dB]\n", getPSNR(target_image, p_image, rect));
    fprintf(fp, "greater_0_flag                        :%d\n", greater_0_flag_sum);
    fprintf(fp, "greater_1_flag                        :%d\n", greater_1_flag_sum);
    fprintf(fp, "sign_flag                             :%d\n", sign_flag_sum);
    fprintf(fp, "mvd_code                              :%d\n\n", mvd_code_sum);

    int max_area_ratio_digits;
    int translation_pell_num = pells[PATCH_CODING_MODE::TRANSLATION_DIFF] + pells[PATCH_CODING_MODE::TRANSLATION_MERGE];
    int affine_pell_num = pells[PATCH_CODING_MODE::AFFINE_DIFF] + pells[PATCH_CODING_MODE::AFFINE_MERGE] + pells[PATCH_CODING_MODE::AFFINE_NEW_MERGE];
    if(translation_pell_num == 0 || affine_pell_num == 0){
        max_area_ratio_digits = 3;
    }else{
        max_area_ratio_digits = 2;
    }

    int max_patch_num_digits = std::max(getNumberOfDigits(translation_patch_num), getNumberOfDigits(affine_patch_num));

    fprintf(fp, "Number of patches ==============================================\n");
    fprintf(fp, "patch num(all)                        :%d\n", patch_num);
    fprintf(fp, "Translation patch num                 :%*d(%*.2f[%%])\n",   max_patch_num_digits, translation_patch_num, max_area_ratio_digits, (double)translation_patch_num / (translation_patch_num + affine_patch_num) * 100);
    fprintf(fp, "Affine patch Num                      :%*d(%*.2f[%%])\n", max_patch_num_digits, affine_patch_num,      max_area_ratio_digits, (double)affine_patch_num / (translation_patch_num + affine_patch_num) * 100);
    fprintf(fp, "Translation area ratio                :%*.2f[%%]\n",   max_area_ratio_digits, (double)translation_pell_num / (translation_pell_num + affine_pell_num) * 100);
    fprintf(fp, "Affine area ratio                     :%*.2f[%%]\n\n", max_area_ratio_digits, (double)affine_pell_num / (translation_pell_num + affine_pell_num) * 100);
    fprintf(fp, "Code amount of patches =========================================\n");
    fprintf(fp, "Translation's code                    :%d\n", mvd_translation_code_sum);
    fprintf(fp, "Affine patch's code                   :%d\n\n", mvd_affine_code_sum);
    fprintf(fp, "Breakdown of coding methods ====================================\n");
    fprintf(fp, "Differential_patch                    :%d\n", differential_counter);
    fprintf(fp, "merge_patch                           :%d\n\n", merge_counter);

    int max_number_of_digits = std::max({getNumberOfDigits(translation_diff), getNumberOfDigits(translation_merge), getNumberOfDigits(affine_diff), getNumberOfDigits(affine_merge), getNumberOfDigits(affine_new_merge)});
    fprintf(fp, "Number of patches by coding methods ============================\n");
    fprintf(fp, "translation differential coding patch :%*d(%2.2f[%%])\n", max_number_of_digits, translation_diff, (double)translation_diff / patch_num * 100);
    fprintf(fp, "translation merge coding patch        :%*d(%2.2f[%%])\n", max_number_of_digits, translation_merge, (double)translation_merge / patch_num * 100);
    fprintf(fp, "Affine differential coding patch      :%*d(%2.2f[%%])\n", max_number_of_digits, affine_diff, (double)affine_diff / patch_num * 100);
    fprintf(fp, "Affine merge coding patch             :%*d(%2.2f[%%])\n", max_number_of_digits, affine_merge, (double)affine_merge / patch_num * 100);
    fprintf(fp, "Affine new merge coding patch         :%*d(%2.2f[%%])\n\n", max_number_of_digits, affine_new_merge, (double)affine_new_merge / patch_num * 100);
    fprintf(fp, "PSNR by coding methods =========================================\n");
    fprintf(fp, "TRANSLATION_DIFF PSNR                 :%.2f[dB]\n", 10 * std::log10(255.0 * 255.0 / (residuals[PATCH_CODING_MODE::TRANSLATION_DIFF] / pells[PATCH_CODING_MODE::TRANSLATION_DIFF])));
    fprintf(fp, "TRANSLATION_MERGE PSNR                :%.2f[dB]\n", 10 * std::log10(255.0 * 255.0 / (residuals[PATCH_CODING_MODE::TRANSLATION_MERGE] / pells[PATCH_CODING_MODE::TRANSLATION_MERGE])));
    fprintf(fp, "AFFINE_DIFF PSNR                      :%.2f[dB]\n", 10 * std::log10(255.0 * 255.0 / (residuals[PATCH_CODING_MODE::AFFINE_DIFF] / pells[PATCH_CODING_MODE::AFFINE_DIFF])));
    fprintf(fp, "AFFINE_MERGE PSNR                     :%.2f[dB]\n", 10 * std::log10(255.0 * 255.0 / (residuals[PATCH_CODING_MODE::AFFINE_MERGE] / pells[PATCH_CODING_MODE::AFFINE_MERGE])));
    fprintf(fp, "AFFINE_NEW_MERGE PSNR                 :%.2f[dB]\n", 10 * std::log10(255.0 * 255.0 / (residuals[PATCH_CODING_MODE::AFFINE_NEW_MERGE] / pells[PATCH_CODING_MODE::AFFINE_NEW_MERGE])));

    if(INTRA_MODE) {
        fprintf(fp, "intra_patch           :%d\n", intra_counter);
        fprintf(fp, "intra_flag_entropy    :%f\n", getEntropy({intra_flag_counter[0], intra_flag_counter[1]}));
    }

    fclose(fp);
}

