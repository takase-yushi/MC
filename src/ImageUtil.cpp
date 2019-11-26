//
// Created by kasph on 2019/05/23.
//

#include <opencv2/core/mat.hpp>
#include <cassert>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "../includes/ImageUtil.h"
#include "../includes/Utils.h"
#include "../includes/CodingTreeUnit.h"
#include "../includes/Reconstruction.h"
#include "../includes/TriangleDivision.h"

/**
 * @fn get_residual(const cv::Mat &target_image, const cv::Mat &predict_image)
 * @brief target_imageとpredict_imageの差分を計算し、差分画像として返す
 * @param target_image 対象画像
 * @param predict_image 予測画像
 * @return 対象画像と予測画像の差分画像
 */
cv::Mat getResidualImage(const cv::Mat &target_image, const cv::Mat &predict_image, int k){
    assert(target_image.cols == predict_image.cols && target_image.rows == predict_image.rows);

    cv::Mat residual_image = cv::Mat::zeros(target_image.rows, target_image.cols, CV_8UC3);

    for(int row = 0 ; row < target_image.rows ; row++){
        for(int col = 0 ; col < target_image.cols ; col++){
            R(residual_image, col, row) = k * abs(R(target_image, col, row) - R(predict_image, col, row));
            G(residual_image, col, row) = k * abs(G(target_image, col, row) - G(predict_image, col, row));
            B(residual_image, col, row) = k * abs(B(target_image, col, row) - B(predict_image, col, row));
        }
    }

    return residual_image;
}

/***
 * @fn double getTriangleResidual(const cv::Mat &target_image, const cv::Mat ref_image, Point3Vec vec)
 * @brief ある三角形を変形した際の残差を返す
 * @param ref_image 参照画像
 * @param target_image 対象画像
 * @param triangle 三角パッチの座標
 * @param vec 動きベクトル
 * @return 残差
 */
double getTriangleResidual(const cv::Mat ref_image, const cv::Mat &target_image, Point3Vec &triangle, std::vector<cv::Point2f> mv, const std::vector<cv::Point2f> &in_triangle_pixels){
    double residual = 0.0;

    cv::Point2f pixels_in_triangle;

    cv::Point2f pp0, pp1, pp2;

    pp0.x = triangle.p1.x + mv[0].x;
    pp0.y = triangle.p1.y + mv[0].y;
    pp1.x = triangle.p2.x + mv[1].x;
    pp1.y = triangle.p2.y + mv[1].y;
    pp2.x = triangle.p3.x + mv[2].x;
    pp2.y = triangle.p3.y + mv[2].y;

    double quantize_step = 4.0;

    cv::Point2f X,a,b,a_later,b_later,X_later;
    double alpha,beta,det;

    double squared_error = 0.0;

    a = triangle.p3 - triangle.p1;
    b = triangle.p2 - triangle.p1;
    det = a.x * b.y - a.y * b.x;

    for(const auto& pixel : in_triangle_pixels) {
        X.x = pixel.x - triangle.p1.x;
        X.y = pixel.y - triangle.p1.y;
        alpha = (X.x * b.y - X.y * b.x) / det;
        beta = (a.x * X.y - a.y * X.x) / det;
        X.x += triangle.p1.x;
        X.y += triangle.p1.y;

        a_later = pp2 - pp0;
        b_later = pp1 - pp0;
        X_later = alpha * a_later + beta * b_later + pp0;

        if (X_later.x >= ref_image.cols - 1) X_later.x = ref_image.cols - 1.001;

        if (X_later.y >= ref_image.rows - 1) X_later.y = ref_image.rows - 1.001;

        if (X_later.x < 0) X_later.x = 0;

        if (X_later.y < 0) X_later.y = 0;

        int x0 = floor(X_later.x);
        double d_x = X_later.x - x0;
        int y0 = floor(X_later.y);
        double d_y = X_later.y - y0;

        int y = (int) floor((M(ref_image, (int) x0    , (int) y0    ) * (1 - d_x) * (1 - d_y)  +
                             M(ref_image, (int) x0 + 1, (int) y0    ) * (    d_x) * (1 - d_y)  +
                             M(ref_image, (int) x0    , (int) y0 + 1) * (1 - d_x) * (    d_y)  +
                             M(ref_image, (int) x0 + 1, (int) y0 + 1) * (    d_x) * (    d_y)) + 0.5);

        squared_error += pow((M(target_image, (int)pixel.x, (int)pixel.y) - (0.299 * y + 0.587 * y + 0.114 * y)), 2);
    }

    return squared_error;
}


/***
 * @fn double getTriangleResidual(const cv::Mat &target_image, const cv::Mat ref_image, Point3Vec vec)
 * @brief ある三角形を変形した際の残差を返す
 * @param ref_image 参照画像
 * @param target_image 対象画像
 * @param triangle 三角パッチの座標
 * @param vec 動きベクトル
 * @return 残差(SAD)
 */
double getTriangleResidual(unsigned char **ref_image, const cv::Mat &target_image, Point3Vec &triangle, std::vector<cv::Point2f> mv, const std::vector<cv::Point2f> &in_triangle_pixels, cv::Rect rect){
    cv::Point2f pp0, pp1, pp2;

    pp0.x = triangle.p1.x + mv[0].x;
    pp0.y = triangle.p1.y + mv[0].y;
    pp1.x = triangle.p2.x + mv[1].x;
    pp1.y = triangle.p2.y + mv[1].y;
    pp2.x = triangle.p3.x + mv[2].x;
    pp2.y = triangle.p3.y + mv[2].y;

    cv::Point2f X,a,b,a_later,b_later,X_later;
    double alpha,beta,det;

    double sad = 0.0;

    a = triangle.p3 - triangle.p1;
    b = triangle.p2 - triangle.p1;
    det = a.x * b.y - a.y * b.x;

    for(const auto& pixel : in_triangle_pixels) {
        X.x = pixel.x - triangle.p1.x;
        X.y = pixel.y - triangle.p1.y;
        alpha = (X.x * b.y - X.y * b.x) / det;
        beta = (a.x * X.y - a.y * X.x) / det;
        X.x += triangle.p1.x;
        X.y += triangle.p1.y;

        a_later = pp2 - pp0;
        b_later = pp1 - pp0;
        X_later = alpha * a_later + beta * b_later + pp0;

        int y = img_ip(ref_image, rect, 4 * X_later.x, 4 * X_later.y, 1);

        sad += fabs(M(target_image, (int)pixel.x, (int)pixel.y) - (0.299 * y + 0.587 * y + 0.114 * y));
    }

    return sad;
}


/**
 * @fn std::vector<std::vector<cv::Mat>> getRefImages(const cv::Mat ref_image, const cv::Mat gauss_ref_image)
 * @brief 参照画像の集まりを返す
 * @param ref_image 参照画像
 * @param gauss_ref_image ガウスニュートン法で使う参照画像
 * @return あつまれ～！
 */
std::vector<std::vector<cv::Mat>> getRefImages(const cv::Mat& ref_image, const cv::Mat& gauss_ref_image){
    std::vector<std::vector<cv::Mat>> ref_images;

    // 参照画像のフィルタ処理（１）
    std::vector<cv::Mat> ref1_levels;
    cv::Mat ref_level_1, ref_level_2, ref_level_3, ref_level_4;
    ref_level_1 = gauss_ref_image;
    ref_level_2 = half(ref_level_1, 2);
    ref_level_3 = half(ref_level_2, 2);
    ref_level_4 = half(ref_level_3, 2);
    ref1_levels.emplace_back(ref_level_4);
    ref1_levels.emplace_back(ref_level_3);
    ref1_levels.emplace_back(ref_level_2);
    ref1_levels.emplace_back(ref_image);

    // 参照画像のフィルタ処理（２）
    std::vector<cv::Mat> ref2_levels;
    cv::Mat ref2_level_1 = getAppliedLPFImage(gauss_ref_image);
    cv::Mat ref2_level_2 = half(ref2_level_1, 2);
    cv::Mat ref2_level_3 = half(ref2_level_2, 1);
    cv::Mat ref2_level_4 = half(ref2_level_3, 1);
    ref2_levels.emplace_back(ref2_level_4);
    ref2_levels.emplace_back(ref2_level_3);
    ref2_levels.emplace_back(ref2_level_2);
    ref2_levels.emplace_back(getAppliedLPFImage(ref_image));

    ref_images.emplace_back(ref1_levels);
    ref_images.emplace_back(ref2_levels);

    return ref_images;
}

/**
 * @fn std::vector<std::vector<cv::Mat>> getTargetImages(const cv::Mat target_image)
 * @brief ガウス・ニュートン法で使う対象画像の集まりを返す
 * @param target_image 対象画像
 * @return vectorのvector(死)
 */
std::vector<std::vector<cv::Mat>> getTargetImages(const cv::Mat target_image){
    std::vector<std::vector<cv::Mat>> target_images;

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

    target_images.emplace_back(target1_levels);
    target_images.emplace_back(target2_levels);

    return target_images;
}

/**
 * @fn EXPAND_ARRAY_TYPE getExpandImages(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, int expand)
 * @brief 拡大した画像をいい感じに格納して返す
 * @param ref_images 参照画像の集合
 * @param target_images 対象画像の集合
 * @param expand 拡大する範囲
 * @return しゅうごう～
 */
EXPAND_ARRAY_TYPE getExpandImages(std::vector<std::vector<cv::Mat>> ref_images, std::vector<std::vector<cv::Mat>> target_images, int expand){
    EXPAND_ARRAY_TYPE expand_images;
    expand_images.resize(ref_images.size());
    for (int filter_num = 0; filter_num < static_cast<int>(ref_images.size()); filter_num++) {
        expand_images[filter_num].resize(ref_images[filter_num].size());
        for (int step = 0; step < static_cast<int>(ref_images[filter_num].size()); step++) {
            expand_images[filter_num][step].resize(4);
        }
    }

    expand += 2;
    for(int filter = 0 ; filter < static_cast<int>(ref_images.size()) ; filter++){
        for(int step = 3 ; step < static_cast<int>(ref_images[filter].size()) ; step++){
            cv::Mat current_target_image = target_images[filter][step];
            cv::Mat current_ref_image = ref_images[filter][step];
#if GAUSS_NEWTON_HEVC_IMAGE

            if(step == 3) {
                expand_images[filter][step][0] = getExpansionHEVCImage(current_ref_image, 4, expand);
                expand_images[filter][step][1] = getExpansionHEVCImage(ref_images[0][3], 4, expand);
                expand_images[filter][step][2] = getExpansionHEVCImage(current_target_image, 4, expand);
                expand_images[filter][step][3] = getExpansionHEVCImage(target_images[0][3], 4, expand);
            } 
#else
            auto **current_target_expand = (unsigned char **) std::malloc(
                    sizeof(unsigned char *) * (current_target_image.cols + expand * 2));
            current_target_expand += expand;
            auto **current_target_org_expand = (unsigned char **) std::malloc(
                    sizeof(unsigned char *) * (current_target_image.cols + expand * 2));
            current_target_org_expand += expand;

            for (int j = -expand; j < current_target_image.cols + expand; j++) {
                current_target_expand[j] = (unsigned char *) std::malloc(
                        sizeof(unsigned char) * (current_target_image.rows + expand * 2));
                current_target_expand[j] += expand;

                current_target_org_expand[j] = (unsigned char *) std::malloc(
                        sizeof(unsigned char) * (current_target_image.rows + expand * 2));
                current_target_org_expand[j] += expand;
            }

            auto **current_ref_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (current_target_image.cols + expand * 2));
            current_ref_expand += expand;
            auto **current_ref_org_expand = (unsigned char **) std::malloc(sizeof(unsigned char *) * (current_target_image.cols + expand * 2));
            current_ref_org_expand += expand;

            for (int j = -expand; j < current_ref_image.cols + expand; j++) {
                if ((current_ref_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (current_target_image.rows + expand * 2))) == nullptr) {
                }
                current_ref_expand[j] += expand;

                (current_ref_org_expand[j] = (unsigned char *) std::malloc(sizeof(unsigned char) * (current_target_image.rows + expand * 2)));
                current_ref_org_expand[j] += expand;
            }

            for (int j = -expand; j < current_target_image.rows + expand; j++) {
                for (int i = -expand; i < current_target_image.cols + expand; i++) {
                    if (j >= 0 && j < current_target_image.rows && i >= 0 && i < current_target_image.cols) {
                        current_target_expand[i][j] = M(current_target_image, i, j);
                        current_ref_expand[i][j] = M(current_ref_image, i, j);

                        current_target_org_expand[i][j] = M(target_images[0][3], i, j);
                        current_ref_org_expand[i][j] = M(ref_images[0][3], i, j);
                    } else {
                        current_target_expand[i][j] = 0;
                        current_ref_expand[i][j] = 0;
                        current_target_org_expand[i][j] = 0;
                        current_ref_org_expand[i][j] = 0;
                    }
                }
            }
            int spread = expand;// 双3次補間を行うために、画像の周り(16+2)=18ピクセルだけ折り返し
            for (int j = 0; j < current_target_image.rows; j++) {
                for (int i = 1; i <= spread; i++) {
                    current_target_expand[-i][j] = current_target_expand[0][j];
                    current_target_expand[current_target_image.cols - 1 + i][j] = current_target_expand[
                            current_target_image.cols - 1][j];
                    current_ref_expand[-i][j] = current_ref_expand[0][j];
                    current_ref_expand[current_target_image.cols - 1 + i][j] = current_ref_expand[
                            current_target_image.cols - 1][j];
                    current_target_org_expand[-i][j] = current_target_org_expand[0][j];
                    current_target_org_expand[current_target_image.cols - 1 + i][j] = current_target_org_expand[
                            current_target_image.cols - 1][j];
                    current_ref_org_expand[-i][j] = current_ref_org_expand[0][j];
                    current_ref_org_expand[current_target_image.cols - 1 + i][j] = current_ref_org_expand[
                            current_target_image.cols - 1][j];
                }
            }
            for (int i = -spread; i < current_target_image.cols + spread; i++) {
                for (int j = 1; j <= spread; j++) {
                    current_target_expand[i][-j] = current_target_expand[i][0];
                    current_target_expand[i][current_target_image.rows - 1 + j] = current_target_expand[i][
                            current_target_image.rows - 1];
                    current_ref_expand[i][-j] = current_ref_expand[i][0];
                    current_ref_expand[i][current_target_image.rows - 1 + j] = current_ref_expand[i][
                            current_target_image.rows - 1];

                    current_target_org_expand[i][-j] = current_target_org_expand[i][0];
                    current_target_org_expand[i][current_target_image.rows - 1 + j] = current_target_org_expand[i][
                            current_target_image.rows - 1];
                    current_ref_org_expand[i][-j] = current_ref_org_expand[i][0];
                    current_ref_org_expand[i][current_target_image.rows - 1 + j] = current_ref_org_expand[i][
                            current_target_image.rows - 1];
                }
            }
                expand_images[filter][step][0] = current_ref_expand;
                expand_images[filter][step][1] = current_ref_org_expand;
                expand_images[filter][step][2] = current_target_expand;
                expand_images[filter][step][3] = current_target_org_expand;
#endif
        }
    }

    return expand_images;
}

/**
 * @fn void freeExpandImages(EXPAND_ARRAY_TYPE expand_images, int expand, int filter_num, int step_num, int rows, int cols)
 * @breif 拡張した画像(mallocで取得)を開放、通称「拡張フリー」をする
 * @param expand_images freeeしたい画像
 * @param expand 拡大した画素数
 * @param filter_num フィルターの個数
 * @param step_num ステップ数
 * @param rows 原画像の横幅
 * @param cols 原画像の縦幅
 */
void freeExpandImages(EXPAND_ARRAY_TYPE expand_images, int expand, int filter_num, int step_num, int rows, int cols){
    for(int filter = 0 ; filter < filter_num ; filter++){
        for(int step = 0 ; step < step_num ; step++){
            int scaled_col = cols / std::pow(2, 3 - step);
            auto current_ref_expand = expand_images[filter][step][0];
            auto current_ref_org_expand = expand_images[filter][step][1];
            auto current_target_expand = expand_images[filter][step][2];
            auto current_target_org_expand = expand_images[filter][step][3];

            for (int d = -expand; d < scaled_col + expand; d++) {
                current_target_expand[d] -= expand;
                current_ref_expand[d] -= expand;
                free(current_ref_expand[d]);
                free(current_target_expand[d]);

                current_target_org_expand[d] -= expand;
                current_ref_org_expand[d] -= expand;
                free(current_ref_org_expand[d]);
                free(current_target_org_expand[d]);
            }

            current_target_expand -= expand;
            current_ref_expand -= expand;
            free(current_target_expand);
            free(current_ref_expand);

            current_target_org_expand -= expand;
            current_ref_org_expand -= expand;
            free(current_target_org_expand);
            free(current_ref_org_expand);
        }
    }

}

void freeHEVCExpandImage(EXPAND_ARRAY_TYPE expand_images, int expand, int filter_num, int step_num, int rows, int cols){
    for(int filter = 0 ; filter < filter_num ; filter++){
        for(int step = 3 ; step < step_num ; step++){
            auto current_ref_expand = expand_images[filter][step][0];
            auto current_ref_org_expand = expand_images[filter][step][1];
            auto current_target_expand = expand_images[filter][step][2];
            auto current_target_org_expand = expand_images[filter][step][3];

            for (int d = -4 * expand; d < 4 * (cols + expand); d++) {
                current_target_expand[d] -= 4 * expand;
                current_ref_expand[d] -= 4 * expand;
                free(current_ref_expand[d]);
                free(current_target_expand[d]);

                current_target_org_expand[d] -= 4 * expand;
                current_ref_org_expand[d] -= 4 * expand;
                free(current_ref_org_expand[d]);
                free(current_target_org_expand[d]);
            }

            current_target_expand -= 4 * expand;
            current_ref_expand -= 4 * expand;
            free(current_target_expand);
            free(current_ref_expand);

            current_target_org_expand -= 4 * expand;
            current_ref_org_expand -= 4 * expand;
            free(current_target_org_expand);
            free(current_ref_org_expand);
        }
    }
}

/**
 * @fn double w(double x)
 * @brief bicubic補間で使う重みを計算する
 * @param x 値
 * @return 重み
 */
double w(double x){
    double absx = fabs(x);

    if (absx <= 1.0) {
        return absx * absx * absx - 2 * absx * absx + 1;
    } else if (absx <= 2.0) {
        return - absx * absx * absx + 5 * absx * absx - 8 * absx + 4;
    } else {
        return 0.0;
    }
}

/**
 * @fn int img_ip(unsigned char **img, int xs, int ys, double x, double y, int mode)
 * @brief x,yの座標の補間値を返す
 * @param img 保管するための原画像
 * @param xs xの最大値
 * @param ys yの最大値
 * @param x 補間するx座標
 * @param y 補間するy座標
 * @param mode 補間モード。モードはImageUtil.hにenumで定義してある
 * @return 補間値
 */
int img_ip(unsigned char **img, cv::Rect rect, double x, double y, int mode){
    int x0, y0;          /* 補間点 (x, y) の整数部分 */
    double dx, dy;       /* 補間点 (x, y) の小数部分 */
    int nx, ny;          /* 双3次補間用のループ変数 */
    double val = 0.0, w(double);

    /*** 補間点(x, y)が原画像の領域外なら, 範囲外を示す -1 を返す ***/
    if (x < rect.x || x > rect.x + rect.width || y < rect.y || y > rect.y + rect.height ) {
        std::cout << "Error in img_ip!" << std::endl;
        std::cout << x << " " << y << std::endl;
        exit(-1);
    }

    /*** 補間点(x, y)の整数部分(x0, y0), 小数部分(dx, dy)を求める ***/
    x0 = (int) floor(x);
    y0 = (int) floor(y);
    dx = x - (double) x0;
    dy = y - (double) y0;

    if(x0 == (rect.width + rect.x)) x0 = (rect.width + rect.x);
    if(y0 == (rect.height + rect.y)) x0 = (rect.height + rect.y);

    /*** mode で指定された補間法に従って補間し，値を val に保存 ***/
    switch(mode) { /* mode = 0 : 最近傍, 1 : 双1次, 2 : 双3次 */
        case 0:  /* 最近傍補間法 --- 式(9.4) */
            if (dx <= 0.5 && dy <= 0.5)     val = img[x0  ][y0  ];
            else if (dx > 0.5 && dy <= 0.5) val = img[x0+1][y0  ];
            else if (dx <= 0.5 && dy > 0.5) val = img[x0  ][y0+1];
            else                            val = img[x0+1][y0+1];
            break;
        case 1: /* 双1次補間法 --- 式(9.8) */
            val = img[x0  ][y0  ] * (1.0 - dx) * (1.0 - dy) +
                  img[x0+1][y0  ] * dx         * (1.0 - dy) +
                  img[x0  ][y0+1] * (1.0 - dx) * dy         +
                  img[x0+1][y0+1] * dx         * dy;
            break;
        case 2: /* 3次補間法 --- 式(9.13) */
            val = 0.0;
            for(ny = -1 ; ny <= 2 ; ny++) {
                for(nx = -1 ; nx <= 2 ; nx++) {
                    val += img[x0 + nx][y0 + ny] * w(nx - dx) * w(ny - dy);
                }
            }
            break;
        default:
            break;
    }

    /*** リミッタを掛けて return ***/
    if (val >= 255.5) return 255;
    else if (val < -0.5) return 0;
    else return (int) (val + 0.5);
}

/**
 * @fn cv::Mat getReconstructionDivisionImage(cv::Mat image, std::vector<CodingTreeUnit *> ctu)
 * @brief CodingTreeをもらって、三角形を書いた画像を返す
 * @param image 下地
 * @param ctu CodingTree
 * @return 画像
 */
cv::Mat getReconstructionDivisionImage(cv::Mat image, std::vector<CodingTreeUnit *> ctu, int block_size_x, int block_size_y) {
    Reconstruction rec(image);
    rec.init(block_size_x, block_size_y, LEFT_DIVIDE);
    puts("");
    rec.reconstructionTriangle(ctu);
    std::vector<Point3Vec> hoge = rec.getTriangleCoordinateList();

    for(const auto foo : hoge) {
        drawTriangle(image, foo.p1, foo.p2, foo.p3, cv::Scalar(255, 255, 255));
    }

    return image;
}

/**
 * @fn cv::Mat getExpansionImage(cv::Mat image, int k)
 * @brief k倍に拡張子，expansion_size分回りを拡張した画像を生成する
 * @param image 画像
 * @param k サイズの倍数
 * @param expansion_size 拡張する画素数
 * @return 拡張した画像
 */
unsigned char** getExpansionImage(cv::Mat image, int k, int expansion_size, IP_MODE mode){
    if(mode == IP_MODE::BICUBIC) {
        int scaled_expansion_size = expansion_size + 2;
        auto **expansion_image_tmp = (unsigned char **) malloc(
                sizeof(unsigned char *) * (image.cols + scaled_expansion_size * 2));
        expansion_image_tmp += scaled_expansion_size;

        for (int i = -scaled_expansion_size; i < image.cols + scaled_expansion_size; i++) {
            expansion_image_tmp[i] = (unsigned char *) malloc(
                    sizeof(unsigned char) * (image.rows + scaled_expansion_size * 2));
            expansion_image_tmp[i] += scaled_expansion_size;
        }

        for (int y = 0; y < image.rows; y++) {
            for (int x = 0; x < image.cols; x++) {
                expansion_image_tmp[x][y] = M(image, x, y);
            }
        }

        // y方向に埋める
        for (int y = 0; y < image.rows; y++) {
            for (int x = -scaled_expansion_size; x < 0; x++) {
                expansion_image_tmp[x][y] = M(image, 0, y);
                expansion_image_tmp[image.cols - x - 1][y] = M(image, image.cols - 1, y);
            }
        }

        // x方向に埋める
        for (int y = -scaled_expansion_size; y < 0; y++) {
            for (int x = -scaled_expansion_size; x < image.cols + scaled_expansion_size; x++) {
                expansion_image_tmp[x][y] = expansion_image_tmp[x][0];
                expansion_image_tmp[x][image.rows - y - 1] = expansion_image_tmp[x][image.rows - 1];
            }
        }

        auto **expansion_image = (unsigned char **) malloc(
                sizeof(unsigned char *) * (k * image.cols + 2 * k * scaled_expansion_size));
        expansion_image += (k * scaled_expansion_size);

        for (int i = -k * expansion_size; i < k * image.cols + 2 * k * expansion_size; i++) {
            expansion_image[i] = (unsigned char *) malloc(
                    sizeof(unsigned char) * (k * image.rows + 2 * k * scaled_expansion_size));
            expansion_image[i] += (k * scaled_expansion_size);
        }

        for (int y = -k * expansion_size; y < k * image.rows + k * expansion_size; y++) {
            for (int x = -k * expansion_size; x < k * image.cols + k * expansion_size; x++) {
                expansion_image[x][y] = img_ip(expansion_image_tmp, cv::Rect(-expansion_size, -expansion_size, image.cols + 2 * expansion_size, image.rows + 2 * expansion_size), (double) x / k, (double) y / k, IP_MODE::BICUBIC);
            }
        }

        for(int i = -scaled_expansion_size ; i < image.cols + scaled_expansion_size ; i++){
            expansion_image_tmp[i] -= scaled_expansion_size;
            free(expansion_image_tmp[i]);
        }
        expansion_image_tmp -= scaled_expansion_size;
        free(expansion_image_tmp);

        return expansion_image;
    }else if(mode == IP_MODE::HEVC){

    }
}


unsigned char** getExpansionHEVCImage(cv::Mat image, int k, int expansion_size){
    // 引き伸ばし＋補間で使うため4画素余分に取る
    int scaled_expansion_size = expansion_size + 4;
    auto **expansion_image_tmp = (unsigned int **) malloc(sizeof(unsigned int *) * (image.cols + scaled_expansion_size * 2));
    expansion_image_tmp += scaled_expansion_size;

    for (int i = -scaled_expansion_size; i < image.cols + scaled_expansion_size; i++) {
        expansion_image_tmp[i] = (unsigned int *) malloc(sizeof(unsigned int) * (image.rows + scaled_expansion_size * 2));
        expansion_image_tmp[i] += scaled_expansion_size;
    }

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            expansion_image_tmp[x][y] = M(image, x, y);
        }
    }

    // y方向に埋める
    for (int y = 0; y < image.rows; y++) {
        for (int x = -scaled_expansion_size; x < 0; x++) {
            expansion_image_tmp[x][y] = M(image, 0, y);
            expansion_image_tmp[image.cols - x - 1][y] = M(image, image.cols - 1, y);
        }
    }

    // x方向に埋める
    for (int y = -scaled_expansion_size; y < 0; y++) {
        for (int x = -scaled_expansion_size; x < image.cols + scaled_expansion_size; x++) {
            expansion_image_tmp[x][y] = expansion_image_tmp[x][0];
            expansion_image_tmp[x][image.rows - y - 1] = expansion_image_tmp[x][image.rows - 1];
        }
    }
    auto **expansion_image = (unsigned int **) malloc(sizeof(unsigned int *) * (k * image.cols + 2 * k * scaled_expansion_size));
    expansion_image += (k * scaled_expansion_size);

    for (int i = -k * scaled_expansion_size; i < k * image.cols + k * scaled_expansion_size; i++) {
        expansion_image[i] = (unsigned int *) malloc(sizeof(unsigned int) * (k * image.rows + 2 * k * scaled_expansion_size));
        expansion_image[i] += (k * scaled_expansion_size);
    }

    // 左上に原画像つっこむ
    int width = k * image.cols + k * scaled_expansion_size;
    int height = k * image.rows + k * scaled_expansion_size;

#pragma omp parallel for
    for(int y = -k * scaled_expansion_size ; y < height ; y+=k){
        for(int x = -k * scaled_expansion_size ; x < width ; x+=k) {
            expansion_image[x][y] = expansion_image_tmp[x / k][y / k];

            if (x >= -k * expansion_size && x < k * image.cols + k * expansion_size) {
                // a
                expansion_image[x + 1][y] = -expansion_image_tmp[x / k - 3][y / k] + 4 * expansion_image_tmp[x / k - 2][y / k] - 10 * expansion_image_tmp[x / k - 1][y / k] + 58 * expansion_image_tmp[x / k][y / k] + 17 * expansion_image_tmp[x / k + 1][y / k] - 5 * expansion_image_tmp[x / k + 2][y / k] + expansion_image_tmp[x / k + 3][y / k];
                // b
                expansion_image[x + 2][y] = -expansion_image_tmp[x / k - 3][y / k] + 4 * expansion_image_tmp[x / k - 2][y / k] - 11 * expansion_image_tmp[x / k - 1][y / k] + 40 * expansion_image_tmp[x / k][y / k] + 40 * expansion_image_tmp[x / k + 1][y / k] - 11 * expansion_image_tmp[x / k + 2][y / k] + 4 * expansion_image_tmp[x / k + 3][y / k] - expansion_image_tmp[x / k + 4][y / k];
                // c
                expansion_image[x + 3][y] =  expansion_image_tmp[x / k - 2][y / k] - 5 * expansion_image_tmp[x / k - 1][y / k] + 17 * expansion_image_tmp[x / k][y / k] + 58 * expansion_image_tmp[x / k + 1][y / k] - 10 * expansion_image_tmp[x / k + 2][y / k] + 4 * expansion_image_tmp[x / k + 3][y / k] - expansion_image_tmp[x / k + 4][y / k];
                // d
                expansion_image[x][y + 1] = -expansion_image_tmp[x / k][y / k - 3] + 4 * expansion_image_tmp[x / k][y / k - 2] -10 * expansion_image_tmp[x / k][y / k - 1] + 58 * expansion_image_tmp[x / k][y / k] + 17 * expansion_image_tmp[x / k][y / k + 1] - 5 * expansion_image_tmp[x / k][y / k + 2] + expansion_image_tmp[x / k][y / k + 3];
                // h
                expansion_image[x][y + 2] = -expansion_image_tmp[x / k][y / k - 3] + 4 * expansion_image_tmp[x / k][y / k - 2] - 11 * expansion_image_tmp[x / k][y / k - 1] + 40 * expansion_image_tmp[x / k][y / k] + 40 * expansion_image_tmp[x / k][y / k + 1] - 11 * expansion_image_tmp[x / k][y / k + 2] + 4 * expansion_image_tmp[x / k][y / k + 3] - expansion_image_tmp[x / k][y / k + 4];
                // n
                expansion_image[x][y + 3] = expansion_image_tmp[x / k][y / k - 2] - 5 * expansion_image_tmp[x / k][y / k - 1] + 17 * expansion_image_tmp[x / k][y / k] + 58 * expansion_image_tmp[x / k][y / k + 1] - 10 * expansion_image_tmp[x / k][y / k + 2] + 4 * expansion_image_tmp[x / k][y / k + 3] - expansion_image_tmp[x / k][y / k + 4];
            }
        }
    }

    width = k * (image.cols + expansion_size);
    height = k * (image.rows + expansion_size);
#pragma omp parallel for
    for(int y = -k * expansion_size ; y < height ; y+=k) {
        for (int x = -k * expansion_size; x < width; x+=k) {
            // e
            expansion_image[x + 1][y + 1] = -expansion_image[x + 1][y - 3 * k] + 4 * expansion_image[x + 1][y - 2 * k] - 10 * expansion_image[x + 1][y - k] + 58 * expansion_image[x + 1][    y] + 17 * expansion_image[x + 1][    y + k] -  5 * expansion_image[x + 1][y + 2 * k] +     expansion_image[x + 1][y + 3 * k];
            // i
            expansion_image[x + 1][y + 2] = -expansion_image[x + 1][y - 3 * k] + 4 * expansion_image[x + 1][y - 2 * k] - 11 * expansion_image[x + 1][y - k] + 40 * expansion_image[x + 1][    y] + 40 * expansion_image[x + 1][    y + k] - 11 * expansion_image[x + 1][y + 2 * k] + 4 * expansion_image[x + 1][y + 3 * k] - expansion_image[x + 1][y + 4 * k];
            // p
            expansion_image[x + 1][y + 3] =                                          expansion_image[x + 1][y - 2 * k] -  5 * expansion_image[x + 1][y - k] + 17 * expansion_image[x + 1][    y] + 58 * expansion_image[x + 1][    y + k] - 10 * expansion_image[x + 1][y + 2 * k] + 4 * expansion_image[x + 1][y + 3 * k] - expansion_image[x + 1][y + 4 * k];
            // f
            expansion_image[x + 2][y + 1] = -expansion_image[x + 2][y - 3 * k] + 4 * expansion_image[x + 2][y - 2 * k] - 10 * expansion_image[x + 2][y - k] + 58 * expansion_image[x + 2][    y] + 17 * expansion_image[x + 2][    y + k] -  5 * expansion_image[x + 2][y + 2 * k] +     expansion_image[x + 2][y + 3 * k];
            // j
            expansion_image[x + 2][y + 2] = -expansion_image[x + 2][y - 3 * k] + 4 * expansion_image[x + 2][y - 2 * k] - 11 * expansion_image[x + 2][y - k] + 40 * expansion_image[x + 2][    y] + 40 * expansion_image[x + 2][    y + k] - 11 * expansion_image[x + 2][y + 2 * k] + 4 * expansion_image[x + 2][y + 3 * k] - expansion_image[x + 2][y + 4 * k];
            // q
            expansion_image[x + 2][y + 3] =                                          expansion_image[x + 2][y - 2 * k] -  5 * expansion_image[x + 2][y - k] + 17 * expansion_image[x + 2][    y] + 58 * expansion_image[x + 2][    y + k] - 10 * expansion_image[x + 2][y + 2 * k] + 4 * expansion_image[x + 2][y + 3 * k] - expansion_image[x + 2][y + 4 * k];
            // g
            expansion_image[x + 3][y + 1] = -expansion_image[x + 3][y - 3 * k] + 4 * expansion_image[x + 3][y - 2 * k] - 10 * expansion_image[x + 3][y - k] + 58 * expansion_image[x + 3][    y] + 17 * expansion_image[x + 3][    y + k] -  5 * expansion_image[x + 3][y + 2 * k] +     expansion_image[x + 3][y + 3 * k];
            // k
            expansion_image[x + 3][y + 2] = -expansion_image[x + 3][y - 3 * k] + 4 * expansion_image[x + 3][y - 2 * k] - 11 * expansion_image[x + 3][y - k] + 40 * expansion_image[x + 3][    y] + 40 * expansion_image[x + 3][    y + k] - 11 * expansion_image[x + 3][y + 2 * k] + 4 * expansion_image[x + 3][y + 3 * k] - expansion_image[x + 3][y + 4 * k];
            // r
            expansion_image[x + 3][y + 3] =                                          expansion_image[x + 3][y - 2 * k] -  5 * expansion_image[x + 3][y - k] + 17 * expansion_image[x + 3][    y] + 58 * expansion_image[x + 3][    y + k] - 10 * expansion_image[x + 3][y + 2 * k] + 4 * expansion_image[x + 3][y + 3 * k] - expansion_image[x + 3][y + 4 * k];

            for(int j = 1 ; j < 4 ; j++){
                for(int i = 1 ; i < 4 ; i++){
                    expansion_image[x + i][y + j] = (expansion_image[x + i][y + j] / 64 + 32) / 64;
                }
            }
        }
    }

    //四捨五入(0～255へ)
#pragma omp parallel for
    for(int y = - k * expansion_size ; y < height ; y+=k){
        for(int x = - k * expansion_size ; x < width ; x+=k){
            // a
            expansion_image[x + 1][    y] = (expansion_image[x + 1][    y] + 32) / 64;
            // b
            expansion_image[x + 2][    y] = (expansion_image[x + 2][    y] + 32) / 64;
            // c
            expansion_image[x + 3][    y] = (expansion_image[x + 3][    y] + 32) / 64;
            // d
            expansion_image[x    ][y + 1] = (expansion_image[x    ][y + 1] + 32) / 64;
            // h
            expansion_image[x    ][y + 2] = (expansion_image[x    ][y + 2] + 32) / 64;
            // n
            expansion_image[x    ][y + 3] = (expansion_image[x    ][y + 3] + 32) / 64;
        }
    }
    unsigned char **ret = (unsigned char **)malloc(sizeof(unsigned char *) * k * (image.cols + 2 * scaled_expansion_size));
    ret += k * scaled_expansion_size;
#pragma omp parallel for
    for(int x = -k * scaled_expansion_size ; x < k * (image.cols + scaled_expansion_size) ; x++) {
        ret[x] = (unsigned char *)malloc(sizeof(unsigned char) * k * (image.rows + 2 * scaled_expansion_size));
        ret[x] += k * scaled_expansion_size;
    }
#pragma omp parallel for
    for(int y = -k * expansion_size ; y < k * (image.rows + expansion_size) ; y++){
        for(int x = -k * expansion_size ; x < k * (image.cols + expansion_size) ; x++){
            ret[x][y] = expansion_image[x][y];
        }
    }
    for(int y = -k * expansion_size ; y < k * (image.rows + expansion_size) ; y++){
        for(int x = -k * scaled_expansion_size ; x <= -k * expansion_size ; x++){
            ret[x][y] = ret[-k * expansion_size][y];
            ret[k*(image.cols + scaled_expansion_size + expansion_size) + x - 1][y] = ret[k * image.cols - 1][y];
        }
    }
    for(int y = -k * scaled_expansion_size ; y < -k * expansion_size ; y++){
        for(int x = -k * scaled_expansion_size ; x < k * (image.cols + scaled_expansion_size); x++){
            ret[x][y] = ret[x][-k * expansion_size + 1];
            ret[x][k * (image.rows + scaled_expansion_size + expansion_size) + y - 1] = ret[x][k * (image.rows - 1 + expansion_size)];
        }
    }

    for(int i = - k * scaled_expansion_size; i < (k * image.cols + k * scaled_expansion_size) ; i++){
        expansion_image[i] -= k * scaled_expansion_size;
        free(expansion_image[i]);
    }
    expansion_image -= k * scaled_expansion_size;
    free(expansion_image);

    for(int i = -scaled_expansion_size ; i < (image.cols + scaled_expansion_size) ; i++) {
        expansion_image_tmp[i] -= scaled_expansion_size;
        free(expansion_image_tmp[i]);
    }
    expansion_image_tmp -= scaled_expansion_size;
    free(expansion_image_tmp);

    std::cout << "scaled_expantion_size:" << k * scaled_expansion_size << std::endl;
    return ret;
}

cv::Mat getExpansionMatHEVCImage(cv::Mat image, int k, int expansion_size){
    unsigned char **expansion_image = getExpansionHEVCImage(image, k, expansion_size);

    cv::Mat out = cv::Mat::zeros(k * (image.rows + 2 * expansion_size), k * (image.cols + 2 * expansion_size), CV_8UC3);

    for(int y = 0 ; y < k * (image.rows + 2 * expansion_size) ; y++){
        for(int x = 0 ; x < k * (image.cols + 2 * expansion_size) ; x++){
            R(out, x, y) = expansion_image[x - k * expansion_size][y - k * expansion_size];
            B(out, x, y) = expansion_image[x - k * expansion_size][y - k * expansion_size];
            G(out, x, y) = expansion_image[x - k * expansion_size][y - k * expansion_size];
        }
    }

    return out;
}

/**
 * @fn cv::Mat getExpandImage(cv::Mat image, int k)
 * @brief k倍に拡張子，expansion_size分回りを拡張した画像(Mat)を生成する
 * @param image 画像
 * @param k サイズの倍数
 * @param expansion_size 拡張する画素数
 * @return 拡張した画像
 */
cv::Mat getExpansionMatImage(cv::Mat &image, int k, int expansion_size, IP_MODE mode){
    unsigned char **img;
    int scaled_expand_size = expansion_size + 2; // 折返しの分
    if((img = (unsigned char **)malloc(sizeof(unsigned char *) * (image.cols + 2 * (scaled_expand_size)))) == NULL) {
        fprintf(stderr, "malloc error");
        exit(1);
    }
    img += scaled_expand_size;

    for(int x = -scaled_expand_size ; x < image.cols + scaled_expand_size ; x++) {
        img[x] = (unsigned char *)malloc(sizeof(unsigned char) * (image.rows + 2 * (scaled_expand_size)));
        img[x] += scaled_expand_size;
    }

    for(int y = 0 ; y < image.rows ; y++){
        for(int x = 0 ; x < image.cols ; x++) {
            img[x][y] = M(image, x, y);
        }
    }

    for(int y = 0 ; y < image.rows ; y++){
        for(int x = -scaled_expand_size ; x < 0 ; x++) {
            img[x][y] = M(image, 0, y);
            img[image.cols + scaled_expand_size + x][y] = M(image, (int)(image.cols - 1), y);
        }
    }

    for (int y = -scaled_expand_size ; y < 0 ; y++) {
        for (int x = -scaled_expand_size ; x < image.cols + scaled_expand_size; x++) {
            img[x][y] = img[x][0];
            img[x][image.rows + expansion_size  + y] = img[x][image.rows - 1];
        }
    }

    cv::Mat expansion_image = cv::Mat::zeros(image.rows * k + 2 * k * expansion_size, image.cols * k + 2 * k * expansion_size, CV_8UC3);

    if(mode == IP_MODE::BICUBIC) {
        for (int y = 0; y < expansion_image.rows; y++) {
            for (int x = 0; x < expansion_image.cols; x++) {
                R(expansion_image, x, y) = img_ip(img, cv::Rect(-expansion_size, -expansion_size, (image.cols + 2 * expansion_size), (image.rows + 2 * expansion_size)), x / (double) k - expansion_size, y / (double) k - expansion_size, 2);
                G(expansion_image, x, y) = img_ip(img, cv::Rect(-expansion_size, -expansion_size, (image.cols + 2 * expansion_size), (image.rows + 2 * expansion_size)), x / (double) k - expansion_size, y / (double) k - expansion_size, 2);
                B(expansion_image, x, y) = img_ip(img, cv::Rect(-expansion_size, -expansion_size, (image.cols + 2 * expansion_size), (image.rows + 2 * expansion_size)), x / (double) k - expansion_size, y / (double) k - expansion_size, 2);
            }
        }
    }else if(mode == IP_MODE::BILINEAR){

        for (int y = 0; y < expansion_image.rows; y++) {
            for (int x = 0; x < expansion_image.cols; x++) {
                R(expansion_image, x, y) = img_ip(img, cv::Rect(-expansion_size, -expansion_size, (image.cols + 2 * expansion_size), (image.rows + 2 * expansion_size)), x / (double) k - expansion_size, y / (double) k - expansion_size, 1);
                G(expansion_image, x, y) = img_ip(img, cv::Rect(-expansion_size, -expansion_size, (image.cols + 2 * expansion_size), (image.rows + 2 * expansion_size)), x / (double) k - expansion_size, y / (double) k - expansion_size, 1);
                B(expansion_image, x, y) = img_ip(img, cv::Rect(-expansion_size, -expansion_size, (image.cols + 2 * expansion_size), (image.rows + 2 * expansion_size)), x / (double) k - expansion_size, y / (double) k - expansion_size, 1);
            }
        }
    }

    for (int x = -scaled_expand_size; x < image.cols + scaled_expand_size; x++) {
        img[x] -= scaled_expand_size;
        free(img[x]);
    }
    img -= scaled_expand_size;
    free(img);

    return expansion_image;
}

/**
 * @fn void freeImage(unsigned char **image, cv::Size image_size, int expansion_size)
 * @brief 拡張画像をfreeする
 * @param image 画素値が格納されていおり，拡張されている画素配列
 * @param image_size 画像のサイズ
 * @param expansion_size 拡張サイズ
 */
void freeImage(unsigned char **image, cv::Size image_size, int expansion_size){
    for(int i = -expansion_size ; i < image_size.height + expansion_size ; i++){
        image[i] -= expansion_size;
        free(image[i]);
    }
    image -= expansion_size;
    free(image);
}

/**
 * @fn bool isMyTriangle(const CodingTreeUnit* ctu, int triangle_index)
 * @brief CodingTreeUnitをさかのぼって、triangle_indexが親に含まれているか確認する
 * @param ctu CTUのカレントノード
 * @param triangle_index 調べるtriangle_index
 * @return 存在していた場合true, 存在しない場合はfalseを返す
 */
bool isMyTriangle(const CodingTreeUnit* ctu, int triangle_index){
    if(ctu == nullptr) return false;

    while(ctu != nullptr) {
        if(triangle_index == ctu->triangle_index) return true;
        ctu = ctu->parentNode;
    }

    return false;
}

/**
 * @fn std::vector<cv::Point2f> getPixelsInTriangle(const Point3Vec& triangle)
 * @brief 三角パッチ内に含まれる画素を返す
 * @param triangle 三角形
 * @param area_flag 境界線がどちらを表すかのフラグ
 * @param triangle_index 三角形の番号
 * @return 画素の集合
 */
std::vector<cv::Point2f> getPixelsInTriangle(const Point3Vec& triangle, const std::vector<std::vector<int>>& area_flag, int triangle_index, CodingTreeUnit* ctu, int block_size_x, int block_size_y){
    std::vector<cv::Point2f> pixels_in_triangle;
    cv::Point2f p0 = triangle.p1, p1 = triangle.p2, p2 = triangle.p3;

    int sx = ceil(std::min({p0.x, p1.x, p2.x}));
    int lx = floor(std::max({p0.x, p1.x, p2.x}));
    int sy = ceil(std::min({p0.y, p1.y, p2.y}));
    int ly = floor(std::max({p0.y, p1.y, p2.y}));

    int width = lx - sx + 1;
    int height = ly - sy + 1;

    pixels_in_triangle.clear();
    cv::Point2i xp;

    for (int j = sy ; j <= ly ; j++) {
        for (int i = sx ; i <= lx; i++) {
            xp.x = i;
            xp.y = j;
            if (isInTriangle(triangle, xp) == 1) {
                int pixel_index = area_flag[xp.x % block_size_x][xp.y % block_size_y];
                if(pixel_index == -1 || pixel_index == triangle_index || isMyTriangle(ctu, pixel_index)){
                    pixels_in_triangle.emplace_back(xp);
                }
            }
        }
    }
    return pixels_in_triangle;
}

/**
 * @fn cv::Mat getAppliedLPFImage(cv::Mat &image)
 * @brief 1:2:1のLPFを適用したMatを返す
 * @param image
 * @return フィルタを適用した画像
 */
cv::Mat getAppliedLPFImage(const cv::Mat &image){
    unsigned char **expansion_image = getExpansionImage(image, 1, 2);

    double weight[3][3];
    weight[0][0] = 0.0625;
    weight[1][0] = 0.125;
    weight[2][0] = 0.0625;

    weight[0][1] = 0.125;
    weight[1][1] = 0.25;
    weight[2][1] = 0.125;

    weight[0][2] = 0.0625;
    weight[1][2] = 0.125;
    weight[2][2] = 0.0625;

    cv::Mat out = cv::Mat::zeros(image.size(), CV_8UC3);
    for(int y = 0 ; y < out.rows ; y++){
        for(int x = 0 ; x < out.cols ; x++){
            double sum = 0.0;
            for(int j = 0 ; j < 3; j++){
                for(int i = 0 ; i < 3 ; i++){
                    sum += weight[i][j] * expansion_image[x + i - 1][y + i - 1];
                }
            }
            sum = round(sum);
            R(out, x, y) = (unsigned char)sum;
            G(out, x, y) = (unsigned char)sum;
            B(out, x, y) = (unsigned char)sum;
        }
    }

    return out;
}
