#include <cmath>

//
// Created by takahiro on 2019/07/19.
//

#include "../includes/SquareDivision.h"
#include <opencv2/core.hpp>
#include <iostream>
#include "../includes/Utils.h"
#include "../includes/ME.hpp"
#include "../includes/CodingTreeUnit.h"
#include "../includes/Reconstruction.h"
#include "../includes/Encode.h"
#include <set>
#include <vector>
#include <utility>
#include <algorithm>
#include <queue>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <random>
#include "../includes/ImageUtil.h"
#include "../includes/Analyzer.h"
#include "../includes/Flags.h"

SquareDivision::SquareDivision(const cv::Mat &refImage, const cv::Mat &targetImage, const cv::Mat &refGaussImage) : target_image(targetImage),
                                                                                                                    ref_image(refImage), ref_gauss_image(refGaussImage) {}



/**
 * @fn void SquareDivision::initSquare(int block_size_x, int block_size_y, int _divide_steps, int _qp, int divide_flag)
 * @brief 四角形を初期化する
 * @param[in] _block_size_x
 * @param[in] _block_size_y
 * @param[in] _divide_steps
 * @param[in] _qp
 * @param[in] _divide_flag
 */
void SquareDivision::initSquare(int _block_size_x, int _block_size_y, int _divide_steps, int _qp, int divide_flag) {
    block_size_x = _block_size_x;
    block_size_y = _block_size_y;
    qp = _qp;
    int block_num_x = ceil((double)target_image.cols / (block_size_x));    //x方向のブロック数
    int block_num_y = ceil((double)target_image.rows / (block_size_y));    //y方向のブロック数
    divide_steps = _divide_steps;
    coded_picture_num = 0;

    corners.clear();
    covered_square.clear();
    squares.clear();

    /*
     *  p1                     p2
     *   *---------------------*
     *   |                     |
     *   |                     |
     *   |                     |
     *   |                     |
     *   |                     |
     *   *---------------------*
     *  p3                     p4
     *
     */

    corner_flag.resize(static_cast<unsigned long>(ref_image.rows));
    for(int i = 0 ; i < ref_image.rows ; i++) {
        corner_flag[i].resize(static_cast<unsigned long>(ref_image.cols));
    }

    for(int y = 0 ; y < ref_image.rows ; y++) {
        for(int x = 0 ; x < ref_image.cols ; x++) {
            corner_flag[y][x] = -1;
        }
    }

    previousMvList.emplace_back();
    // すべての頂点を入れる
    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {
        //y列目の上側の頂点を入れる
        for (int block_x = 0 ; block_x < block_num_x; block_x++) {
            int nx = block_x * (block_size_x);    //ブロックの左上のx座標
            int ny = block_y * (block_size_y);    //ブロックの左上のy座標

            corners.emplace_back(nx, ny);
            corner_flag[ny][nx] = static_cast<int>(corners.size() - 1);

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());

            nx = (block_x + 1) * (block_size_x) - 1;   //ブロックの右上のx座標
            ny = (block_y) * (block_size_y);          //ブロックの右上のy座標

            corners.emplace_back(nx, ny);
            corner_flag[ny][nx] = static_cast<int>(corners.size() - 1);

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
        }

        //y列目の下側の頂点を入れる
        for (int block_x = 0 ; block_x < block_num_x; block_x++) {
            int nx = block_x * (block_size_x);              //ブロックの左下のx座標
            int ny = (block_y + 1) * (block_size_y) - 1;    //ブロックの左下のy座標

            corners.emplace_back(nx, ny);
            corner_flag[ny][nx] = static_cast<int>(corners.size() - 1);

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());

            nx = (block_x + 1) * (block_size_x) - 1;    //ブロックの右下のx座標
            ny = (block_y + 1) * (block_size_y) - 1;    //ブロックの右下のy座標

            corners.emplace_back(nx, ny);
            corner_flag[ny][nx] = static_cast<int>(corners.size() - 1);

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
        }
    }

    // 過去のMVを残すやつを初期化
    for(auto node : previousMvList[coded_picture_num]) {
        node->node1 = node->node2 = node->node3 = node->node4 = nullptr;
        node->mv1 = cv::Point2f(0.0, 0.0);
    }

    std::cout << "block_num_y:" << block_num_y << std::endl;
    std::cout << "block_num_x:" << block_num_x << std::endl;

    covered_square.resize(static_cast<unsigned long>(4 * (block_num_x) * (block_num_y)));  //頂点の個数はブロック数×4

    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {
        for(int block_x = 0 ; block_x < block_num_x ; block_x++) {
            //頂点番号
            int p1_idx;
            int p2_idx;
            int p3_idx;
            int p4_idx;
            p1_idx = 2 * block_x + 4 * block_num_x * block_y;
            p2_idx = p1_idx + 1;
            p3_idx = p1_idx + 2 * block_num_x;
            p4_idx = p3_idx + 1;

            int squareIndex = insertSquare(p1_idx, p2_idx, p3_idx, p4_idx);
            addCoveredSquare(p1_idx, p2_idx, p3_idx, p4_idx, squareIndex); // p1/p2/p3はsquareIndex番目の四角形に含まれている
        }
    }

    for(int i = 0 ; i < isCodedSquare.size() ; i++) {
        isCodedSquare[i] = false;
    }

    predicted_buf.emplace_back(cv::Mat::zeros(ref_image.size()/8, CV_8UC3));
    predicted_buf.emplace_back(cv::Mat::zeros(ref_image.size()/4, CV_8UC3));
    predicted_buf.emplace_back(cv::Mat::zeros(ref_image.size()/2, CV_8UC3));
    predicted_buf.emplace_back(cv::Mat::zeros(ref_image.size(), CV_8UC3));

    ref_images = getRefImages(ref_image, ref_gauss_image);
    target_images = getTargetImages(target_image);

    // この1bitは手法フラグ(warpingかtranslation),もう1bitはマージフラグ分です
    if(PRED_MODE == NEWTON && !GAUSS_NEWTON_TRANSLATION_ONLY) flags_code++;
    if (MERGE_MODE) flags_code++;

    int expansion_size = SERACH_RANGE;
    int scaled_expansion_size = expansion_size + 2;
    if(HEVC_REF_IMAGE) expansion_ref = getExpansionMatHEVCImage(ref_image, 4, expansion_size);
    else expansion_ref = getExpansionMatImage(ref_image, 4, scaled_expansion_size);

    ref_hevc = getExpansionHEVCImage(ref_image, 4, expansion_size);

    cv::Mat tmp_mat = getExpansionMatImage(ref_image, 1, scaled_expansion_size);

    expansion_ref_uchar = (unsigned char **)malloc(sizeof(unsigned char *) * tmp_mat.cols);
    expansion_ref_uchar += scaled_expansion_size;
    for(int x = 0; x < tmp_mat.cols ; x++){
        expansion_ref_uchar[x - scaled_expansion_size] = (unsigned char *)malloc(sizeof(unsigned char) * tmp_mat.rows);
        expansion_ref_uchar[x - scaled_expansion_size] += scaled_expansion_size;
    }

    for(int y = 0 ; y < tmp_mat.rows ; y++){
        for(int x = 0 ; x < tmp_mat.cols ; x++){
            expansion_ref_uchar[x - scaled_expansion_size][y - scaled_expansion_size] = M(tmp_mat, x, y);
        }
    }

    //参照ブロックを入れる
    int square_index = 0;

    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {                     //
        for(int block_x = 0 ; block_x < block_num_x ; block_x++) {                 //   ---------------     ---------------     ---------------
            //頂点番号                                                             //   |             |     |             |     |             |
            int p1_idx;                                                            //   |             |     |             |     |             |
            int p2_idx;                                                            //   |             |     |             |     |             |
            int p3_idx;                                                            //   |          5●|     |          4●|     |3●          |
                                                                                   //   ---------------     ---------------     ---------------
            p1_idx = 2 * block_x + 4 * block_num_x * block_y;                      //
            p2_idx = p1_idx + 1;                                                   //   ---------------     ---------------
            p3_idx = p1_idx + 2 * block_num_x;                                     //   |             |     | p1       p2 |
                                                                                   //   |             |     |             |
            cv::Point2f sp = corners[p3_idx];                                      //   |             |     |             |
            sp.x--; sp.y++;                                                        //   |          2●|     | p3          |
                                                                                   //   ---------------     ---------------
            int sp1_idx = getCornerIndex(sp);                                      //
            int sp2_idx, sp3_idx, sp4_idx, sp5_idx ;                               //   ---------------
            if(sp1_idx != -1) {                                                    //   |          1●|
                // 1の頂点を入れる                    　             　　　　　　　//   |             |
                reference_block_list[square_index].emplace_back(sp1_idx);          //   |             |
            }                                                                      //   |             |
            // 2の頂点を入れる                                       　　　　　　　//   ---------------
            sp.y--;
            sp2_idx = getCornerIndex(sp);
            if(sp2_idx != -1) {
                reference_block_list[square_index].emplace_back(sp2_idx);
            }
            // 3の頂点を入れる
            sp = corners[p2_idx];
            sp.x++; sp.y--;
            sp3_idx = getCornerIndex(sp);
            if(sp3_idx != -1) {
                reference_block_list[square_index].emplace_back(sp3_idx);
            }
            // 4の頂点を入れる
            sp.x--;
            sp4_idx = getCornerIndex(sp);
            if(sp4_idx != -1) {
                reference_block_list[square_index].emplace_back(sp4_idx);
            }
            // 5の頂点を入れる
            sp = corners[p1_idx];
            sp.x--; sp.y--;
            sp5_idx = getCornerIndex(sp);
            if(sp5_idx != -1) {
                reference_block_list[square_index].emplace_back(sp5_idx);
            }
            if(sp2_idx != -1) {
                merge_reference_block_list[square_index].emplace_back(sp2_idx);
            }
            if(sp4_idx != -1) {
                merge_reference_block_list[square_index].emplace_back(sp4_idx);
            }
            if(sp3_idx != -1) {
                merge_reference_block_list[square_index].emplace_back(sp3_idx);
            }
            if(sp1_idx != -1) {
                merge_reference_block_list[square_index].emplace_back(sp1_idx);
            }
            if(sp5_idx != -1) {
                merge_reference_block_list[square_index].emplace_back(sp5_idx);
            }
            square_index++;
        }
    }

//    for(int i = 0 ; i < reference_block_list.size() ; i++) {
//        std::cout << "square_index : " << i << std::endl << "reference_block : ";
//        for(auto rbl : reference_block_list[i]) {
//            std::set<int> tmp_s;
//            tmp_s = covered_square[rbl];
//            for(auto idx : tmp_s)
//                std::cout << idx << ", ";
//        }
//        std::cout << std::endl;
//    }
//
    std::cout << "reference_block_list.size = " << reference_block_list.size() << std::endl;
}

/**
 * @fn std::vector<Point4Vec> getSquareCoordinateList()
 * @brief 現在存在する四角形の集合(座標)を返す
 * @return 四角形の集合（座標）
 */
std::vector<Point4Vec> SquareDivision::getSquareCoordinateList() {
    std::vector<Point4Vec> vec;

    for(int i = 0 ; i < squares.size() ; i++) {
        if(!isCodedSquare[i]) continue;
        Square square = squares[i];
        vec.emplace_back(corners[square.p1_idx], corners[square.p2_idx], corners[square.p3_idx], corners[square.p4_idx]);
    }

    return vec;
}

/**
 * @fn std::vector<Square> SquareDivision::getSquareIndexList()
 * @brief 現在存在する四角形の集合(インデックス)を返す
 * @return 四角形の集合（インデックス）
 */
std::vector<Square> SquareDivision::getSquareIndexList() {
    std::vector<Square> v;
    for(int i = 0 ; i < squares.size() ; i++) {
        if(!isCodedSquare[i]) continue;
        v.emplace_back(squares[i]);
    }
    return v;
}

/**
 * @fn std::vector<Point4Vec> getAllSquareCoordinateList()
 * @brief 現在存在するすべての四角形の集合(座標)を返す（※論理削除されたパッチも含まれています）
 * @return 四角形の集合（座標）
 */
std::vector<Point4Vec> SquareDivision::getAllSquareCoordinateList() {
    std::vector<Point4Vec> vec;

    for(int i = 0 ; i < squares.size() ; i++) {
        Square square = squares[i];
        vec.emplace_back(corners[square.p1_idx], corners[square.p2_idx], corners[square.p3_idx], corners[square.p4_idx]);
    }

    return vec;
}

/**
 * @fn std::vector<Square> SquareDivision::getAllSquareIndexList()
 * @brief 現在存在する四角形の集合(インデックス)を返す（※論理削除されたパッチも含まれています）
 * @return 四角形の集合（インデックス）
 */
std::vector<Square> SquareDivision::getAllSquareIndexList() {
    std::vector<Square> v;
    for(int i = 0 ; i < squares.size() ; i++) {
        v.emplace_back(squares[i]);
    }
    return v;
}


std::vector<Point4Vec> SquareDivision::getSquares() {
    std::vector<Point4Vec> ss;

    cv::Point2f p1, p2, p3, p4;
    for(auto & square : squares){
        p1 = corners[square.p1_idx];
        p2 = corners[square.p2_idx];
        p3 = corners[square.p3_idx];
        p4 = corners[square.p4_idx];
        ss.emplace_back(Point4Vec(p1, p2, p3, p4));
    }

    return ss;
}

/**
 * @fn std::vector<cv::Point2f> SquareDivision::getCorners()
 * @brief 頂点の集合を返す
 * @return 頂点
 */
std::vector<cv::Point2f> SquareDivision::getCorners() {
    return corners;
}

/**
 * @fn int SquareDivision::insertSquare(int p1_idx, int p2_idx, int p3_idx, int p4_idx)
 * @brief 四角形を追加する
 * @param[in] p1_idx 頂点1の座標のインデックス
 * @param[in] p2_idx 頂点2の座標のインデックス
 * @param[in] p3_idx 頂点3の座標のインデックス
 * @param[in] p4_idx 頂点4の座標のインデックス
 * @return 挿入した四角形が格納されているインデックス
 */
int SquareDivision::insertSquare(int p1_idx, int p2_idx, int p3_idx, int p4_idx) {
    Square square(p1_idx, p2_idx, p3_idx, p4_idx, static_cast<int>(squares.size()));

    squares.emplace_back(square);
    isCodedSquare.emplace_back(false);
    square_gauss_results.emplace_back();
    square_gauss_results[square_gauss_results.size() - 1].residual_translation = -1.0;
    reference_block_list.emplace_back();
    merge_reference_block_list.emplace_back();

    return static_cast<int>(squares.size() - 1);
}

/**
 * @fn void SquareDivision::eraseSquare(int t_idx)
 * @brief 四角パッチに関わる情報を削除する
 * @param t_idx 四角パッチの番号
 */
void SquareDivision::eraseSquare(int s_idx){
    Square square = squares[s_idx];
    isCodedSquare.erase(isCodedSquare.begin() + s_idx);
    squares.erase(squares.begin() + s_idx);
    square_gauss_results.erase(square_gauss_results.begin() + s_idx);
    reference_block_list.erase(reference_block_list.begin() + s_idx);
    merge_reference_block_list.erase(merge_reference_block_list.begin() + s_idx);
}


/***
 * @fn void SquareDivision::addCoveredSquare(int p1_idx, int p2_idx, int p3_idx, int p4_idx, int square_no)
 * @brief ある頂点を含む四角形のインデックスの情報を更新する
 * @param[in] p1_idx 頂点1の座標のインデックス
 * @param[in] p2_idx 頂点2の座標のインデックス
 * @param[in] p3_idx 頂点3の座標のインデックス
 * @param[in] p4_idx 頂点4の座標のインデックス
 * @param[in] square_no 四角形のインデックス
 */
void SquareDivision::addCoveredSquare(int p1_idx, int p2_idx, int p3_idx, int p4_idx, int square_no) {
    covered_square[p1_idx] = square_no;
    covered_square[p2_idx] = square_no;
    covered_square[p3_idx] = square_no;
    covered_square[p4_idx] = square_no;
}

/**
 * @fn double SquareDivision::getDistance(const cv::Point2f &a, const cv::Point2f &b)
 * @brief 2点間の距離を返す
 * @param[in] a 点1ベクトル
 * @param[in] b 点2のベクトル
 * @return 2点間の距離（スカラー）
 */
double SquareDivision::getDistance(const cv::Point2f &a, const cv::Point2f &b){
    cv::Point2f v = a - b;
    return std::sqrt(v.x * v.x + v.y * v.y);
}


/**
 * @fn std::vector<Point4Vec> SquareDivision::getIdxCoveredSquareCoordinateList(int idx)
 * @brief 指定された頂点が含まれる四角形の集合を返す
 * @param[in] target_vertex_idx 頂点のインデックス
 * @return 四角形の集合(座標で返される)
 */
std::vector<Point4Vec> SquareDivision::getIdxCoveredSquareCoordinateList(int target_vertex_idx) {
//    std::set<int> same_corners = same_corner_list[target_vertex_idx];

    std::set<int> s;
    std::set<int> tmp_s;
//    for(auto same_corner : same_corners){
//        tmp_s = covered_square[same_corner];
//        for(auto idx : tmp_s) s.emplace(idx);
//    }
    std::vector<Point4Vec> v(s.size());

    for(auto square_idx : s) {
        Square square = squares[square_idx];
        v.emplace_back(corners[square.p1_idx], corners[square.p2_idx], corners[square.p3_idx], corners[square.p4_idx]);
    }

    return v;
}

/**
 * @fn std::vector<int> SquareDivision::getIdxCoveredSquareIndexList(int idx)
 * @brief 指定の頂点を含む四角形の集合（頂点番号）を返す
 * @param[in] idx 頂点のインデックス
 * @return 四角形の集合（座標）
 */
std::vector<int> SquareDivision::getIdxCoveredSquareIndexList(int target_vertex_idx) {
//    std::set<int> same_corners = same_corner_list[target_vertex_idx];

    std::set<int> s;
    std::set<int> tmp_s;
//    for(auto same_corner : same_corners){
//        tmp_s = covered_square[same_corner];
//        for(auto idx : tmp_s) s.emplace(idx);
//    }
    std::vector<int> v;

    for(auto square_idx : s) {
        v.emplace_back(square_idx);
    }

    std::sort(v.begin(), v.end());

    return v;
}


/**
 * @fn int SquareDivision::getOrAddCornerIndex(cv::Point2f p)
 * @brief 頂点が格納されているインデックスを返す。頂点が存在しない場合、その頂点を頂点集合に追加した後インデックスを返す
 * @param[in] p 追加する頂点の座標
 * @return 頂点番号
 */
int SquareDivision::getOrAddCornerIndex(cv::Point2f p) {
    if(corner_flag[(int)(p.y)][(int)(p.x)] != -1) return corner_flag[(int)(p.y)][(int)(p.x)]; //すでに頂点がある場合
    corners.emplace_back(p);
    covered_square.emplace_back();
    corner_flag[(int)(p.y)][(int)(p.x)] = static_cast<int>(corners.size() - 1);
    return static_cast<int>(corners.size() - 1);
}


/**
 * @fn int SquareDivision::getCornerIndex(cv::Point2f p)
 * @brief 頂点が格納されているインデックスを返す。頂点が存在しない場合、-1を返す
 * @param[in] 頂点の座標
 * @return 頂点番号
 */
int SquareDivision::getCornerIndex(cv::Point2f p) {
    if(0 <= p.x && p.x < 1920 && 0 <= p.y && p.y < 1024) {
        if (corner_flag[(int) (p.y)][(int) (p.x)] != -1)
            return corner_flag[(int) (p.y)][(int) (p.x)]; //すでに頂点がある場合
        return -1;
    }
    return -1;
}

/**
 * @fn void SquareDivision::eraseCornerFlag(Point4Vec s1, Point4Vec s2, Point4Vec s3, Point4Vec s4)
 * @brief 4分割後の不要な頂点が格納されているインデックスを-1で初期化する。
 * @param[in] 四角形の頂点の集合 s1
 * @param[in] 四角形の頂点の集合 s2
 * @param[in] 四角形の頂点の集合 s3
 * @param[in] 四角形の頂点の集合 s4
 *  ----   ----
 *  | s1| | s2|
 *  ----  ----
 *  ----   ----
 *  | s3| | s4|
 *  ----  ----
 */
void SquareDivision::eraseCornerFlag(Point4Vec s1, Point4Vec s2, Point4Vec s3, Point4Vec s4) {
    corner_flag[s1.p2.y][s1.p2.x] = -1;
    corner_flag[s1.p3.y][s1.p3.x] = -1;
    corner_flag[s1.p4.y][s1.p4.x] = -1;

    corner_flag[s2.p1.y][s2.p1.x] = -1;
    corner_flag[s2.p3.y][s2.p3.x] = -1;
    corner_flag[s2.p4.y][s2.p4.x] = -1;

    corner_flag[s3.p1.y][s3.p1.x] = -1;
    corner_flag[s3.p2.y][s3.p2.x] = -1;
    corner_flag[s3.p4.y][s3.p4.x] = -1;

    corner_flag[s4.p1.y][s4.p1.x] = -1;
    corner_flag[s4.p2.y][s4.p2.x] = -1;
    corner_flag[s4.p3.y][s4.p3.x] = -1;
}

/**
 * @fn void SquareDivision::addCornerAndSquare(Square square, int square_index)
 * @brief 長方形を正方形に分割する
 * @param square
 * @param type
 * @return
 */
void SquareDivision::addCornerAndSquare(Square square){

    cv::Point2f p1 = corners[square.p1_idx];
    cv::Point2f p2 = corners[square.p2_idx];
    cv::Point2f p3 = corners[square.p3_idx];    //p4は使わないので宣言していない

    cv::Point2f x = (p2 - p1) / 2.0;
    x.x -= 0.5;                            //  a       e g      b
    cv::Point2f a = p1;                    //   -----------------
    cv::Point2f c = p3;                    //   |               |
                                           //   |               |
    cv::Point2f e = a + x;                 //   -----------------
    cv::Point2f f = c + x;                 //  c       f h      d
    cv::Point2f g = e;    g.x++;           //
    cv::Point2f h = f;    h.x++;           //

    int e_idx = getOrAddCornerIndex(e);
    int f_idx = getOrAddCornerIndex(f);
    int g_idx = getOrAddCornerIndex(g);
    int h_idx = getOrAddCornerIndex(h);

    int a_idx = square.p1_idx;
    int b_idx = square.p2_idx;
    int c_idx = square.p3_idx;
    int d_idx = square.p4_idx;

    int s1_idx = insertSquare(a_idx, e_idx, c_idx, f_idx);
    int s2_idx = insertSquare(g_idx, b_idx, h_idx, d_idx);

    addCoveredSquare(a_idx, e_idx, c_idx, f_idx, s1_idx);
    addCoveredSquare(g_idx, b_idx, h_idx, d_idx, s2_idx);
}

/**
 * @fn bool SquareDivision::split(std::vector<std::vector<std::vector<unsigned char **>>> expand_images, CodingTreeUnit* ctu, CollocatedMvTree* cmt, Point4Vec square, int square_index, int type, int steps)
 * @brief 与えられた四角形を分割するか判定し，必要な場合は分割を行う
 * @details この関数は再帰的に呼び出され，そのたびに分割を行う
 * @param gaussRefImage ガウス・ニュートン法の参照画像
 * @param ctu CodingTreeUnitのrootノード
 * @oaran cmt 時間予測用のCollocatedMvTreeのノード(collocatedmvtree→cmt)
 * @param square 四角形の各点の座標
 * @param square_index 四角形のindex
 * @param square_number 4つに分割したときの四角形の番号　0:左上, 1:右上, 2:左下, 3:右下, 4:初期ブロック(ctu_width * ctu_height の四角形)
 * @param steps 分割回数
 * @return 分割した場合はtrue, そうでない場合falseを返す
 */
bool SquareDivision::split(std::vector<std::vector<std::vector<unsigned char **>>> expand_images, CodingTreeUnit* ctu, CollocatedMvTree* cmt, Point4Vec square, int square_index, int square_number, int steps) {


    double RMSE_before_subdiv = 0.0;
    double error_warping, error_translation;
    double cost_warping, cost_translation, cost_before_subdiv;
    int code_length_warping, code_length_translation, code_length;
    std::vector<cv::Point2f> mvd_warping, mvd_translation, mvd;
    int selected_index_warping, selected_index_translation, selected_index;
    MV_CODE_METHOD method_warping, method_translation, method_flag;
    cv::Point2f p1 = square.p1;
    cv::Point2f p2 = square.p2;
    cv::Point2f p3 = square.p3;
    cv::Point2f p4 = square.p4;

    Point4Vec targetSquare(p1, p2, p3, p4);
    int square_size = 0;
    bool translation_flag;

    std::vector<cv::Point2f> dummy;
    std::vector<cv::Point2f> gauss_result_warping;
    cv::Point2f gauss_result_translation;

    int warping_limit = 2; // 6: 64×64まで  4:32×32まで  2:16×16まで  0:8×8まで

    if(cmt == nullptr) {
        cmt = previousMvList[0][square_index];
    }

    if(square_gauss_results[square_index].residual_translation > 0) {
        GaussResult result_before = square_gauss_results[square_index];
        gauss_result_warping = result_before.mv_warping;
        gauss_result_translation = result_before.mv_translation;
        square_size = result_before.square_size;
        translation_flag = result_before.translation_flag;
        error_translation = result_before.residual_translation;
        error_warping = result_before.residual_warping;
        ctu->error_bm = result_before.residual_bm;
        if(translation_flag) {
            ctu->error_newton = result_before.residual_translation;
            RMSE_before_subdiv = result_before.residual_translation;
        }
        else {
            ctu->error_newton = result_before.residual_warping;
            RMSE_before_subdiv = result_before.residual_warping;
        }

        if(PRED_MODE == NEWTON) {
            std::tie(cost_translation, code_length_translation, mvd_translation, selected_index_translation,
                     method_translation) = getMVD(
                    {gauss_result_translation, gauss_result_translation, gauss_result_translation}, error_translation,
                    square_index, square_number, cmt->mv1, ctu, true, dummy, steps);
#if !GAUSS_NEWTON_TRANSLATION_ONLY
            std::tie(cost_warping, code_length_warping, mvd_warping, selected_index_warping, method_warping) = getMVD(
                    gauss_result_warping, error_warping,
                    square_index, square_number, cmt->mv1, ctu, false, dummy, steps);
#endif
            if (cost_translation < cost_warping || (steps < warping_limit) || GAUSS_NEWTON_TRANSLATION_ONLY) {
                cost_before_subdiv = cost_translation;
                code_length = code_length_translation;
                mvd = mvd_translation;
                selected_index = selected_index_translation;
                method_flag = method_translation;
                square_gauss_results[square_index].translation_flag = true;
                square_gauss_results[square_index].method = method_translation;
                translation_flag = true;
            } else {
                cost_before_subdiv = cost_warping;
                code_length = code_length_warping;
                mvd = mvd_warping;
                selected_index = selected_index_warping;
                method_flag = method_warping;
                square_gauss_results[square_index].translation_flag = false;
                square_gauss_results[square_index].method = method_warping;
                translation_flag = false;
            }

//        if(square_gauss_results[square_index].translation_flag) {
//            std::tie(cost_before_subdiv, code_length, mvd, selected_index, method_flag) = getMVD(
//                    {gauss_result_translation, gauss_result_translation, gauss_result_translation}, error_translation,
//                    square_index, square_number, cmt->mv1, ctu, true, dummy, steps);
//        }else{
//            std::tie(cost_before_subdiv, code_length, mvd, selected_index, method_flag) = getMVD(
//                    square_gauss_results[square_index].mv_warping, error_warping,
//                    square_index, square_number, cmt->mv1, ctu, false, dummy, steps);
//        }
        } else if(PRED_MODE == BM) {
            std::tie(cost_before_subdiv, code_length, mvd, selected_index, method_flag) = getMVD(
                    {gauss_result_translation, gauss_result_translation, gauss_result_translation}, error_translation,
                    square_index, square_number, cmt->mv1, ctu, true, dummy, steps);
        }
    }else {
        if(PRED_MODE == NEWTON) {
            if(GAUSS_NEWTON_INIT_VECTOR) {
                std::vector<cv::Point2f> tmp_bm_mv;
                std::vector<double> tmp_bm_errors;
//                std::tie(tmp_bm_mv, tmp_bm_errors) = fullpellBlockMatching(square, target_image, expansion_ref, square_index, ctu);
                std::tie(gauss_result_warping, gauss_result_translation, error_warping, error_translation, square_size) = Square_GaussNewton(
                        ref_images, target_images, expand_images, targetSquare, square_index, ctu, tmp_bm_mv[2], ref_hevc);
#if USE_BM_translation_MV
                gauss_result_translation = tmp_bm_mv[2];
                error_translation = tmp_bm_errors[2];
#endif
            }else{
                std::tie(gauss_result_warping, gauss_result_translation, error_warping, error_translation, square_size) = Square_GaussNewton(
                        ref_images, target_images, expand_images, targetSquare, square_index, ctu, cv::Point2f(-1000, -1000), ref_hevc);
            }

            square_gauss_results[square_index].mv_translation = gauss_result_translation;
            square_gauss_results[square_index].mv_warping = gauss_result_warping;
            square_gauss_results[square_index].residual_translation = error_translation;
            square_gauss_results[square_index].residual_warping = error_warping;
            square_gauss_results[square_index].square_size = square_size;

            std::tie(cost_translation, code_length_translation, mvd_translation, selected_index_translation, method_translation) = getMVD(
                    {gauss_result_translation, gauss_result_translation, gauss_result_translation}, error_translation,
                    square_index, square_number, cmt->mv1, ctu, true, dummy, steps);
#if !GAUSS_NEWTON_TRANSLATION_ONLY
            std::tie(cost_warping, code_length_warping, mvd_warping, selected_index_warping, method_warping)= getMVD(
                    gauss_result_warping, error_warping,
                    square_index, square_number, cmt->mv1, ctu, false, dummy, steps);
#endif
//            std::cout << "cost_translation : " << cost_translation << ", cost_warping : " << cost_warping;
            if(cost_translation < cost_warping || (steps < warping_limit)|| GAUSS_NEWTON_TRANSLATION_ONLY){
//                std::cout << ", translation, " << (method_translation ? "MERGE" : "SPATIAL") << std::endl;
                cost_before_subdiv = cost_translation;
                code_length = code_length_translation;
                mvd = mvd_translation;
                selected_index = selected_index_translation;
                method_flag = method_translation;
                square_gauss_results[square_index].translation_flag = true;
                square_gauss_results[square_index].method = method_translation;
                translation_flag = true;
            }else{
//                std::cout << ", warping    , " << (method_warping ? "MERGE" : "SPATIAL") << std::endl;
                cost_before_subdiv = cost_warping;
                code_length = code_length_warping;
                mvd = mvd_warping;
                selected_index = selected_index_warping;
                method_flag = method_warping;
                square_gauss_results[square_index].translation_flag = false;
                square_gauss_results[square_index].method = method_warping;
                translation_flag = false;
            }

        }else if(PRED_MODE == BM) {
            std::vector<cv::Point2f> tmp_bm_mv;
            std::vector<double> tmp_bm_errors;
#if RD_BLOCK_MATCHING
            std::tie(tmp_bm_mv, tmp_bm_errors) = blockMatching(square, target_image, expansion_ref, square_index, ctu);
#else
            std::tie(tmp_bm_mv, tmp_bm_errors) = ::blockMatching(square, target_image, expansion_ref);
#endif
            square_gauss_results[square_index].residual_bm = tmp_bm_errors[2];
            ctu->error_bm = tmp_bm_errors[2];
            gauss_result_warping = tmp_bm_mv;
            gauss_result_translation = tmp_bm_mv[2];
            RMSE_before_subdiv = tmp_bm_errors[2];
            error_translation = tmp_bm_errors[2];
            square_gauss_results[square_index].mv_warping = gauss_result_warping;
            square_gauss_results[square_index].mv_translation = gauss_result_translation;
            square_gauss_results[square_index].square_size = square_size;
            square_gauss_results[square_index].residual_bm = RMSE_before_subdiv;
            square_gauss_results[square_index].translation_flag = true;
            translation_flag = true;
            std::tie(cost_before_subdiv, code_length, mvd, selected_index, method_flag) = getMVD(
                    {gauss_result_translation, gauss_result_translation, gauss_result_translation}, error_translation,
                    square_index, square_number, cmt->mv1, ctu, true, dummy, steps);
        }
    }

    if(method_flag == MV_CODE_METHOD::MERGE) {
        //参照ブロックを作るときのためにマージのベクトルを入れる
        square_gauss_results[square_index].mv_translation = mvd[0];
        square_gauss_results[square_index].mv_warping = mvd;
        gauss_result_translation = mvd[0];
        gauss_result_warping = mvd;
    }

    std::vector<cv::Point2f> mv;
    if (translation_flag) {
        mv.emplace_back(gauss_result_translation);
        mv.emplace_back(gauss_result_translation);
        mv.emplace_back(gauss_result_translation);
    } else {
        mv = gauss_result_warping;
    }

    ctu->mv1 = mv[0];
    ctu->mv2 = mv[1];
    ctu->mv3 = mv[2];
    ctu->square_index = square_index;
    ctu->code_length = code_length;
    ctu->collocated_mv = cmt->mv1;
    ctu->translation_flag = translation_flag;
    ctu->method = method_flag;
    ctu->ref_square_idx = selected_index;

    if(method_flag == SPATIAL) {
        ctu->mvds.clear();
        if(ctu->translation_flag) {
            ctu->mvds.emplace_back(mvd[0]);
            ctu->mvds.emplace_back(mvd[0]);
            ctu->mvds.emplace_back(mvd[0]);
        }else{
            ctu->mvds = mvd;
        }
    }

    if(steps <= 0){
        isCodedSquare[square_index] = true;
        return false;
    }

    SplitResult split_squares = getSplitSquare(p1, p2, p3, p4, 1);

    SplitResult split_sub_squares1 = getSplitSquare(split_squares.s1.p1, split_squares.s1.p2, split_squares.s1.p3, split_squares.s1.p4, split_squares.s_type);
    SplitResult split_sub_squares2 = getSplitSquare(split_squares.s2.p1, split_squares.s2.p2, split_squares.s2.p3, split_squares.s2.p4, split_squares.s_type);

    std::vector<Point4Vec> subdiv_target_squares;

    subdiv_target_squares.emplace_back(split_sub_squares1.s1);
    subdiv_target_squares.emplace_back(split_sub_squares1.s2);
    subdiv_target_squares.emplace_back(split_sub_squares2.s1);
    subdiv_target_squares.emplace_back(split_sub_squares2.s2);

    double RMSE_after_subdiv = 0.0;
    std::vector<GaussResult> split_mv_result(subdiv_target_squares.size());

    int s1_p1_idx = getOrAddCornerIndex(split_squares.s1.p1);
    int s1_p2_idx = getOrAddCornerIndex(split_squares.s1.p2);
    int s1_p3_idx = getOrAddCornerIndex(split_squares.s1.p3);
    int s1_p4_idx = getOrAddCornerIndex(split_squares.s1.p4);
    addCornerAndSquare(Square(s1_p1_idx, s1_p2_idx, s1_p3_idx, s1_p4_idx));

    int s2_p1_idx = getOrAddCornerIndex(split_squares.s2.p1);
    int s2_p2_idx = getOrAddCornerIndex(split_squares.s2.p2);
    int s2_p3_idx = getOrAddCornerIndex(split_squares.s2.p3);
    int s2_p4_idx = getOrAddCornerIndex(split_squares.s2.p4);
    addCornerAndSquare(Square(s2_p1_idx, s2_p2_idx, s2_p3_idx, s2_p4_idx));

    isCodedSquare[square_index] = false;

    int square_indexes[] = {(int)squares.size() - 4, (int)squares.size() - 3, (int)squares.size() - 2, (int)squares.size() - 1};

//    std::cout << "reference_block_list.size = " << reference_block_list.size() << std::endl;

    //分割後の隣接ブロックを入れる
    for (int j = 0; j < (int) subdiv_target_squares.size(); j++) {
        addReferenceBlock(subdiv_target_squares[j], square_indexes[j]);
    }

    ctu->node1 = new CodingTreeUnit();
    ctu->node1->square_index = squares.size() - 4;
    ctu->node1->parentNode = ctu;
    ctu->node2 = new CodingTreeUnit();
    ctu->node2->square_index = squares.size() - 3;
    ctu->node2->parentNode = ctu;
    ctu->node3 = new CodingTreeUnit();
    ctu->node3->square_index = squares.size() - 2;
    ctu->node3->parentNode = ctu;
    ctu->node4 = new CodingTreeUnit();
    ctu->node4->square_index = squares.size() - 1;
    ctu->node4->parentNode = ctu;

    std::vector<CodingTreeUnit*> ctus{ctu->node1, ctu->node2, ctu->node3, ctu->node4};
#if !MVD_DEBUG_LOG
//    #pragma omp parallel for
#endif
    cv::Point2f mv_translation;
    std::vector<cv::Point2f> mv_warping;
    std::vector<cv::Point2f> tmp_bm_mv;
    std::vector<double> tmp_bm_errors;
    double tmp_error_newton;
    cv::Point2f original_mv_translation[4];
    std::vector<cv::Point2f> original_mv_warping[4];
    double cost_after_subdivs[4];
    MV_CODE_METHOD method_flags[4];
    CollocatedMvTree *cmts[4];

    cmts[0]  = (cmt->node1 == nullptr ? cmt : cmt->node1);
    cmts[1]  = (cmt->node2 == nullptr ? cmt : cmt->node2);
    cmts[2]  = (cmt->node3 == nullptr ? cmt : cmt->node3);
    cmts[3]  = (cmt->node4 == nullptr ? cmt : cmt->node4);
    for (int j = 0; j < (int) subdiv_target_squares.size(); j++) {
        if(PRED_MODE == NEWTON){
            if(GAUSS_NEWTON_INIT_VECTOR) {
//                std::tie(tmp_bm_mv, tmp_bm_errors) = fullpellBlockMatching(subdiv_target_squares[j], target_image, expansion_ref, square_indexes[j], ctus[j]);
                std::tie(mv_warping, mv_translation, error_warping, error_translation,square_size) = Square_GaussNewton(
                        ref_images, target_images, expand_images, subdiv_target_squares[j], square_indexes[j], ctus[j], tmp_bm_mv[2], ref_hevc);
#if USE_BM_TRANSLATION_MV
                error_translation_tmp = tmp_bm_errors[2];
                mv_translation_tmp = tmp_bm_mv[2];
#endif
            }else{
                std::tie(mv_warping, mv_translation, error_warping, error_translation, square_size) = Square_GaussNewton(
                        ref_images, target_images, expand_images, subdiv_target_squares[j], square_indexes[j], ctus[j], cv::Point2f(-1000, -1000), ref_hevc);
            }

            square_gauss_results[square_indexes[j]].mv_translation = mv_translation;
            square_gauss_results[square_indexes[j]].mv_warping = mv_warping;

            std::tie(cost_translation, code_length_translation, mvd_translation, std::ignore, method_translation) = getMVD(
                    {mv_translation, mv_translation, mv_translation}, error_translation,
                    square_indexes[j], j, cmts[j]->mv1, ctus[j], true, dummy, steps - 2);
#if !GAUSS_NEWTON_TRANSLATION_ONLY

            std::tie(cost_warping, code_length_warping, mvd_warping, std::ignore, method_warping) = getMVD(
                    mv_warping, error_warping,
                    square_indexes[j], j, cmts[j]->mv1, ctus[j], false, dummy, steps - 2);
#endif
//            std::cout << "cost_translation_tmp : " << cost_translation_tmp << ", cost_warping_tmp : " << cost_warping_tmp << std::endl;

            mvd.clear();
            if(cost_translation < cost_warping || (steps - 2 < warping_limit) || GAUSS_NEWTON_TRANSLATION_ONLY){
                square_gauss_results[square_indexes[j]].translation_flag = true;
                cost_after_subdivs[j] = cost_translation;
                method_flags[j] = method_translation;
                mvd = mvd_translation;
                split_mv_result[j] = GaussResult(mv_warping, mv_translation, error_warping, error_translation, square_size, true, error_translation);
            }else{
                square_gauss_results[square_indexes[j]].translation_flag = false;
                cost_after_subdivs[j] = cost_warping;
                method_flags[j] = method_warping;
                mvd = mvd_warping;
                split_mv_result[j] = GaussResult(mv_warping, mv_translation, error_warping, error_translation, square_size, false, error_warping);
            }

        }else if(PRED_MODE == BM){
#if RD_BLOCK_MATCHING
            std::tie(tmp_bm_mv, tmp_bm_errors) = blockMatching(subdiv_target_squares[j], target_image, expansion_ref, square_indexes[j], ctus[j]);
#else
            std::tie(tmp_bm_mv, tmp_bm_errors) = ::blockMatching(subdiv_target_squares[j], target_image, expansion_ref);
#endif
            mv_warping = tmp_bm_mv;
            mv_translation = tmp_bm_mv[2];
            error_translation = tmp_bm_errors[2];
            error_warping = tmp_bm_errors[2];
            square_size = (double)1e6;

            split_mv_result[j] = GaussResult(mv_warping, mv_translation, error_warping, error_translation, square_size, true, tmp_bm_errors[2]);

            square_gauss_results[square_indexes[j]].translation_flag = true;
            square_gauss_results[square_indexes[j]].mv_translation = mv_translation;

            std::tie(cost_translation, code_length_translation, std::ignore, std::ignore, method_translation) = getMVD(
                    {mv_translation, mv_translation, mv_translation}, error_translation,
                    square_indexes[j], j, cmts[j]->mv1, ctus[j], true, dummy, steps - 2);
            cost_after_subdivs[j] = cost_translation;
            method_flags[j] = method_translation;
        }

        //分割後の参照ブロックを作るときのために一旦マージのベクトルを入れる
        if(method_flags[j] == MV_CODE_METHOD::MERGE) {
            if(split_mv_result[j].translation_flag) {
                gauss_result_translation = mvd[0];
                original_mv_translation[j] = square_gauss_results[square_indexes[j]].mv_translation;
                square_gauss_results[square_indexes[j]].mv_translation = gauss_result_translation;
            }else{
                original_mv_warping[j] = square_gauss_results[square_indexes[j]].mv_warping;
                square_gauss_results[square_indexes[j]].mv_warping = mvd;
            }
        }
        isCodedSquare[square_indexes[j]] = true;
    }
    for(int i = 0 ; i < 4 ; i++) isCodedSquare[square_indexes[i]] = false;

    double lambda = getLambdaPred(qp, (translation_flag ? 1.0 : 1.0));
//    double lambda = getLambdaMode(qp);

    double alpha = 1.0;
    std::cout << "before   : " << cost_before_subdiv << "    after : " << alpha * (cost_after_subdivs[0] + cost_after_subdivs[1] + cost_after_subdivs[2] + cost_after_subdivs[3]) << std::endl;
//    std::cout << "D before : " << cost_before_subdiv - lambda * code_length<< "    D after : " << alpha * (cost_after_subdivs[0] + cost_after_subdivs[1] + cost_after_subdivs[2] + cost_after_subdivs[3] - lambda * (code_lengthes[0] + code_lengthes[1] + code_lengthes[2] + code_lengthes[3])) << std::endl;
//    std::cout << "R before : " << code_length<< "         R after : " << alpha * (code_lengthes[0] + code_lengthes[1] + code_lengthes[2] + code_length[3]) << ", mv : " << square_gauss_results[square_index].mv_translation << std::endl;
//    std::cout << "after1 D : " << alpha * (cost_after_subdivs[0] - lambda * code_lengthes[0]) << ", R : " << alpha * (code_lengthes[0]) << ", method : " << method_flags[0] << ", mv : " << square_gauss_results[square_indexes[0]].mv_translation << ", square_index : " << square_indexes[0] << std::endl;
//    std::cout << "after2 D : " << alpha * (cost_after_subdivs[1] - lambda * code_lengthes[1]) << ", R : " << alpha * (code_lengthes[1]) << ", method : " << method_flags[1] << ", mv : " << square_gauss_results[square_indexes[1]].mv_translation << ", square_index : " << square_indexes[1] << std::endl;
//    std::cout << "after3 D : " << alpha * (cost_after_subdivs[2] - lambda * code_lengthes[2]) << ", R : " << alpha * (code_lengthes[2]) << ", method : " << method_flags[2] << ", mv : " << square_gauss_results[square_indexes[2]].mv_translation << ", square_index : " << square_indexes[2] << std::endl;
//    std::cout << "after4 D : " << alpha * (cost_after_subdivs[3] - lambda * code_lengthes[3]) << ", R : " << alpha * (code_lengthes[3]) << ", method : " << method_flags[3] << ", mv : " << square_gauss_results[square_indexes[3]].mv_translation << ", square_index : " << square_indexes[3] << std::endl <<std::endl;
    if(cost_before_subdiv >= alpha * (cost_after_subdivs[0] + cost_after_subdivs[1] + cost_after_subdivs[2] + cost_after_subdivs[3])) {
        ctu->split_cu_flag = true;
        for (int j = 0; j < (int) subdiv_target_squares.size(); j++) {
            // j個目の四角形
            if (method_flags[j] == MV_CODE_METHOD::MERGE) {
                if (split_mv_result[j].translation_flag) {
                    split_mv_result[j].mv_translation = original_mv_translation[j];
                } else {
                    split_mv_result[j].mv_warping = original_mv_warping[j];
                }
            }
            square_gauss_results[square_indexes[j]] = split_mv_result[j];
            split(expand_images, ctus[j], cmts[j], subdiv_target_squares[j], square_indexes[j], j, steps - 2);
        }

        return true;
    }else{
        //4分割により追加された頂点12個を消す
        corners.erase(corners.end() - 12, corners.end());
        eraseCornerFlag(split_sub_squares1.s1, split_sub_squares1.s2, split_sub_squares2.s1, split_sub_squares2.s2);
        isCodedSquare[square_index] = true;
        ctu->node1 = ctu->node2 = ctu->node3 = ctu->node4 = nullptr;
        ctu->method = method_flag;
        eraseSquare(squares.size() - 1);
        eraseSquare(squares.size() - 1);
        eraseSquare(squares.size() - 1);
        eraseSquare(squares.size() - 1);
        covered_square.erase(covered_square.end() - 12, covered_square.end());
        addCoveredSquare(squares[square_index].p1_idx,squares[square_index].p2_idx,squares[square_index].p3_idx,squares[square_index].p4_idx, square_index);

//        std::cout << (ctu->method == MERGE ? "MERGE" : "SPATIAL") << " " << (ctu->translation_flag ? "TRANSLATION" : "WARPING") << " "  << ctu->mv1 << " " << ctu->mv2 << " " << ctu->mv3 << std::endl;
        return false;
    }

}

/**
 * @fn SquareDivision::SplitResult SquareDivision::getSplitSquare(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4, int type)
 * @details ４点の座標とtypeを受け取り，分割した形状を返す
 * @param p1 頂点１の座標
 * @param p2 頂点２の座標
 * @param p3 頂点３の座標
 * @param p4 頂点4の座標
 * @param type 分割形状
 * @return 分割結果
 */
SquareDivision::SplitResult SquareDivision::getSplitSquare(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4, int type){
    cv::Point2f a, b, c, d, e, f, g, h;

    switch(type) {
        case 1:
        {
            cv::Point2f y = (p3 - p1) / 2.0;         //  a                b
            y.y -= 0.5;                              //   -----------------
            a = p1;                                  //   |               |
            b = p2;                                  // e |               | f
            c = p3;                                  // g |               | h
            d = p4;                                  //   |               |
            e = g = a + y;                           //   -----------------
            f = h = b + y;                           //  c                d
            g.y++;
            h.y++;

            return {Point4Vec(a, b, e, f), Point4Vec(g, h, c, d), 2};
        }
        case 2:
        {
            cv::Point2f x = (p2 - p1) / 2.0;          //  a       e g      b
            x.x -= 0.5;                               //   -----------------
            a = p1;                                   //   |               |
            b = p2;                                   //   |               |
            c = p3;                                   //   -----------------
            d = p4;                                   //  c       f h      d
            e = g = a + x;                            //
            f = h = c + x;                            //
            g.x++;
            h.x++;

            return {Point4Vec(a, e, c, f), Point4Vec(g, b, h, d), 1};
        }
        default:
            return {Point4Vec(p1, p2, p3, p4), Point4Vec(p1, p2, p3, p4), 1};
    }

}

void SquareDivision::addReferenceBlock(Point4Vec subdiv_target_square, int square_index) {
    //追加する頂点を宣言                                           //
    cv::Point2f sp1 = subdiv_target_square.p3;                 //   ---------------     ---------------     ---------------
    cv::Point2f sp2 = subdiv_target_square.p3;                 //   |             |     |             |     |             |
    cv::Point2f sp3 = subdiv_target_square.p2;                 //   |             |     |             |     |             |
    cv::Point2f sp4 = subdiv_target_square.p2;                 //   |             |     |             |     |             |
    cv::Point2f sp5 = subdiv_target_square.p1;                 //   |        sp5●|     |        sp4●|     |●sp3        |
    //頂点の座標を調整                                             //   ---------------     ---------------     ---------------
    sp1.x--; sp1.y++;                                              //
    sp2.x--;                                                       //   ---------------     ---------------
    sp3.x++; sp3.y--;                                              //   |           　|     | p1       p2 |
    sp4.y--;                                                       //   |             |     |             |
    sp5.x--; sp5.y--;                                              //   |             |     |             |
    //頂点インデックスを取得                                       //   |        sp2●|     | p3          |
    int sp1_idx = getCornerIndex(sp1);                             //   ---------------     ---------------
    int sp2_idx = getCornerIndex(sp2);                             //   ---------------
    int sp3_idx = getCornerIndex(sp3);                             //   |        sp1●|
    int sp4_idx = getCornerIndex(sp4);                             //   |             |
    int sp5_idx = getCornerIndex(sp5);                             //   |             |
    //   |             |
    if(sp1_idx != -1) {                                            //   ---------------
        // 1の頂点を入れる
        reference_block_list[square_index].emplace_back(sp1_idx);
    }
    else {
        //隣が同じstep以上分割されていない場合も候補ブロックを5個にするために2回追加
        cv::Point2f sp1_2 = sp1;
        sp1_2.y--;
        sp1.x++;
        for(int i = 0 ; i < 7 ; i++) {
            sp1.x -= 8;
            if((sp1_idx = getCornerIndex(sp1)) != -1) {
                reference_block_list[square_index].emplace_back(sp1_idx);
                break;
            }
            sp1_2.y += 8;
            if((sp1_idx = getCornerIndex(sp1_2)) != -1) {
                reference_block_list[square_index].emplace_back(sp1_idx);
                break;
            }
        }
    }
    if(sp2_idx != -1) {
        // 2の頂点を入れる
        reference_block_list[square_index].emplace_back(sp2_idx);
    }
    else {
        for(int i = 0 ; i < 7 ; i++) {
            sp2.y += 8;
            if((sp2_idx = getCornerIndex(sp2)) != -1) {
                reference_block_list[square_index].emplace_back(sp2_idx);
                break;
            }
        }
    }
    if(sp3_idx != -1) {
        // 3の頂点を入れる
        reference_block_list[square_index].emplace_back(sp3_idx);
    }
    else {
        cv::Point2f sp3_2 = sp3;
        sp3_2.y++;
        sp3.x--;
        for(int i = 0 ; i < 7 ; i++) {
            sp3.x += 8;
            if((sp3_idx = getCornerIndex(sp3)) != -1) {
                reference_block_list[square_index].emplace_back(sp3_idx);
                break;
            }
            sp3_2.y -= 8;
            if((sp3_idx = getCornerIndex(sp3_2)) != -1) {
                reference_block_list[square_index].emplace_back(sp3_idx);
                break;
            }
        }
    }
    if(sp4_idx != -1) {
        // 4の頂点を入れる
        reference_block_list[square_index].emplace_back(sp4_idx);
    }
    else {
        for(int i = 0 ; i < 7 ; i++) {
            sp4.x += 8;
            if((sp4_idx = getCornerIndex(sp4)) != -1) {
                reference_block_list[square_index].emplace_back(sp4_idx);
                break;
            }
        }
    }
    if(sp5_idx != -1) {
        // 5の頂点を入れる
        reference_block_list[square_index].emplace_back(sp5_idx);
    }
    else {
        cv::Point2f sp5_2 = sp5;
        sp5.y++;
        sp5_2.x++;
        for(int i = 0 ; i < 7 ; i++) {
            sp5.y -= 8;
            if((sp5_idx = getCornerIndex(sp5)) != -1) {
                reference_block_list[square_index].emplace_back(sp5_idx);
                break;
            }
            sp5_2.x -= 8;
            if((sp5_idx = getCornerIndex(sp5_2)) != -1) {
                reference_block_list[square_index].emplace_back(sp5_idx);
                break;
            }
        }
    }
    if(sp2_idx != -1) {
        merge_reference_block_list[square_index].emplace_back(sp2_idx);
    }
    if(sp4_idx != -1) {
        merge_reference_block_list[square_index].emplace_back(sp4_idx);
    }
    if(sp3_idx != -1) {
        merge_reference_block_list[square_index].emplace_back(sp3_idx);
    }
    if(sp1_idx != -1) {
        merge_reference_block_list[square_index].emplace_back(sp1_idx);
    }
    if(sp5_idx != -1) {
        merge_reference_block_list[square_index].emplace_back(sp5_idx);
    }
//        std::cout << std::endl;
//
//    std::cout << "square_index : " << square_indexes[j] << ", reference_block_list[" << square_indexes[j] << "].size : " << reference_block_list[square_indexes[j]].size() << std::endl << "reference_block : ";
//    for(auto rbl : reference_block_list[square_indexes[j]]) {
//        std::cout << covered_square[rbl] << ", ";
//    }
//    std::cout << std::endl;
}

/**
 * @fn std::vector<int> SquareDivision::getSpatialSquareList(int square_idx, bool translation_flag)
 * @brief square_idx番目の四角形の参照候補ブロックの動きベクトルを返す
 * @param[in] square_idx 符号化対照ブロックのインデックス
 * @param[in] translation_flag 符号化対照ブロックのflag
 * @return 候補ブロックの動きベクトルを返す
 */
std::tuple< std::vector<std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >>>, std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> > SquareDivision::getSpatialSquareList(int square_idx, bool translation_flag) {
    //隣接するブロックを取得する
    std::vector<int> reference_vertexes = reference_block_list[square_idx];
    int i, j;
    std::vector<int> tmp_reference_block;
    for (i = 0 ; i < reference_vertexes.size() ; i++) {
        tmp_reference_block.emplace_back(covered_square[reference_vertexes[i]]);
    }

//    std::cout << "square_index : " << square_idx << ", tmp_reference_block_size : " << tmp_reference_block.size() << std::endl << "tmp_reference_block : ";
//    for(int i = 0 ; i < tmp_reference_block.size() ; i++) {
//        std::cout << tmp_reference_block[i] << ", ";
//    }
//    std::cout << std::endl;

    std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> vectors;
    std::vector<cv::Point2f> tmp_vectors;
    std::vector<std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >>> warping_vectors;
    std::vector<std::vector<cv::Point2f>> tmp_warping_vectors;
    //参照可能候補に入れるかどうかを判定する配列
    bool is_in_flag[5] = {true, true, true, true, true};

    std::vector<int> reference_block_list;

    //平行移動とワーピングの動きベクトル
    for(i = 0 ; i < tmp_reference_block.size() ; i++) {
        int reference_block_index = tmp_reference_block[i];
        if(!isCodedSquare[reference_block_index]) {  //符号化済みでないブロックも参照候補リストに入れているのでその場合は空のものを入れておく
            tmp_vectors.emplace_back();
            tmp_warping_vectors.emplace_back();
            is_in_flag[i] = false;                    //符号化済みでないものは入れないのでfalseにする
            continue;
        }
        if(square_gauss_results[reference_block_index].translation_flag) { //参照候補ブロックが平行移動の場合
            cv::Point2f current_mv = square_gauss_results[reference_block_index].mv_translation;
            if(translation_flag) {
                tmp_vectors.emplace_back(current_mv);
            } else {
                std::vector<cv::Point2f> v{current_mv, current_mv, current_mv};
                tmp_warping_vectors.emplace_back(v);
            }
        } else {  //参照候補ブロックがワーピングの場合
            cv::Point2f current_mv1 = square_gauss_results[reference_block_index].mv_warping[0];
            cv::Point2f current_mv2 = square_gauss_results[reference_block_index].mv_warping[1];
            cv::Point2f current_mv3 = square_gauss_results[reference_block_index].mv_warping[2];
#if MVD_DEBUG_LOG
            std::cout << "target_square_coordinate:";
            std::cout << corners[squares[square_idx].p1_idx] << " ";
            std::cout << corners[squares[square_idx].p2_idx] << " ";
            std::cout << corners[squares[square_idx].p3_idx] << " ";
            std::cout << corners[squares[square_idx].p4_idx] << std::endl;
            std::cout << "ref_square_coordinate:";
            std::cout << corners[squares[tmp_reference_block[i]].p1_idx] << " ";
            std::cout << corners[squares[tmp_reference_block[i]].p2_idx] << " ";
            std::cout << corners[squares[tmp_reference_block[i]].p3_idx] << " ";
            std::cout << corners[squares[tmp_reference_block[i]].p4_idx] <<std::endl;
            std::cout << "ref_square_mvs:";
            std::cout << current_mv1 << " " << current_mv2 << " " << current_mv3 << std::endl;
#endif
            std::vector<cv::Point2f> ref_mvs{current_mv1, current_mv2, current_mv3};
            Square target_square = squares[square_idx];
            cv::Point2f pp1 = corners[target_square.p1_idx], pp2 = corners[target_square.p2_idx], pp3 = corners[target_square.p3_idx], pp4 = corners[target_square.p4_idx];
            Square ref_square = squares[reference_block_index];
            std::vector<cv::Point2f> ref_square_coordinates{corners[ref_square.p1_idx], corners[ref_square.p2_idx], corners[ref_square.p3_idx], corners[ref_square.p4_idx]};
            if(translation_flag) {
                std::vector<cv::Point2f> target_square_coordinates{cv::Point2f((pp1 + pp2 + pp3 + pp4) / 4.0)};
                std::vector<cv::Point2f> mvs = getPredictedWarpingMv(ref_square_coordinates, ref_mvs, target_square_coordinates);
                tmp_vectors.emplace_back(mvs[0]);
            } else {
                std::vector<cv::Point2f> target_square_coordinates;
                target_square_coordinates.emplace_back(pp1);
                target_square_coordinates.emplace_back(pp2);
                target_square_coordinates.emplace_back(pp3);
                std::vector<cv::Point2f> mvs = getPredictedWarpingMv(ref_square_coordinates, ref_mvs, target_square_coordinates);
                std::vector<cv::Point2f> v{mvs[0], mvs[1], mvs[2]};
                tmp_warping_vectors.emplace_back(v);
            }
        }
    }

    if(translation_flag) {
        for (j = 0; j < tmp_vectors.size(); j++) {
            //重複していない場合
            if (is_in_flag[j]) {
                for ( i = j + 1; i < tmp_vectors.size(); i++) {
                    //同一動き情報をもっている場合は配列をoff(false)にする
                    if (is_in_flag[i] && tmp_vectors[j] == tmp_vectors[i])
                        is_in_flag[i] = false;
                }
            }
        }
    } else {
        for (j = 0; j < tmp_warping_vectors.size(); j++) {
            if (is_in_flag[j]) {
                for (i = j + 1; i < tmp_warping_vectors.size(); i++) {
                    //同一動き情報をもっている場合は配列をoff(false)にする
                    if (is_in_flag[i] && tmp_warping_vectors[j] == tmp_warping_vectors[i])
                        is_in_flag[i] = false;
                }
            }
        }
    }
    //重複がなく，符号化済みのブロックのみ入れる
    if(translation_flag) {
        for (j = 0 ; j < tmp_vectors.size() ; j++) {
            if (is_in_flag[j])
                vectors.emplace_back(tmp_vectors[j], SPATIAL);
        }
    } else {
        for (j = 0 ; j < tmp_warping_vectors.size() ; j++) {
            if (is_in_flag[j]) {
                std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> v;
                v.emplace_back(tmp_warping_vectors[j][0], SPATIAL);
                v.emplace_back(tmp_warping_vectors[j][1], SPATIAL);
                v.emplace_back(tmp_warping_vectors[j][2], SPATIAL);
                warping_vectors.emplace_back(v);
            }
        }
    }

//    std::cout << "square_index : " << square_idx << ", method : " << "SPATIAL" << ", SquareList_size : " << (translation_flag ? vectors.size() : warping_vectors.size()) << std::endl;

    return {warping_vectors, vectors};
}


/**
 * @fn std::vector<int> SquareDivision::getMergeSquareList(int square_idx, bool translation_flag, Point4Vec coordinate, cv::Point2f image_size)
 * @brief square_idx番目の四角形の参照候補ブロックの動きベクトルを返す
 * @param[in] square_idx 符号化対照ブロックのインデックス
 * @param[in] translation_flag 符号化対照ブロックのflag
 * @param[in] coordinate 符号化対照ブロックの頂点の座標
 * @param[in] image_size 画像のサイズ
 * @return 候補ブロックの動きベクトルを返す
 */
std::tuple< std::vector<std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >>>, std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> > SquareDivision::getMergeSquareList(int square_idx, bool translation_flag, Point4Vec coordinate) {
    //隣接するブロックを取得する
    std::vector<int> reference_vertexes = merge_reference_block_list[square_idx];
    int i, j;
    std::vector<int> tmp_reference_block;
    for (i = 0 ; i < reference_vertexes.size() ; i++) {
        tmp_reference_block.emplace_back(covered_square[reference_vertexes[i]]);
    }

//    std::cout << "reference_block_size : " << reference_block.size() << ", ";

    std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> vectors;
    std::vector<cv::Point2f> tmp_merge_vectors;
    std::vector<std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >>> warping_vectors;
    std::vector<std::vector<cv::Point2f>> tmp_warping_merge_vectors;
    //参照可能候補に入れるかどうかを判定する配列
    bool is_in_flag[5] = {true, true, true, true, true};

    std::vector<int> reference_block_list;

    //平行移動とワーピングの動きベクトル
    for(i = 0 ; i < tmp_reference_block.size() ; i++) {
        int reference_block_index = tmp_reference_block[i];
        if(!isCodedSquare[reference_block_index]) {  //符号化済みでないブロックも参照候補リストに入れているのでその場合は空のものを入れておく
            tmp_merge_vectors.emplace_back();
            tmp_warping_merge_vectors.emplace_back();
            is_in_flag[i] = false;                    //符号化済みでないものは入れないのでfalseにする
            continue;
        }
        if(square_gauss_results[reference_block_index].translation_flag) { //参照候補ブロックが平行移動の場合
            cv::Point2f current_mv = square_gauss_results[reference_block_index].mv_translation;
            if(translation_flag) {
                //画像の外に出てしまう場合は候補に入れない
                if(current_mv.x + coordinate.p1.x < -SERACH_RANGE || current_mv.y + coordinate.p1.y < -SERACH_RANGE || current_mv.x + coordinate.p4.x >= target_image.cols + SERACH_RANGE || current_mv.y + coordinate.p4.y >= target_image.rows + SERACH_RANGE) {
                    tmp_merge_vectors.emplace_back();
                    is_in_flag[i] = false;
                }
                else tmp_merge_vectors.emplace_back(current_mv);
            } else {
                if ((current_mv.x + coordinate.p1.x < -SERACH_RANGE || current_mv.y + coordinate.p1.y < -SERACH_RANGE || current_mv.x + coordinate.p1.x >= target_image.cols + SERACH_RANGE || current_mv.y + coordinate.p1.y >= target_image.rows + SERACH_RANGE) ||
                    (current_mv.x + coordinate.p2.x < -SERACH_RANGE || current_mv.y + coordinate.p2.y < -SERACH_RANGE || current_mv.x + coordinate.p2.x >= target_image.cols + SERACH_RANGE || current_mv.y + coordinate.p2.y >= target_image.rows + SERACH_RANGE) ||
                    (current_mv.x + coordinate.p3.x < -SERACH_RANGE || current_mv.y + coordinate.p3.y < -SERACH_RANGE || current_mv.x + coordinate.p3.x >= target_image.cols + SERACH_RANGE || current_mv.y + coordinate.p3.y >= target_image.rows + SERACH_RANGE) ||
                    (current_mv.x + coordinate.p4.x < -SERACH_RANGE || current_mv.y + coordinate.p4.y < -SERACH_RANGE || current_mv.x + coordinate.p4.x >= target_image.cols + SERACH_RANGE || current_mv.y + coordinate.p4.y >= target_image.rows + SERACH_RANGE)) {
                    tmp_warping_merge_vectors.emplace_back();
                    is_in_flag[i] = false;
                }
                else {
                    std::vector<cv::Point2f> v{current_mv, current_mv, current_mv};
                    tmp_warping_merge_vectors.emplace_back(v);
                }
            }
        } else {  //参照候補ブロックがワーピングの場合
            cv::Point2f current_mv1 = square_gauss_results[reference_block_index].mv_warping[0];
            cv::Point2f current_mv2 = square_gauss_results[reference_block_index].mv_warping[1];
            cv::Point2f current_mv3 = square_gauss_results[reference_block_index].mv_warping[2];
#if MVD_DEBUG_LOG
            std::cout << "target_square_coordinate:";
            std::cout << corners[squares[square_idx].p1_idx] << " ";
            std::cout << corners[squares[square_idx].p2_idx] << " ";
            std::cout << corners[squares[square_idx].p3_idx] << " ";
            std::cout << corners[squares[square_idx].p4_idx] << std::endl;
            std::cout << "ref_square_coordinate:";
            std::cout << corners[squares[tmp_reference_block[i]].p1_idx] << " ";
            std::cout << corners[squares[tmp_reference_block[i]].p2_idx] << " ";
            std::cout << corners[squares[tmp_reference_block[i]].p3_idx] << " ";
            std::cout << corners[squares[tmp_reference_block[i]].p4_idx] <<std::endl;
            std::cout << "ref_square_mvs:";
            std::cout << current_mv1 << " " << current_mv2 << " " << current_mv3 << std::endl;
#endif
            std::vector<cv::Point2f> ref_mvs{current_mv1, current_mv2, current_mv3};
            Square target_square = squares[square_idx];
            cv::Point2f pp1 = coordinate.p1, pp2 = coordinate.p2, pp3 = coordinate.p3, pp4 = coordinate.p4;
            Square ref_square = squares[reference_block_index];
            std::vector<cv::Point2f> ref_square_coordinates{corners[ref_square.p1_idx], corners[ref_square.p2_idx], corners[ref_square.p3_idx], corners[ref_square.p4_idx]};
            if(translation_flag) {
                std::vector<cv::Point2f> target_square_coordinates{cv::Point2f((pp1 + pp2 + pp3 + pp4) / 4.0)};
                std::vector<cv::Point2f> mvs = getPredictedWarpingMv(ref_square_coordinates, ref_mvs, target_square_coordinates);
                if(mvs[0].x + coordinate.p1.x < -SERACH_RANGE || mvs[0].y + coordinate.p1.y < -SERACH_RANGE || mvs[0].x + coordinate.p4.x >= target_image.cols + SERACH_RANGE || mvs[0].y + coordinate.p4.y >= target_image.rows + SERACH_RANGE) {
                    tmp_merge_vectors.emplace_back();
                    is_in_flag[i] = false;
                }
                else tmp_merge_vectors.emplace_back(mvs[0]);
            } else {
                std::vector<cv::Point2f> target_square_coordinates;
                target_square_coordinates.emplace_back(pp1);
                target_square_coordinates.emplace_back(pp2);
                target_square_coordinates.emplace_back(pp3);
                std::vector<cv::Point2f> mvs = getPredictedWarpingMv(ref_square_coordinates, ref_mvs, target_square_coordinates);
                cv::Point2f p4 = pp3 + mvs[2] + pp2 + mvs[1] - pp1 - mvs[0]; //右下の頂点の変形後の座標
                if ((mvs[0].x + coordinate.p1.x < -SERACH_RANGE || mvs[0].y + coordinate.p1.y < -SERACH_RANGE || mvs[0].x + coordinate.p1.x >= target_image.cols + SERACH_RANGE || mvs[0].y + coordinate.p1.y >= target_image.rows + SERACH_RANGE) ||
                    (mvs[1].x + coordinate.p2.x < -SERACH_RANGE || mvs[1].y + coordinate.p2.y < -SERACH_RANGE || mvs[1].x + coordinate.p2.x >= target_image.cols + SERACH_RANGE || mvs[1].y + coordinate.p2.y >= target_image.rows + SERACH_RANGE) ||
                    (mvs[2].x + coordinate.p3.x < -SERACH_RANGE || mvs[2].y + coordinate.p3.y < -SERACH_RANGE || mvs[2].x + coordinate.p3.x >= target_image.cols + SERACH_RANGE || mvs[2].y + coordinate.p3.y >= target_image.rows + SERACH_RANGE) ||
                    (p4.x < -SERACH_RANGE || p4.y < -SERACH_RANGE || p4.x >= target_image.cols + SERACH_RANGE || p4.y >= target_image.rows + SERACH_RANGE)) {
                    tmp_warping_merge_vectors.emplace_back();
                    is_in_flag[i] = false;
                }
                else {
                    std::vector<cv::Point2f> v{mvs[0], mvs[1], mvs[2]};
                    tmp_warping_merge_vectors.emplace_back(v);
                }
            }
        }
    }

    if(tmp_reference_block.size() == 2) {
        if(translation_flag) {
            //同一動き情報をもっている場合は配列をoff(false)にする
            //③
            if (is_in_flag[0] && is_in_flag[1] && tmp_merge_vectors[0] == tmp_merge_vectors[1])
                is_in_flag[1] = false;
        } else {
            //③
            if (is_in_flag[0] && is_in_flag[1] && tmp_warping_merge_vectors[0] == tmp_warping_merge_vectors[1])
                is_in_flag[1] = false;
        }
    }
    else if(tmp_reference_block.size() == 3) {
        if(translation_flag) {
            //①
            if(is_in_flag[0] && is_in_flag[1] && tmp_merge_vectors[0] == tmp_merge_vectors[1])
               is_in_flag[1] = false;
            //④
            if((is_in_flag[0] && is_in_flag[2] && tmp_merge_vectors[0] == tmp_merge_vectors[2]) ||
               (is_in_flag[1] && is_in_flag[2] && tmp_merge_vectors[1] == tmp_merge_vectors[2]))
                is_in_flag[2] = false;
        } else {
            //①
            if(is_in_flag[0] && is_in_flag[1] && tmp_warping_merge_vectors[0] == tmp_warping_merge_vectors[1])
               is_in_flag[1] = false;
            //④
            if((is_in_flag[0] && is_in_flag[2] && tmp_warping_merge_vectors[0] == tmp_warping_merge_vectors[2]) ||
               (is_in_flag[1] && is_in_flag[2] && tmp_warping_merge_vectors[1] == tmp_warping_merge_vectors[2]))
                is_in_flag[2] = false;
        }
    }
    else if(tmp_reference_block.size() == 4) {
        bool flag = false;
        cv::Point2f p2 = corners[squares[square_idx].p2_idx];
        if(p2.x == target_image.cols - 1) flag = true;

        if(flag) { //右上にブロックがない場合
            if(translation_flag) {
                //①
                if (is_in_flag[0] && is_in_flag[1] && tmp_merge_vectors[0] == tmp_merge_vectors[1])
                    is_in_flag[1] = false;
                //③
                if (is_in_flag[0] && is_in_flag[2] && tmp_merge_vectors[0] == tmp_merge_vectors[2])
                    is_in_flag[2] = false;
                //④
                if ((is_in_flag[0] && is_in_flag[3] && tmp_merge_vectors[0] == tmp_merge_vectors[3]) ||
                    (is_in_flag[1] && is_in_flag[3] && tmp_merge_vectors[1] == tmp_merge_vectors[3]))
                    is_in_flag[3] = false;
            } else {
                //①
                if (is_in_flag[0] && is_in_flag[1] && tmp_warping_merge_vectors[0] == tmp_warping_merge_vectors[1])
                    is_in_flag[1] = false;
                //③
                if (is_in_flag[0] && is_in_flag[2] && tmp_warping_merge_vectors[0] == tmp_warping_merge_vectors[2])
                    is_in_flag[2] = false;
                //④
                if ((is_in_flag[0] && is_in_flag[3] && tmp_warping_merge_vectors[0] == tmp_warping_merge_vectors[3]) ||
                    (is_in_flag[1] && is_in_flag[3] && tmp_warping_merge_vectors[1] == tmp_warping_merge_vectors[3]))
                    is_in_flag[3] = false;
            }
        }
        else {
            if(translation_flag) {
                //①
                if (is_in_flag[0] && is_in_flag[1] && tmp_merge_vectors[0] == tmp_merge_vectors[1])
                    is_in_flag[1] = false;
                //②
                if (is_in_flag[1] && is_in_flag[2] && tmp_merge_vectors[1] == tmp_merge_vectors[2])
                    is_in_flag[2] = false;
                //④
                if ((is_in_flag[0] && is_in_flag[3] && tmp_merge_vectors[0] == tmp_merge_vectors[3]) ||
                    (is_in_flag[1] && is_in_flag[3] && tmp_merge_vectors[1] == tmp_merge_vectors[3]))
                    is_in_flag[3] = false;
            } else {
                //①
                if (is_in_flag[0] && is_in_flag[1] && tmp_warping_merge_vectors[0] == tmp_warping_merge_vectors[1])
                    is_in_flag[1] = false;
                //②
                if (is_in_flag[1] && is_in_flag[2] && tmp_warping_merge_vectors[1] == tmp_warping_merge_vectors[2])
                    is_in_flag[2] = false;
                //④
                if ((is_in_flag[0] && is_in_flag[3] && tmp_warping_merge_vectors[0] == tmp_warping_merge_vectors[3]) ||
                    (is_in_flag[1] && is_in_flag[3] && tmp_warping_merge_vectors[1] == tmp_warping_merge_vectors[3]))
                    is_in_flag[3] = false;
            }
        }
    }
    else if(tmp_reference_block.size() == 5) {
        if(translation_flag) {
            //①
            if (is_in_flag[0] && is_in_flag[1] && tmp_merge_vectors[0] == tmp_merge_vectors[1])
                is_in_flag[1] = false;
            //②
            if (is_in_flag[1] && is_in_flag[2] && tmp_merge_vectors[1] == tmp_merge_vectors[2])
                is_in_flag[2] = false;
            //③
            if (is_in_flag[0] && is_in_flag[3] && tmp_merge_vectors[0] == tmp_merge_vectors[3])
                is_in_flag[3] = false;
            //④
            if ((is_in_flag[0] && is_in_flag[4] && tmp_merge_vectors[0] == tmp_merge_vectors[4]) ||
                (is_in_flag[1] && is_in_flag[4] && tmp_merge_vectors[1] == tmp_merge_vectors[4]))
                is_in_flag[4] = false;
        } else {
            //①
            if (is_in_flag[0] && is_in_flag[1] && tmp_warping_merge_vectors[0] == tmp_warping_merge_vectors[1])
                is_in_flag[1] = false;
            //②
            if (is_in_flag[1] && is_in_flag[2] && tmp_warping_merge_vectors[1] == tmp_warping_merge_vectors[2])
                is_in_flag[2] = false;
            //③
            if (is_in_flag[0] && is_in_flag[3] && tmp_warping_merge_vectors[0] == tmp_warping_merge_vectors[3])
                is_in_flag[3] = false;
            //④
            if ((is_in_flag[0] && is_in_flag[4] && tmp_warping_merge_vectors[0] == tmp_warping_merge_vectors[4]) ||
                (is_in_flag[1] && is_in_flag[4] && tmp_warping_merge_vectors[1] == tmp_warping_merge_vectors[4]))
                is_in_flag[4] = false;
        }
    }
    //重複がなく，符号化済みのブロックのみ入れる
    if(translation_flag) {
        for (j = 0; j < tmp_merge_vectors.size(); j++) {
            if (is_in_flag[j])
                vectors.emplace_back(tmp_merge_vectors[j], MERGE);
        }
    } else {
        for (j = 0; j < tmp_warping_merge_vectors.size(); j++) {
            if (is_in_flag[j]) {
                std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> v;
                v.emplace_back(tmp_warping_merge_vectors[j][0], MERGE);
                v.emplace_back(tmp_warping_merge_vectors[j][1], MERGE);
                v.emplace_back(tmp_warping_merge_vectors[j][2], MERGE);
                warping_vectors.emplace_back(v);
            }
        }
    }

//    std::cout << "square_index : " << square_idx << ", method : " << "MERGE" << ", SquareList_size : " << (translation_flag ? vectors.size() : warping_vectors.size()) << std::endl;

    return {warping_vectors, vectors};
}


/**
 * @fn void SquareDivision::constructPreviousCodingTree(std::vector<CodingTreeUnit*> trees, int pic_num)
 * @brief 過去の動きベクトルを参照するためのTreeを構築する
 * @param trees 分割形状
 * @param pic_num 何枚目のPピクチャか
 */
void SquareDivision::constructPreviousCodingTree(std::vector<CodingTreeUnit*> trees, int pic_num) {

    for(int i = 0 ; i < squares.size() ; i++) {
        previousMvList[0][i]->mv1 = cv::Point2f(0, 0);
        previousMvList[0][i]->mv2 = cv::Point2f(0, 0);
        previousMvList[0][i]->mv3 = cv::Point2f(0, 0);

        auto* node1 = new CollocatedMvTree();
        node1->mv1 = cv::Point2f(0, 0);
        node1->mv2 = cv::Point2f(0, 0);
        node1->mv3 = cv::Point2f(0, 0);

        node1->node1 = new CollocatedMvTree();
        node1->node1->mv1 = cv::Point2f(0, 0);
        node1->node1->mv2 = cv::Point2f(0, 0);
        node1->node1->mv3 = cv::Point2f(0, 0);

        node1->node2 = new CollocatedMvTree();
        node1->node2->mv1 = cv::Point2f(0, 0);
        node1->node2->mv2 = cv::Point2f(0, 0);
        node1->node2->mv3 = cv::Point2f(0, 0);

        node1->node3 = new CollocatedMvTree();
        node1->node3->mv1 = cv::Point2f(0, 0);
        node1->node3->mv2 = cv::Point2f(0, 0);
        node1->node3->mv3 = cv::Point2f(0, 0);

        node1->node4 = new CollocatedMvTree();
        node1->node4->mv1 = cv::Point2f(0, 0);
        node1->node4->mv2 = cv::Point2f(0, 0);
        node1->node4->mv3 = cv::Point2f(0, 0);

        previousMvList[pic_num][i]->node1 = node1;

        auto* node2 = new CollocatedMvTree();
        node2->mv1 = cv::Point2f(0, 0);
        node2->mv2 = cv::Point2f(0, 0);
        node2->mv3 = cv::Point2f(0, 0);

        node2->node1 = new CollocatedMvTree();
        node2->node1->mv1 = cv::Point2f(0, 0);
        node2->node1->mv2 = cv::Point2f(0, 0);
        node2->node1->mv3 = cv::Point2f(0, 0);

        node2->node2 = new CollocatedMvTree();
        node2->node2->mv1 = cv::Point2f(0, 0);
        node2->node2->mv2 = cv::Point2f(0, 0);
        node2->node2->mv3 = cv::Point2f(0, 0);

        node2->node3 = new CollocatedMvTree();
        node2->node3->mv1 = cv::Point2f(0, 0);
        node2->node3->mv2 = cv::Point2f(0, 0);
        node2->node3->mv3 = cv::Point2f(0, 0);

        node2->node4 = new CollocatedMvTree();
        node2->node4->mv1 = cv::Point2f(0, 0);
        node2->node4->mv2 = cv::Point2f(0, 0);
        node2->node4->mv3 = cv::Point2f(0, 0);

        previousMvList[pic_num][i]->node2 = node2;

        auto* node3 = new CollocatedMvTree();
        node3->mv1 = cv::Point2f(0, 0);
        node3->mv2 = cv::Point2f(0, 0);
        node3->mv3 = cv::Point2f(0, 0);

        node3->node1 = new CollocatedMvTree();
        node3->node1->mv1 = cv::Point2f(0, 0);
        node3->node1->mv2 = cv::Point2f(0, 0);
        node3->node1->mv3 = cv::Point2f(0, 0);

        node3->node2 = new CollocatedMvTree();
        node3->node2->mv1 = cv::Point2f(0, 0);
        node3->node2->mv2 = cv::Point2f(0, 0);
        node3->node2->mv3 = cv::Point2f(0, 0);

        node3->node3 = new CollocatedMvTree();
        node3->node3->mv1 = cv::Point2f(0, 0);
        node3->node3->mv2 = cv::Point2f(0, 0);
        node3->node3->mv3 = cv::Point2f(0, 0);

        node3->node4 = new CollocatedMvTree();
        node3->node4->mv1 = cv::Point2f(0, 0);
        node3->node4->mv2 = cv::Point2f(0, 0);
        node3->node4->mv3 = cv::Point2f(0, 0);

        previousMvList[pic_num][i]->node3 = node3;

        auto* node4 = new CollocatedMvTree();
        node4->mv1 = cv::Point2f(0, 0);
        node4->mv2 = cv::Point2f(0, 0);
        node4->mv3 = cv::Point2f(0, 0);

        node4->node1 = new CollocatedMvTree();
        node4->node1->mv1 = cv::Point2f(0, 0);
        node4->node1->mv2 = cv::Point2f(0, 0);
        node4->node1->mv3 = cv::Point2f(0, 0);

        node4->node2 = new CollocatedMvTree();
        node4->node2->mv1 = cv::Point2f(0, 0);
        node4->node2->mv2 = cv::Point2f(0, 0);
        node4->node2->mv3 = cv::Point2f(0, 0);

        node4->node3 = new CollocatedMvTree();
        node4->node3->mv1 = cv::Point2f(0, 0);
        node4->node3->mv2 = cv::Point2f(0, 0);
        node4->node3->mv3 = cv::Point2f(0, 0);

        node4->node4 = new CollocatedMvTree();
        node4->node4->mv1 = cv::Point2f(0, 0);
        node4->node4->mv2 = cv::Point2f(0, 0);
        node4->node4->mv3 = cv::Point2f(0, 0);

        previousMvList[pic_num][i]->node4 = node4;
    }

}


/**
 * @fn void SquareDivision::constructPreviousCodingTree(std::vector<CollocatedMvTree*> trees)
 * @brief 木を再帰的に呼び出し構築する
 * @param codingTree 分割結果を表す木
 * @param constructedTree 構築するための木
 */
void SquareDivision::constructPreviousCodingTree(CodingTreeUnit* codingTree, CollocatedMvTree* constructedTree) {
    constructedTree->mv1 = codingTree->mv1;
    constructedTree->mv2 = codingTree->mv2;
    constructedTree->mv3 = codingTree->mv3;

    if(codingTree->node1 != nullptr) {
        constructedTree->node1 = new CollocatedMvTree();
        constructPreviousCodingTree(codingTree->node1, constructedTree->node1);
    }
    if(codingTree->node2 != nullptr) {
        constructedTree->node2 = new CollocatedMvTree();
        constructPreviousCodingTree(codingTree->node2, constructedTree->node2);
    }
    if(codingTree->node3 != nullptr) {
        constructedTree->node3 = new CollocatedMvTree();
        constructPreviousCodingTree(codingTree->node3, constructedTree->node3);
    }
    if(codingTree->node4 != nullptr) {
        constructedTree->node4 = new CollocatedMvTree();
        constructPreviousCodingTree(codingTree->node4, constructedTree->node4);
    }


}

/**
 * @fn bool isVectorExists(const std::vector<std::tuple<cv::Point2f, int, MV_CODE_METHOD>> &vectors, const cv::Point2f &mv)
 * @brief mvがvectorsに含まれるか判定する
 * @param vectors なんかやばめのtupleを持ってるvector
 * @param mv 動きベクトル
 * @return vectorsにmvが含まれていればtrue, 存在しなければfalse
 */
bool SquareDivision::isMvExists(const std::vector<std::pair<cv::Point2f, MV_CODE_METHOD>> &vectors, const cv::Point2f &mv){
    for(auto vector : vectors) {
        if(vector.first == mv) {
            return true;
        }
    }
    return false;
}

/**
 * @fn std::tuple<cv::Point2f, int, MV_CODE_METHOD> RD(int square_idx, CodingTreeUnit* ctu)
 * @brief RDを行い，最適な差分ベクトルを返す
 * @param[in] mv 動きベクトル
 * @param[in] square_idx 四角パッチの番号
 * @param[in] residual そのパッチの残差
 * @param[in] ctu CodingTreeUnit 符号木
 * @return 差分ベクトル，参照したパッチ，空間or時間のフラグのtuple
 */
std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD> SquareDivision::getMVD(std::vector<cv::Point2f> mv, double residual, int square_idx, int square_number, cv::Point2f &collocated_mv, CodingTreeUnit* ctu, bool translation_flag, std::vector<cv::Point2f> &pixels, int steps){
    std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> vectors;
    std::vector<std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >>> warping_vectors;   // ベクトルとモードを表すフラグのペア
    // 空間予測と時間予測の候補を取り出す
    std::tie(warping_vectors, vectors) = getSpatialSquareList(square_idx, translation_flag);

#if MVD_DEBUG_LOG
    std::cout << corners[squares[square_idx].p1_idx] << " " << corners[squares[square_idx].p2_idx] << " " << corners[squares[square_idx].p3_idx] << " " << corners[squares[square_idx].p4_idx] << std::endl;
#endif

    if(translation_flag) {
        if(!isMvExists(vectors, collocated_mv)) {
            vectors.emplace_back(collocated_mv, Collocated);
        }
    }
    else {
        std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> v;
        v.emplace_back(collocated_mv, Collocated);
        v.emplace_back(collocated_mv, Collocated);
        v.emplace_back(collocated_mv, Collocated);
        warping_vectors.emplace_back(v);
    }

    if(vectors.size() < 2) {
        vectors.emplace_back(cv::Point2f(0.0, 0.0), SPATIAL);
    }
    if(warping_vectors.size() < 2) {
        std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> v;
        v.emplace_back(cv::Point2f(0.0, 0.0), SPATIAL);
        v.emplace_back(cv::Point2f(0.0, 0.0), SPATIAL);
        v.emplace_back(cv::Point2f(0.0, 0.0), SPATIAL);
        warping_vectors.emplace_back(v);
    }

    double lambda = getLambdaPred(qp, (translation_flag ? 1.0 : 1.0));

    //                      コスト, 差分ベクトル, 番号, タイプ
    std::vector<std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum, Flags> > results;
    if(translation_flag) { // 平行移動成分に関してはこれまで通りにやる
        for (int i = 0; i < vectors.size(); i++) {
            std::pair<cv::Point2f, MV_CODE_METHOD> vector = vectors[i];
            cv::Point2f current_mv = vector.first;

            FlagsCodeSum flag_code_sum(0, 0, 0, 0);
            Flags flags;

            cv::Point2f mvd = current_mv - mv[0];
#if MVD_DEBUG_LOG
            std::cout << "target_vector_idx       :" << i << std::endl;
            std::cout << "diff_target_mv(translation):" << current_mv << std::endl;
            std::cout << "encode_mv(translation)     :" << mv[0] << std::endl;
#endif
            mvd = getQuantizedMv(mvd, 4);

            // 正負の判定(使ってません！！！)
            bool is_x_minus = mvd.x < 0;
            bool is_y_minus = mvd.y < 0;

            flags.x_sign_flag.emplace_back(is_x_minus);
            flags.y_sign_flag.emplace_back(is_y_minus);

#if MVD_DEBUG_LOG
            std::cout << "mvd(translation)           :" << mvd << std::endl;
#endif
            mvd.x = std::fabs(mvd.x);
            mvd.y = std::fabs(mvd.y);

            mvd *= 4;
            int abs_x = mvd.x;
            int abs_y = mvd.y;
#if MVD_DEBUG_LOG
            std::cout << "4 * mvd(translation)       :" << mvd << std::endl;
#endif

            // 動きベクトル差分の絶対値が0より大きいのか？
            bool is_x_greater_than_zero = abs_x > 0;
            bool is_y_greater_than_zero = abs_y > 0;

            flags.x_greater_0_flag.emplace_back(is_x_greater_than_zero);
            flags.y_greater_0_flag.emplace_back(is_y_greater_than_zero);

            flag_code_sum.countGreater0Code();
            flag_code_sum.countGreater0Code();
            flag_code_sum.setXGreater0Flag(is_x_greater_than_zero);
            flag_code_sum.setYGreater0Flag(is_y_greater_than_zero);

            // 動きベクトル差分の絶対値が1より大きいのか？
            bool is_x_greater_than_one = abs_x > 1;
            bool is_y_greater_than_one = abs_y > 1;

            flags.x_greater_1_flag.emplace_back(is_x_greater_than_one);
            flags.y_greater_1_flag.emplace_back(is_y_greater_than_one);

            int mvd_code_length = 2;
            if (is_x_greater_than_zero) {
                mvd_code_length += 1;

                if (is_x_greater_than_one) {
                    int mvd_x_minus_2 = mvd.x - 2.0;
                    mvd.x -= 2.0;
                    mvd_code_length += getExponentialGolombCodeLength((int) mvd_x_minus_2, 0);
                    flag_code_sum.addMvdCodeLength(getExponentialGolombCodeLength((int) mvd_x_minus_2, 0));
                }

                flag_code_sum.countGreater1Code();
                flag_code_sum.setXGreater1Flag(is_x_greater_than_one);
                flag_code_sum.countSignFlagCode();
            }

            if (is_y_greater_than_zero) {
                mvd_code_length += 1;

                if (is_y_greater_than_one) {
                    int mvd_y_minus_2 = mvd.y - 2.0;
                    mvd.y -= 2.0;
                    mvd_code_length += getExponentialGolombCodeLength((int) mvd_y_minus_2, 0);
                    flag_code_sum.addMvdCodeLength(getExponentialGolombCodeLength((int) mvd_y_minus_2, 0));
                }

                flag_code_sum.countGreater1Code();
                flag_code_sum.setYGreater1Flag(is_y_greater_than_one);
                flag_code_sum.countSignFlagCode();
            }

//            std::cout << "mvd_code_length" << mvd_code_length << std::endl;

            // 参照箇所符号化
            int reference_index = i; //std::get<1>(vector);
            int reference_index_code_length = getUnaryCodeLength(reference_index);

            // 各種フラグ分を(3*2)bit足してます
            double rd = residual + lambda * (mvd_code_length + reference_index_code_length + flags_code);

            std::vector<cv::Point2f> mvds{mvd};
            // 結果に入れる
            results.emplace_back(rd, mvd_code_length + reference_index_code_length + flags_code, mvds, i, vector.second, flag_code_sum, flags);
        }
    }else{
        for (int i = 0; i < warping_vectors.size(); i++) {
            std::vector<cv::Point2f> mvds;
            mvds.emplace_back(warping_vectors[i][0].first - mv[0]);
            mvds.emplace_back(warping_vectors[i][1].first - mv[1]);
            mvds.emplace_back(warping_vectors[i][2].first - mv[2]);

            int mvd_code_length = 6;
            FlagsCodeSum flag_code_sum(0, 0, 0, 0);
            Flags flags;
            for(int j = 0 ; j < mvds.size() ; j++){

#if MVD_DEBUG_LOG
                std::cout << "target_vector_idx       :" << j << std::endl;
                std::cout << "diff_target_mv(warping) :" << warping_vectors[i][0].first << ", " << warping_vectors[i][1].first << ", " << warping_vectors[i][2].first << std::endl;
                std::cout << "encode_mv(warping)      :" << mv[j] << std::endl;
#endif
                cv::Point2f mvd = getQuantizedMv(mvds[j], 4);

                // 正負の判定
                bool is_x_minus = mvd.x < 0;
                bool is_y_minus = mvd.y < 0;
                flags.x_sign_flag.emplace_back(is_x_minus);
                flags.y_sign_flag.emplace_back(is_y_minus);

                mvd.x = std::fabs(mvd.x);
                mvd.y = std::fabs(mvd.y);
#if MVD_DEBUG_LOG
                std::cout << "mvd(warping)            :" << mvd << std::endl;
#endif
                mvd *= 4;
                mvds[j] = mvd;

#if MVD_DEBUG_LOG
                std::cout << "4 * mvd(warping)        :" << mvd << std::endl;
#endif
                int abs_x = mvd.x;
                int abs_y = mvd.y;

                // 動きベクトル差分の絶対値が0より大きいのか？
                bool is_x_greater_than_zero = abs_x > 0;
                bool is_y_greater_than_zero = abs_y > 0;

                flags.x_greater_0_flag.emplace_back(is_x_greater_than_zero);
                flags.y_greater_0_flag.emplace_back(is_y_greater_than_zero);

                flag_code_sum.countGreater0Code();
                flag_code_sum.countGreater0Code();
                flag_code_sum.setXGreater0Flag(is_x_greater_than_zero);
                flag_code_sum.setYGreater0Flag(is_y_greater_than_zero);

                // 動きベクトル差分の絶対値が1より大きいのか？
                bool is_x_greater_than_one = abs_x > 1;
                bool is_y_greater_than_one = abs_y > 1;

                flags.x_greater_1_flag.emplace_back(is_x_greater_than_one);
                flags.y_greater_1_flag.emplace_back(is_y_greater_than_one);

                if(is_x_greater_than_zero){
                    mvd_code_length += 1;

                    if(is_x_greater_than_one){
                        int mvd_x_minus_2 = mvd.x - 2.0;
                        mvd.x -= 2.0;
                        mvd_code_length += getExponentialGolombCodeLength((int) mvd_x_minus_2, 0);
                        flag_code_sum.addMvdCodeLength(getExponentialGolombCodeLength((int) mvd_x_minus_2, 0));
                    }

                    flag_code_sum.countGreater1Code();
                    flag_code_sum.setXGreater1Flag(is_x_greater_than_one);
                    flag_code_sum.countSignFlagCode();
                }

                if(is_y_greater_than_zero){
                    mvd_code_length += 1;

                    if(is_y_greater_than_one){
                        int mvd_y_minus_2 = mvd.y - 2.0;
                        mvd.y -= 2.0;
                        mvd_code_length +=  getExponentialGolombCodeLength((int) mvd_y_minus_2, 0);
                        flag_code_sum.addMvdCodeLength(getExponentialGolombCodeLength((int) mvd_y_minus_2, 0));
                    }
                    flag_code_sum.countGreater1Code();
                    flag_code_sum.setYGreater1Flag(is_y_greater_than_one);
                    flag_code_sum.countSignFlagCode();
                }
                mvds[j].x = mvd.x;
                mvds[j].y = mvd.y;
            }

            // 参照箇所符号化
            int reference_index = i; //std::get<1>(vector);
            int reference_index_code_length = getUnaryCodeLength(reference_index);

            // 各種フラグ分を(3*2)bit足してます
            double rd = residual + lambda * (mvd_code_length + reference_index_code_length + flags_code);

            // 結果に入れる
            results.emplace_back(rd, mvd_code_length + reference_index_code_length + flags_code, mvds, i, warping_vectors[i][0].second, flag_code_sum, flags);
        }
    }

#if MERGE_MODE
    // マージ符号化
    // マージで参照する動きベクトルを使って残差を求め直す
    Square current_square_coordinate = squares[square_idx];
    cv::Point2f p1 = corners[current_square_coordinate.p1_idx];
    cv::Point2f p2 = corners[current_square_coordinate.p2_idx];
    cv::Point2f p3 = corners[current_square_coordinate.p3_idx];
    cv::Point2f p4 = corners[current_square_coordinate.p4_idx];
    Point4Vec coordinate = Point4Vec(p1, p2, p3, p4);
    std::vector<cv::Point2f> pixels_in_square;
    if(pixels.empty()) {
        pixels_in_square = getPixelsInSquare(coordinate);
    }else{
        pixels_in_square = pixels;
    }

//    if(steps == 2) {
//        //CU内部の四角形を参照できないように符号化済みフラグをfalseにする
//        if (square_number != 4) {
//            for (int i = 0; i < square_number; i++) {
//                isCodedSquare[square_idx - (i + 1)] = false;
//            }
//        }
//    }

    //マージ候補のリストを作成
    warping_vectors.clear(); vectors.clear();
    std::tie(warping_vectors, vectors) = getMergeSquareList(square_idx, translation_flag, coordinate);

    if(vectors.size() < 5) {
        vectors.emplace_back(cv::Point2f(0.0, 0.0), MERGE);
    }
    if(warping_vectors.size() < 5) {
        std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> v;
        v.emplace_back(cv::Point2f(0.0, 0.0), MERGE);
        v.emplace_back(cv::Point2f(0.0, 0.0), MERGE);
        v.emplace_back(cv::Point2f(0.0, 0.0), MERGE);
        warping_vectors.emplace_back(v);
    }

//    if(steps == 2) {
//        //マージ候補は作成できたので，符号化済みフラグをtrueにする
//        if (square_number != 4) {
//            for (int i = 0; i < square_number; i++) {
//                isCodedSquare[square_idx - (i + 1)] = true;
//            }
//        }
//    }

    int merge_count = 0;

    if(translation_flag) {
        for (int i = 0; i < vectors.size(); i++) {
            std::pair<cv::Point2f, MV_CODE_METHOD> merge_vector = vectors[i];
            cv::Point2f current_mv = merge_vector.first;
            std::vector<cv::Point2f> mvds;
            std::vector<cv::Point2f> mvs;

            mvs.emplace_back(current_mv);
            mvs.emplace_back(current_mv);
            mvs.emplace_back(current_mv);

            double ret_residual = getSquareResidual_Pred(target_image, coordinate, mvs, pixels_in_square, ref_hevc);
            double rd = (ret_residual + lambda * (getUnaryCodeLength(merge_count) + flags_code)) * MERGE_ALPHA;
            results.emplace_back(rd, getUnaryCodeLength(merge_count) + flags_code, mvs, merge_count, merge_vector.second, FlagsCodeSum(0, 0, 0, 0), Flags());
            merge_count++;
        }
    }else {
        for (int i = 0; i < warping_vectors.size(); i++) {
            std::vector<cv::Point2f> mvs;

            mvs.emplace_back(warping_vectors[i][0].first);
            mvs.emplace_back(warping_vectors[i][1].first);
            mvs.emplace_back(warping_vectors[i][2].first);

            double ret_residual = getSquareResidual_Pred(target_image, coordinate, mvs, pixels_in_square, ref_hevc);
            double rd = (ret_residual + lambda * (getUnaryCodeLength(merge_count) + flags_code)) * MERGE_ALPHA;
            results.emplace_back(rd, getUnaryCodeLength(merge_count) + flags_code, mvs, merge_count, warping_vectors[i][0].second, FlagsCodeSum(0, 0, 0, 0), Flags());
            merge_count++;
        }
    }
#endif

    // RDしたスコアが小さい順にソート
    std::sort(results.begin(), results.end(), [](const std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum, Flags >& a, const std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum, Flags>& b){
        return std::get<0>(a) < std::get<0>(b);
    });

    //4分割の判定の為にRDコスト(mode)を計算し直す
    double cost = std::get<0>(results[0]);
    int code_length = std::get<1>(results[0]);
    std::vector<cv::Point2f> mvds = std::get<2>(results[0]);
    int selected_idx = std::get<3>(results[0]);
    MV_CODE_METHOD method = std::get<4>(results[0]);
    FlagsCodeSum flag_code_sum = std::get<5>(results[0]);
    Flags result_flags = std::get<6>(results[0]);
    ctu->x_greater_0_flag = result_flags.x_greater_0_flag;
    ctu->y_greater_0_flag = result_flags.y_greater_0_flag;
    ctu->x_greater_1_flag = result_flags.x_greater_1_flag;
    ctu->y_greater_1_flag = result_flags.y_greater_1_flag;
    ctu->x_sign_flag = result_flags.x_sign_flag;
    ctu->y_sign_flag = result_flags.y_sign_flag;
    ctu->flags_code_sum = flag_code_sum;
    if(method != MV_CODE_METHOD::MERGE) {
        (ctu->mvds_x).clear();
        (ctu->mvds_y).clear();
        (ctu->original_mvds_x).clear();
        (ctu->original_mvds_y).clear();

        if (translation_flag) {
            (ctu->mvds_x).emplace_back(mvds[0].x);
            (ctu->mvds_y).emplace_back(mvds[0].y);
        } else {
            for (int i = 0; i < 3; i++) {
                (ctu->mvds_x).emplace_back(mvds[i].x);
                (ctu->mvds_y).emplace_back(mvds[i].y);
            }
        }
    }

#if SPLIT_USE_SSE
    double RDCost;
    lambda = getLambdaMode(qp);
    if(method == MV_CODE_METHOD::MERGE){
        if(translation_flag) {
            double ret_residual = getSquareResidual_Mode(target_image, mvds, pixels_in_square, expansion_ref);
            RDCost = ret_residual + lambda * code_length;
        }else{
            double ret_residual = getSquareResidual_Mode(target_image, mvds, pixels_in_square, expansion_ref);
            RDCost = ret_residual + lambda * code_length;
        }
    }
    else if(MV_CODE_METHOD::MERGE_Collocated) {
        if(translation_flag) {
            double ret_residual = getSquareResidual_Mode(target_image, mvds, pixels_in_square, expansion_ref);
            RDCost = ret_residual + lambda * code_length;
        }
    }
    else {
        std::pair<cv::Point2f, MV_CODE_METHOD> vector = vectors[selected_idx];
        cv::Point2f current_mv = vector.first;
        // TODO: ワーピング対応
        if (translation_flag) {
            double ret_residual = getSquareResidual_Mode(target_image, {current_mv, current_mv, current_mv}, pixels_in_square, expansion_ref);
            RDCost = ret_residual + lambda * code_length;
        } else {
            double ret_residual = getSquareResidual_Mode(target_image, {current_mv, current_mv, current_mv}, pixels_in_square, expansion_ref);
            RDCost = ret_residual + lambda * code_length;
        }
    }

    return {RDCost, code_length, mvds, selected_idx, method};
#endif
    return {cost, code_length, mvds, selected_idx, method};
}


/**
 * @fn double  SquareDivision::getRDCost(std::vector<cv::Point2f> mv, double residual, int square_idx, cv::Point2f &collocated_mv, CodingTreeUnit* ctu, bool translation_flag, std::vector<cv::Point2f> &pixels, std::vector<int> spatial_squares)
 * @brief RDを行い，最適な差分ベクトルを返す
 * @param[in] mv 動きベクトル
 * @param[in] square_idx 四角パッチの番号
 * @param[in] residual そのパッチの残差
 * @param[in] ctu CodingTreeUnit 符号木
 * @return RDコスト
 */
double  SquareDivision::getRDCost(std::vector<cv::Point2f> mv, double residual, int square_idx, cv::Point2f &collocated_mv, CodingTreeUnit* ctu, std::vector<cv::Point2f> &pixels, std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> vectors){
    if(vectors.size() < 2) {
        vectors.emplace_back(cv::Point2f(0.0, 0.0), Collocated);
    }

    double lambda = getLambdaPred(qp, 1.0);

    //                      コスト, 差分ベクトル, 番号, タイプ
    std::vector<std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum, Flags> > results;
    for (int i = 0; i < vectors.size(); i++) {
        std::pair<cv::Point2f, MV_CODE_METHOD> vector = vectors[i];
        cv::Point2f current_mv = vector.first;
        FlagsCodeSum flag_code_sum(0, 0, 0, 0);
        Flags flags;

        cv::Point2f mvd = current_mv - mv[0];
        mvd = getQuantizedMv(mvd, 4);

        // 正負の判定(使ってません！！！)
        bool is_x_minus = mvd.x < 0;
        bool is_y_minus = mvd.y < 0;

        flags.x_sign_flag.emplace_back(is_x_minus);
        flags.y_sign_flag.emplace_back(is_y_minus);

        mvd.x = std::fabs(mvd.x);
        mvd.y = std::fabs(mvd.y);

        mvd *= 4;
        int abs_x = mvd.x;
        int abs_y = mvd.y;

        // 動きベクトル差分の絶対値が0より大きいのか？
        bool is_x_greater_than_zero = abs_x > 0;
        bool is_y_greater_than_zero = abs_y > 0;

        flags.x_greater_0_flag.emplace_back(is_x_greater_than_zero);
        flags.y_greater_0_flag.emplace_back(is_y_greater_than_zero);

        flag_code_sum.countGreater0Code();
        flag_code_sum.countGreater0Code();
        flag_code_sum.setXGreater0Flag(is_x_greater_than_zero);
        flag_code_sum.setYGreater0Flag(is_y_greater_than_zero);

        // 動きベクトル差分の絶対値が1より大きいのか？
        bool is_x_greater_than_one = abs_x > 1;
        bool is_y_greater_than_one = abs_y > 1;

        flags.x_greater_1_flag.emplace_back(is_x_greater_than_one);
        flags.y_greater_1_flag.emplace_back(is_y_greater_than_one);

        int mvd_code_length = 2;
        if (is_x_greater_than_zero) {
            mvd_code_length += 1;

            if (is_x_greater_than_one) {
                int mvd_x_minus_2 = mvd.x - 2.0;
                mvd.x -= 2.0;
                mvd_code_length += getExponentialGolombCodeLength((int) mvd_x_minus_2, 0);
                flag_code_sum.addMvdCodeLength(getExponentialGolombCodeLength((int) mvd_x_minus_2, 0));
            }

            flag_code_sum.countGreater1Code();
            flag_code_sum.setXGreater1Flag(is_x_greater_than_one);
            flag_code_sum.countSignFlagCode();
        }

        if (is_y_greater_than_zero) {
            mvd_code_length += 1;

            if (is_y_greater_than_one) {
                int mvd_y_minus_2 = mvd.y - 2.0;
                mvd.y -= 2.0;
                mvd_code_length += getExponentialGolombCodeLength((int) mvd_y_minus_2, 0);
                flag_code_sum.addMvdCodeLength(getExponentialGolombCodeLength((int) mvd_y_minus_2, 0));
            }

            flag_code_sum.countGreater1Code();
            flag_code_sum.setYGreater1Flag(is_y_greater_than_one);
            flag_code_sum.countSignFlagCode();
        }

        // 参照箇所符号化
        int reference_index = i; //std::get<1>(vector);
        int reference_index_code_length = getUnaryCodeLength(reference_index);

        // 各種フラグ分を(3*2)bit足してます
        double rd = residual + lambda * (mvd_code_length + reference_index_code_length + flags_code);

        std::vector<cv::Point2f> mvds{mvd};
        // 結果に入れる
        results.emplace_back(rd, mvd_code_length + reference_index_code_length, mvds, i, vector.second, flag_code_sum, flags);
    }

    // RDしたスコアが小さい順にソート
    std::sort(results.begin(), results.end(), [](const std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum, Flags >& a, const std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum, Flags>& b){
        return std::get<0>(a) < std::get<0>(b);
    });
    double cost = std::get<0>(results[0]);
    return cost;
}


/**
 * @fn cv::Point2f SquareDivision::getQuantizedMv(cv::Point2f mv, int quantize_step)
 * @param mv 動きベクトル
 * @param quantize_step 量子化ステップ幅
 * @return 量子化済みの動きベクトル
 */
cv::Point2f SquareDivision::getQuantizedMv(cv::Point2f &mv, double quantize_step){
    cv::Point2f ret(mv.x, mv.y);

    double half_quantize_step = 1.0 / quantize_step / 2.0;
    if(ret.x < 0) {
        ret.x -= half_quantize_step;
    }else{
        ret.x += half_quantize_step;
    }

    if(ret.y < 0) {
        ret.y -= half_quantize_step;
    }else{
        ret.y += half_quantize_step;
    }

    ret.x = (int)(ret.x * quantize_step);
    ret.y = (int)(ret.y * quantize_step);

    ret.x /= quantize_step;
    ret.y /= quantize_step;

    return ret;
}

cv::Mat SquareDivision::getPredictedDiagonalImageFromCtu(std::vector<CodingTreeUnit*> ctus){
    cv::Mat out = cv::Mat::zeros(ref_image.size(), CV_8UC3);
    for(int i = 0 ; i < ctus.size() ; i++) {
        getPredictedDiagonalImageFromCtu(ctus[i], out);
    }

    return out;
}

void SquareDivision::getPredictedDiagonalImageFromCtu(CodingTreeUnit* ctu, const cv::Mat &out){

    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int square_index = ctu->square_index;
        Square square_corner_idx = squares[square_index];
        Point4Vec square(corners[square_corner_idx.p1_idx], corners[square_corner_idx.p2_idx], corners[square_corner_idx.p3_idx], corners[square_corner_idx.p4_idx]);
        std::vector<cv::Point2f> pixels = getPixelsInSquare(square);
        std::random_device rnd;     // 非決定的な乱数生成器
        std::mt19937 mt(rnd());
        int r = mt() % 256;
        int g = mt() % 256;
        int b = mt() % 256;

        for (const auto& pixel : pixels) {
            R(out, (int) pixel.x, (int) pixel.y) = r;
            G(out, (int) pixel.x, (int) pixel.y) = g;
            B(out, (int) pixel.x, (int) pixel.y) = b;
        }

        return;
    }

    if(ctu->node1 != nullptr) getPredictedDiagonalImageFromCtu(ctu->node1, out);
    if(ctu->node2 != nullptr) getPredictedDiagonalImageFromCtu(ctu->node2, out);
    if(ctu->node3 != nullptr) getPredictedDiagonalImageFromCtu(ctu->node3, out);
    if(ctu->node4 != nullptr) getPredictedDiagonalImageFromCtu(ctu->node4, out);
}

cv::Mat SquareDivision::getPredictedImageFromCtu(std::vector<CodingTreeUnit*> ctus){
    cv::Mat out = cv::Mat::zeros(ref_image.size(), CV_8UC3);

#pragma omp parallel for
    for(int i = 0 ; i < ctus.size() ; i++) {
        getPredictedImageFromCtu(ctus[i], out);
    }

    return out;
}

void SquareDivision::getPredictedImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int square_index = ctu->square_index;
        cv::Point2f mv = ctu->mv1;
        Square square_corner_idx = squares[square_index];
        Point4Vec square(corners[square_corner_idx.p1_idx], corners[square_corner_idx.p2_idx], corners[square_corner_idx.p3_idx], corners[square_corner_idx.p4_idx]);

        std::vector<cv::Point2f> mvs;
        if(ctu->translation_flag){
            mvs.emplace_back(mv);
            mvs.emplace_back(mv);
            mvs.emplace_back(mv);
        }else{
            mvs.emplace_back(ctu->mv1);
            mvs.emplace_back(ctu->mv2);
            mvs.emplace_back(ctu->mv3);
        }

        getPredictedImage(expansion_ref_uchar, target_image, out, square, mvs, ref_hevc);
        return;
    }

    if(ctu->node1 != nullptr) getPredictedImageFromCtu(ctu->node1, out);
    if(ctu->node2 != nullptr) getPredictedImageFromCtu(ctu->node2, out);
    if(ctu->node3 != nullptr) getPredictedImageFromCtu(ctu->node3, out);
    if(ctu->node4 != nullptr) getPredictedImageFromCtu(ctu->node4, out);
}
//select  0 : 四角形の輪郭が白になる , 1 : 四角形の頂点が赤くなる
cv::Mat SquareDivision::getPredictedColorImageFromCtu(std::vector<CodingTreeUnit*> ctus, double original_psnr, int select){
    cv::Mat out = cv::Mat::zeros(ref_image.size(), CV_8UC3);

    std::vector<cv::Scalar> colors;

    colors.emplace_back(YELLOW);
    colors.emplace_back(BLUE);
    colors.emplace_back(GREEN);
    colors.emplace_back(LIGHT_BLUE);
    colors.emplace_back(RED);
    colors.emplace_back(PURPLE);

//#pragma omp parallel for
    for(int i = 0 ; i < ctus.size() ; i++) {
        getPredictedColorImageFromCtu(ctus[i], out, original_psnr, colors);
    }

    std::vector<Point4Vec> ss = getSquareCoordinateList();
    for(const auto &t : ss) {
        drawSquare(out, t.p1, t.p2, t.p3, t.p4, cv::Scalar(255, 255, 255), select);
    }

    return out;
}

void SquareDivision::getPredictedColorImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out, double original_psnr, std::vector<cv::Scalar> &colors){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int square_index = ctu->square_index;
        cv::Point2f mv = ctu->mv1;
        Square square_corner_idx = squares[square_index];
        Point4Vec square(corners[square_corner_idx.p1_idx], corners[square_corner_idx.p2_idx], corners[square_corner_idx.p3_idx], corners[square_corner_idx.p4_idx]);

        std::vector<cv::Point2f> mvs{mv, mv, mv};
        std::vector<cv::Point2f> pixels = getPixelsInSquare(square);

        if(ctu->translation_flag) {
            if(ctu->method == MV_CODE_METHOD::MERGE){
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    G(out, (int)pixel.x, (int)pixel.y) = 0;
                    B(out, (int)pixel.x, (int)pixel.y) = 0;
                }
            } else if(ctu->method == MV_CODE_METHOD::MERGE_Collocated) {
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    G(out, (int)pixel.x, (int)pixel.y) = 0;
                    B(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                }
            } else{
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = 0;
                    G(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    B(out, (int)pixel.x, (int)pixel.y) = 0;
                }
            }

        } else{
            if(ctu->method == MV_CODE_METHOD::MERGE){
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    G(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    B(out, (int)pixel.x, (int)pixel.y) = 0;
                }
            } else if(ctu->method == MV_CODE_METHOD::MERGE_Collocated) {
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = 0;
                    G(out, (int)pixel.x, (int)pixel.y) = 0;
                    B(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                }
            } else{
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = 0;
                    G(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    B(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                }
            }
//            getPredictedImage(expansion_ref_uchar, target_image, out, square, mvs, ref_hevc);
        }
        return;
    }

    if(ctu->node1 != nullptr) getPredictedColorImageFromCtu(ctu->node1, out, original_psnr, colors);
    if(ctu->node2 != nullptr) getPredictedColorImageFromCtu(ctu->node2, out, original_psnr, colors);
    if(ctu->node3 != nullptr) getPredictedColorImageFromCtu(ctu->node3, out, original_psnr, colors);
    if(ctu->node4 != nullptr) getPredictedColorImageFromCtu(ctu->node4, out, original_psnr, colors);
}

int SquareDivision::getCtuCodeLength(std::vector<CodingTreeUnit*> ctus) {
    int code_length_sum = 0;
    for(auto & ctu : ctus){
        code_length_sum += getCtuCodeLength(ctu);
    }
    return code_length_sum;
}

int SquareDivision::getCtuCodeLength(CodingTreeUnit *ctu){

    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        return ctu->code_length;
    }

    // ここで足している1はsplit_cu_flag分です
    return 1 + getCtuCodeLength(ctu->node1) + getCtuCodeLength(ctu->node2) + getCtuCodeLength(ctu->node3) + getCtuCodeLength(ctu->node4);
}


cv::Mat SquareDivision::getMvImage(std::vector<CodingTreeUnit*> ctus){
    cv::Mat out = target_image.clone();

    for(auto square : getSquareCoordinateList()){
        drawSquare(out, square.p1, square.p2, square.p3, square.p4, cv::Scalar(255, 255, 255), 0);
    }

    for(int i = 0 ; i < ctus.size() ; i++){   //i番目のctuを書いていく
        drawMvImage(out, ctus[i]);
    }

    return out;
}

void SquareDivision::drawMvImage(cv::Mat &out, CodingTreeUnit *ctu){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        Square s = squares[ctu->square_index];
        cv::Point2f p1 = corners[s.p1_idx];
        cv::Point2f p2 = corners[s.p2_idx];
        cv::Point2f p3 = corners[s.p3_idx];
        cv::Point2f p4 = corners[s.p4_idx];

        if(ctu->translation_flag) {
            cv::Point2f g = (p1 + p2 + p3 + p4) / 4.0;

            cv::line(out, g, g + ctu->mv1, GREEN);
        }else{
            cv::line(out, p1, p1 + ctu->mv1, GREEN);
            cv::line(out, p2, p2 + ctu->mv2, GREEN);
            cv::line(out, p3, p3 + ctu->mv3, GREEN);
        }
    }

    if(ctu->node1 != nullptr) drawMvImage(out, ctu->node1);
    if(ctu->node2 != nullptr) drawMvImage(out, ctu->node2);
    if(ctu->node3 != nullptr) drawMvImage(out, ctu->node3);
    if(ctu->node4 != nullptr) drawMvImage(out, ctu->node4);
}

SquareDivision::SquareDivision() {}

SquareDivision::SplitResult::SplitResult(const Point4Vec &s1, const Point4Vec &s2, int type) : s1(s1), s2(s2), s_type(type) {}

SquareDivision::GaussResult::GaussResult(const std::vector<cv::Point2f> &mvWarping, const cv::Point2f &mvTranslation,
                                         double residual_warping, double residual_translation, int squareSize, bool translationFlag, double residualBm) : mv_warping(
        mvWarping), mv_translation(mvTranslation), residual_warping(residual_warping), residual_translation(residual_translation), square_size(squareSize), translation_flag(translationFlag), residual_bm(residualBm) {}

SquareDivision::GaussResult::GaussResult() {}

std::tuple<std::vector<cv::Point2f>, std::vector<double>> SquareDivision::blockMatching(Point4Vec square, const cv::Mat& target_image, cv::Mat expansion_ref_image, int square_index, CodingTreeUnit *ctu) {
    double sx, sy, lx, ly;
    cv::Point2f sp1, sp4;

    sp1 = square.p1;
    sp4 = square.p4;

    sx = 4 * sp1.x;
    sy = 4 * sp1.y;
    lx = 4 * sp4.x + 3;
    ly = 4 * sp4.y + 3;

    cv::Point2f mv_tmp(0.0, 0.0); //ブロックの動きベクトル
    int SX = SERACH_RANGE;                 // ブロックマッチングの探索範囲(X)
    int SY = SERACH_RANGE;                 // ブロックマッチングの探索範囲(Y)
    int neighbor_pixels = BLOCKMATCHING_NEIGHBOR_PIXELS;     //1 : 近傍 1 画素,  2 : 近傍 2 画素,   n : 近傍 n 画素

    double rd, sad;
    double rd_min = 1e9, sad_min = 1e9;

    cv::Point2f mv_min;
    int spread_quarter = 4 * SERACH_RANGE;
    int s = 4;                   //4 : Full-pel, 2 : Half-pel, 1 : Quarter-pel
    std::vector<cv::Point2f> pixels = getPixelsInSquare(square);
    std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> vectors;
    std::tie(std::ignore, vectors) = getSpatialSquareList(square_index, true);

    for(int j = -SY * 4 ; j <= SY * 4 ; j += s) {            //j : y方向のMV
        for(int i = -SX * 4 ; i <= SX * 4 ; i += s) {        //i : x方向のMV
            //探索範囲が画像上かどうか判定
            if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
               && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                sad = 0.0;
                for(const auto& pixel : pixels) {
                        sad += fabs(R(expansion_ref_image, i + (int)(4 * pixel.x) + spread_quarter, j + (int)(4 * pixel.y) + spread_quarter) - R(target_image, (int)(pixel.x), (int)(pixel.y)));
                }
                cv::Point2f cmt = cv::Point2f(0.0, 0.0);
                cv::Point2f mv  = cv::Point2f((double)i/4.0, (double)j/4.0);
//                std::tie(rd, std::ignore,std::ignore,std::ignore,std::ignore) = getMVD({mv, mv, mv}, e, square_index, cmt, ctu, true, pixels, spatial_squares);
                rd = getRDCost({mv, mv, mv}, sad, square_index, cmt, ctu, pixels, vectors);
                if(rd_min > rd){
                    sad_min = sad;
                    rd_min = rd;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }

    std::vector<cv::Point2f> mvs;
    std::vector<double> errors;
    mvs.emplace_back(mv_min.x, mv_min.y);
    errors.emplace_back(sad_min);

    mv_tmp.x = mv_min.x * 4;
    mv_tmp.y = mv_min.y * 4;

    s = 2;
    for(int j = - neighbor_pixels * s + mv_tmp.y ; j <= neighbor_pixels * s + mv_tmp.y ; j += s){            //j : y方向のMV
        for(int i = - neighbor_pixels * s + mv_tmp.x ; i <= neighbor_pixels * s + mv_tmp.x ; i += s){        //i : x方向のMV
            if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
               && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                sad = 0.0;
                for(const auto& pixel : pixels) {
                    sad += fabs(R(expansion_ref_image, i + (int)(4 * pixel.x) + spread_quarter, j + (int)(4 * pixel.y) + spread_quarter) - R(target_image, (int)(pixel.x), (int)(pixel.y)));
                }
                cv::Point2f cmt = cv::Point2f(0.0, 0.0);
                cv::Point2f mv  = cv::Point2f((double)i/4.0, (double)j/4.0);
//                std::tie(rd, std::ignore,std::ignore,std::ignore,std::ignore) = getMVD({mv, mv, mv}, e, square_index, cmt, ctu, true, pixels, spatial_squares);
                rd = getRDCost({mv, mv, mv}, sad, square_index, cmt, ctu, pixels, vectors);
                if(rd_min > rd){
                    sad_min = sad;
                    rd_min = rd;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }

    mvs.emplace_back(mv_min.x, mv_min.y);
    errors.emplace_back(sad_min);
    mv_tmp.x = mv_min.x * 4;
    mv_tmp.y = mv_min.y * 4;

    s = 1;

    for(int j = - neighbor_pixels * s + mv_tmp.y ; j <= neighbor_pixels * s + mv_tmp.y ; j += s){            //j : y方向のMV
        for(int i = - neighbor_pixels * s + mv_tmp.x ; i <= neighbor_pixels * s + mv_tmp.x ; i += s){        //i : x方向のMV
            if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
               && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                sad = 0.0;
                for(const auto& pixel : pixels) {
                    sad += fabs(R(expansion_ref_image, i + (int)(4 * pixel.x) + spread_quarter, j + (int)(4 * pixel.y) + spread_quarter) - R(target_image, (int)(pixel.x), (int)(pixel.y)));
                }
                cv::Point2f cmt = cv::Point2f(0.0, 0.0);
                cv::Point2f mv  = cv::Point2f((double)i/4.0, (double)j/4.0);
//                std::tie(rd, std::ignore,std::ignore,std::ignore,std::ignore) = getMVD({mv, mv, mv}, e, square_index, cmt, ctu, true, pixels, spatial_squares);
                rd = getRDCost({mv, mv, mv}, sad, square_index, cmt, ctu, pixels, vectors);
                if(rd_min > rd){
                    sad_min = sad;
                    rd_min = rd;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }

//    if(824 <= square_index && square_index <= 827) {
//        std::cout << "mv : " <<  mv_min << std::endl << std::endl;
//        for(const auto& pixel : pixels) {
//            std::cout << "p : " <<  (int)(R(expansion_ref_image, (int)(mv_min.x) + (int)(4 * pixel.x) + spread_quarter, (int)(mv_min.y) + (int)(4 * pixel.y) + spread_quarter))<<
//            ", target : " << (int)(R(target_image, (int)(pixel.x), (int)(pixel.y))) << std::endl;
//        }
//        std::cout << std::endl;
//    }

    errors.emplace_back(sad_min);
    mvs.emplace_back(mv_min.x, mv_min.y);

    return std::make_tuple(mvs, errors);
}

bool SquareDivision::isMvExists(const std::vector<Point3Vec> &vectors, const std::vector<cv::Point2f> &mvs) {
    for(const auto& vector : vectors){
        if(vector.p1 == mvs[0] && vector.p2 == mvs[1] && vector.p3 == mvs[2]) return true;
    }

    return false;
}

SquareDivision::~SquareDivision() {
    std::vector<cv::Point2f>().swap(corners);
    std::vector<int>().swap(covered_square);
    std::vector<std::vector<int> >().swap(reference_block_list);
    std::vector<std::vector<int> >().swap(merge_reference_block_list);
    std::vector<std::vector<int> >().swap(corner_flag);
    std::vector<bool>().swap(isCodedSquare);
    std::vector<std::vector<CollocatedMvTree*>>().swap(previousMvList);
    std::vector<cv::Mat>().swap(predicted_buf);
    std::vector<GaussResult>().swap(square_gauss_results);
    std::vector<std::vector<cv::Mat>>().swap(ref_images);
    std::vector<std::vector<cv::Mat>>().swap(target_images);

    int scaled_expansion_size = SERACH_RANGE + 2;
    for(int i = -scaled_expansion_size ; i < target_image.cols + scaled_expansion_size ; i++){
        expansion_ref_uchar[i] -= scaled_expansion_size;
        free(expansion_ref_uchar[i]);
    }
    expansion_ref_uchar -= scaled_expansion_size;
    free(expansion_ref_uchar);

    for(int i = 4 * (4 + SERACH_RANGE) ; i < 4 * (ref_image.cols + (4 + SERACH_RANGE)) ; i++) {
        ref_hevc[i] -= 4 * (4 + SERACH_RANGE);
        free(ref_hevc[i]);
    }
    ref_hevc -= 4 * (4 + SERACH_RANGE);
    free(ref_hevc);

}
