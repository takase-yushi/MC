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
    neighbor_vtx.clear();
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

    corner_flag.resize(static_cast<unsigned long>(ref_image.rows * 2));
    for(int i = 0 ; i < ref_image.rows * 2 ; i++) {
        corner_flag[i].resize(static_cast<unsigned long>(ref_image.cols * 2));
    }

    for(int y = 0 ; y < ref_image.rows * 2; y++) {
        for(int x = 0 ; x < ref_image.cols * 2; x++) {
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
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            same_corner_list.emplace_back();
            same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1); // 他と共有している頂点は、自分の番号だけ入れる
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());

            nx = (block_x + 1) * (block_size_x) - 1;   //ブロックの右上のx座標
            ny = (block_y) * (block_size_y);          //ブロックの右上のy座標

            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            same_corner_list.emplace_back();
            same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1);
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
        }

        //y列目の下側の頂点を入れる
        for (int block_x = 0 ; block_x < block_num_x; block_x++) {
            int nx = block_x * (block_size_x);              //ブロックの左下のx座標
            int ny = (block_y + 1) * (block_size_y) - 1;    //ブロックの左下のy座標

            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            same_corner_list.emplace_back();
            same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1);;
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());

            nx = (block_x + 1) * (block_size_x) - 1;    //ブロックの右下のx座標
            ny = (block_y + 1) * (block_size_y) - 1;    //ブロックの右下のy座標

            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            same_corner_list.emplace_back();
            same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1);
            neighbor_vtx.emplace_back();

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
            addNeighborVertex(p1_idx, p2_idx, p3_idx, p4_idx);
            addCoveredSquare(p1_idx, p2_idx, p3_idx, p4_idx, squareIndex); // p1/p2/p3はsquareIndex番目の四角形に含まれている
        }
    }

    for(int i = 0 ; i < isCodedSquare.size() ; i++) {
        isCodedSquare[i] = false;
    }

    delete_flag.resize(squares.size());
    for(int i = 0 ; i < delete_flag.size() ; i++) {
        delete_flag[i] = false;
    }

    predicted_buf.emplace_back(cv::Mat::zeros(ref_image.size()/8, CV_8UC3));
    predicted_buf.emplace_back(cv::Mat::zeros(ref_image.size()/4, CV_8UC3));
    predicted_buf.emplace_back(cv::Mat::zeros(ref_image.size()/2, CV_8UC3));
    predicted_buf.emplace_back(cv::Mat::zeros(ref_image.size(), CV_8UC3));

    ref_images = getRefImages(ref_image, ref_gauss_image);
    target_images = getTargetImages(target_image);

    int expansion_size = 16;
    int scaled_expansion_size = expansion_size + 2;
    if(HEVC_REF_IMAGE) expansion_ref = getExpansionMatHEVCImage(ref_image, 4, expansion_size);
    else expansion_ref = getExpansionMatImage(ref_image, 4, scaled_expansion_size);

    ref_hevc = getExpansionHEVCImage(ref_image, 4, 16);

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

    // 0行目
    for(int block_x = 1 ; block_x < (block_num_x * 2) - 1; block_x+=2){              //   ---------------     ---------------
        int p1_idx = block_x;                                                        //   |          p1 |     | p2          |
        int p2_idx = block_x + 1;                                                    //   |             |     |             |
        same_corner_list[p2_idx].emplace(p1_idx);                                    //   |             |     |             |
    }                                                                                //   |             |     |             |
                                                                                     //   ---------------     ---------------
    for(int block_y = 1 ; block_y < (2 * block_num_y - 1) ; block_y+=2){             //
        for(int block_x = 1 ; block_x < (block_num_x * 2 - 1) ; block_x+=2){         //   ---------------     ---------------
            int p1_idx = block_x +     2 * block_num_x * block_y;                    //   |             |     |             |
            int p2_idx = block_x + 1 + 2 * block_num_x * block_y;                    //   |             |     |             |
            same_corner_list[p2_idx].emplace(p1_idx);                                //   |             |     |             |
                                                                                     //   |          p1 |     | p2          |
             int p3_idx = p1_idx + 2 * block_num_x;                                  //   ---------------     ---------------
             int p4_idx = p3_idx + 1;                                                //
                                                                                     //   ---------------     ---------------
            same_corner_list[p2_idx].emplace(p3_idx);                                //   |          p3 |     | p4          |
                                                                                     //   |             |     |             |
            same_corner_list[p3_idx].emplace(p1_idx);                                //   |             |     |             |
            same_corner_list[p3_idx].emplace(p2_idx);                                //   |             |     |             |
                                                                                     //   ---------------     ---------------
            same_corner_list[p4_idx].emplace(p1_idx);
            same_corner_list[p4_idx].emplace(p2_idx);
        }
    }

    std::cout << same_corner_list.size() << std::endl;
    // 最下行目                                                                      //   ---------------     ---------------
    for(int block_x = 1 ; block_x < (block_num_x * 2) - 1; block_x+=2){              //   |          p1 |     | p2          |
        int p1_idx = block_x + 2 * block_num_x * (2 * block_num_y - 1);              //   |             |     |             |
        int p2_idx = block_x + 1;                                                    //   |             |     |             |
        same_corner_list[p2_idx].emplace(p1_idx);                                    //   |             |     |             |
    }                                                                                //   ---------------     ---------------
}

/**
 * @fn std::vector<Point4Vec> getSquareCoordinateList()
 * @brief 現在存在する四角形の集合(座標)を返す
 * @return 四角形の集合（座標）
 */
std::vector<Point4Vec> SquareDivision::getSquareCoordinateList() {
    std::vector<Point4Vec> vec;

    for(int i = 0 ; i < squares.size() ; i++) {
        if(delete_flag[i] || !isCodedSquare[i]) continue;
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
        if(delete_flag[i]) continue;
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
    std::vector<Point4Vec> ts;

    cv::Point2f p1, p2, p3, p4;
    for(auto & square : squares){
        p1 = corners[square.p1_idx];
        p2 = corners[square.p2_idx];
        p3 = corners[square.p3_idx];
        p4 = corners[square.p4_idx];
        ts.emplace_back(Point4Vec(p1, p2, p3, p4));
    }

    return ts;
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
    std::vector<std::pair<cv::Point2f, int> > v;
    v.emplace_back(corners[p1_idx], p1_idx);
    v.emplace_back(corners[p2_idx], p2_idx);
    v.emplace_back(corners[p3_idx], p3_idx);
    v.emplace_back(corners[p4_idx], p4_idx);

    // ラスタスキャン順でソート
    sort(v.begin(), v.end(), [](const std::pair<cv::Point2f, int> &a1, const std::pair<cv::Point2f, int> &a2) {
        if (a1.first.y != a2.first.y) {
            return a1.first.y < a2.first.y;
        } else {
            return a1.first.x < a2.first.x;
        }
    });

    Square square(v[0].second, v[1].second, v[2].second, v[3].second, static_cast<int>(squares.size()));

    squares.emplace_back(square);
    isCodedSquare.emplace_back(false);
    square_gauss_results.emplace_back();
    square_gauss_results[square_gauss_results.size() - 1].residual = -1.0;
    delete_flag.emplace_back(false);

    return static_cast<int>(squares.size() - 1);
}

/**
 * @fn void SquareDivision::eraseSquare(int t_idx)
 * @brief 四角パッチに関わる情報を削除する
 * @param t_idx 四角パッチの番号
 */
void SquareDivision::eraseSquare(int s_idx){
    Square square = squares[s_idx];
    removeSquareNeighborVertex(square.p1_idx, square.p2_idx, square.p3_idx, square.p4_idx);
    removeSquareCoveredSquare(square.p1_idx, square.p2_idx, square.p3_idx, square.p4_idx, s_idx);
    isCodedSquare.erase(isCodedSquare.begin() + s_idx);
    squares.erase(squares.begin() + s_idx);
    square_gauss_results.erase(square_gauss_results.begin() + s_idx);
    delete_flag.erase(delete_flag.begin() + s_idx);
}

/**
 * @fn void SquareDivision::addNeighborVertex(int p1_idx, int p2_idx, int p3_idx, int p4_idx)
 * @brief p1, p2, p3, p4の隣接頂点情報を更新する
 * @param[in] p1_idx 頂点1の座標のインデックス
 * @param[in] p2_idx 頂点2の座標のインデックス
 * @param[in] p3_idx 頂点3の座標のインデックス
 * @param[in] p4_idx 頂点4の座標のインデックス
 */
void SquareDivision::addNeighborVertex(int p1_idx, int p2_idx, int p3_idx, int p4_idx) {
    neighbor_vtx[p1_idx].emplace(p2_idx);
    neighbor_vtx[p2_idx].emplace(p1_idx);

    neighbor_vtx[p1_idx].emplace(p3_idx);
    neighbor_vtx[p3_idx].emplace(p1_idx);

    neighbor_vtx[p2_idx].emplace(p4_idx);
    neighbor_vtx[p4_idx].emplace(p2_idx);

    neighbor_vtx[p3_idx].emplace(p4_idx);
    neighbor_vtx[p4_idx].emplace(p3_idx);

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
    covered_square[p1_idx].emplace(square_no);
    covered_square[p2_idx].emplace(square_no);
    covered_square[p3_idx].emplace(square_no);
    covered_square[p4_idx].emplace(square_no);
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
 * @fn std::vector<int> SquareDivision::getNeighborVertexIndexList(int idx)
 * @brief 指定された頂点に隣接する頂点（インデックス）の集合を返す
 * @param[in] idx 頂点のインデックス
 * @return 頂点の集合（インデックス）
 */
std::vector<int> SquareDivision::getNeighborVertexIndexList(int idx) {
    std::set<int> s = neighbor_vtx[idx];
    std::vector<int> v(s.size());

    for(const auto e : s) {
        v.emplace_back(e);
    }

    return v;
}

/**
 * @fn std::vector<cv::Point2f> SquareDivision::getNeighborVertexCoordinateList(int idx)
 * @brief 指定された頂点に隣接する頂点の集合（座標）を返す
 * @param[in] idx 頂点のインデックス
 * @return 頂点の集合（座標）
 */
std::vector<cv::Point2f> SquareDivision::getNeighborVertexCoordinateList(int idx) {
    std::set<int> s = neighbor_vtx[idx];
    std::vector<cv::Point2f> v(s.size());

    for(const auto e : s) {
        v.emplace_back(corners[e]);
    }

    return v;
}

/**
 * @fn std::vector<Point4Vec> SquareDivision::getIdxCoveredSquareCoordinateList(int idx)
 * @brief 指定された頂点が含まれる四角形の集合を返す
 * @param[in] target_vertex_idx 頂点のインデックス
 * @return 四角形の集合(座標で返される)
 */
std::vector<Point4Vec> SquareDivision::getIdxCoveredSquareCoordinateList(int target_vertex_idx) {
    std::set<int> same_corners = same_corner_list[target_vertex_idx];

    std::set<int> s;
    std::set<int> tmp_s;
    for(auto same_corner : same_corners){
        tmp_s = covered_square[same_corner];
        for(auto idx : tmp_s) s.emplace(idx);
    }
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
    std::set<int> same_corners = same_corner_list[target_vertex_idx];

    std::set<int> s;
    std::set<int> tmp_s;
    for(auto same_corner : same_corners){
        tmp_s = covered_square[same_corner];
        for(auto idx : tmp_s) s.emplace(idx);
    }
    std::vector<int> v(s.size());

    for(auto square_idx : s) {
        v.emplace_back(square_idx);
    }

    std::sort(v.begin(), v.end());

    return v;
}

/**
 * @fn void SquareDivision::removeSquareNeighborVertex(int p1_idx, int p2_idx, int p3_idx, int p4_idx)
 * @brief 指定された四角形に含まれる頂点隣接ノード集合から、自分以外のノードを消す
 * @details 日本語が難しいからコードで理解して
 * @param p1_idx
 * @param p2_idx
 * @param p3_idx
 * @param p4_idx
 */
void SquareDivision::removeSquareNeighborVertex(int p1_idx, int p2_idx, int p3_idx, int p4_idx) {
    neighbor_vtx[p1_idx].erase(p2_idx);
    neighbor_vtx[p1_idx].erase(p3_idx);
    neighbor_vtx[p2_idx].erase(p1_idx);
    neighbor_vtx[p2_idx].erase(p4_idx);
    neighbor_vtx[p3_idx].erase(p1_idx);
    neighbor_vtx[p3_idx].erase(p4_idx);
    neighbor_vtx[p4_idx].erase(p2_idx);
    neighbor_vtx[p4_idx].erase(p3_idx);
}

/**
 * @fn void SquareDivision::removeSquareCoveredSquare(int p1_idx, int p2_idx, int p3_idx, int p4_idx, int square_idx)
 * @brief p1, p2, p3, p4を含む四角形の集合から, square_idx番目の四角形を消す
 * @param p1_idx 頂点1のインデックス
 * @param p2_idx 頂点2のインデックス
 * @param p3_idx 頂点3のインデックス
 * @param p4_idx 頂点4のインデックス
 * @param square_idx 削除対象の四角形のインデックス
 */
void SquareDivision::removeSquareCoveredSquare(int p1_idx, int p2_idx, int p3_idx, int p4_idx, int square_idx) {
    covered_square[p1_idx].erase(square_idx);
    covered_square[p2_idx].erase(square_idx);
    covered_square[p3_idx].erase(square_idx);
    covered_square[p4_idx].erase(square_idx);
}

/**
 * @fn int SquareDivision::getOrAddCornerIndex(cv::Point2f p)
 * @brief 頂点が格納されているインデックスを返す。頂点が存在しない場合、その頂点を頂点集合に追加した後インデックスを返す
 * @param[in] p 追加する頂点の座標
 * @return 頂点番号
 */
int SquareDivision::getOrAddCornerIndex(cv::Point2f p) {
    if(corner_flag[(int)(p.y * 2)][(int)(p.x * 2)] != -1) return corner_flag[(int)(p.y * 2)][(int)(p.x * 2)]; //すでに頂点がある場合
    corners.emplace_back(p);
    neighbor_vtx.emplace_back();
    covered_square.emplace_back();
    corner_flag[(int)(p.y * 2)][(int)(p.x * 2)] = static_cast<int>(corners.size() - 1);
    same_corner_list.emplace_back();
    same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1);
    return static_cast<int>(corners.size() - 1);
}


/**
 * @fn int SquareDivision::getCornerIndex(cv::Point2f p)
 * @brief 頂点が格納されているインデックスを返す。頂点が存在しない場合、-1を返す
 * @param[in] 頂点の座標
 * @return 頂点番号
 */
int SquareDivision::getCornerIndex(cv::Point2f p) {
    if(p.x != -1 && p.y != -1) {
        if (corner_flag[(int) (p.y * 2)][(int) (p.x * 2)] != -1)
            return corner_flag[(int) (p.y * 2)][(int) (p.x * 2)]; //すでに頂点がある場合
        return -1;
    }
    return -1;
}

/**
 * @fn void SquareDivision::addCornerAndSquare(Square square, int square_index)
 * @brief
 * @param square
 * @param square_index
 * @param type
 * @return
 */
void SquareDivision::addCornerAndSquare(Square square, int square_index){

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

    removeSquareNeighborVertex(square.p1_idx, square.p2_idx, square.p3_idx, square.p4_idx);
    removeSquareCoveredSquare( square.p1_idx, square.p2_idx, square.p3_idx, square.p4_idx, square_index);

    addNeighborVertex(a_idx, e_idx, c_idx, f_idx);
    addNeighborVertex(g_idx, b_idx, h_idx, d_idx);

    addCoveredSquare(a_idx, e_idx, c_idx, f_idx, s1_idx);
    addCoveredSquare(g_idx, b_idx, h_idx, d_idx, s2_idx);

    same_corner_list[g_idx].emplace(e_idx);

    same_corner_list[h_idx].emplace(f_idx);

    isCodedSquare[square_index] = false;
    delete_flag[square_index] = true;
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
 * @param type 分割方向
 * @param steps 分割回数
 * @return 分割した場合はtrue, そうでない場合falseを返す
 */
bool SquareDivision::split(std::vector<std::vector<std::vector<unsigned char **>>> expand_images, CodingTreeUnit* ctu, CollocatedMvTree* cmt, Point4Vec square, int square_index, int type, int steps) {
    if(steps <= 0) return false;

    double RMSE_before_subdiv = 0.0;
    double error_parallel;
    cv::Point2f p1 = square.p1;
    cv::Point2f p2 = square.p2;
    cv::Point2f p3 = square.p3;
    cv::Point2f p4 = square.p4;

    int square_size = 0;
    bool parallel_flag;

    std::vector<cv::Point2f> dummy;
    std::vector<cv::Point2f> gauss_result_warping;
    cv::Point2f gauss_result_parallel;


    if(cmt == nullptr) {
        cmt = previousMvList[0][square_index];
    }

    if(square_gauss_results[square_index].residual > 0) {
        GaussResult result_before = square_gauss_results[square_index];
        gauss_result_parallel = result_before.mv_parallel;
        RMSE_before_subdiv = result_before.residual;
        square_size = result_before.square_size;
        parallel_flag = result_before.parallel_flag;
        if(parallel_flag){
            error_parallel = result_before.residual;
        }
        ctu->error_bm = result_before.residual_bm;
        ctu->error_newton = result_before.residual_newton;
    }else {
        std::vector<cv::Point2f> tmp_bm_mv;
        double tmp_bm_error;
        std::tie(tmp_bm_mv, tmp_bm_error) = blockMatching(square, target_image, expansion_ref, square_index, ctu);
        square_gauss_results[square_index].residual_bm = tmp_bm_error;
        ctu->error_bm = tmp_bm_error;
        gauss_result_warping = tmp_bm_mv;
        gauss_result_parallel = tmp_bm_mv[2];
        RMSE_before_subdiv = tmp_bm_error;
        error_parallel = tmp_bm_error;
        square_gauss_results[square_index].mv_parallel = gauss_result_parallel;
        square_gauss_results[square_index].square_size = square_size;
        square_gauss_results[square_index].residual = RMSE_before_subdiv;
        square_gauss_results[square_index].parallel_flag = true;
        parallel_flag = true;
    }

    std::vector<cv::Point2f> mvd;
    int selected_index;
    MV_CODE_METHOD method_flag;
    double cost_before_subdiv;
    int code_length;

    if(square_gauss_results[square_index].parallel_flag) {
        std::tie(cost_before_subdiv, code_length, mvd, selected_index, method_flag) = getMVD(
                gauss_result_parallel, error_parallel,
                square_index, cmt->mv1, ctu, dummy);
    }

    std::vector<cv::Point2i> ret_gauss2;

    cv::Point2f mv;
    if(parallel_flag){
        mv = gauss_result_parallel;
    }
    ctu->mv1 = mv;
    ctu->square_index = square_index;
    ctu->code_length = code_length;
    ctu->collocated_mv = cmt->mv1;
    ctu->parallel_flag = parallel_flag;
    ctu->method = method_flag;

    SplitResult split_squares = getSplitSquare(p1, p2, p3, p4, type);

    SplitResult split_sub_squares1 = getSplitSquare(split_squares.s1.p1, split_squares.s1.p2, split_squares.s1.p3, split_squares.s1.p4, split_squares.s_type);
    SplitResult split_sub_squares2 = getSplitSquare(split_squares.s2.p1, split_squares.s2.p2, split_squares.s2.p3, split_squares.s2.p4, split_squares.s_type);

    std::vector<Point4Vec> subdiv_ref_squares, subdiv_target_squares;
    subdiv_ref_squares.emplace_back(split_sub_squares1.s1);
    subdiv_ref_squares.emplace_back(split_sub_squares1.s2);
    subdiv_ref_squares.emplace_back(split_sub_squares2.s1);
    subdiv_ref_squares.emplace_back(split_sub_squares2.s2);

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
    addCornerAndSquare(Square(s1_p1_idx, s1_p2_idx, s1_p3_idx, s1_p4_idx), square_index);

    int s2_p1_idx = getOrAddCornerIndex(split_squares.s2.p1);
    int s2_p2_idx = getOrAddCornerIndex(split_squares.s2.p2);
    int s2_p3_idx = getOrAddCornerIndex(split_squares.s2.p3);
    int s2_p4_idx = getOrAddCornerIndex(split_squares.s2.p4);
    addCornerAndSquare(Square(s2_p1_idx, s2_p2_idx, s2_p3_idx, s2_p4_idx), square_index);

                                                                         //  -----------------
    //2分割後の隣接する頂点を追加                                        //  |               |
    same_corner_list[s2_p1_idx].emplace(s1_p3_idx);                      //  |s1_p3     s1_p4|
                                                                         //  -----------------
    same_corner_list[s2_p2_idx].emplace(s1_p4_idx);                      //  -----------------
                                                                         //  |s2_p1     s2_p2|
                                                                         //  |               |
    //4分割後の隣接する頂点を追加                                        //  -----------------
    //p4とp3、p2とp1に関してはaddCornerAndSquareで追加しているのでここでは追加しない。
    int sub1_s1_p4_idx = getOrAddCornerIndex(split_sub_squares1.s1.p4);  //     ---------------  ---------------
    int sub2_s1_p2_idx = getOrAddCornerIndex(split_sub_squares2.s1.p2);  //     |             |  |             |
                                                                         //     |             |  |             |
    int sub1_s2_p3_idx = getOrAddCornerIndex(split_sub_squares1.s2.p3);  //     |             |  |             |
    int sub2_s2_p1_idx = getOrAddCornerIndex(split_sub_squares2.s2.p1);  //     |   sub1_s1_p4|  |sub1_s2_p3   |
                                                                         //     ---------------  ---------------
    same_corner_list[sub2_s1_p2_idx].emplace(sub1_s1_p4_idx);            //     ---------------  ---------------
    same_corner_list[sub2_s1_p2_idx].emplace(sub1_s2_p3_idx);            //     |   sub2_s1_p2|  |sub2_s2_p1   |
                                                                         //     |             |  |             |
    same_corner_list[sub2_s2_p1_idx].emplace(sub1_s1_p4_idx);            //     |             |  |             |
    same_corner_list[sub2_s2_p1_idx].emplace(sub1_s2_p3_idx);            //     |             |  |             |
                                                                         //     ---------------  ---------------

    //4分割後の隣接するパッチの頂点を追加                                // if != -1 のとき
    int sub1_s1_p3_idx = getCornerIndex(split_sub_squares1.s1.p3);       //              |          sp3|  |sp4          |
    int sub2_s1_p1_idx = getCornerIndex(split_sub_squares2.s1.p1);       //              ---------------  ---------------
                                                                         //
    int sub1_s1_p2_idx = getCornerIndex(split_sub_squares1.s1.p2);       //      ---     ---------------  ---------------
    int sub1_s2_p1_idx = getCornerIndex(split_sub_squares1.s2.p1);       //        |     |   sub1_s1_p2|  |sub1_s2_p1   |
                                                                         //        |     |             |  |             |
    cv::Point2f sp1 = split_sub_squares1.s1.p3;                          //        |     |             |  |             |
    cv::Point2f sp2 = split_sub_squares2.s1.p1;                          //     sp1|     |sub1_s1_p3   |  |             |
    cv::Point2f sp3 = split_sub_squares1.s1.p2;                          //      ---     ---------------  ---------------
    cv::Point2f sp4 = split_sub_squares1.s2.p1;                          //      ---     ---------------  ---------------
    //それぞれspの座標を合わせる                                         //     sp2|     |sub2_s1_p1   |  |             |
    sp1.x--; sp2.x--; sp3.y--; sp4.y--;                                  //        |     |             |  |             |
    //頂点がある場合はそのインデックスをもらってくる(無いときは-1)       //        |     |             |  |             |
    int sp1_idx = getCornerIndex(sp1);                                   //        |     |             |  |             |
    int sp2_idx = getCornerIndex(sp2);                                   //      ---     ---------------  ---------------
    int sp3_idx = getCornerIndex(sp3);                                   //
    int sp4_idx = getCornerIndex(sp4);                                   //　-1 のとき

    if(sp1_idx != -1) {
        same_corner_list[sub1_s1_p3_idx].emplace(sp1_idx);
        same_corner_list[sub2_s1_p1_idx].emplace(sp1_idx);
    }

    if(sp2_idx != -1) {
        same_corner_list[sub1_s1_p3_idx].emplace(sp2_idx);
        same_corner_list[sub2_s1_p1_idx].emplace(sp2_idx);
    }

    if(sp3_idx != -1) {
        same_corner_list[sub1_s1_p2_idx].emplace(sp3_idx);
        same_corner_list[sub1_s2_p1_idx].emplace(sp3_idx);
    }

    if(sp4_idx != -1) {
        same_corner_list[sub1_s1_p2_idx].emplace(sp4_idx);
        same_corner_list[sub1_s2_p1_idx].emplace(sp4_idx);
    }

    int square_indexes[] = {(int)squares.size() - 4, (int)squares.size() - 3, (int)squares.size() - 2, (int)squares.size() - 1};

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
    #pragma omp parallel for
#endif
    for (int j = 0; j < (int) subdiv_ref_squares.size(); j++) {
        double error_warping_tmp, error_parallel_tmp;
        int square_size_tmp;
        cv::Point2f mv_parallel_tmp;
        std::vector<cv::Point2f> mv_warping_tmp;
        std::vector<cv::Point2f> tmp_bm_mv;
        double tmp_bm_error;
        double cost_warping_tmp, cost_parallel_tmp;
        double tmp_error_newton;
        MV_CODE_METHOD method_warping_tmp, method_parallel_tmp;

        std::tie(tmp_bm_mv, tmp_bm_error) = blockMatching(subdiv_target_squares[j], target_image, expansion_ref, square_indexes[j], ctus[j]);
        mv_parallel_tmp = tmp_bm_mv[2];
        error_parallel_tmp = tmp_bm_error;
        square_size_tmp = (double)1e6;

        split_mv_result[j] = GaussResult(mv_parallel_tmp, error_parallel_tmp, square_size_tmp, true, tmp_bm_error, tmp_error_newton);
    }

    double cost_after_subdiv1;
    int code_length1;
    CollocatedMvTree *cmt_left_left, *cmt_left_right, *cmt_right_left, *cmt_right_right;

    cmt_left_left    = (cmt->node1 == nullptr ? cmt : cmt->node1);
    cmt_left_right   = (cmt->node2 == nullptr ? cmt : cmt->node2);
    cmt_right_left   = (cmt->node3 == nullptr ? cmt : cmt->node3);
    cmt_right_right  = (cmt->node4 == nullptr ? cmt : cmt->node4);

    MV_CODE_METHOD method_flag1, method_flag2, method_flag3, method_flag4;
    if(split_mv_result[0].parallel_flag) {
        std::tie(cost_after_subdiv1, code_length1, mvd, selected_index, method_flag1) = getMVD(
                split_mv_result[0].mv_parallel, split_mv_result[0].residual, square_indexes[0], cmt_left_left->mv1, ctu->node1, dummy);
    }

    double cost_after_subdiv2;
    int code_length2;
    if(split_mv_result[1].parallel_flag){
        std::tie(cost_after_subdiv2, code_length2, mvd, selected_index, method_flag2) = getMVD(
                split_mv_result[1].mv_parallel, split_mv_result[1].residual, square_indexes[1], cmt_left_right->mv1, ctu->node2, dummy);
    }

    double cost_after_subdiv3;
    int code_length3;
    if(split_mv_result[2].parallel_flag) {
        std::tie(cost_after_subdiv3, code_length3, mvd, selected_index, method_flag3) = getMVD(
                split_mv_result[2].mv_parallel, split_mv_result[2].residual, square_indexes[2], cmt_right_left->mv1, ctu->node3, dummy);
    }

    double cost_after_subdiv4;
    int code_length4;
    if(split_mv_result[3].parallel_flag){
        std::tie(cost_after_subdiv4, code_length4, mvd, selected_index, method_flag4) = getMVD(
                split_mv_result[3].mv_parallel, split_mv_result[3].residual, square_indexes[3], cmt_right_right->mv1, ctu->node4, dummy);
    }
    double alpha = 1;
    std::cout << "before:" << cost_before_subdiv << " after:" << alpha * (cost_after_subdiv1 + cost_after_subdiv2 + cost_after_subdiv3 + cost_after_subdiv4) << std::endl;
    if(cost_before_subdiv >= alpha * (cost_after_subdiv1 + cost_after_subdiv2 + cost_after_subdiv3 + cost_after_subdiv4)) {
        ctu->split_cu_flag = true;

        int s1_idx = squares.size() - 4;
        int s2_idx = squares.size() - 3;
        int s3_idx = squares.size() - 2;
        int s4_idx = squares.size() - 1;

        // 1つ目の頂点追加
        ctu->node1->square_index = s1_idx;
        if(split_mv_result[0].parallel_flag) {
            ctu->node1->mv1 = split_mv_result[0].mv_parallel;
            ctu->node1->mv2 = split_mv_result[0].mv_parallel;
            ctu->node1->mv3 = split_mv_result[0].mv_parallel;
        }
        ctu->node1->code_length = code_length1;
        ctu->node1->parallel_flag = split_mv_result[0].parallel_flag;
        ctu->node1->method = method_flag1;
        square_gauss_results[s1_idx] = split_mv_result[0];
        isCodedSquare[s1_idx] = true;
        bool result = split(expand_images, ctu->node1, cmt_left_left, split_sub_squares1.s1, s1_idx, 1, steps - 2);

        // 2つ目の四角形
        ctu->node2->square_index = s2_idx;
        if(split_mv_result[1].parallel_flag){
            ctu->node2->mv1 = split_mv_result[1].mv_parallel;
            ctu->node2->mv2 = split_mv_result[1].mv_parallel;
            ctu->node2->mv3 = split_mv_result[1].mv_parallel;
        }
        ctu->node2->code_length = code_length2;
        ctu->node2->parallel_flag = split_mv_result[1].parallel_flag;
        ctu->node2->method = method_flag2;

        square_gauss_results[s2_idx] = split_mv_result[1];
        isCodedSquare[s2_idx] = true;
        result = split(expand_images, ctu->node2, cmt_left_right, split_sub_squares1.s2, s2_idx, 1, steps - 2);

        // 3つ目の四角形
        ctu->node3->square_index = s3_idx;
        if(split_mv_result[2].parallel_flag) {
            ctu->node3->mv1 = split_mv_result[2].mv_parallel;
            ctu->node3->mv2 = split_mv_result[2].mv_parallel;
            ctu->node3->mv3 = split_mv_result[2].mv_parallel;
        }
        ctu->node3->code_length = code_length3;
        ctu->node3->parallel_flag = split_mv_result[2].parallel_flag;
        ctu->node3->method = method_flag3;
        square_gauss_results[s3_idx] = split_mv_result[2];
        isCodedSquare[s3_idx] = true;
        result = split(expand_images, ctu->node3, cmt_right_left, split_sub_squares2.s1, s3_idx, 1, steps - 2);

        // 4つ目の四角形
        ctu->node4->square_index = s4_idx;
        if(split_mv_result[3].parallel_flag) {
            ctu->node4->mv1 = split_mv_result[3].mv_parallel;
            ctu->node4->mv2 = split_mv_result[3].mv_parallel;
            ctu->node4->mv3 = split_mv_result[3].mv_parallel;
        }
        ctu->node4->code_length = code_length4;
        ctu->node4->parallel_flag = split_mv_result[3].parallel_flag;
        ctu->node4->method = method_flag4;
        square_gauss_results[s4_idx] = split_mv_result[3];
        isCodedSquare[s4_idx] = true;
        result = split(expand_images, ctu->node4, cmt_right_right, split_sub_squares2.s2, s4_idx, 1, steps - 2);

        return true;
    }else{
        //4分割により追加された頂点12個を消す
        same_corner_list.erase(same_corner_list.end() - 12,same_corner_list.end());
        corners.erase(corners.end() - 12, corners.end());
        isCodedSquare[square_index] = true;
        delete_flag[square_index] = false;
        ctu->node1 = ctu->node2 = ctu->node3 = ctu->node4 = nullptr;
        ctu->method = method_flag;
        eraseSquare(squares.size() - 1);
        eraseSquare(squares.size() - 1);
        eraseSquare(squares.size() - 1);
        eraseSquare(squares.size() - 1);
        addNeighborVertex(squares[square_index].p1_idx,squares[square_index].p2_idx,squares[square_index].p3_idx,squares[square_index].p4_idx);
        addCoveredSquare(squares[square_index].p1_idx,squares[square_index].p2_idx,squares[square_index].p3_idx,squares[square_index].p4_idx, square_index);
        #if MVD_DEBUG_LOG
        std::cout << "square_index:" << square_index << std::endl;
        std::cout << "p1_idx:" << squares[square_index].p1_idx << " p2_idx:" << squares[square_index].p2_idx << " p3_idx:" << squares[square_index].p3_idx << " p4_idx:" << squares[square_index].p4_idx << std::endl;
        std::cout << "p1:" << corners[squares[square_index].p1_idx] << " p2:" << corners[squares[square_index].p2_idx] << " p3:" << corners[squares[square_index].p3_idx] << " p4:" << corners[squares[square_index].p4_idx] << std::endl;
#endif

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
            break;
    }

}

/**
 * @fn std::vector<int> SquareDivision::getSpatialSquareList(int s_idx)
 * @brief t_idx番目の四角形の空間予測動きベクトル候補を返す
 * @param[in] t_idx 四角パッチのインデックス
 * @return 候補のパッチの番号を返す
 */
std::vector<int> SquareDivision::getSpatialSquareList(int s_idx){
    Square square = squares[s_idx];
    std::set<int> spatialSquares;
    std::vector<int> list1 = getIdxCoveredSquareIndexList(square.p1_idx);
    std::vector<int> list2 = getIdxCoveredSquareIndexList(square.p2_idx);
    std::vector<int> list3 = getIdxCoveredSquareIndexList(square.p3_idx);
    std::vector<int> list4 = getIdxCoveredSquareIndexList(square.p4_idx);

    std::set<int> mutualIndexSet1, mutualIndexSet2, mutualIndexSet3, mutualIndexSet4;

#if MVD_DEBUG_LOG
    std::cout << "p1:" << squares[s_idx].p1_idx << std::endl;
    for(auto item : list1){
        std::cout << item << std::endl;
    }
    puts("");

    std::cout << "p2:" << squares[s_idx].p2_idx << std::endl;
    for(auto item : list2){
        std::cout << item << std::endl;
    }
    puts("");

    std::cout << "p3:" << squares[s_idx].p3_idx << std::endl;
    for(auto item : list3){
        std::cout << item << std::endl;
    }
    puts("");

    std::cout << "p4:" << squares[s_idx].p4_idx << std::endl;
    for(auto item : list4){
        std::cout << item << std::endl;
    }
    puts("");

    std::cout << "s_idx:" << s_idx << std::endl;
    puts("");

#endif

    for(auto idx : list1) if(isCodedSquare[idx] && idx != s_idx) mutualIndexSet1.emplace(idx);
    for(auto idx : list2) if(isCodedSquare[idx] && idx != s_idx) mutualIndexSet2.emplace(idx);
    for(auto idx : list3) if(isCodedSquare[idx] && idx != s_idx) mutualIndexSet3.emplace(idx);
    for(auto idx : list4) if(isCodedSquare[idx] && idx != s_idx) mutualIndexSet4.emplace(idx);

    for(auto idx : mutualIndexSet1) spatialSquares.emplace(idx);
    for(auto idx : mutualIndexSet2) spatialSquares.emplace(idx);
    for(auto idx : mutualIndexSet3) spatialSquares.emplace(idx);
    for(auto idx : mutualIndexSet4) spatialSquares.emplace(idx);

    std::vector<int> ret;

    for(auto idx : spatialSquares){
        ret.emplace_back(idx);
    }

    return ret;
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

        auto* node1 = new CollocatedMvTree();
        node1->mv1 = cv::Point2f(0, 0);

        node1->node1 = new CollocatedMvTree();
        node1->node1->mv1 = cv::Point2f(0, 0);

        node1->node2 = new CollocatedMvTree();
        node1->node2->mv1 = cv::Point2f(0, 0);

        node1->node3 = new CollocatedMvTree();
        node1->node3->mv1 = cv::Point2f(0, 0);

        node1->node4 = new CollocatedMvTree();
        node1->node4->mv1 = cv::Point2f(0, 0);

        previousMvList[pic_num][i]->node1 = node1;

        auto* node2 = new CollocatedMvTree();
        node2->mv1 = cv::Point2f(0, 0);

        node2->node1 = new CollocatedMvTree();
        node2->node1->mv1 = cv::Point2f(0, 0);

        node2->node2 = new CollocatedMvTree();
        node2->node2->mv1 = cv::Point2f(0, 0);

        node2->node3 = new CollocatedMvTree();
        node2->node3->mv1 = cv::Point2f(0, 0);

        node2->node4 = new CollocatedMvTree();
        node2->node4->mv1 = cv::Point2f(0, 0);

        previousMvList[pic_num][i]->node2 = node2;

        auto* node3 = new CollocatedMvTree();
        node3->mv1 = cv::Point2f(0, 0);

        node3->node1 = new CollocatedMvTree();
        node3->node1->mv1 = cv::Point2f(0, 0);

        node3->node2 = new CollocatedMvTree();
        node3->node2->mv1 = cv::Point2f(0, 0);

        node3->node3 = new CollocatedMvTree();
        node3->node3->mv1 = cv::Point2f(0, 0);

        node3->node4 = new CollocatedMvTree();
        node3->node4->mv1 = cv::Point2f(0, 0);
        node3->node4->mv2 = cv::Point2f(0, 0);

        previousMvList[pic_num][i]->node3 = node3;

        auto* node4 = new CollocatedMvTree();
        node4->mv1 = cv::Point2f(0, 0);

        node4->node1 = new CollocatedMvTree();
        node4->node1->mv1 = cv::Point2f(0, 0);

        node4->node2 = new CollocatedMvTree();
        node4->node2->mv1 = cv::Point2f(0, 0);

        node4->node3 = new CollocatedMvTree();
        node4->node3->mv1 = cv::Point2f(0, 0);

        node4->node4 = new CollocatedMvTree();
        node4->node4->mv1 = cv::Point2f(0, 0);

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
        if(std::get<0>(vector) == mv) {
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
std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD> SquareDivision::getMVD(cv::Point2f mv, double residual, int square_idx, cv::Point2f &collocated_mv, CodingTreeUnit* ctu, std::vector<cv::Point2f> &pixels){
    // 空間予測と時間予測の候補を取り出す
    std::vector<int> spatial_squares = getSpatialSquareList(square_idx);
    int spatial_square_size = static_cast<int>(spatial_squares.size());
    std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> vectors; // ベクトルとモードを表すフラグのペア

    // すべてのベクトルを格納する．
    for(int i = 0 ; i < spatial_square_size ; i++) {
        int spatial_square_index = spatial_squares[i];
        GaussResult spatial_square = square_gauss_results[spatial_square_index];

        if(spatial_square.parallel_flag){
            if(!isMvExists(vectors, spatial_square.mv_parallel)) {
                vectors.emplace_back(spatial_square.mv_parallel, SPATIAL);
            }
        }
    }

#if MVD_DEBUG_LOG
    std::cout << corners[squares[square_idx].p1_idx] << " " << corners[squares[square_idx].p2_idx] << " " << corners[squares[square_idx].p3_idx] << " " << corners[squares[square_idx].p4_idx] << std::endl;
    #endif

    if(!isMvExists(vectors, collocated_mv)) vectors.emplace_back(collocated_mv, SPATIAL);

    if(vectors.size() < 2) vectors.emplace_back(cv::Point2f(0.0, 0.0), Collocated);

    double lambda = getLambdaPred(qp);

    //                      コスト, 差分ベクトル, 番号, タイプ
    std::vector<std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum> > results;
    for(int i = 0 ; i < vectors.size() ; i++) {
        std::pair<cv::Point2f, MV_CODE_METHOD> vector = vectors[i];
        cv::Point2f current_mv = vector.first;

        FlagsCodeSum flag_code_sum(0, 0, 0, 0);
        cv::Point2f mvd = current_mv - mv;
#if MVD_DEBUG_LOG
        std::cout << "target_vector_idx       :" << i << std::endl;
        std::cout << "diff_target_mv(parallel):" << current_mv << std::endl;
        std::cout << "encode_mv(parallel)     :" << mv << std::endl;
#endif
        mvd = getQuantizedMv(mvd, 4);
#if MVD_DEBUG_LOG
        std::cout << "mvd(parallel)           :" << mvd << std::endl;
#endif
        mvd.x = std::fabs(mvd.x);
        mvd.y = std::fabs(mvd.y);

        mvd *= 4;
        int abs_x = mvd.x;
        int abs_y = mvd.y;
#if MVD_DEBUG_LOG
        std::cout << "4 * mvd(parallel)       :" << mvd << std::endl;
#endif

        // 動きベクトル差分の絶対値が0より大きいのか？
        bool is_x_greater_than_zero = abs_x > 0;
        bool is_y_greater_than_zero = abs_y > 0;
        flag_code_sum.countGreater0Code();
        flag_code_sum.countGreater0Code();
        flag_code_sum.setXGreater0Flag(is_x_greater_than_zero);
        flag_code_sum.setYGreater0Flag(is_y_greater_than_zero);
        // 動きベクトル差分の絶対値が1より大きいのか？
        bool is_x_greater_than_one = abs_x > 1;
        bool is_y_greater_than_one = abs_y > 1;
        // 正負の判定(使ってません！！！)
        bool is_x_minus = mvd.x < 0;
        bool is_y_minus = mvd.y < 0;
        int mvd_code_length = 2;
        if(is_x_greater_than_zero){
            mvd_code_length += 1;
            if(is_x_greater_than_one){
                int mvd_x_minus_2 = mvd.x - 2.0;
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
                mvd_code_length += getExponentialGolombCodeLength((int) mvd_y_minus_2, 0);
                flag_code_sum.addMvdCodeLength(getExponentialGolombCodeLength((int) mvd_y_minus_2, 0));
            }
            flag_code_sum.countGreater1Code();
            flag_code_sum.setYGreater1Flag(is_y_greater_than_one);
            flag_code_sum.countSignFlagCode();
        }
        // 参照箇所符号化
        int reference_index = std::get<1>(vector);
        int reference_index_code_length = getUnaryCodeLength(reference_index);
        // 各種フラグ分を(3*2)bit足してます
        double rd = residual + lambda * (mvd_code_length + reference_index_code_length);
        std::vector<cv::Point2f> mvds{mvd};
        // 結果に入れる
        results.emplace_back(rd, mvd_code_length + reference_index_code_length, mvds, i, vector.second, flag_code_sum);
    }

    // マージ符号化
    // マージで参照する動きベクトルを使って残差を求め直す
    Square current_square_coordinate = squares[square_idx];
    cv::Point2f p1 = corners[current_square_coordinate.p1_idx];
    cv::Point2f p2 = corners[current_square_coordinate.p2_idx];
    cv::Point2f p3 = corners[current_square_coordinate.p3_idx];
    cv::Point2f p4 = corners[current_square_coordinate.p4_idx];
    Point4Vec coordinate = Point4Vec(p1, p2, p3, p4);
    vectors.clear();

    std::vector<cv::Point2f> pixels_in_square;
    std::vector<std::pair<cv::Point2f, MV_CODE_METHOD>> merge_vectors;
    if(pixels.empty()) {
         pixels_in_square = getPixelsInSquare(coordinate);
    }else{
        pixels_in_square = pixels;
    }

    for(int i = 0 ; i < spatial_square_size ; i++) {
        int spatial_square_index = spatial_squares[i];
        GaussResult spatial_square = square_gauss_results[spatial_square_index];
        std::vector<cv::Point2f> mvds;

        if(spatial_square.parallel_flag){
            if(!isMvExists(merge_vectors, spatial_square.mv_parallel)) {
                merge_vectors.emplace_back(spatial_square.mv_parallel, MERGE);
                double ret_residual = getSquareResidual(expansion_ref_uchar, target_image, mv, pixels_in_square, ref_hevc);
                double rd = ret_residual + lambda * (getUnaryCodeLength(i) + 1);
                results.emplace_back(rd, getUnaryCodeLength(i) + 1, mvds, results.size(), MERGE, FlagsCodeSum(0, 0, 0, 0));
            }
        }
    }

    // RDしたスコアが小さい順にソート
    std::sort(results.begin(), results.end(), [](const std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum >& a, const std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum>& b){
        return std::get<0>(a) < std::get<0>(b);
    });
    double cost = std::get<0>(results[0]);
    int code_length = std::get<1>(results[0]);
    std::vector<cv::Point2f> mvds = std::get<2>(results[0]);
    int selected_idx = std::get<3>(results[0]);
    MV_CODE_METHOD method = std::get<4>(results[0]);
    FlagsCodeSum flag_code_sum = std::get<5>(results[0]);

#if MVD_DEBUG_LOG
    puts("Result ===========================================");
    std::cout << "code_length:" << code_length << std::endl;
    std::cout << "cost       :" << cost << std::endl;
    if(method != MERGE){
        std::cout << "mvd        :" << mvds[0] << std::endl;
    }
    puts("");
#endif

    ctu->flags_code_sum = flag_code_sum;
    if(method != MERGE) {
        (ctu->mvds_x).clear();
        (ctu->mvds_y).clear();
        (ctu->original_mvds_x).clear();
        (ctu->original_mvds_y).clear();

        (ctu->mvds_x).emplace_back(mvds[0].x);
        (ctu->mvds_y).emplace_back(mvds[0].y);

    }

    return {cost, code_length, mvds, selected_idx, method};
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

        getPredictedImage(expansion_ref_uchar, target_image, out, square, mv, ref_hevc);
        return;
    }

    if(ctu->node1 != nullptr) getPredictedImageFromCtu(ctu->node1, out);
    if(ctu->node2 != nullptr) getPredictedImageFromCtu(ctu->node2, out);
    if(ctu->node3 != nullptr) getPredictedImageFromCtu(ctu->node3, out);
    if(ctu->node4 != nullptr) getPredictedImageFromCtu(ctu->node4, out);
}

cv::Mat SquareDivision::getPredictedColorImageFromCtu(std::vector<CodingTreeUnit*> ctus, double original_psnr){
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

    return out;
}

void SquareDivision::getPredictedColorImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out, double original_psnr, std::vector<cv::Scalar> &colors){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int square_index = ctu->square_index;
        cv::Point2f mv = ctu->mv1;
        Square square_corner_idx = squares[square_index];
        Point4Vec square(corners[square_corner_idx.p1_idx], corners[square_corner_idx.p2_idx], corners[square_corner_idx.p3_idx], corners[square_corner_idx.p4_idx]);

        std::vector<cv::Point2f> pixels = getPixelsInSquare(square);
//        double residual = getSquareResidual(expansion_ref_uchar, target_image, square, mvs, pixels, cv::Rect(-16, -16, target_image.cols + 2 * 16, target_image.rows + 2 * 16));
//        double mse = residual / (pixels.size());
//        double psnr = 10 * std::log10(255.0 * 255.0 / mse);

//        if(psnr < 25.0){
//            for(auto pixel : pixels) {
//                R(out, (int)pixel.x, (int)pixel.y) = colors[0][0];
//                G(out, (int)pixel.x, (int)pixel.y) = colors[0][1];
//                B(out, (int)pixel.x, (int)pixel.y) = colors[0][2];
//            }
//        }else if(psnr < 26.0){
//            for(auto pixel : pixels) {
//                R(out, (int)pixel.x, (int)pixel.y) = colors[1][0];
//                G(out, (int)pixel.x, (int)pixel.y) = colors[1][1];
//                B(out, (int)pixel.x, (int)pixel.y) = colors[1][2];
//            }
//        }else if(psnr < 27.0){
//            for(auto pixel : pixels) {
//                R(out, (int)pixel.x, (int)pixel.y) = colors[2][0];
//                G(out, (int)pixel.x, (int)pixel.y) = colors[2][1];
//                B(out, (int)pixel.x, (int)pixel.y) = colors[2][2];
//            }
//        }else if(psnr < 28.0){
//            for(auto pixel : pixels) {
//                R(out, (int)pixel.x, (int)pixel.y) = colors[3][0];
//                G(out, (int)pixel.x, (int)pixel.y) = colors[3][1];
//                B(out, (int)pixel.x, (int)pixel.y) = colors[3][2];
//            }
//        }else if(psnr < 29.0){
//            for(auto pixel : pixels) {
//                R(out, (int)pixel.x, (int)pixel.y) = colors[4][0];
//                G(out, (int)pixel.x, (int)pixel.y) = colors[4][1];
//                B(out, (int)pixel.x, (int)pixel.y) = colors[4][2];
//            }
//        }else if(psnr < 30.0){
//            for(auto pixel : pixels) {
//                R(out, (int)pixel.x, (int)pixel.y) = colors[5][0];
//                G(out, (int)pixel.x, (int)pixel.y) = colors[5][1];
//                B(out, (int)pixel.x, (int)pixel.y) = colors[5][2];
//            }
        if(!ctu->parallel_flag) {
            for(auto pixel : pixels) {
                R(out, (int)pixel.x, (int)pixel.y) = 255;
                G(out, (int)pixel.x, (int)pixel.y) = 0;
                B(out, (int)pixel.x, (int)pixel.y) = 0;
            }
        }else{
            getPredictedImage(expansion_ref_uchar, target_image, out, square, mv, ref_hevc);
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
        return 1+ctu->code_length;
    }

    // ここで足している1はsplit_cu_flag分です
    return 1 + getCtuCodeLength(ctu->node1) + getCtuCodeLength(ctu->node2) + getCtuCodeLength(ctu->node3) + getCtuCodeLength(ctu->node4);
}


cv::Mat SquareDivision::getMvImage(std::vector<CodingTreeUnit*> ctus){
    cv::Mat out = target_image.clone();

    for(auto square : getSquareCoordinateList()){
        drawSquare(out, square.p1, square.p2, square.p3, square.p4, cv::Scalar(255, 255, 255));
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

        cv::Point2f g = (p1 + p2 + p3 + p4) / 4.0;

        cv::line(out, g, g+ctu->mv1, GREEN);
    }

    if(ctu->node1 != nullptr) drawMvImage(out, ctu->node1);
    if(ctu->node2 != nullptr) drawMvImage(out, ctu->node2);
    if(ctu->node3 != nullptr) drawMvImage(out, ctu->node3);
    if(ctu->node4 != nullptr) drawMvImage(out, ctu->node4);
}

SquareDivision::SquareDivision() {}

SquareDivision::SplitResult::SplitResult(const Point4Vec &s1, const Point4Vec &s2, int type) : s1(s1), s2(s2), s_type(type) {}

SquareDivision::GaussResult::GaussResult(const cv::Point2f &mvParallel, double residual, int squareSize, bool parallelFlag, double residualBm, double residualNewton) :
                                  mv_parallel(mvParallel), residual(residual), square_size(squareSize), parallel_flag(parallelFlag), residual_bm(residualBm), residual_newton(residualNewton) {}

SquareDivision::GaussResult::GaussResult() {}

std::tuple<std::vector<cv::Point2f>, double> SquareDivision::blockMatching(Point4Vec square, const cv::Mat& target_image, cv::Mat expansion_ref_image, int square_index, CodingTreeUnit *ctu) {
    double sx, sy, lx, ly;
    cv::Point2f sp1, sp2, sp3, sp4;

    sp1 = square.p1;
    sp2 = square.p2;
    sp3 = square.p3;
    sp4 = square.p4;

    sx = 4 * std::min({sp1.x, sp2.x, sp3.x, sp4.x});
    sy = 4 * std::min({sp1.y, sp2.y, sp3.y, sp4.y});
    lx = 4 * std::max({sp1.x, sp2.x, sp3.x, sp4.x}) + 3;
    ly = 4 * std::max({sp1.y, sp2.y, sp3.y, sp4.y}) + 3;

    cv::Point2f mv_tmp(0.0, 0.0); //ブロックの動きベクトル
    int SX = 16; // ブロックマッチングの探索範囲(X)
    int SY = 16; // ブロックマッチングの探索範囲(Y)

    double rd, e;
    double rd_min = 1e9, e_min = 1e9;

    cv::Point2f mv_min;
    int spread_quarter = 64;
    int s = 4;                   //4 : Full-pel, 2 : Half-pel, 1 : Quarter-pel
    std::vector<cv::Point2f> pixels = getPixelsInSquare(square);

    for(int j = -SY * 4 ; j <= SY * 4 ; j += s) {            //j : y方向のMV
        for(int i = -SX * 4 ; i <= SX * 4 ; i += s) {        //i : x方向のMV
            //探索範囲が画像上かどうか判定
            if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
               && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                e = 0.0;
                for (int y = (int) (round(sy) / 4); y <= (int) (round(ly) / 4); y++) {
                    for (int x = (int) (round(sx) / 4); x <= (int) (round(lx) / 4); x++) {
                        e += fabs(R(expansion_ref_image, i + 4 * x + spread_quarter, j + 4 * y + spread_quarter) - R(target_image, x, y));
                    }
                }
                cv::Point2f cmt = cv::Point2f(0.0, 0.0);
                std::tie(rd, std::ignore,std::ignore,std::ignore,std::ignore) = getMVD(cv::Point2f((double)i/4.0, (double)j/4.0), e, square_index, cmt, ctu, pixels);
                if(rd_min > rd){
                    e_min = e;
                    rd_min = rd;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }

    std::vector<cv::Point2f> mvs;
    mvs.emplace_back(mv_min.x, mv_min.y);

    mv_tmp.x = mv_min.x * 4;
    mv_tmp.y = mv_min.y * 4;

    s = 2;
    for(int j = - 2 * s + mv_tmp.y ; j <= 2 * s + mv_tmp.y ; j += s){            //j : y方向のMV
        for(int i = - 2 * s + mv_tmp.x ; i <= 2 * s + mv_tmp.x ; i += s){        //i : x方向のMV
            if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
               && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                e = 0.0;
                for (int y = (int) (round(sy) / 4); y <= (int) (round(ly) / 4); y++) {
                    for (int x = (int) (round(sx) / 4); x <= (int) (round(lx) / 4); x++) {
                        e += fabs(R(expansion_ref_image, i + 4 * x + spread_quarter, j + 4 * y + spread_quarter) - R(target_image, x, y));
                    }
                }
                cv::Point2f cmt = cv::Point2f(0.0, 0.0);
                std::tie(rd, std::ignore,std::ignore,std::ignore,std::ignore) = getMVD(cv::Point2f((double)i/4.0, (double)j/4.0), e, square_index, cmt, ctu, pixels);
                if(rd_min > rd){
                    e_min = e;
                    rd_min = rd;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }

    mvs.emplace_back(mv_min.x, mv_min.y);
    mv_tmp.x = mv_min.x * 4;
    mv_tmp.y = mv_min.y * 4;

    s = 1;

    for(int j = - 2 * s + mv_tmp.y ; j <= 2 * s + mv_tmp.y ; j += s){            //j : y方向のMV
        for(int i = - 2 * s + mv_tmp.x ; i <= 2 * s + mv_tmp.x ; i += s){        //i : x方向のMV
            if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
               && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                e = 0.0;
                for (int y = (int) (round(sy) / 4); y <= (int) (round(ly) / 4); y++) {
                    for (int x = (int) (round(sx) / 4); x <= (int) (round(lx) / 4); x++) {
                        e += fabs(R(expansion_ref_image, i + 4 * x + spread_quarter, j + 4 * y + spread_quarter) - R(target_image, x, y));
                    }
                }
                cv::Point2f cmt = cv::Point2f(0.0, 0.0);
                std::tie(rd, std::ignore,std::ignore,std::ignore,std::ignore) = getMVD(cv::Point2f((double)i/4.0, (double)j/4.0), e, square_index, cmt, ctu, pixels);
                if(rd_min > rd){
                    e_min = e;
                    rd_min = rd;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }

    double error = e_min;
    mvs.emplace_back(mv_min.x, mv_min.y);

    return std::make_tuple(mvs, error);
}