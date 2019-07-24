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


//TODO 四角形対応
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
    for(int block_y = 0 ; block_y <= block_num_y ; block_y++) {
        //y列目の上側の頂点を入れる
        for (int block_x = 0 ; block_x <= block_num_x; block_x++) {
            int nx = block_x * (block_size_x);    //ブロックの左上のx座標
            int ny = block_y * (block_size_y);    //ブロックの左上のy座標

            if(nx < 0) nx = 0;
            if(target_image.cols <= nx) nx = target_image.cols - 1;
            if(ny < 0) ny = 0;
            if(target_image.rows <= ny) ny = target_image.rows - 1;
            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());

            if(block_x == block_num_x) continue;    //x座標が画像の端のときはcontinue

            nx = (block_x + 1) * (block_size_x) - 1;   //ブロックの右上のx座標
            ny = (block_y) * (block_size_y);          //ブロックの右上のy座標

            if(nx < 0) nx = 0;
            if(target_image.cols <= nx) nx = target_image.cols - 1;
            if(ny < 0) ny = 0;
            if(target_image.rows <= ny) ny = target_image.rows - 1;
            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
        }

        if(block_y == block_num_y) continue;    //y座標が画像の端のときはcontinue
        //y列目の下側の頂点を入れる
        for (int block_x = 0 ; block_x <= block_num_x; block_x++) {
            int nx = block_x * (block_size_x);              //ブロックの左下のx座標
            int ny = (block_y + 1) * (block_size_y) - 1;    //ブロックの左下のy座標

            if(nx < 0) nx = 0;
            if(target_image.cols <= nx) nx = target_image.cols - 1;
            if(ny < 0) ny = 0;
            if(target_image.rows <= ny) ny = target_image.rows - 1;
            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());

            if(block_x == block_num_x) continue;

            nx = (block_x + 1) * (block_size_x) - 1;    //ブロックの右下のx座標
            ny = (block_y + 1) * (block_size_y) - 1;    //ブロックの右下のy座標

            if(nx < 0) nx = 0;
            if(target_image.cols <= nx) nx = target_image.cols - 1;
            if(ny < 0) ny = 0;
            if(target_image.rows <= ny) ny = target_image.rows - 1;
            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
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

    covered_square.resize(static_cast<unsigned long>((block_num_x + 1) * (block_num_y + 1)));

    //なにしてる？？？
    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {
        for(int block_x = 0 ; block_x < block_num_x ; block_x++) {
            //頂点番号
            int p1_idx;
            int p2_idx;
            int p3_idx;
            int p4_idx;
            p1_idx = 2 * block_x + (2 * block_y) * (2 * (block_num_x) + 1);
            p2_idx = p1_idx + 1;
            p3_idx = p1_idx + ((block_num_x) * 2 + 1 );
            p4_idx = p3_idx + 1;

            int squareIndex = insertSquare(p1_idx, p2_idx, p3_idx, p4_idx);
            addNeighborVertex(p1_idx, p2_idx, p3_idx, p4_idx);
            addCoveredSquare(p1_idx, p2_idx, p3_idx, p4_idx, squareIndex); // p1/p2/p3はsquareIndex番目の四角形に含まれている

//            int p4_idx = p2_idx;
//            int p5_idx = p3_idx;
//            int p6_idx = p3_idx + 1;
//
//            squareIndex = insertSquare(p4_idx, p5_idx, p6_idx, TYPE2);
//            addNeighborVertex(p4_idx, p5_idx, p6_idx);
//            addCoveredSquare(p4_idx, p5_idx, p6_idx, squareIndex);
//
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
 * @fn std::vector<Point3Vec> getAllSquareCoordinateList()
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
//    covered_square.emplace_back();
    isCodedSquare.emplace_back(false);
    square_gauss_results.emplace_back();
    square_gauss_results[square_gauss_results.size() - 1].residual = -1.0;
    delete_flag.emplace_back(false);

    return static_cast<int>(squares.size() - 1);
}
//TODO 四角形対応
/**
 * @fn void SquareDivision::eraseSquare(int t_idx)
 * @brief 四角パッチに関わる情報を削除する
 * @param t_idx 四角パッチの番号
 */
void SquareDivision::eraseSquare(int t_idx){
    Square square = squares[t_idx];
    removeSquareNeighborVertex(square.p1_idx, square.p2_idx, square.p3_idx, square.p4_idx);
    removeSquareCoveredSquare(square.p1_idx, square.p2_idx, square.p3_idx, square.p4_idx, t_idx);
    isCodedSquare.erase(isCodedSquare.begin() + t_idx);
    squares.erase(squares.begin() + t_idx);
//    covered_square.erase(covered_square.begin() + t_idx);
    square_gauss_results.erase(square_gauss_results.begin() + t_idx);
    delete_flag.erase(delete_flag.begin() + t_idx);
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
    std::set<int> s = covered_square[target_vertex_idx];
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
    std::set<int> s = covered_square[target_vertex_idx];
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
 * @fn int SquareDivision::getCornerIndex(cv::Point2f p)
 * @brief 頂点が格納されているインデックスを返す。頂点が存在しない場合、その頂点を頂点集合に追加した後インデックスを返す
 * @param[in] p 追加する頂点の座標
 * @return 頂点番号
 */
int SquareDivision::getCornerIndex(cv::Point2f p) {
    if(corner_flag[(int)(p.y * 2)][(int)(p.x * 2)] != -1) return corner_flag[(int)(p.y * 2)][(int)(p.x * 2)]; //すでに頂点がある場合
    corners.emplace_back(p);
    neighbor_vtx.emplace_back();
    covered_square.emplace_back();
    corner_flag[(int)(p.y * 2)][(int)(p.x * 2)] = static_cast<int>(corners.size() - 1);
    return static_cast<int>(corners.size() - 1);
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
    cv::Point2f p3 = corners[square.p3_idx];
    cv::Point2f p4 = corners[square.p4_idx];

    cv::Point2f x = (p2 - p1) / 2.0;
    cv::Point2f y = (p3 - p1) / 2.0;

    cv::Point2f a = p1;
    cv::Point2f b = p2;
    cv::Point2f c = p3;
    cv::Point2f d = p4;
    cv::Point2f e = a + y;
    cv::Point2f f = b + y;
    cv::Point2f g = a + y;    g.y++;
    cv::Point2f h = b + y;    h.y++;

    int e_idx = getCornerIndex(e);
    int f_idx = getCornerIndex(f);
    int g_idx = getCornerIndex(g);
    int h_idx = getCornerIndex(h);

    int a_idx = square.p1_idx;
    int b_idx = square.p2_idx;
    int c_idx = square.p3_idx;
    int d_idx = square.p4_idx;

    int s1_idx = insertSquare(a_idx, b_idx, e_idx, f_idx);
    int s2_idx = insertSquare(g_idx, h_idx, c_idx, d_idx);

    removeSquareNeighborVertex(square.p1_idx, square.p2_idx, square.p3_idx, square.p4_idx);
    removeSquareCoveredSquare( square.p1_idx, square.p2_idx, square.p3_idx, square.p3_idx, square_index);

    addNeighborVertex(a_idx, b_idx, e_idx, f_idx);
    addNeighborVertex(g_idx, h_idx, c_idx, d_idx);

    addCoveredSquare(a_idx, b_idx, e_idx, f_idx, s1_idx);
    addCoveredSquare(g_idx, h_idx, c_idx, d_idx, s2_idx);

    isCodedSquare[square_index] = false;
    delete_flag[square_index] = true;
}
//TODO 四角形対応
/**
 * @fn bool SquareDivision::split(cv::Mat &gaussRefImage, CodingTreeUnit* ctu, Point4Vec square, int square_index, int type, int steps)
 * @brief 与えられたトライアングルを分割するか判定し，必要な場合は分割を行う
 * @details この関数は再帰的に呼び出され，そのたびに分割を行う
 * @param gaussRefImage ガウス・ニュートン法の参照画像
 * @param ctu CodingTreeUnitのrootノード
 * @oaran cmt 時間予測用のCollocatedMvTreeのノード(collocatedmvtree→cmt)
 * @param square 四角形の各点の座標
 * @param square_index 三角形のindex
 * @param type 分割方向
 * @param steps 分割回数
 * @return 分割した場合はtrue, そうでない場合falseを返す
 */
bool SquareDivision::split(std::vector<std::vector<std::vector<unsigned char **>>> expand_images, CodingTreeUnit* ctu, CollocatedMvTree* cmt, Point4Vec square, int square_index, int type, int steps, std::vector<std::vector<int>> &diagonal_line_area_flag) {
    if(steps <= 0) return false;

    double RMSE_before_subdiv = 0.0;
    double error_warping, error_parallel;
    cv::Point2f p1 = square.p1;
    cv::Point2f p2 = square.p2;
    cv::Point2f p3 = square.p3;

    Point3Vec targetSquare(p1, p2, p3);
    int square_size = 0;
    bool parallel_flag;
    int num;

    std::vector<cv::Point2f> dummy;
    std::vector<cv::Point2f> gauss_result_warping;
    cv::Point2f gauss_result_parallel;


    if(cmt == nullptr) {
        cmt = previousMvList[0][square_index];
    }

    if(square_gauss_results[square_index].residual > 0) {
        GaussResult result_before = square_gauss_results[square_index];
        gauss_result_warping = result_before.mv_warping;
        gauss_result_parallel = result_before.mv_parallel;
        RMSE_before_subdiv = result_before.residual;
        square_size = result_before.square_size;
        parallel_flag = result_before.parallel_flag;
        if(parallel_flag){
            error_parallel = result_before.residual;
        }else{
            error_warping = result_before.residual;
        }
        ctu->error_bm = result_before.residual_bm;
        ctu->error_newton = result_before.residual_newton;
    }else {
        if(PRED_MODE == NEWTON) {
            if(GAUSS_NEWTON_INIT_VECTOR) {
                std::vector<cv::Point2f> tmp_bm_mv;
                std::vector<double> tmp_bm_errors;
                std::tie(tmp_bm_mv, tmp_bm_errors) = blockMatching(square, target_image, expansion_ref,
                                                                   diagonal_line_area_flag, square_index, ctu);
                std::tie(gauss_result_warping, gauss_result_parallel, error_warping, error_parallel, square_size) = GaussNewton(ref_images, target_images, expand_images, targetSquare,
                                                                                                                                diagonal_line_area_flag, square_index, ctu, block_size_x,
                                                                                                                                block_size_y, tmp_bm_mv[2], ref_hevc);
            }else{
                std::tie(gauss_result_warping, gauss_result_parallel, error_warping, error_parallel, square_size) = GaussNewton(ref_images, target_images, expand_images, targetSquare,
                                                                                                                                diagonal_line_area_flag, square_index, ctu, block_size_x,
                                                                                                                                block_size_y, cv::Point2f(-1000, -1000), ref_hevc);
            }

            square_gauss_results[square_index].mv_warping = gauss_result_warping;
            square_gauss_results[square_index].mv_parallel = gauss_result_parallel;
            square_gauss_results[square_index].square_size = square_size;
            square_gauss_results[square_index].residual = RMSE_before_subdiv;

            int cost_warping, cost_parallel;
            MV_CODE_METHOD method_warping, method_parallel;

            std::tie(cost_parallel, std::ignore, std::ignore, std::ignore, std::ignore) = getMVD(
                    {gauss_result_parallel, gauss_result_parallel, gauss_result_parallel}, error_parallel,
                    square_index, cmt->mv1, diagonal_line_area_flag, ctu);

#if !GAUSS_NEWTON_PARALLEL_ONLY
            std::tie(cost_warping, std::ignore, std::ignore, std::ignore, method_warping) = getMVD(
                    triangle_gauss_results[triangle_index].mv_warping, error_warping,
                    triangle_index, cmt->mv1, diagonal_line_area_flag, ctu, false, dummy);
#endif
            if(cost_parallel < cost_warping || GAUSS_NEWTON_PARALLEL_ONLY){
                square_gauss_results[square_index].parallel_flag = true;
                square_gauss_results[square_index].residual = error_parallel;
                square_gauss_results[square_index].method = method_parallel;
                parallel_flag = true;
            }else{
                square_gauss_results[square_index].parallel_flag = false;
                square_gauss_results[square_index].residual = error_warping;
                square_gauss_results[square_index].method = method_warping;
                parallel_flag = false;
            }

        }else if(PRED_MODE == BM) {
            std::vector<cv::Point2f> tmp_bm_mv;
            std::vector<double> tmp_bm_errors;
            std::tie(tmp_bm_mv, tmp_bm_errors) = blockMatching(square, target_image, expansion_ref,
                                                               diagonal_line_area_flag, square_index, ctu);
            square_gauss_results[square_index].residual_bm = tmp_bm_errors[2];
            ctu->error_bm = tmp_bm_errors[2];
            gauss_result_warping = tmp_bm_mv;
            gauss_result_parallel = tmp_bm_mv[2];
            RMSE_before_subdiv = tmp_bm_errors[2];
            error_parallel = tmp_bm_errors[2];
            square_gauss_results[square_index].mv_warping = gauss_result_warping;
            square_gauss_results[square_index].mv_parallel = gauss_result_parallel;
            square_gauss_results[square_index].square_size = square_size;
            square_gauss_results[square_index].residual = RMSE_before_subdiv;
            square_gauss_results[square_index].parallel_flag = true;
            parallel_flag = true;
        }
    }

    std::vector<cv::Point2f> mvd;
    int selected_index;
    MV_CODE_METHOD method_flag;
    double cost_before_subdiv;
    int code_length;

    if(square_gauss_results[square_index].parallel_flag) {
        std::tie(cost_before_subdiv, code_length, mvd, selected_index, method_flag) = getMVD(
                {gauss_result_parallel, gauss_result_parallel, gauss_result_parallel}, error_parallel,
                square_index, cmt->mv1, diagonal_line_area_flag, ctu, true, dummy);
    }else{
        std::tie(cost_before_subdiv, code_length, mvd, selected_index, method_flag) = getMVD(
                square_gauss_results[square_index].mv_warping, error_warping,
                square_index, cmt->mv1, diagonal_line_area_flag, ctu, false, dummy);
        std::cout << cost_before_subdiv << std::endl;
    }

    std::vector<cv::Point2i> ret_gauss2;

    std::vector<cv::Point2f> mv;
    if(parallel_flag){
        mv.emplace_back(gauss_result_parallel);
        mv.emplace_back(gauss_result_parallel);
        mv.emplace_back(gauss_result_parallel);
    }else{
        mv = gauss_result_warping;
    }
    ctu->mv1 = mv[0];
    ctu->square_index = square_index;
    ctu->code_length = code_length;
    ctu->collocated_mv = cmt->mv1;
    ctu->parallel_flag = parallel_flag;
    ctu->method = method_flag;

    SplitResult split_squares = getSplitSquare(p1, p2, p3, p4, type);

    SplitResult split_sub_squares1 = getSplitSquare(split_squares.s1.p1, split_squares.s1.p2, split_squares.s1.p3, split_squares.s1.p4, split_squares.s_type);
    SplitResult split_sub_squares2 = getSplitSquare(split_squares.s2.p1, split_squares.s2.p2, split_squares.s2.p3, split_squares.s2.p4, split_squares.s_type);

    std::vector<Point3Vec> subdiv_ref_squares, subdiv_target_squares;
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

//    int p1_idx = getCornerIndex(p1);
//    int p2_idx = getCornerIndex(p2);
//    int p3_idx = getCornerIndex(p3);
//    addCornerAndSquare(Square(p1_idx, p2_idx, p3_idx), square_index, type);

    int t1_p1_idx = getCornerIndex(split_squares.t1.p1);
    int t1_p2_idx = getCornerIndex(split_squares.t1.p2);
    int t1_p3_idx = getCornerIndex(split_squares.t1.p3);
    addCornerAndSquare(Square(t1_p1_idx, t1_p2_idx, t1_p3_idx), square_index, split_squares.t1_type);

    int t2_p1_idx = getCornerIndex(split_squares.t2.p1);
    int t2_p2_idx = getCornerIndex(split_squares.t2.p2);
    int t2_p3_idx = getCornerIndex(split_squares.t2.p3);
    addCornerAndSquare(Square(t2_p1_idx, t2_p2_idx, t2_p3_idx), square_index, split_squares.t2_type);

    int square_indexes[] = {(int)squares.size() - 4, (int)squares.size() - 3, (int)squares.size() - 2, (int)squares.size() - 1};

    std::vector<std::vector<int>> prev_area_flag(diagonal_line_area_flag);

    // 分割回数が偶数回目のとき斜線の更新を行う
    int sx = ceil( std::min({square.p1.x, square.p2.x, square.p3.x}));
    int lx = floor(std::max({square.p1.x, square.p2.x, square.p3.x}));
    int sy = ceil( std::min({square.p1.y, square.p2.y, square.p3.y}));
    int ly = floor(std::max({square.p1.y, square.p2.y, square.p3.y}));

    int width =  (lx - sx) / 2 + 1;
    int height = (ly - sy) / 2 + 1;

    bool flag = true;
    int a, b, c, d;
    if(type == 1) {    //四角形
        for (int x = 0 ; x < width  ; x++) {
            diagonal_line_area_flag[(x + sx) % block_size_x][(x + sy) % block_size_y] = (x % 2 == 0 ? square_indexes[0] : square_indexes[2]);
            flag = !flag;
        }
    }else if(type == 2) {    //長方形
        for (int x = 0 ; x < width ; x++) {
            diagonal_line_area_flag[(sx + width + x) % block_size_x][(sy + height + x) % block_size_y] = (x % 2 == 0 ? square_indexes[1] : square_indexes[3]);
            flag = !flag;
        }

    }

    ctu->node1 = new CodingTreeUnit();
    ctu->node1->triangle_index = squares.size() - 4;
    ctu->node1->parentNode = ctu;
    ctu->node2 = new CodingTreeUnit();
    ctu->node2->triangle_index = squares.size() - 3;
    ctu->node2->parentNode = ctu;
    ctu->node3 = new CodingTreeUnit();
    ctu->node3->triangle_index = squares.size() - 2;
    ctu->node3->parentNode = ctu;
    ctu->node4 = new CodingTreeUnit();
    ctu->node4->triangle_index = squares.size() - 1;
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
        std::vector<double> tmp_bm_errors;
        double cost_warping_tmp, cost_parallel_tmp;
        double tmp_error_newton;
        MV_CODE_METHOD method_warping_tmp, method_parallel_tmp;
        if(PRED_MODE == NEWTON){
            if(GAUSS_NEWTON_INIT_VECTOR) {
                std::tie(tmp_bm_mv, tmp_bm_errors) = blockMatching(subdiv_target_squares[j], target_image,
                                                                   expansion_ref, diagonal_line_area_flag,
                                                                   square_indexes[j], ctus[j]);
                std::tie(mv_warping_tmp, mv_parallel_tmp, error_warping_tmp, error_parallel_tmp,square_size_tmp) = GaussNewton(
                        ref_images, target_images, expand_images, subdiv_target_squares[j], diagonal_line_area_flag,
                        square_indexes[j], ctus[j], block_size_x, block_size_y,
                        tmp_bm_mv[2], ref_hevc);


            }else{
                std::tie(mv_warping_tmp, mv_parallel_tmp, error_warping_tmp, error_parallel_tmp, square_size_tmp) = GaussNewton(
                        ref_images, target_images, expand_images, subdiv_target_squares[j], diagonal_line_area_flag,
                        square_indexes[j], ctus[j], block_size_x, block_size_y,
                        cv::Point2f(-1000, -1000), ref_hevc);
            }

            std::tie(cost_parallel_tmp,std::ignore, std::ignore, std::ignore, method_parallel_tmp) = getMVD(
                    {mv_parallel_tmp, mv_parallel_tmp, mv_parallel_tmp}, error_parallel_tmp,
                    square_indexes[j], cmt->mv1, diagonal_line_area_flag, ctus[j], true, dummy);
#if !GAUSS_NEWTON_PARALLEL_ONLY

            std::tie(cost_warping_tmp, std::ignore, std::ignore, std::ignore, method_warping_tmp) = getMVD(
                    mv_warping_tmp, error_warping_tmp,
                    triangle_indexes[j], cmt->mv1, diagonal_line_area_flag, ctus[j], false, dummy);
#endif

            if(cost_parallel_tmp < cost_warping_tmp || GAUSS_NEWTON_PARALLEL_ONLY){
                square_gauss_results[square_indexes[j]].parallel_flag = true;
                split_mv_result[j] = GaussResult(mv_warping_tmp, mv_parallel_tmp, error_parallel_tmp, square_size_tmp, true, error_parallel_tmp, error_warping_tmp);
            }else{
                square_gauss_results[square_indexes[j]].parallel_flag = false;
                split_mv_result[j] = GaussResult(mv_warping_tmp, mv_parallel_tmp, error_warping_tmp, square_size_tmp, false, error_parallel_tmp, error_warping_tmp);
            }

        }else if(PRED_MODE == BM){
            std::tie(tmp_bm_mv, tmp_bm_errors) = blockMatching(subdiv_target_squares[j], target_image, expansion_ref, diagonal_line_area_flag, square_indexes[j], ctu);
            mv_warping_tmp = tmp_bm_mv;
            mv_parallel_tmp = tmp_bm_mv[2];
            error_parallel_tmp = tmp_bm_errors[2];
            square_size_tmp = (double)1e6;

            split_mv_result[j] = GaussResult(mv_warping_tmp, mv_parallel_tmp, error_parallel_tmp, square_size_tmp, true, tmp_bm_errors[2], tmp_error_newton);
        }

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
                {split_mv_result[0].mv_parallel, split_mv_result[0].mv_parallel, split_mv_result[0].mv_parallel},
                split_mv_result[0].residual,
                square_indexes[0], cmt_left_left->mv1, diagonal_line_area_flag, ctu->node1, true, dummy);
    }else{
        std::tie(cost_after_subdiv1, code_length1, mvd, selected_index, method_flag1) = getMVD(
                split_mv_result[0].mv_warping, split_mv_result[0].residual,
                square_indexes[0], cmt_left_left->mv1, diagonal_line_area_flag, ctu->node1, false, dummy);
    }

    double cost_after_subdiv2;
    int code_length2;
    if(split_mv_result[1].parallel_flag){
        std::tie(cost_after_subdiv2, code_length2, mvd, selected_index, method_flag2) = getMVD(
                {split_mv_result[1].mv_parallel, split_mv_result[1].mv_parallel, split_mv_result[1].mv_parallel}, split_mv_result[1].residual,
                square_indexes[1], cmt_left_right->mv1, diagonal_line_area_flag, ctu->node2, true, dummy);

    }else{
        std::tie(cost_after_subdiv2, code_length2, mvd, selected_index, method_flag2) = getMVD(
                split_mv_result[1].mv_warping, split_mv_result[1].residual,
                square_indexes[1], cmt_left_right->mv1, diagonal_line_area_flag, ctu->node2, false, dummy);
    }

    double cost_after_subdiv3;
    int code_length3;
    if(split_mv_result[2].parallel_flag) {
        std::tie(cost_after_subdiv3, code_length3, mvd, selected_index, method_flag3) = getMVD(
                {split_mv_result[2].mv_parallel, split_mv_result[2].mv_parallel, split_mv_result[2].mv_parallel},
                split_mv_result[2].residual,
                square_indexes[2], cmt_right_left->mv1, diagonal_line_area_flag, ctu->node3, true, dummy);
    }else{
        std::tie(cost_after_subdiv3, code_length3, mvd, selected_index, method_flag3) = getMVD(
                split_mv_result[2].mv_warping, split_mv_result[2].residual,
                square_indexes[2], cmt_right_left->mv1, diagonal_line_area_flag, ctu->node3, false, dummy);
    }

    double cost_after_subdiv4;
    int code_length4;
    if(split_mv_result[3].parallel_flag){
        std::tie(cost_after_subdiv4, code_length4, mvd, selected_index, method_flag4) = getMVD(
                {split_mv_result[3].mv_parallel, split_mv_result[3].mv_parallel, split_mv_result[3].mv_parallel}, split_mv_result[3].residual,
                square_indexes[3], cmt_right_right->mv1, diagonal_line_area_flag, ctu->node4, true, dummy);
    }else{
        std::tie(cost_after_subdiv4, code_length4, mvd, selected_index, method_flag4) = getMVD(
                split_mv_result[3].mv_warping, split_mv_result[3].residual,
                square_indexes[3], cmt_right_right->mv1, diagonal_line_area_flag, ctu->node4, false, dummy);
    }

    double alpha = 1;
    std::cout << "before:" << cost_before_subdiv << " after:" << alpha * (cost_after_subdiv1 + cost_after_subdiv2 + cost_after_subdiv3 + cost_after_subdiv4) << std::endl;
    if(cost_before_subdiv >= alpha * (cost_after_subdiv1 + cost_after_subdiv2 + cost_after_subdiv3 + cost_after_subdiv4)) {
        ctu->split_cu_flag = true;

        int t1_idx = squares.size() - 4;
        int t2_idx = squares.size() - 3;
        int t3_idx = squares.size() - 2;
        int t4_idx = squares.size() - 1;

        // 1つ目の頂点追加
        ctu->node1->triangle_index = t1_idx;
        if(split_mv_result[0].parallel_flag) {
            ctu->node1->mv1 = split_mv_result[0].mv_parallel;
            ctu->node1->mv2 = split_mv_result[0].mv_parallel;
            ctu->node1->mv3 = split_mv_result[0].mv_parallel;
        }else{
            ctu->node1->mv1 = split_mv_result[0].mv_warping[0];
            ctu->node1->mv2 = split_mv_result[0].mv_warping[1];
            ctu->node1->mv3 = split_mv_result[0].mv_warping[2];
        }
        ctu->node1->code_length = code_length1;
        ctu->node1->parallel_flag = split_mv_result[0].parallel_flag;
        ctu->node1->method = method_flag1;
        square_gauss_results[t1_idx] = split_mv_result[0];
        isCodedSquare[t1_idx] = true;
        bool result = split(expand_images, ctu->node1, cmt_left_left, split_sub_squares1.t1, t1_idx,split_sub_squares1.t1_type, steps - 2, diagonal_line_area_flag);

        // 2つ目の三角形
        ctu->node2->triangle_index = t2_idx;
        if(split_mv_result[1].parallel_flag){
            ctu->node2->mv1 = split_mv_result[1].mv_parallel;
            ctu->node2->mv2 = split_mv_result[1].mv_parallel;
            ctu->node2->mv3 = split_mv_result[1].mv_parallel;
        }else{
            ctu->node2->mv1 = split_mv_result[1].mv_warping[0];
            ctu->node2->mv2 = split_mv_result[1].mv_warping[1];
            ctu->node2->mv3 = split_mv_result[1].mv_warping[2];
        }
        ctu->node2->code_length = code_length2;
        ctu->node2->parallel_flag = split_mv_result[1].parallel_flag;
        ctu->node2->method = method_flag2;

        square_gauss_results[t2_idx] = split_mv_result[1];
        isCodedSquare[t2_idx] = true;
        result = split(expand_images, ctu->node2, cmt_left_right, split_sub_squares1.t2, t2_idx, split_sub_squares1.t2_type, steps - 2, diagonal_line_area_flag);

        // 3つ目の三角形
        ctu->node3->triangle_index = t3_idx;
        if(split_mv_result[2].parallel_flag) {
            ctu->node3->mv1 = split_mv_result[2].mv_parallel;
            ctu->node3->mv2 = split_mv_result[2].mv_parallel;
            ctu->node3->mv3 = split_mv_result[2].mv_parallel;
        }else{
            ctu->node3->mv1 = split_mv_result[2].mv_warping[0];
            ctu->node3->mv2 = split_mv_result[2].mv_warping[1];
            ctu->node3->mv3 = split_mv_result[2].mv_warping[2];
        }
        ctu->node3->code_length = code_length3;
        ctu->node3->parallel_flag = split_mv_result[2].parallel_flag;
        ctu->node3->method = method_flag3;
        square_gauss_results[t3_idx] = split_mv_result[2];
        isCodedSquare[t3_idx] = true;
        result = split(expand_images, ctu->node3, cmt_right_left, split_sub_squares2.t1, t3_idx, split_sub_squares2.t1_type, steps - 2, diagonal_line_area_flag);

        // 4つ目の三角形
        ctu->node4->triangle_index = t4_idx;
        if(split_mv_result[3].parallel_flag) {
            ctu->node4->mv1 = split_mv_result[3].mv_parallel;
            ctu->node4->mv2 = split_mv_result[3].mv_parallel;
            ctu->node4->mv3 = split_mv_result[3].mv_parallel;
        }else{
            ctu->node4->mv1 = split_mv_result[3].mv_warping[0];
            ctu->node4->mv2 = split_mv_result[3].mv_warping[1];
            ctu->node4->mv3 = split_mv_result[3].mv_warping[2];
        }
        ctu->node4->code_length = code_length4;
        ctu->node4->parallel_flag = split_mv_result[3].parallel_flag;
        ctu->node4->method = method_flag4;
        square_gauss_results[t4_idx] = split_mv_result[3];
        isCodedSquare[t4_idx] = true;
        result = split(expand_images, ctu->node4, cmt_right_right, split_sub_squares2.t2, t4_idx, split_sub_squares2.t2_type, steps - 2, diagonal_line_area_flag);

        return true;
    }else{
        isCodedSquare[square_index] = true;
        delete_flag[square_index] = false;
        ctu->node1 = ctu->node2 = ctu->node3 = ctu->node4 = nullptr;
        ctu->method = method_flag;
        diagonal_line_area_flag = prev_area_flag;
        eraseSquare(squares.size() - 1);
        eraseSquare(squares.size() - 1);
        eraseSquare(squares.size() - 1);
        eraseSquare(squares.size() - 1);
        addNeighborVertex(squares[square_index].p1_idx,squares[square_index].p2_idx,squares[square_index].p3_idx,squares[square_index].p4_idx);
        addCoveredSquare(squares[square_index].p1_idx,squares[square_index].p2_idx,squares[square_index].p3_idx,squares[square_index].p4_idx, square_index);
        #if MVD_DEBUG_LOG
        std::cout << "triangle_index:" << triangle_index << std::endl;
        std::cout << "p1_idx:" << triangles[triangle_index].first.p1_idx << " p2_idx:" << triangles[triangle_index].first.p2_idx << " p3_idx:" << triangles[triangle_index].first.p3_idx << std::endl;
        std::cout << "p1:" << corners[triangles[triangle_index].first.p1_idx] << " p2:" << corners[triangles[triangle_index].first.p2_idx] << " p3:" << corners[triangles[triangle_index].first.p3_idx] << std::endl;
#endif

        return false;
    }

}

/**
 * @fn SquareDivision::SplitResult SquareDivision::getSplitSquare(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4, int type)
 * @details ３点の座標とtypeを受け取り，分割した形状を返す
 * @param p1 頂点１の座標
 * @param p2 頂点２の座標
 * @param p3 頂点３の座標
 * @param p3 頂点4の座標
 * @param type 分割形状
 * @return 分割結果
 */
SquareDivision::SplitResult SquareDivision::getSplitSquare(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4, int type){
    cv::Point2f a, b, c, d, e, f, g, h;

    switch(type) {
        case 1:
        {
            cv::Point2f x = (p2 - p1) / 2;           //  a       e g      b
                                                     //   -----------------
            a = p1;                                  //   |               |
            b = p2;                                  //   |               |
            c = p3;                                  //   |               |
            d = p4;                                  //   |               |
            e = g = a + x;                           //   -----------------
            f = h = c + x;                           //  c       f h      d
            g.x++;
            h.x++;

            return {Point4Vec(a, e, c, f), Point4Vec(g, b, h, d), 2};
        }
        case 2:
        {
            cv::Point2f y = (p1 - p3) / 2;            //    a         b
                                                      //     ---------
            a = p1;                                   //     |       |
            b = p2;                                   //   e |       | f
            c = p3;                                   //   g |       | h
            d = p4;                                   //     |       |
            e = g = a + y;                            //     ---------
            f = h = b + y;                            //    c         d
            g.y++;
            h.y++;

            return {Point4Vec(a, b, e, f), Point4Vec(g, h, c, d), 1};
        }
        default:
            break;
    }

}
//TODO 四角形対応
/**
 * @fn std::vector<int> getSpatialMvList()
 * @brief t_idx番目の三角形の空間予測動きベクトル候補を返す
 * @param[in] t_idx 三角パッチのインデックス
 * @return 候補のパッチの番号を返す
 */
std::vector<int> SquareDivision::getSpatialSquareList(int t_idx){
    Square square = squares[t_idx];
    std::set<int> spatialTriangles;
    std::vector<int> list1 = getIdxCoveredSquareIndexList(square.p1_idx);
    std::vector<int> list2 = getIdxCoveredSquareIndexList(square.p2_idx);
    std::vector<int> list3 = getIdxCoveredSquareIndexList(square.p3_idx);

    std::set<int> mutualIndexSet1, mutualIndexSet2, mutualIndexSet3;

#if MVD_DEBUG_LOG
    std::cout << "p1:" << squares[t_idx].p1_idx << std::endl;
    for(auto item : list1){
        std::cout << item << std::endl;
    }
    puts("");

    std::cout << "p2:" << squares[t_idx].p2_idx << std::endl;
    for(auto item : list2){
        std::cout << item << std::endl;
    }
    puts("");
    std::cout << "p3:" << squares[t_idx].p3_idx << std::endl;

    for(auto item : list3){
        std::cout << item << std::endl;
    }
    std::cout << "t_idx:" << t_idx << std::endl;
    puts("");

#endif

    for(auto idx : list1) if(isCodedSquare[idx] && idx != t_idx) mutualIndexSet1.emplace(idx);
    for(auto idx : list2) if(isCodedSquare[idx] && idx != t_idx) mutualIndexSet2.emplace(idx);
    for(auto idx : list3) if(isCodedSquare[idx] && idx != t_idx) mutualIndexSet3.emplace(idx);

    for(auto idx : mutualIndexSet1) spatialSquares.emplace(idx);
    for(auto idx : mutualIndexSet2) spatialSquares.emplace(idx);
    for(auto idx : mutualIndexSet3) spatialSquares.emplace(idx);

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
        previousMvList[0][i]->mv2 = cv::Point2f(0, 0);
        previousMvList[0][i]->mv3 = cv::Point2f(0, 0);

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

//TODO 四角形対応
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
//TODO 四角形対応
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
//TODO 四角形対応
/**
 * @fn std::tuple<cv::Point2f, int, MV_CODE_METHOD> RD(int square_idx, CodingTreeUnit* ctu)
 * @brief RDを行い，最適な差分ベクトルを返す
 * @param[in] mv 動きベクトル
 * @param[in] square_idx 三角パッチの番号
 * @param[in] residual そのパッチの残差
 * @param[in] ctu CodingTreeUnit 符号木
 * @param[in] area_flag 含まれる画素のフラグ
 * @return 差分ベクトル，参照したパッチ，空間or時間のフラグのtuple
 */
std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD> SquareDivision::getMVD(std::vector<cv::Point2f> mv, double residual, int square_idx, cv::Point2f &collocated_mv, const std::vector<std::vector<int>> &area_flag, CodingTreeUnit* ctu, bool parallel_flag, std::vector<cv::Point2f> &pixels){
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
        }else{
            // 隣接パッチがワーピングで予想されている場合、そのパッチの0番の動きベクトルを候補とする
            if(!isMvExists(vectors, spatial_square.mv_warping[0])){
                vectors.emplace_back(spatial_square.mv_warping[0], SPATIAL);
            }
        }
    }

#if MVD_DEBUG_LOG
    std::cout << corners[triangles[triangle_idx].first.p1_idx] << " " << corners[triangles[triangle_idx].first.p2_idx] << " " << corners[triangles[triangle_idx].first.p3_idx] << std::endl;
    #endif

    if(!isMvExists(vectors, collocated_mv)) vectors.emplace_back(collocated_mv, SPATIAL);

    if(vectors.size() < 2) vectors.emplace_back(cv::Point2f(0.0, 0.0), Collocated);

    double lambda = getLambdaPred(qp);

    //                      コスト, 差分ベクトル, 番号, タイプ
    std::vector<std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum> > results;
    for(int i = 0 ; i < vectors.size() ; i++) {
        std::pair<cv::Point2f, MV_CODE_METHOD> vector = vectors[i];
        cv::Point2f current_mv = vector.first;
        // TODO: ワーピング対応
        if(parallel_flag) { // 平行移動成分に関してはこれまで通りにやる
            FlagsCodeSum flag_code_sum(0, 0, 0, 0);

            cv::Point2f mvd = current_mv - mv[0];
#if MVD_DEBUG_LOG
            std::cout << "target_vector_idx       :" << i << std::endl;
            std::cout << "diff_target_mv(parallel):" << current_mv << std::endl;
            std::cout << "encode_mv(parallel)     :" << mv[0] << std::endl;
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
        }else{
            std::vector<cv::Point2f> mvds;
            mvds.emplace_back(current_mv - mv[0]);
            mvds.emplace_back(current_mv - mv[1]);
            mvds.emplace_back(current_mv - mv[2]);

            int mvd_code_length = 6;
            FlagsCodeSum flag_code_sum(0, 0, 0, 0);

            for(int j = 0 ; j < mvds.size() ; j++){

#if MVD_DEBUG_LOG
                std::cout << "target_vector_idx       :" << j << std::endl;
                std::cout << "diff_target_mv(warping) :" << current_mv << std::endl;
                std::cout << "encode_mv(warping)      :" << mv[j] << std::endl;
#endif

                cv::Point2f mvd = getQuantizedMv(mvds[j], 4);
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

                flag_code_sum.countGreater0Code();
                flag_code_sum.countGreater0Code();
                flag_code_sum.setXGreater0Flag(is_x_greater_than_zero);
                flag_code_sum.setYGreater0Flag(is_y_greater_than_zero);

                // 動きベクトル差分の絶対値が1より大きいのか？
                bool is_x_greater_than_one = abs_x > 1;
                bool is_y_greater_than_one = abs_y > 1;


                // 正負の判定(これもつかってません！！！）
                bool is_x_minus = mvd.x < 0;
                bool is_y_minus = mvd.y < 0;

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
                        mvd_code_length +=  getExponentialGolombCodeLength((int) mvd_y_minus_2, 0);
                        flag_code_sum.addMvdCodeLength(getExponentialGolombCodeLength((int) mvd_y_minus_2, 0));
                    }
                    flag_code_sum.countGreater1Code();
                    flag_code_sum.setYGreater1Flag(is_y_greater_than_one);
                    flag_code_sum.countSignFlagCode();
                }
            }

            // 参照箇所符号化
            int reference_index = std::get<1>(vector);
            int reference_index_code_length = getUnaryCodeLength(reference_index);

            // 各種フラグ分を(3*2)bit足してます
            double rd = residual + lambda * (mvd_code_length + reference_index_code_length);

            // 結果に入れる
            results.emplace_back(rd, mvd_code_length + reference_index_code_length, mvds, i, vector.second, flag_code_sum);
        }
    }

    // マージ符号化
    // マージで参照する動きベクトルを使って残差を求め直す
    Square current_square_coordinate = squares[square_idx];
    cv::Point2f p1 = corners[current_square_coordinate.p1_idx];
    cv::Point2f p2 = corners[current_square_coordinate.p2_idx];
    cv::Point2f p3 = corners[current_square_coordinate.p3_idx];
    Point3Vec coordinate = Point3Vec(p1, p2, p3);
    vectors.clear();

    std::vector<cv::Point2f> pixels_in_triangle;
    std::vector<std::pair<cv::Point2f, MV_CODE_METHOD>> merge_vectors;
    if(pixels.empty()) {
         pixels_in_square = getPixelsInSquare(coordinate, area_flag, square_idx, ctu,
                                                                          block_size_x, block_size_y);
    }else{
        pixels_in_square = pixels;
    }

    for(int i = 0 ; i < spatial_square_size ; i++) {
        int spatial_triangle_index = spatial_squares[i];
        GaussResult spatial_triangle = square_gauss_results[spatial_triangle_index];
        std::vector<cv::Point2f> mvds;

        if(spatial_triangle.parallel_flag){
            if(!isMvExists(merge_vectors, spatial_triangle.mv_parallel)) {
                merge_vectors.emplace_back(spatial_triangle.mv_parallel, MERGE);
                double ret_residual = getTriangleResidual(ref_image, target_image, coordinate, mv, pixels_in_triangle);
                double rd = ret_residual + lambda * (getUnaryCodeLength(i) + 1);
                results.emplace_back(rd, getUnaryCodeLength(i) + 1, mvds, results.size(), MERGE, FlagsCodeSum(0, 0, 0, 0));
            }
        }else{
            if(!isMvExists(merge_vectors, spatial_square.mv_warping[0])) {
                merge_vectors.emplace_back(spatial_square.mv_warping[0], MERGE);
                double ret_residual = getSquareResidual(ref_image, target_image, coordinate, mv, pixels_in_square);
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
        if(parallel_flag) {
            std::cout << "mvd        :" << mvds[0] << std::endl;
        }else{
            for(auto mvd : mvds){
                std::cout << "mvd        :" << mvd << std::endl;
            }
        }
    }
    puts("");
#endif

    ctu->flags_code_sum = flag_code_sum;
    if(method != MERGE) {
        (ctu->mvds_x).clear();
        (ctu->mvds_y).clear();
        (ctu->original_mvds_x).clear();
        (ctu->original_mvds_y).clear();

        if (parallel_flag) {
            (ctu->mvds_x).emplace_back(mvds[0].x);
            (ctu->mvds_y).emplace_back(mvds[0].y);
        } else {
            for (int i = 0; i < 3; i++) {
                (ctu->mvds_x).emplace_back(mvds[i].x);
                (ctu->mvds_y).emplace_back(mvds[i].y);
            }
        }
    }

    return {cost, code_length, mvds, selected_idx, method};
}
//TODO 四角形対応
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
//TODO 四角形対応
cv::Mat SquareDivision::getPredictedDiagonalImageFromCtu(std::vector<CodingTreeUnit*> ctus, std::vector<std::vector<std::vector<int>>> &area_flag){
    cv::Mat out = cv::Mat::zeros(ref_image.size(), CV_8UC3);
    for(int i = 0 ; i < ctus.size() ; i++) {
        getPredictedDiagonalImageFromCtu(ctus[i], area_flag[(int)i/2], out);
    }

    return out;
}

void SquareDivision::getPredictedDiagonalImageFromCtu(CodingTreeUnit* ctu, std::vector<std::vector<int>> &area_flag, const cv::Mat &out){

    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int triangle_index = ctu->triangle_index;
        Square square_corner_idx = squares[triangle_index];
        Point4Vec square(corners[square_corner_idx.p1_idx], corners[square_corner_idx.p2_idx], corners[square_corner_idx.p3_idx], corners[square_corner_idx.p4_idx]);
        std::vector<cv::Point2f> pixels = getPixelsInSquare(square, area_flag, triangle_index, ctu, block_size_x, block_size_y);
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

    if(ctu->node1 != nullptr) getPredictedDiagonalImageFromCtu(ctu->node1, area_flag, out);
    if(ctu->node2 != nullptr) getPredictedDiagonalImageFromCtu(ctu->node2, area_flag, out);
    if(ctu->node3 != nullptr) getPredictedDiagonalImageFromCtu(ctu->node3, area_flag, out);
    if(ctu->node4 != nullptr) getPredictedDiagonalImageFromCtu(ctu->node4, area_flag, out);
}
//TODO 四角形対応
cv::Mat SquareDivision::getPredictedImageFromCtu(std::vector<CodingTreeUnit*> ctus, std::vector<std::vector<std::vector<int>>> &area_flag){
    cv::Mat out = cv::Mat::zeros(ref_image.size(), CV_8UC3);

#pragma omp parallel for
    for(int i = 0 ; i < ctus.size() ; i++) {
        getPredictedImageFromCtu(ctus[i], out, area_flag[i/2]);
    }

    return out;
}
//TODO 四角形対応
void SquareDivision::getPredictedImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out, std::vector<std::vector<int>> &area_flag){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int square_index = ctu->square_index;
        cv::Point2f mv = ctu->mv1;
        Square square_corner_idx = squares[square_index];
        Point4Vec square(corners[square_corner_idx.p1_idx], corners[square_corner_idx.p2_idx], corners[square_corner_idx.p3_idx], corners[square_corner_idx.p4_idx]);

        std::vector<cv::Point2f> mvs;
        if(ctu->parallel_flag){
            mvs.emplace_back(mv);
            mvs.emplace_back(mv);
            mvs.emplace_back(mv);
        }else{
            mvs.emplace_back(ctu->mv1);
            mvs.emplace_back(ctu->mv2);
            mvs.emplace_back(ctu->mv3);
        }

        getPredictedImage(expansion_ref_uchar, target_image, out, square, mvs, 16, area_flag, ctu->square_index, ctu, cv::Rect(0, 0, block_size_x, block_size_y), ref_hevc);
        return;
    }

    if(ctu->node1 != nullptr) getPredictedImageFromCtu(ctu->node1, out, area_flag);
    if(ctu->node2 != nullptr) getPredictedImageFromCtu(ctu->node2, out, area_flag);
    if(ctu->node3 != nullptr) getPredictedImageFromCtu(ctu->node3, out, area_flag);
    if(ctu->node4 != nullptr) getPredictedImageFromCtu(ctu->node4, out, area_flag);
}
//TODO 四角形対応
cv::Mat SquareDivision::getPredictedColorImageFromCtu(std::vector<CodingTreeUnit*> ctus, std::vector<std::vector<std::vector<int>>> &area_flag, double original_psnr){
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
        getPredictedColorImageFromCtu(ctus[i], out, area_flag[i/2], original_psnr, colors);
    }

    return out;
}
//TODO 四角形対応
void SquareDivision::getPredictedColorImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out, std::vector<std::vector<int>> &area_flag, double original_psnr, std::vector<cv::Scalar> &colors){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int square_index = ctu->triangle_index;
        cv::Point2f mv = ctu->mv1;
        Square square_corner_idx = squares[square_index];
        Point4Vec square(corners[square_corner_idx.p1_idx], corners[square_corner_idx.p2_idx], corners[square_corner_idx.p3_idx], corners[square_corner_idx.p4_idx]);

        std::vector<cv::Point2f> mvs{mv, mv, mv};
        std::vector<cv::Point2f> pixels = getPixelsInSquare(square, area_flag, square_index, ctu, block_size_x, block_size_y);
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
            getPredictedImage(expansion_ref_uchar, target_image, out, square, mvs, 16, area_flag, ctu->square_index, ctu, cv::Rect(0, 0, block_size_x, block_size_y), ref_hevc);
        }
        return;
    }

    if(ctu->node1 != nullptr) getPredictedColorImageFromCtu(ctu->node1, out, area_flag, original_psnr, colors);
    if(ctu->node2 != nullptr) getPredictedColorImageFromCtu(ctu->node2, out, area_flag, original_psnr, colors);
    if(ctu->node3 != nullptr) getPredictedColorImageFromCtu(ctu->node3, out, area_flag, original_psnr, colors);
    if(ctu->node4 != nullptr) getPredictedColorImageFromCtu(ctu->node4, out, area_flag, original_psnr, colors);
}
//TODO 四角形対応
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

//TODO 四角形対応
cv::Mat SquareDivision::getMvImage(std::vector<CodingTreeUnit*> ctus){
    cv::Mat out = target_image.clone();

    for(auto square : getSquareCoordinateList()){
        drawSquare(out, square.p1, square.p2, square.p3, cv::Scalar(255, 255, 255));
    }

    for(int i = 0 ; i < ctus.size() ; i++){
        drawMvImage(out, ctus[i]);
    }

    return out;
}
//TODO 四角形対応
void SquareDivision::drawMvImage(cv::Mat &out, CodingTreeUnit *ctu){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        Square t = squares[ctu->triangle_index];
        cv::Point2f p1 = corners[t.p1_idx];
        cv::Point2f p2 = corners[t.p2_idx];
        cv::Point2f p3 = corners[t.p3_idx];

        cv::Point2f g = (p1 + p2 + p3) / 3.0;

        cv::line(out, g, g+ctu->mv1, GREEN);
    }

    if(ctu->node1 != nullptr) drawMvImage(out, ctu->node1);
    if(ctu->node2 != nullptr) drawMvImage(out, ctu->node2);
    if(ctu->node3 != nullptr) drawMvImage(out, ctu->node3);
    if(ctu->node4 != nullptr) drawMvImage(out, ctu->node4);
}

SquareDivision::SquareDivision() {}

SquareDivision::SplitResult::SplitResult(const Point3Vec &t1, const Point3Vec &t2, int t1Type, int t2Type) : t1(t1),
                                                                                                               t2(t2),
                                                                                                               t1_type(t1Type),
                                                                                                               t2_type(t2Type) {}

SquareDivision::GaussResult::GaussResult(const std::vector<cv::Point2f> &mvWarping, const cv::Point2f &mvParallel,
                                           double residual, int triangleSize, bool parallelFlag, double residualBm, double residualNewton) : mv_warping(
        mvWarping), mv_parallel(mvParallel), residual(residual), square_size(squareSize), parallel_flag(parallelFlag), residual_bm(residualBm), residual_newton(residualNewton) {}

SquareDivision::GaussResult::GaussResult() {}

std::tuple<std::vector<cv::Point2f>, std::vector<double>> SquareDivision::blockMatching(Point3Vec triangle, const cv::Mat& target_image, cv::Mat expansion_ref_image, std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, cv::Point2f fullpell_initial_vector) {
    double sx, sy, lx, ly;
    cv::Point2f tp1, tp2, tp3;
    tp1 = triangle.p1;
    tp2 = triangle.p2;
    tp3 = triangle.p3;

    sx = std::min({tp1.x, tp2.x, tp3.x});
    sy = std::min({tp1.y, tp2.y, tp3.y});
    lx = std::max({tp1.x, tp2.x, tp3.x});
    ly = std::max({tp1.y, tp2.y, tp3.y});

    int width = lx - sx + 1;
    int height = ly - sy + 1;

    sx = sx * 4;
    sy = sy * 4;
    lx = sx + width * 4 - 1;
    ly = sy + height * 4 - 1;

    cv::Point2f mv_tmp(0.0, 0.0); //三角パッチの動きベクトル
    int SX = 16; // ブロックマッチングの探索範囲(X)
    int SY = 16; // ブロックマッチングの探索範囲(Y)

    double e = 1e9, error_min = 1e9, rd_min = 1e9, rd = 1e9;
    int e_count;
    cv::Point2f mv_min;
    int spread_quarter = 64;
    int s = 1;                   //4 : Full-pel, 2 : Half-pel, 1 : Quarter-pel
    std::vector<cv::Point2f> pixels = getPixelsInTriangle(triangle, area_flag, triangle_index, ctu, 128, 128);

    if(fullpell_initial_vector.x == -10000 && fullpell_initial_vector.y == -10000){
        for(int j = -SY * 4 ; j <= SY * 4 ; j += s) {            //j : y方向のMV
            for(int i = -SX * 4 ; i <= SX * 4 ; i += s) {        //i : x方向のMV
                //探索範囲が画像上かどうか判定
                if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
                   && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                    e = 0.0;
                    e_count = 0;
                    for(auto &pixel : pixels) {
                        int ref_x = std::max((int)(4 * pixel.x), 0);
                        ref_x = (i + ref_x + spread_quarter);
                        int ref_y = std::max((int)((4 * pixel.y)), 0);
                        ref_y = (j + ref_y + spread_quarter);
                        e += fabs(R(expansion_ref_image, ref_x, ref_y) - R(target_image, (int)pixel.x, (int)pixel.y));
                        e_count++;
                    }
                }

                cv::Point2f cmt = cv::Point2f(0.0, 0.0);
                std::tie(rd, std::ignore,std::ignore,std::ignore,std::ignore) = getMVD({cv::Point2f((double)i/4.0, (double)j/4.0), cv::Point2f((double)i/4.0, (double)j/4.0), cv::Point2f((double)i/4.0, (double)j/4.0)}, e, triangle_index, cmt, area_flag, ctu, true, pixels);
                if(rd_min > rd){
                    error_min = e;
                    rd_min = rd;
                    mv_min.x = (double)i / 4.0;
                    mv_min.y = (double)j / 4.0;
                }
            }
        }
    }else{
        mv_min.x = (fullpell_initial_vector.x > 0 ? (int)(fullpell_initial_vector.x + 0.5) : (int) (fullpell_initial_vector.x - 0.5));
        mv_min.y = (fullpell_initial_vector.y > 0 ? (int)(fullpell_initial_vector.y + 0.5) : (int) (fullpell_initial_vector.y - 0.5));
    }

    std::vector<cv::Point2f> mvs;
    std::vector<double> errors;
    mvs.emplace_back(mv_min.x, mv_min.y);
    errors.emplace_back(error_min);
    mvs.emplace_back(mv_min.x, mv_min.y);
    errors.emplace_back(error_min);
    mvs.emplace_back(mv_min.x, mv_min.y);
    errors.emplace_back(error_min);

    return std::make_tuple(mvs, errors);
}