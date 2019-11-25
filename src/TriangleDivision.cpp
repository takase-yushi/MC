#include <cmath>

//
// Created by kasph on 2019/04/08.
//

#include "../includes/TriangleDivision.h"
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

TriangleDivision::TriangleDivision(const cv::Mat &refImage, const cv::Mat &targetImage, const cv::Mat &refGaussImage) : target_image(targetImage),
                                                                                                                        ref_image(refImage), ref_gauss_image(refGaussImage) {}



/**
 * @fn void TriangleDivision::initTriangle(int block_size_x, int block_size_y, int _divide_steps, int _qp, int divide_flag)
 * @brief 三角形を初期化する
 * @param[in] _block_size_x
 * @param[in] _block_size_y
 * @param[in] _divide_steps
 * @param[in] _qp
 * @param[in] _divide_flag
 */
void TriangleDivision::initTriangle(int _block_size_x, int _block_size_y, int _divide_steps, int _qp, int divide_flag) {
    block_size_x = _block_size_x;
    block_size_y = _block_size_y;
    qp = _qp;
    int block_num_x = ceil((double)target_image.cols / (block_size_x));
    int block_num_y = ceil((double)target_image.rows / (block_size_y));
    divide_steps = _divide_steps;
    coded_picture_num = 0;

    corners.clear();
    neighbor_vtx.clear();
    covered_triangle.clear();
    triangles.clear();

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
        for (int block_x = 0 ; block_x < block_num_x; block_x++) {
            int nx = block_x * (block_size_x);
            int ny = block_y * (block_size_y);

            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            same_corner_list.emplace_back();
            same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1); // 他と共有している頂点は、自分の番号だけ入れる
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());

            nx = (block_x + 1) * (block_size_x) - 1;
            ny = (block_y) * (block_size_y);

            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            same_corner_list.emplace_back();
            same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1);
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
        }

        for (int block_x = 0 ; block_x < block_num_x; block_x++) {
            int nx = block_x * (block_size_x);
            int ny = (block_y + 1) * (block_size_y) - 1;

            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            same_corner_list.emplace_back();
            same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1);;
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());

            if(block_x == block_num_x) continue;

            nx = (block_x + 1) * (block_size_x) - 1;
            ny = (block_y + 1) * (block_size_y) - 1;

            if(nx < 0) nx = 0;
            if(target_image.cols <= nx) nx = target_image.cols - 1;
            if(ny < 0) ny = 0;
            if(target_image.rows <= ny) ny = target_image.rows - 1;
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
        node->mv2 = cv::Point2f(0.0, 0.0);
        node->mv3 = cv::Point2f(0.0, 0.0);
    }

    std::cout << "block_num_y:" << block_num_y << std::endl;
    std::cout << "block_num_x:" << block_num_x << std::endl;

    covered_triangle.resize(static_cast<unsigned long>((block_num_x * 2) * (block_num_y * 2)));

    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {
        for(int block_x = 0 ; block_x < block_num_x ; block_x++) {
            int p1_idx;
            int p2_idx;
            int p3_idx;
            int p4_idx;
            if(divide_flag == LEFT_DIVIDE) {
                p1_idx = 2 * block_x + (2 * block_y) * ((block_num_x) * 2);
                p2_idx = p1_idx + 1;
                p3_idx = p1_idx + ((block_num_x) * 2);

                int triangleIndex = insertTriangle(p1_idx, p2_idx, p3_idx, TYPE1);
                addNeighborVertex(p1_idx, p2_idx, p3_idx);
                addCoveredTriangle(p1_idx, p2_idx, p3_idx, triangleIndex); // p1/p2/p3はtriangleIndex番目の三角形に含まれている

                int p4_idx = p2_idx;
                int p5_idx = p3_idx;
                int p6_idx = p3_idx + 1;

                triangleIndex = insertTriangle(p4_idx, p5_idx, p6_idx, TYPE2);
                addNeighborVertex(p4_idx, p5_idx, p6_idx);
                addCoveredTriangle(p4_idx, p5_idx, p6_idx, triangleIndex);
            }else{
                int triangleIndex = insertTriangle(p1_idx, p2_idx, p4_idx, TYPE1);
                addNeighborVertex(p1_idx, p2_idx, p4_idx);
                addCoveredTriangle(p1_idx, p2_idx, p4_idx, triangleIndex);

                triangleIndex = insertTriangle(p1_idx, p3_idx, p4_idx, TYPE2);
                addNeighborVertex(p1_idx, p3_idx, p4_idx);
                addCoveredTriangle(p1_idx, p3_idx, p4_idx, triangleIndex);
            }
        }
    }

    for(int i = 0 ; i < isCodedTriangle.size() ; i++) {
        isCodedTriangle[i] = false;
    }

    delete_flag.resize(triangles.size());
    for(int i = 0 ; i < delete_flag.size() ; i++) {
        delete_flag[i] = false;
    }

    predicted_buf.emplace_back(cv::Mat::zeros(ref_image.size()/8, CV_8UC3));
    predicted_buf.emplace_back(cv::Mat::zeros(ref_image.size()/4, CV_8UC3));
    predicted_buf.emplace_back(cv::Mat::zeros(ref_image.size()/2, CV_8UC3));
    predicted_buf.emplace_back(cv::Mat::zeros(ref_image.size(), CV_8UC3));

    ref_images = getRefImages(ref_image, ref_gauss_image);
    target_images = getTargetImages(target_image);

    int expansion_size = SEARCH_RANGE;
    int scaled_expansion_size = expansion_size + 2;
    if(HEVC_REF_IMAGE) expansion_ref = getExpansionMatHEVCImage(ref_image, 4, expansion_size);
    else expansion_ref = getExpansionMatImage(ref_image, 4, scaled_expansion_size);

    ref_hevc = getExpansionHEVCImage(ref_image, 4, SEARCH_RANGE);

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
    for(int block_x = 1 ; block_x < (block_num_x * 2) - 1; block_x+=2){
        int p1_idx = block_x;
        int p2_idx = block_x + 1;
        same_corner_list[p1_idx].emplace(p2_idx);
        same_corner_list[p2_idx].emplace(p1_idx);
    }

    for(int block_y = 1 ; block_y < (block_num_y * 2) - 1; block_y+=2){
        for(int block_x = 1 ; block_x < (block_num_x * 2) - 1; block_x+=2){
            int p1_idx = block_x +     2 * block_num_x * block_y;
            int p2_idx = block_x + 1 + 2 * block_num_x * block_y;

            int p3_idx = p1_idx + 2 * block_num_x;
            int p4_idx = p3_idx + 1;

            same_corner_list[p1_idx].emplace(p2_idx);
            same_corner_list[p1_idx].emplace(p3_idx);
            same_corner_list[p1_idx].emplace(p4_idx);
            same_corner_list[p2_idx].emplace(p1_idx);
            same_corner_list[p2_idx].emplace(p3_idx);
            same_corner_list[p2_idx].emplace(p4_idx);
            same_corner_list[p3_idx].emplace(p1_idx);
            same_corner_list[p3_idx].emplace(p2_idx);
            same_corner_list[p3_idx].emplace(p4_idx);
            same_corner_list[p4_idx].emplace(p1_idx);
            same_corner_list[p4_idx].emplace(p2_idx);
            same_corner_list[p4_idx].emplace(p3_idx);
        }
    }

    std::cout << same_corner_list.size() << std::endl;
    // 0行目
    for(int block_x = 1 ; block_x < (block_num_x * 2) - 1; block_x+=2){
        int p1_idx = block_x + 2 * block_num_x * (2 * block_num_y - 1);
        int p2_idx = p1_idx + 1;
        same_corner_list[p1_idx].emplace(p2_idx);
        same_corner_list[p2_idx].emplace(p1_idx);
    }

    // イントラ用のMatを作る
    intra_tmp_image = cv::Mat::zeros(target_image.rows, target_image.cols, CV_8UC3);

    intra_flag.resize(target_image.cols);
    for(int i = 0 ; i < intra_flag.size() ; i++){
        intra_flag[i].resize(target_image.rows);
    }
}

/**
 * @fn std::vector<Point3Vec> getTriangleCoordinateList()
 * @brief 現在存在する三角形の集合(座標)を返す
 * @return 三角形の集合（座標）
 */
std::vector<Point3Vec> TriangleDivision::getTriangleCoordinateList() {
    std::vector<Point3Vec> vec;

    for(int i = 0 ; i < triangles.size() ; i++) {
        if(delete_flag[i] || !isCodedTriangle[i]) continue;
        Triangle triangle = triangles[i].first;
        vec.emplace_back(corners[triangle.p1_idx], corners[triangle.p2_idx], corners[triangle.p3_idx]);
    }

    return vec;
}

/**
 * @fn std::vector<Triangle> TriangleDivision::getTriangleIndexList()
 * @brief 現在存在する三角形の集合(インデックス)を返す
 * @return 三角形の集合（インデックス）
 */
std::vector<Triangle> TriangleDivision::getTriangleIndexList() {
    std::vector<Triangle> v;
    for(int i = 0 ; i < triangles.size() ; i++) {
        if(delete_flag[i]) continue;
        v.emplace_back(triangles[i].first);
    }
    return v;
}

/**
 * @fn std::vector<Point3Vec> getAllTriangleCoordinateList()
 * @brief 現在存在するすべての三角形の集合(座標)を返す（※論理削除されたパッチも含まれています）
 * @return 三角形の集合（座標）
 */
std::vector<Point3Vec> TriangleDivision::getAllTriangleCoordinateList() {
    std::vector<Point3Vec> vec;

    for(int i = 0 ; i < triangles.size() ; i++) {
        Triangle triangle = triangles[i].first;
        vec.emplace_back(corners[triangle.p1_idx], corners[triangle.p2_idx], corners[triangle.p3_idx]);
    }

    return vec;
}

/**
 * @fn std::vector<Triangle> TriangleDivision::getAllTriangleIndexList()
 * @brief 現在存在する三角形の集合(インデックス)を返す（※論理削除されたパッチも含まれています）
 * @return 三角形の集合（インデックス）
 */
std::vector<Triangle> TriangleDivision::getAllTriangleIndexList() {
    std::vector<Triangle> v;
    for(int i = 0 ; i < triangles.size() ; i++) {
        v.emplace_back(triangles[i].first);
    }
    return v;
}


std::vector<std::pair<Point3Vec, int> > TriangleDivision::getTriangles() {
    std::vector<std::pair<Point3Vec, int> > ts;

    cv::Point2f p1, p2, p3;
    for(auto & triangle : triangles){
        p1 = corners[triangle.first.p1_idx];
        p2 = corners[triangle.first.p2_idx];
        p3 = corners[triangle.first.p3_idx];
        ts.emplace_back(Point3Vec(p1, p2, p3), triangle.second);
    }

    return ts;
}

/**
 * @fn std::vector<cv::Point2f> TriangleDivision::getCorners()
 * @brief 頂点の集合を返す
 * @return 頂点
 */
std::vector<cv::Point2f> TriangleDivision::getCorners() {
    return corners;
}

/**
 * @fn int TriangleDivision::insertTriangle(int p1_idx, int p2_idx, int p3_idx, int type)
 * @brief 三角形を追加する
 * @param[in] p1_idx 頂点1の座標のインデックス
 * @param[in] p2_idx 頂点2の座標のインデックス
 * @param[in] p3_idx 頂点3の座標のインデックス
 * @param[in] type 分割タイプ
 * @return 挿入した三角形が格納されているインデックス
 */
int TriangleDivision::insertTriangle(int p1_idx, int p2_idx, int p3_idx, int type) {
    std::vector<std::pair<cv::Point2f, int> > v;
    v.emplace_back(corners[p1_idx], p1_idx);
    v.emplace_back(corners[p2_idx], p2_idx);
    v.emplace_back(corners[p3_idx], p3_idx);

    // ラスタスキャン順でソート
    sort(v.begin(), v.end(), [](const std::pair<cv::Point2f, int> &a1, const std::pair<cv::Point2f, int> &a2) {
        if (a1.first.y != a2.first.y) {
            return a1.first.y < a2.first.y;
        } else {
            return a1.first.x < a2.first.x;
        }
    });

    Triangle triangle(v[0].second, v[1].second, v[2].second, static_cast<int>(triangles.size()));

    triangles.emplace_back(triangle, type);
    isCodedTriangle.emplace_back(false);
    triangle_gauss_results.emplace_back();
    triangle_gauss_results[triangle_gauss_results.size() - 1].residual = -1.0;
    delete_flag.emplace_back(false);

    return static_cast<int>(triangles.size() - 1);
}

/**
 * @fn void TriangleDivision::eraseTriangle(int t_idx)
 * @brief 三角パッチに関わる情報を削除する
 * @param t_idx 三角パッチの番号
 */
void TriangleDivision::eraseTriangle(int t_idx){
    Triangle triangle = triangles[t_idx].first;
    removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
    removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, t_idx);
    isCodedTriangle.erase(isCodedTriangle.begin() + t_idx);
    triangles.erase(triangles.begin() + t_idx);
//    covered_triangle.erase(covered_triangle.begin() + t_idx);
    triangle_gauss_results.erase(triangle_gauss_results.begin() + t_idx);
    delete_flag.erase(delete_flag.begin() + t_idx);
}

/**
 * @fn void TriangleDivision::addNeighborVertex(int p1_idx, int p2_idx, int p3_idx, int divide_flag)
 * @brief p1, p2, p3の隣接頂点情報を更新する
 * @param[in] p1_idx 頂点1の座標のインデックス
 * @param[in] p2_idx 頂点2の座標のインデックス
 * @param[in] p3_idx 頂点3の座標のインデックス
 */
void TriangleDivision::addNeighborVertex(int p1_idx, int p2_idx, int p3_idx) {
    neighbor_vtx[p1_idx].emplace(p2_idx);
    neighbor_vtx[p2_idx].emplace(p1_idx);

    neighbor_vtx[p1_idx].emplace(p3_idx);
    neighbor_vtx[p3_idx].emplace(p1_idx);

    neighbor_vtx[p2_idx].emplace(p3_idx);
    neighbor_vtx[p3_idx].emplace(p2_idx);

}

/***
 * @fn void TriangleDivision::addCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_no)
 * @brief ある頂点を含む三角形のインデックスの情報を更新する
 * @param[in] p1_idx 頂点1の座標のインデックス
 * @param[in] p2_idx 頂点2の座標のインデックス
 * @param[in] p3_idx 頂点3の座標のインデックス
 * @param[in] triangle_no 三角形のインデックス
 */
void TriangleDivision::addCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_no) {
    covered_triangle[p1_idx].emplace(triangle_no);
    covered_triangle[p2_idx].emplace(triangle_no);
    covered_triangle[p3_idx].emplace(triangle_no);
}

/**
 * @fn double TriangleDivision::getDistance(const cv::Point2f &a, const cv::Point2f &b)
 * @brief 2点間の距離を返す
 * @param[in] a 点1ベクトル
 * @param[in] b 点2のベクトル
 * @return 2点間の距離（スカラー）
 */
double TriangleDivision::getDistance(const cv::Point2f &a, const cv::Point2f &b){
    cv::Point2f v = a - b;
    return std::sqrt(v.x * v.x + v.y * v.y);
}

/**
 * @fn std::vector<int> TriangleDivision::getNeighborVertexIndexList(int idx)
 * @brief 指定された頂点に隣接する頂点（インデックス）の集合を返す
 * @param[in] idx 頂点のインデックス
 * @return 頂点の集合（インデックス）
 */
std::vector<int> TriangleDivision::getNeighborVertexIndexList(int idx) {
    std::set<int> s = neighbor_vtx[idx];
    std::vector<int> v(s.size());

    for(const auto e : s) {
        v.emplace_back(e);
    }

    return v;
}

/**
 * @fn std::vector<cv::Point2f> TriangleDivision::getNeighborVertexCoordinateList(int idx)
 * @brief 指定された頂点に隣接する頂点の集合（座標）を返す
 * @param[in] idx 頂点のインデックス
 * @return 頂点の集合（座標）
 */
std::vector<cv::Point2f> TriangleDivision::getNeighborVertexCoordinateList(int idx) {
    std::set<int> s = neighbor_vtx[idx];
    std::vector<cv::Point2f> v(s.size());

    for(const auto e : s) {
        v.emplace_back(corners[e]);
    }

    return v;
}

/**
 * @fn std::vector<Point3Vec> TriangleDivision::getIdxCoveredTriangleCoordinateList(int idx)
 * @brief 指定された頂点が含まれる三角形の集合を返す
 * @param[in] target_vertex_idx 頂点のインデックス
 * @return 三角形の集合(座標で返される)
 */
std::vector<Point3Vec> TriangleDivision::getIdxCoveredTriangleCoordinateList(int target_vertex_idx) {
    std::set<int> same_corners = same_corner_list[target_vertex_idx];

    std::set<int> s;
    std::set<int> tmp_s;
    for(auto same_corner : same_corners){
        tmp_s = covered_triangle[same_corner];
        for(auto idx : tmp_s) s.emplace(idx);
    }
    std::vector<Point3Vec> v(s.size());

    for(auto triangle_idx : s) {
        Triangle triangle = triangles[triangle_idx].first;
        v.emplace_back(corners[triangle.p1_idx], corners[triangle.p2_idx], corners[triangle.p3_idx]);
    }

    return v;
}

/**
 * @fn std::vector<int> TriangleDivision::getIdxCoveredTriangleIndexList(int idx)
 * @brief 指定の頂点を含む三角形の集合（頂点番号）を返す
 * @param[in] idx 頂点のインデックス
 * @return 三角形の集合（座標）
 */
std::vector<int> TriangleDivision::getIdxCoveredTriangleIndexList(int target_vertex_idx) {
    std::set<int> same_corners = same_corner_list[target_vertex_idx];

    std::set<int> s;
    std::set<int> tmp_s;
    for(auto same_corner : same_corners){
        tmp_s = covered_triangle[same_corner];
        for(auto idx : tmp_s) s.emplace(idx);
    }
    std::vector<int> v;

    for(auto triangle_idx : s) {
        v.emplace_back(triangle_idx);
    }

    std::sort(v.begin(), v.end());

    return v;
}

/**
 * @fn void TriangleDivision::removeTriangleNeighborVertex(int p1_idx, int p2_idx, int p3_idx)
 * @brief 指定された三角形に含まれる頂点隣接ノード集合から、自分以外のノードを消す
 * @details 日本語が難しいからコードで理解して
 * @param p1_idx
 * @param p2_idx
 * @param p3_idx
 */
void TriangleDivision::removeTriangleNeighborVertex(int p1_idx, int p2_idx, int p3_idx) {
    neighbor_vtx[p1_idx].erase(p2_idx);
    neighbor_vtx[p1_idx].erase(p3_idx);
    neighbor_vtx[p2_idx].erase(p1_idx);
    neighbor_vtx[p2_idx].erase(p3_idx);
    neighbor_vtx[p3_idx].erase(p1_idx);
    neighbor_vtx[p3_idx].erase(p2_idx);
}

/**
 * @fn void TriangleDivision::removeTriangleCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_idx)
 * @brief p1, p2, p3を含む三角形の集合から, triangle_idx番目の三角形を消す
 * @param p1_idx 頂点1のインデックス
 * @param p2_idx 頂点2のインデックス
 * @param p3_idx 頂点3のインデックス
 * @param triangle_idx 削除対象の三角形のインデックス
 */
void TriangleDivision::removeTriangleCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_idx) {
    covered_triangle[p1_idx].erase(triangle_idx);
    covered_triangle[p2_idx].erase(triangle_idx);
    covered_triangle[p3_idx].erase(triangle_idx);
}

/**
 * @fn int TriangleDivision::getCornerIndex(cv::Point2f p)
 * @brief 頂点が格納されているインデックスを返す。頂点が存在しない場合、その頂点を頂点集合に追加した後インデックスを返す
 * @param[in] p 追加する頂点の座標
 * @return 頂点番号
 */
int TriangleDivision::getCornerIndex(cv::Point2f p) {
    if(corner_flag[(int)(p.y * 2)][(int)(p.x * 2)] != -1) return corner_flag[(int)(p.y * 2)][(int)(p.x * 2)];
    corners.emplace_back(p);
    neighbor_vtx.emplace_back();
    covered_triangle.emplace_back();
    corner_flag[(int)(p.y * 2)][(int)(p.x * 2)] = static_cast<int>(corners.size() - 1);
    same_corner_list.emplace_back();
    same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1);
    return static_cast<int>(corners.size() - 1);
}


/**
 *
 * @param triangle
 * @param triangle_index
 * @param type
 * @return
 */
void TriangleDivision::addCornerAndTriangle(Triangle triangle, int triangle_index, int type){
    switch(type) {
        case DIVIDE::TYPE1:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p2 - p1) / 2.0;
            cv::Point2f y = (p3 - p1) / 2.0;

            cv::Point2f a = p1;
            cv::Point2f b = p2;
            cv::Point2f c = a + x + y;
            cv::Point2f d = p3;

            int c_idx = getCornerIndex(c);

            int a_idx = triangle.p1_idx;
            int b_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, c_idx, TYPE5);
            int t2_idx = insertTriangle(a_idx, c_idx, d_idx, TYPE6);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, c_idx);
            addNeighborVertex(a_idx, c_idx, d_idx);

            addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
            addCoveredTriangle(a_idx, c_idx, d_idx, t2_idx);

        }
            break;
        case DIVIDE::TYPE2:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p2 - p3) / 2.0;
            cv::Point2f y = (p1 - p3) / 2.0;

            cv::Point2f a = p1;
            cv::Point2f b = p3 + x + y;
            cv::Point2f c = p2;
            cv::Point2f d = p3;

            int b_idx = getCornerIndex(b);

            int a_idx = triangle.p1_idx;
            int c_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, d_idx, TYPE8);
            int t2_idx = insertTriangle(b_idx, c_idx, d_idx, TYPE7);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, d_idx);
            addNeighborVertex(b_idx, c_idx, d_idx);

            addCoveredTriangle(a_idx, b_idx, d_idx, t1_idx);
            addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);
        }
            break;
        case DIVIDE::TYPE3:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p1 - p2) / 2.0;
            cv::Point2f y = (p3 - p2) / 2.0;

            cv::Point2f a = p1;
            cv::Point2f b = p2;
            cv::Point2f c = p2 + x + y;
            cv::Point2f d = p3;

            int c_idx = getCornerIndex(c);

            int a_idx = triangle.p1_idx;
            int b_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, c_idx, TYPE5);
            int t2_idx = insertTriangle(b_idx, c_idx, d_idx, TYPE8);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, c_idx);
            addNeighborVertex(b_idx, c_idx, d_idx);

            addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
            addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);

        }
            break;
        case DIVIDE::TYPE4:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p3 - p2) / 2.0;
            cv::Point2f y = (p1 - p2) / 2.0;

            cv::Point2f a = p1;
            cv::Point2f b = p2 + x + y;
            cv::Point2f c = p2;
            cv::Point2f d = p3;

            int b_idx = getCornerIndex(b);

            int a_idx = triangle.p1_idx;
            int c_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, c_idx, TYPE6);
            int t2_idx = insertTriangle(b_idx, c_idx, d_idx, TYPE7);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, c_idx);
            addNeighborVertex(b_idx, c_idx, d_idx);

            addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
            addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);

        }
            break;
        case DIVIDE::TYPE5:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p2 - p1) / 2.0;
            x.x = (int)x.x;

            cv::Point2f b1 = p1 + x;
            cv::Point2f b2 = p1 + x;
            b2.x += 1;
            b1.y = (int)b1.y;
            b2.y = (int)b2.y;

            cv::Point2f d1 = p3;
            cv::Point2f d2 = p3;
            d1.x = (int)d1.x;
            d1.y = (int)d1.y;
            d2.x = ceil(d2.x);
            d2.y = (int)(d2.y);

            int b1_idx = getCornerIndex(b1);
            int b2_idx = getCornerIndex(b2);
            int d1_idx = getCornerIndex(d1);
            int d2_idx = getCornerIndex(d2);

            int a_idx = triangle.p1_idx;
            int c_idx = triangle.p2_idx;

            int t1_idx = insertTriangle(a_idx, b1_idx, d1_idx, TYPE3);
            int t2_idx = insertTriangle(b2_idx, c_idx, d2_idx, TYPE1);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b1_idx, d1_idx);
            addNeighborVertex(b2_idx, c_idx, d2_idx);

            addCoveredTriangle(a_idx, b1_idx, d1_idx, t1_idx);
            addCoveredTriangle(b2_idx, c_idx, d2_idx, t2_idx);

            same_corner_list[b1_idx].emplace(b2_idx);
            same_corner_list[b2_idx].emplace(b1_idx);

            same_corner_list[d1_idx].emplace(d2_idx);
            same_corner_list[d2_idx].emplace(d1_idx);

        }
            break;
        case DIVIDE::TYPE6:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f y = (p3 - p1) / 2.0;
            y.y = (int)y.y;

            cv::Point2f b1 = p1 + y;
            cv::Point2f b2 = p1 + y;
            b2.y += 1;

            cv::Point2f c1 = p2;
            cv::Point2f c2 = p2;
            c1.x = (int)c1.x;
            c1.y = (int)c1.y;
            c2.x = (int)(c2.x);
            c2.y = ceil(c2.y);

            int b1_idx = getCornerIndex(b1);
            int b2_idx = getCornerIndex(b2);
            int c1_idx = getCornerIndex(c1);
            int c2_idx = getCornerIndex(c2);

            int a_idx = triangle.p1_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b1_idx, c1_idx, TYPE4);
            int t2_idx = insertTriangle(b2_idx, c2_idx, d_idx, TYPE1);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b1_idx, c1_idx);
            addNeighborVertex(b2_idx, c2_idx, d_idx);

            addCoveredTriangle(a_idx, b1_idx, c1_idx, t1_idx);
            addCoveredTriangle(b2_idx, c2_idx, d_idx, t2_idx);

            same_corner_list[b1_idx].emplace(b2_idx);
            same_corner_list[b2_idx].emplace(b1_idx);

            same_corner_list[c1_idx].emplace(c2_idx);
            same_corner_list[c2_idx].emplace(c1_idx);
        }
            break;
        case DIVIDE::TYPE7:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p3 - p2) / 2.0;
            x.x = (int)x.x;

            cv::Point2f a1 = p1, a2 = p1;
            a1.x = (int)a1.x;
            a1.y = ceil(a1.y);
            a2.x = ceil(a2.x);
            a2.y = ceil(a2.y);

            cv::Point2f c1 = p2 + x;
            cv::Point2f c2 = p2 + x;
            c2.x += 1;

            int a1_idx = getCornerIndex(a1);
            int a2_idx = getCornerIndex(a2);
            int c1_idx = getCornerIndex(c1);
            int c2_idx = getCornerIndex(c2);

            int b_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a1_idx, b_idx, c1_idx, TYPE2);
            int t2_idx = insertTriangle(a2_idx, c2_idx, d_idx, TYPE4);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a1_idx, b_idx, c1_idx);
            addNeighborVertex(a2_idx, c2_idx, d_idx);

            addCoveredTriangle(a1_idx, b_idx, c1_idx, t1_idx);
            addCoveredTriangle(a2_idx, c2_idx, d_idx, t2_idx);

            same_corner_list[a1_idx].emplace(a2_idx);
            same_corner_list[a2_idx].emplace(a1_idx);

            same_corner_list[c1_idx].emplace(c2_idx);
            same_corner_list[c2_idx].emplace(c1_idx);

        }
            break;
        case DIVIDE::TYPE8:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f y = (p3 - p1) / 2.0;

            cv::Point2f a1 = p2;
            cv::Point2f a2 = p2;
            a1.x = ceil(a1.x);
            a1.y = (int)a1.y;
            a2.x = ceil(a2.x);
            a2.y = ceil(a2.y);

            cv::Point2f c1 = p1 + y;
            cv::Point2f c2 = p1 + y;
            c1.y = (int)c1.y;
            c2.y = ceil(c2.y);

            int a1_idx = getCornerIndex(a1);
            int a2_idx = getCornerIndex(a2);
            int c1_idx = getCornerIndex(c1);
            int c2_idx = getCornerIndex(c2);

            int b_idx = triangle.p1_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(b_idx, a1_idx, c1_idx, TYPE2);
            int t2_idx = insertTriangle(a2_idx, c2_idx, d_idx, TYPE3);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(b_idx, a1_idx, c1_idx);
            addNeighborVertex(a2_idx, c2_idx, d_idx);

            addCoveredTriangle(b_idx, a1_idx, c1_idx, t1_idx);
            addCoveredTriangle(a2_idx, c2_idx, d_idx, t2_idx);

            same_corner_list[a1_idx].emplace(a2_idx);
            same_corner_list[a2_idx].emplace(a1_idx);

            same_corner_list[c1_idx].emplace(c2_idx);
            same_corner_list[c2_idx].emplace(c1_idx);
        }
            break;
        default:
            break;
    }

    isCodedTriangle[triangle_index] = false;
    delete_flag[triangle_index] = true;
}

/**
 * @fn bool TriangleDivision::split(cv::Mat &gaussRefImage, CodingTreeUnit* ctu, Point3Vec triangle, int triangle_index, int type, int steps)
 * @brief 与えられたトライアングルを分割するか判定し，必要な場合は分割を行う
 * @details この関数は再帰的に呼び出され，そのたびに分割を行う
 * @param gaussRefImage ガウス・ニュートン法の参照画像
 * @param ctu CodingTreeUnitのrootノード
 * @oaran cmt 時間予測用のCollocatedMvTreeのノード(collocatedmvtree→cmt)
 * @param triangle 三角形の各点の座標
 * @param triangle_index 三角形のindex
 * @param type 分割方向
 * @param steps 分割回数
 * @return 分割した場合はtrue, そうでない場合falseを返す
 */
bool TriangleDivision::split(std::vector<std::vector<std::vector<unsigned char **>>> expand_images, CodingTreeUnit* ctu, CollocatedMvTree* cmt, Point3Vec triangle, int triangle_index, int type, int steps, std::vector<std::vector<int>> &diagonal_line_area_flag) {


    double RMSE_before_subdiv = 0.0;
    double error_warping, error_translation;
    cv::Point2f p1 = triangle.p1;
    cv::Point2f p2 = triangle.p2;
    cv::Point2f p3 = triangle.p3;

    Point3Vec targetTriangle(p1, p2, p3);
    int triangle_size = 0;
    bool translation_flag;
    int num;

    std::vector<cv::Point2f> dummy;
    std::vector<cv::Point2f> gauss_result_warping;
    cv::Point2f gauss_result_translation;

    int warping_limit = 4;

    if(cmt == nullptr) {
        cmt = previousMvList[0][triangle_index];
    }

    if(triangle_gauss_results[triangle_index].residual > 0) {
        GaussResult result_before = triangle_gauss_results[triangle_index];
        gauss_result_warping = result_before.original_mv_warping;
        gauss_result_translation = result_before.original_mv_translation;
        RMSE_before_subdiv = result_before.residual;
        triangle_size = result_before.triangle_size;
        translation_flag = result_before.translation_flag;
        if(translation_flag){
            error_translation = result_before.residual;
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
                std::tie(tmp_bm_mv, tmp_bm_errors) = fullpellBlockMatching(triangle, target_image, expansion_ref,
                                                                   diagonal_line_area_flag, triangle_index, ctu);
                std::tie(gauss_result_warping, gauss_result_translation, error_warping, error_translation, triangle_size) = GaussNewton(ref_images, target_images, expand_images, targetTriangle,
                                                      diagonal_line_area_flag, triangle_index, ctu, block_size_x,
                                                      block_size_y, tmp_bm_mv[2], ref_hevc);
#if USE_BM_TRANSLATION_MV
                gauss_result_translation = tmp_bm_mv[2];
                error_translation = tmp_bm_errors[2];
#endif
            }else{
                std::tie(gauss_result_warping, gauss_result_translation, error_warping, error_translation, triangle_size) = GaussNewton(ref_images, target_images, expand_images, targetTriangle,
                                                      diagonal_line_area_flag, triangle_index, ctu, block_size_x,
                                                      block_size_y, cv::Point2f(-1000, -1000), ref_hevc);
            }

            triangle_gauss_results[triangle_index].mv_warping = gauss_result_warping;
            triangle_gauss_results[triangle_index].mv_translation = gauss_result_translation;
            triangle_gauss_results[triangle_index].original_mv_warping = gauss_result_warping;
            triangle_gauss_results[triangle_index].original_mv_translation = gauss_result_translation;
            triangle_gauss_results[triangle_index].triangle_size = triangle_size;
            triangle_gauss_results[triangle_index].residual = RMSE_before_subdiv;

            int cost_warping, cost_translation;
            MV_CODE_METHOD method_warping, method_translation;
            std::tie(cost_translation, std::ignore, std::ignore, std::ignore, method_translation) = getMVD(
                    {gauss_result_translation, gauss_result_translation, gauss_result_translation}, error_translation,
                    triangle_index, cmt->mv1, diagonal_line_area_flag, ctu, true, dummy);
#if !GAUSS_NEWTON_TRANSLATION_ONLY
            std::tie(cost_warping, std::ignore, std::ignore, std::ignore, method_warping) = getMVD(
                    triangle_gauss_results[triangle_index].mv_warping, error_warping,
                    triangle_index, cmt->mv1, diagonal_line_area_flag, ctu, false, dummy);
#endif
            if(cost_translation < cost_warping || (steps <= warping_limit)|| GAUSS_NEWTON_TRANSLATION_ONLY){
                triangle_gauss_results[triangle_index].translation_flag = true;
                triangle_gauss_results[triangle_index].residual = error_translation;
                triangle_gauss_results[triangle_index].method = method_translation;
                translation_flag = true;
            }else{
                triangle_gauss_results[triangle_index].translation_flag = false;
                triangle_gauss_results[triangle_index].residual = error_warping;
                triangle_gauss_results[triangle_index].method = method_warping;
                translation_flag = false;
            }

        }else if(PRED_MODE == BM) {
            std::vector<cv::Point2f> tmp_bm_mv;
            std::vector<double> tmp_bm_errors;
            std::tie(tmp_bm_mv, tmp_bm_errors) = blockMatching(triangle, target_image, expansion_ref,
                                                               diagonal_line_area_flag, triangle_index, ctu);
            triangle_gauss_results[triangle_index].residual_bm = tmp_bm_errors[2];
            ctu->error_bm = tmp_bm_errors[2];
            gauss_result_warping = tmp_bm_mv;
            gauss_result_translation = tmp_bm_mv[2];
            RMSE_before_subdiv = tmp_bm_errors[2];
            error_translation = tmp_bm_errors[2];
            triangle_gauss_results[triangle_index].mv_warping = gauss_result_warping;
            triangle_gauss_results[triangle_index].mv_translation = gauss_result_translation;
            triangle_gauss_results[triangle_index].triangle_size = triangle_size;
            triangle_gauss_results[triangle_index].residual = RMSE_before_subdiv;
            triangle_gauss_results[triangle_index].translation_flag = true;
            translation_flag = true;
        }
    }

    std::vector<cv::Point2f> mvd;
    int selected_index;
    MV_CODE_METHOD method_flag;
    double cost_before_subdiv;
    int code_length;

    if(triangle_gauss_results[triangle_index].translation_flag) {
        std::tie(cost_before_subdiv, code_length, mvd, selected_index, method_flag) = getMVD(
                {gauss_result_translation, gauss_result_translation, gauss_result_translation}, error_translation,
                triangle_index, cmt->mv1, diagonal_line_area_flag, ctu, true, dummy);
    }else{
        std::tie(cost_before_subdiv, code_length, mvd, selected_index, method_flag) = getMVD(
                triangle_gauss_results[triangle_index].mv_warping, error_warping,
                triangle_index, cmt->mv1, diagonal_line_area_flag, ctu, false, dummy);
    }

    std::vector<cv::Point2i> ret_gauss2;

    if(method_flag == MV_CODE_METHOD::MERGE || method_flag == MV_CODE_METHOD::MERGE2) {
        if(triangle_gauss_results[triangle_index].translation_flag) {
            ctu->original_mv1 = triangle_gauss_results[triangle_index].original_mv_translation;
            ctu->original_mv2 = triangle_gauss_results[triangle_index].original_mv_translation;
            ctu->original_mv3 = triangle_gauss_results[triangle_index].original_mv_translation;
        }else{
            ctu->original_mv1 = triangle_gauss_results[triangle_index].original_mv_warping[0];
            ctu->original_mv2 = triangle_gauss_results[triangle_index].original_mv_warping[1];
            ctu->original_mv3 = triangle_gauss_results[triangle_index].original_mv_warping[2];
        }

        triangle_gauss_results[triangle_index].mv_translation = mvd[0];
        triangle_gauss_results[triangle_index].mv_warping = mvd;
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
    ctu->triangle_index = triangle_index;
    ctu->code_length = code_length;
    ctu->collocated_mv = cmt->mv1;
    ctu->translation_flag = translation_flag;
    ctu->method = method_flag;
    ctu->ref_triangle_idx = selected_index;

    if(method_flag == SPATIAL) {
        ctu->mvds.clear();
        if(ctu->translation_flag) {
            ctu->mvds.emplace_back(mvd[0]);
            ctu->mvds.emplace_back(mvd[0]);
            ctu->mvds.emplace_back(mvd[0]);
        }else{
            ctu->mvds.emplace_back(mvd[0]);
            ctu->mvds.emplace_back(mvd[1]);
            ctu->mvds.emplace_back(mvd[2]);
        }
    }

    if(steps <= 0){
        isCodedTriangle[triangle_index] = true;

        if(ctu->method != MV_CODE_METHOD::INTRA && ctu->method != MV_CODE_METHOD::INTRA) {
            std::vector<cv::Point2f> mvs;
            if(ctu->translation_flag){
                mvs.emplace_back(ctu->mv1);
                mvs.emplace_back(ctu->mv1);
                mvs.emplace_back(ctu->mv1);
            }else{
                mvs.emplace_back(ctu->mv1);
                mvs.emplace_back(ctu->mv2);
                mvs.emplace_back(ctu->mv3);
            }
            getPredictedImage(expansion_ref_uchar, target_image, intra_tmp_image, triangle, mvs, SEARCH_RANGE, diagonal_line_area_flag, ctu->triangle_index, ctu, cv::Rect(0, 0, block_size_x, block_size_y), ref_hevc);
        }

        std::vector<cv::Point2f> pixels = getPixelsInTriangle(triangle, diagonal_line_area_flag, triangle_index, ctu, block_size_x, block_size_y);
        for(const auto& p : pixels) intra_flag[p.x][p.y] = true;

        ctu->node1 = ctu->node2 = ctu->node3 = ctu->node4 = nullptr;

        return false;
    }

    SplitResult split_triangles = getSplitTriangle(p1, p2, p3, type);

    SplitResult split_sub_triangles1 = getSplitTriangle(split_triangles.t1.p1, split_triangles.t1.p2, split_triangles.t1.p3, split_triangles.t1_type);
    SplitResult split_sub_triangles2 = getSplitTriangle(split_triangles.t2.p1, split_triangles.t2.p2, split_triangles.t2.p3, split_triangles.t2_type);

    std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
    subdiv_ref_triangles.emplace_back(split_sub_triangles1.t1);
    subdiv_ref_triangles.emplace_back(split_sub_triangles1.t2);
    subdiv_ref_triangles.emplace_back(split_sub_triangles2.t1);
    subdiv_ref_triangles.emplace_back(split_sub_triangles2.t2);

    subdiv_target_triangles.emplace_back(split_sub_triangles1.t1);
    subdiv_target_triangles.emplace_back(split_sub_triangles1.t2);
    subdiv_target_triangles.emplace_back(split_sub_triangles2.t1);
    subdiv_target_triangles.emplace_back(split_sub_triangles2.t2);

    double RMSE_after_subdiv = 0.0;
    std::vector<GaussResult> split_mv_result(subdiv_target_triangles.size());

//    int p1_idx = getCornerIndex(p1);
//    int p2_idx = getCornerIndex(p2);
//    int p3_idx = getCornerIndex(p3);
//    addCornerAndTriangle(Triangle(p1_idx, p2_idx, p3_idx), triangle_index, type);

    int t1_p1_idx = getCornerIndex(split_triangles.t1.p1);
    int t1_p2_idx = getCornerIndex(split_triangles.t1.p2);
    int t1_p3_idx = getCornerIndex(split_triangles.t1.p3);
    addCornerAndTriangle(Triangle(t1_p1_idx, t1_p2_idx, t1_p3_idx), triangle_index, split_triangles.t1_type);

    int t2_p1_idx = getCornerIndex(split_triangles.t2.p1);
    int t2_p2_idx = getCornerIndex(split_triangles.t2.p2);
    int t2_p3_idx = getCornerIndex(split_triangles.t2.p3);
    addCornerAndTriangle(Triangle(t2_p1_idx, t2_p2_idx, t2_p3_idx), triangle_index, split_triangles.t2_type);

    int triangle_indexes[] = {(int)triangles.size() - 4, (int)triangles.size() - 3, (int)triangles.size() - 2, (int)triangles.size() - 1};

    std::vector<std::vector<int>> prev_area_flag(diagonal_line_area_flag);

    // 分割回数が偶数回目のとき斜線の更新を行う
    int sx = ceil( std::min({triangle.p1.x, triangle.p2.x, triangle.p3.x}));
    int lx = floor(std::max({triangle.p1.x, triangle.p2.x, triangle.p3.x}));
    int sy = ceil( std::min({triangle.p1.y, triangle.p2.y, triangle.p3.y}));
    int ly = floor(std::max({triangle.p1.y, triangle.p2.y, triangle.p3.y}));

    int width =  (lx - sx) / 2 + 1;
    int height = (ly - sy) / 2 + 1;

    bool flag = true;
    int a, b, c, d;
    if(type == TYPE1) {
        for (int x = 0 ; x < width  ; x++) {
            diagonal_line_area_flag[(x + sx) % block_size_x][(x + sy) % block_size_y] = (x % 2 == 0 ? triangle_indexes[0] : triangle_indexes[2]);
            flag = !flag;
        }
    }else if(type == TYPE2) {
        for (int x = 0 ; x < width ; x++) {
            diagonal_line_area_flag[(sx + width + x) % block_size_x][(sy + height + x) % block_size_y] = (x % 2 == 0 ? triangle_indexes[1] : triangle_indexes[3]);
            flag = !flag;
        }

    }else if(type == TYPE3){
        for(int x = 0 ; x < width ; x++){
            diagonal_line_area_flag[(sx + width + x) % block_size_x][(sy + height - x - 1) % block_size_y] = (x % 2 == 0 ? triangle_indexes[1] : triangle_indexes[2]);
            flag = !flag;
        }
    }else if(type == TYPE4){
        for(int x = 0 ; x < width ; x++){
            diagonal_line_area_flag[(x + sx) % block_size_x][(ly - x) % block_size_y] = (x % 2 == 0 ? triangle_indexes[1] : triangle_indexes[2]);
            flag = !flag;
        }
    }

    ctu->node1 = new CodingTreeUnit();
    ctu->node1->triangle_index = triangles.size() - 4;
    ctu->node1->parentNode = ctu;
    ctu->node2 = new CodingTreeUnit();
    ctu->node2->triangle_index = triangles.size() - 3;
    ctu->node2->parentNode = ctu;
    ctu->node3 = new CodingTreeUnit();
    ctu->node3->triangle_index = triangles.size() - 2;
    ctu->node3->parentNode = ctu;
    ctu->node4 = new CodingTreeUnit();
    ctu->node4->triangle_index = triangles.size() - 1;
    ctu->node4->parentNode = ctu;

    std::vector<CodingTreeUnit*> ctus{ctu->node1, ctu->node2, ctu->node3, ctu->node4};
#if !MVD_DEBUG_LOG
//    #pragma omp parallel for
#endif
    for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
        double error_warping_tmp, error_translation_tmp;
        int triangle_size_tmp;
        cv::Point2f mv_translation_tmp;
        std::vector<cv::Point2f> mv_warping_tmp;
        std::vector<cv::Point2f> tmp_bm_mv;
        std::vector<double> tmp_bm_errors;
        double cost_warping_tmp, cost_translation_tmp;
        double tmp_error_newton;
        MV_CODE_METHOD method_warping_tmp, method_translation_tmp;
        if(PRED_MODE == NEWTON){
            if(GAUSS_NEWTON_INIT_VECTOR) {
                std::tie(tmp_bm_mv, tmp_bm_errors) = fullpellBlockMatching(subdiv_target_triangles[j], target_image,
                                                                   expansion_ref, diagonal_line_area_flag,
                                                                   triangle_indexes[j], ctus[j]);
                std::tie(mv_warping_tmp, mv_translation_tmp, error_warping_tmp, error_translation_tmp,triangle_size_tmp) = GaussNewton(
                        ref_images, target_images, expand_images, subdiv_target_triangles[j], diagonal_line_area_flag,
                        triangle_indexes[j], ctus[j], block_size_x, block_size_y,
                        tmp_bm_mv[2], ref_hevc);

#if USE_BM_TRANSLATION_MV
                error_translation_tmp = tmp_bm_errors[2];
                mv_translation_tmp = tmp_bm_mv[2];
#endif
            }else{
                std::tie(mv_warping_tmp, mv_translation_tmp, error_warping_tmp, error_translation_tmp, triangle_size_tmp) = GaussNewton(
                        ref_images, target_images, expand_images, subdiv_target_triangles[j], diagonal_line_area_flag,
                        triangle_indexes[j], ctus[j], block_size_x, block_size_y,
                        cv::Point2f(-1000, -1000), ref_hevc);
            }

            std::vector<cv::Point2f> mvd_translation, mvd_warping;
            // TODO: cmt直す
            std::tie(cost_translation_tmp,std::ignore, mvd_translation, std::ignore, method_translation_tmp) = getMVD(
                    {mv_translation_tmp, mv_translation_tmp, mv_translation_tmp}, error_translation_tmp,
                    triangle_indexes[j], cmt->mv1, diagonal_line_area_flag, ctus[j], true, dummy);
#if !GAUSS_NEWTON_TRANSLATION_ONLY

            std::tie(cost_warping_tmp, std::ignore, mvd_warping, std::ignore, method_warping_tmp) = getMVD(
                    mv_warping_tmp, error_warping_tmp,
                    triangle_indexes[j], cmt->mv1, diagonal_line_area_flag, ctus[j], false, dummy);
#endif
            if(cost_translation_tmp < cost_warping_tmp || (steps - 2 <= warping_limit) || GAUSS_NEWTON_TRANSLATION_ONLY){
                triangle_gauss_results[triangle_indexes[j]].translation_flag = true;
                triangle_gauss_results[triangle_indexes[j]].mv_translation = mv_translation_tmp;
                triangle_gauss_results[triangle_indexes[j]].original_mv_translation = mv_translation_tmp;
                split_mv_result[j] = GaussResult(mv_warping_tmp, mv_translation_tmp, error_translation_tmp, triangle_size_tmp, true, error_translation_tmp, error_warping_tmp);

                if(method_translation_tmp == MV_CODE_METHOD::MERGE || method_translation_tmp == MV_CODE_METHOD::MERGE2){
                    triangle_gauss_results[triangle_indexes[j]].mv_translation = mvd[0];
                }
            }else{
                 triangle_gauss_results[triangle_indexes[j]].translation_flag = false;
                triangle_gauss_results[triangle_indexes[j]].mv_warping = mv_warping_tmp;
                triangle_gauss_results[triangle_indexes[j]].original_mv_warping = mv_warping_tmp;
                split_mv_result[j] = GaussResult(mv_warping_tmp, mv_translation_tmp, error_warping_tmp, triangle_size_tmp, false, error_translation_tmp, error_warping_tmp);

                if(method_warping_tmp == MV_CODE_METHOD::MERGE || method_warping_tmp == MV_CODE_METHOD::MERGE2){
                    triangle_gauss_results[triangle_indexes[j]].mv_warping = mvd;
                }
            }

        }else if(PRED_MODE == BM){
            std::tie(tmp_bm_mv, tmp_bm_errors) = blockMatching(subdiv_target_triangles[j], target_image, expansion_ref, diagonal_line_area_flag, triangle_indexes[j], ctus[j]);
            mv_warping_tmp = tmp_bm_mv;
            mv_translation_tmp = tmp_bm_mv[2];
            error_translation_tmp = tmp_bm_errors[2];
            triangle_size_tmp = (double)1e6;

            split_mv_result[j] = GaussResult(mv_warping_tmp, mv_translation_tmp, error_translation_tmp, triangle_size_tmp, true, tmp_bm_errors[2], tmp_error_newton);

            triangle_gauss_results[triangle_indexes[j]].translation_flag = true;
            triangle_gauss_results[triangle_indexes[j]].original_mv_translation = mv_translation_tmp;
        }

        isCodedTriangle[triangle_indexes[j]] = true;
    }

    for(int i = 0 ; i < 4 ; i++){
        isCodedTriangle[triangle_indexes[i]] = false;
    }

    double cost_after_subdiv1;
    int code_length1;
    CollocatedMvTree *cmt_left_left, *cmt_left_right, *cmt_right_left, *cmt_right_right;

    cmt_left_left    = (cmt->node1 == nullptr ? cmt : cmt->node1);
    cmt_left_right   = (cmt->node2 == nullptr ? cmt : cmt->node2);
    cmt_right_left   = (cmt->node3 == nullptr ? cmt : cmt->node3);
    cmt_right_right  = (cmt->node4 == nullptr ? cmt : cmt->node4);

    MV_CODE_METHOD method_flag1, method_flag2, method_flag3, method_flag4;
    if(split_mv_result[0].translation_flag) {
        std::tie(cost_after_subdiv1, code_length1, mvd, selected_index, method_flag1) = getMVD(
                {split_mv_result[0].mv_translation, split_mv_result[0].mv_translation, split_mv_result[0].mv_translation},
                split_mv_result[0].residual,
                triangle_indexes[0], cmt_left_left->mv1, diagonal_line_area_flag, ctu->node1, true, dummy);

        if(method_flag1 == MV_CODE_METHOD::MERGE || method_flag1 == MV_CODE_METHOD::MERGE2) {
            if(split_mv_result[0].translation_flag) {
                gauss_result_translation = mvd[0];
                triangle_gauss_results[triangle_indexes[0]].mv_translation = gauss_result_translation;
            }else{
                triangle_gauss_results[triangle_indexes[0]].mv_warping = mvd;
            }
        }
    }else{
        std::tie(cost_after_subdiv1, code_length1, mvd, selected_index, method_flag1) = getMVD(
                split_mv_result[0].mv_warping, split_mv_result[0].residual,
                triangle_indexes[0], cmt_left_left->mv1, diagonal_line_area_flag, ctu->node1, false, dummy);
    }
    isCodedTriangle[triangle_indexes[0]] = true;

    double cost_after_subdiv2;
    int code_length2;
    if(split_mv_result[1].translation_flag){
        std::tie(cost_after_subdiv2, code_length2, mvd, selected_index, method_flag2) = getMVD(
                {split_mv_result[1].mv_translation, split_mv_result[1].mv_translation, split_mv_result[1].mv_translation}, split_mv_result[1].residual,
                triangle_indexes[1], cmt_left_right->mv1, diagonal_line_area_flag, ctu->node2, true, dummy);

        if(method_flag2 == MV_CODE_METHOD::MERGE || method_flag2 == MV_CODE_METHOD::MERGE2) {
            if(split_mv_result[1].translation_flag) {
                gauss_result_translation = mvd[0];
                triangle_gauss_results[triangle_indexes[1]].mv_translation = gauss_result_translation;
            }else{
                triangle_gauss_results[triangle_indexes[1]].mv_warping = mvd;
            }
        }
    }else{
        std::tie(cost_after_subdiv2, code_length2, mvd, selected_index, method_flag2) = getMVD(
                split_mv_result[1].mv_warping, split_mv_result[1].residual,
                triangle_indexes[1], cmt_left_right->mv1, diagonal_line_area_flag, ctu->node2, false, dummy);
    }
    isCodedTriangle[triangle_indexes[1]] = true;

    double cost_after_subdiv3;
    int code_length3;
    if(split_mv_result[2].translation_flag) {
        std::tie(cost_after_subdiv3, code_length3, mvd, selected_index, method_flag3) = getMVD(
                {split_mv_result[2].mv_translation, split_mv_result[2].mv_translation, split_mv_result[2].mv_translation},
                split_mv_result[2].residual,
                triangle_indexes[2], cmt_right_left->mv1, diagonal_line_area_flag, ctu->node3, true, dummy);

        if(method_flag3 == MV_CODE_METHOD::MERGE || method_flag3 == MV_CODE_METHOD::MERGE2) {
            if(split_mv_result[2].translation_flag) {
                gauss_result_translation = mvd[0];
                triangle_gauss_results[triangle_indexes[2]].mv_translation = gauss_result_translation;
            }else{
                triangle_gauss_results[triangle_indexes[2]].mv_warping = mvd;
            }
        }
    }else{
        std::tie(cost_after_subdiv3, code_length3, mvd, selected_index, method_flag3) = getMVD(
                split_mv_result[2].mv_warping, split_mv_result[2].residual,
                triangle_indexes[2], cmt_right_left->mv1, diagonal_line_area_flag, ctu->node3, false, dummy);
    }
    isCodedTriangle[triangle_indexes[2]] = true;

    double cost_after_subdiv4;
    int code_length4;
    if(split_mv_result[3].translation_flag){
        std::tie(cost_after_subdiv4, code_length4, mvd, selected_index, method_flag4) = getMVD(
                {split_mv_result[3].mv_translation, split_mv_result[3].mv_translation, split_mv_result[3].mv_translation}, split_mv_result[3].residual,
                triangle_indexes[3], cmt_right_right->mv1, diagonal_line_area_flag, ctu->node4, true, dummy);

        if(method_flag4 == MV_CODE_METHOD::MERGE || method_flag4 == MV_CODE_METHOD::MERGE2) {
            if(split_mv_result[3].translation_flag) {
                gauss_result_translation = mvd[0];
                triangle_gauss_results[triangle_indexes[3]].mv_translation = gauss_result_translation;
            }else{
                triangle_gauss_results[triangle_indexes[3]].mv_warping = mvd;
            }
        }
    }else{
        std::tie(cost_after_subdiv4, code_length4, mvd, selected_index, method_flag4) = getMVD(
                split_mv_result[3].mv_warping, split_mv_result[3].residual,
                triangle_indexes[3], cmt_right_right->mv1, diagonal_line_area_flag, ctu->node4, false, dummy);
    }
    isCodedTriangle[triangle_indexes[3]] = true;

    double alpha = 1;

#if DISPLAY_RD_COST
    std::cout << "before:" << cost_before_subdiv << " after:" << alpha * (cost_after_subdiv1 + cost_after_subdiv2 + cost_after_subdiv3 + cost_after_subdiv4) << std::endl;
#endif

    if(cost_before_subdiv >= alpha * (cost_after_subdiv1 + cost_after_subdiv2 + cost_after_subdiv3 + cost_after_subdiv4)) {

        for(int i = 0 ; i < 4 ; i++){
            isCodedTriangle[triangle_indexes[i]] = false;
        }
        ctu->split_cu_flag = true;

        int t1_idx = triangles.size() - 4;
        int t2_idx = triangles.size() - 3;
        int t3_idx = triangles.size() - 2;
        int t4_idx = triangles.size() - 1;

        // 1つ目の頂点追加
        ctu->node1->triangle_index = t1_idx;
        ctu->node1->code_length = code_length1;
        ctu->node1->translation_flag = split_mv_result[0].translation_flag;
        ctu->node1->method = method_flag1;
        triangle_gauss_results[t1_idx] = split_mv_result[0];

        int next_step = steps - 2;
        if(ctu->node1->method == MV_CODE_METHOD::INTRA && INTRA_LIMIT_MODE){
            next_step = 0;
        }
        bool result = split(expand_images, ctu->node1, cmt_left_left, split_sub_triangles1.t1, t1_idx,split_sub_triangles1.t1_type, next_step, diagonal_line_area_flag);

        // 2つ目の三角形
        ctu->node2->triangle_index = t2_idx;
        ctu->node2->code_length = code_length2;
        ctu->node2->translation_flag = split_mv_result[1].translation_flag;
        ctu->node2->method = method_flag2;

        triangle_gauss_results[t2_idx] = split_mv_result[1];
        next_step = steps - 2;
        if(ctu->node2->method == MV_CODE_METHOD::INTRA && INTRA_LIMIT_MODE){
            next_step = 0;
        }
        result = split(expand_images, ctu->node2, cmt_left_right, split_sub_triangles1.t2, t2_idx, split_sub_triangles1.t2_type, next_step, diagonal_line_area_flag);

        // 3つ目の三角形
        ctu->node3->triangle_index = t3_idx;
        ctu->node3->code_length = code_length3;
        ctu->node3->translation_flag = split_mv_result[2].translation_flag;
        ctu->node3->method = method_flag3;
        triangle_gauss_results[t3_idx] = split_mv_result[2];
        next_step = steps - 2;
        if(ctu->node3->method == MV_CODE_METHOD::INTRA && INTRA_LIMIT_MODE){
            next_step = 0;
        }
        result = split(expand_images, ctu->node3, cmt_right_left, split_sub_triangles2.t1, t3_idx, split_sub_triangles2.t1_type, next_step, diagonal_line_area_flag);

        // 4つ目の三角形
        ctu->node4->triangle_index = t4_idx;
        ctu->node4->code_length = code_length4;
        ctu->node4->translation_flag = split_mv_result[3].translation_flag;
        ctu->node4->method = method_flag4;
        triangle_gauss_results[t4_idx] = split_mv_result[3];
        next_step = steps - 2;
        if(ctu->node4->method == MV_CODE_METHOD::INTRA && INTRA_LIMIT_MODE){
            next_step = 0;
        }
        result = split(expand_images, ctu->node4, cmt_right_right, split_sub_triangles2.t2, t4_idx, split_sub_triangles2.t2_type, next_step, diagonal_line_area_flag);

        return true;
    }else{
        isCodedTriangle[triangle_index] = true;
        delete_flag[triangle_index] = false;
        for(int i = 0 ; i < 4 ; i++) isCodedTriangle[triangle_indexes[i]] = false;
        ctu->node1 = ctu->node2 = ctu->node3 = ctu->node4 = nullptr;
        ctu->method = method_flag;
        diagonal_line_area_flag = prev_area_flag;
        eraseTriangle(triangles.size() - 1);
        eraseTriangle(triangles.size() - 1);
        eraseTriangle(triangles.size() - 1);
        eraseTriangle(triangles.size() - 1);
        addNeighborVertex(triangles[triangle_index].first.p1_idx,triangles[triangle_index].first.p2_idx,triangles[triangle_index].first.p3_idx);
        addCoveredTriangle(triangles[triangle_index].first.p1_idx,triangles[triangle_index].first.p2_idx,triangles[triangle_index].first.p3_idx, triangle_index);

        if(method_flag != MV_CODE_METHOD::INTRA) {
            std::vector<cv::Point2f> mvs;
            if(ctu->translation_flag){
                mvs.emplace_back(ctu->mv1);
                mvs.emplace_back(ctu->mv1);
                mvs.emplace_back(ctu->mv1);
            }else{
                mvs.emplace_back(ctu->mv1);
                mvs.emplace_back(ctu->mv2);
                mvs.emplace_back(ctu->mv3);
            }
            getPredictedImage(expansion_ref_uchar, target_image, intra_tmp_image, triangle, mvs, SEARCH_RANGE, diagonal_line_area_flag, ctu->triangle_index, ctu, cv::Rect(0, 0, block_size_x, block_size_y), ref_hevc);
        }
        std::vector<cv::Point2f> pixels = getPixelsInTriangle(triangle, diagonal_line_area_flag, triangle_index, ctu, block_size_x, block_size_y);
        for(const auto& p : pixels) intra_flag[p.x][p.y] = true;

//        std::cout << (ctu->method == MERGE ? "MERGE" : "SPATIAL") << " " << (ctu->parallel_flag ? "PARALLEL" : "WARPING") << " "  << ctu->mv1 << " " << ctu->mv2 << " " << ctu->mv3 << std::endl;
        return false;
    }

}

void TriangleDivision::storeIntraImage(){

    cv::imwrite(getProjectDirectory(OS) + "/intra.png", intra_tmp_image);

}

/**
 * @fn TriangleDivision::SplitResult TriangleDivision::getSplitTriangle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, int type)
 * @details ３点の座標とtypeを受け取り，分割した形状を返す
 * @param p1 頂点１の座標
 * @param p2 頂点２の座標
 * @param p3 頂点３の座標
 * @param type 分割形状
 * @return 分割結果
 */
TriangleDivision::SplitResult TriangleDivision::getSplitTriangle(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, int type){
    cv::Point2f a, b, c, d;

    switch(type) {
        case DIVIDE::TYPE1:
        {
            cv::Point2f x = (p2 - p1) / 2;
            cv::Point2f y = (p3 - p1) / 2;

            a = p1;
            b = p2;
            c = a + x + y;
            d = p3;

            return {Point3Vec(a, b, c), Point3Vec(a, c, d), TYPE5, TYPE6};
        }
        case DIVIDE::TYPE2:
        {
            cv::Point2f x = (p2 - p3) / 2.0;
            cv::Point2f y = (p1 - p3) / 2.0;
            a = p1;
            b = p3 + x + y;
            c = p2;
            d = p3;

            return {Point3Vec(a, b, d), Point3Vec(b, c, d), TYPE8, TYPE7};
        }
        case DIVIDE::TYPE3:
        {
            cv::Point2f x = (p1 - p2) / 2.0;
            cv::Point2f y = (p3 - p2) / 2.0;

            a = p1;
            b = p2;
            c = p2 + x + y;
            d = p3;

            return {Point3Vec(a, b, c), Point3Vec(b, c, d), TYPE5, TYPE8};
        }
        case DIVIDE::TYPE4:
        {
            cv::Point2f x = (p3 - p2) / 2.0;
            cv::Point2f y = (p1 - p2) / 2.0;

            a = p1;
            b = p2 + x + y;
            c = p2;
            d = p3;

            return {Point3Vec(a, b, c), Point3Vec(b, c, d), TYPE6, TYPE7};
        }
        case DIVIDE::TYPE5:
        {
            cv::Point2f x = (p2 - p1) / 2.0;
            x.x = (int)x.x;

            a = p1;

            cv::Point2f b1 = p1 + x;
            cv::Point2f b2 = p1 + x;
            b2.x += 1;
            b1.y = (int)b1.y;
            b2.y = (int)b2.y;

            c = p2;

            cv::Point2f d1 = p3;
            cv::Point2f d2 = p3;
            d1.x = (int)d1.x;
            d1.y = (int)d1.y;
            d2.x = ceil(d2.x);
            d2.y = (int)(d2.y);

            return {Point3Vec(a, b1, d1), Point3Vec(b2, c, d2), TYPE3, TYPE1};
        }
        case DIVIDE::TYPE6:
        {
            cv::Point2f y = (p3 - p1) / 2.0;
            y.y = (int)y.y;

            a = p1;
            cv::Point2f b1 = p1 + y;
            cv::Point2f b2 = p1 + y;
            b2.y += 1;

            cv::Point2f c1 = p2;
            cv::Point2f c2 = p2;
            c1.x = (int)c1.x;
            c1.y = (int)c1.y;
            c2.x = (int)(c2.x);
            c2.y = ceil(c2.y);

            d = p3;

            return {Point3Vec(a, b1, c1), Point3Vec(b2, c2, d), TYPE4, TYPE1};
        }
        case DIVIDE::TYPE7:
        {
            cv::Point2f x = (p3 - p2) / 2.0;
            x.x = (int)x.x;

            cv::Point2f a1 = p1;
            cv::Point2f a2 = p1;
            a1.x = (int)a1.x;
            a1.y = ceil(a1.y);
            a2.x = ceil(a2.x);
            a2.y = ceil(a2.y);

            b = p2;

            cv::Point2f c1 = p2 + x;
            cv::Point2f c2 = p2 + x;
            c2.x += 1;

            d = p3;

            return {Point3Vec(a1, b, c1), Point3Vec(a2, c2, d), TYPE2, TYPE4};
        }
        case DIVIDE::TYPE8:
        {
            cv::Point2f y = (p3 - p1) / 2.0;

            cv::Point2f a1 = p2;
            cv::Point2f a2 = p2;
            a1.x = ceil(a1.x);
            a1.y = (int)a1.y;
            a2.x = ceil(a2.x);
            a2.y = ceil(a2.y);

            b = p1;

            cv::Point2f c1 = p1 + y;
            cv::Point2f c2 = p1 + y;
            c1.y = (int)c1.y;
            c2.y = ceil(c2.y);

            d = p3;

            return {Point3Vec(b, a1, c1), Point3Vec(a2, c2, d), TYPE2, TYPE3};
        }
        default:
            break;
    }
}

/**
 * @fn std::vector<int> getSpatialMvList()
 * @brief t_idx番目の三角形の空間予測動きベクトル候補を返す
 * @param[in] t_idx 三角パッチのインデックス
 * @return 候補のパッチの番号を返す
 */
std::vector<int> TriangleDivision::getSpatialTriangleList(int t_idx){
    std::pair<Triangle, int> triangle = triangles[t_idx];
    std::set<int> spatialTriangles;
    std::vector<int> list1 = getIdxCoveredTriangleIndexList(triangle.first.p1_idx);
    std::vector<int> list2 = getIdxCoveredTriangleIndexList(triangle.first.p2_idx);
    std::vector<int> list3 = getIdxCoveredTriangleIndexList(triangle.first.p3_idx);

    std::set<int> mutualIndexSet1, mutualIndexSet2, mutualIndexSet3;

#if MVD_DEBUG_LOG
    std::cout << "p1:" << triangles[t_idx].first.p1_idx << std::endl;
    for(auto item : list1){
        std::cout << item << std::endl;
    }
    puts("");

    std::cout << "p2:" << triangles[t_idx].first.p2_idx << std::endl;
    for(auto item : list2){
        std::cout << item << std::endl;
    }
    puts("");
    std::cout << "p3:" << triangles[t_idx].first.p3_idx << std::endl;

    for(auto item : list3){
        std::cout << item << std::endl;
    }
    std::cout << "t_idx:" << t_idx << std::endl;
    puts("");

#endif

    for(auto idx : list1) if(isCodedTriangle[idx] && idx != t_idx) mutualIndexSet1.emplace(idx);
    for(auto idx : list2) if(isCodedTriangle[idx] && idx != t_idx) mutualIndexSet2.emplace(idx);
    for(auto idx : list3) if(isCodedTriangle[idx] && idx != t_idx) mutualIndexSet3.emplace(idx);

    for(auto idx : mutualIndexSet1) spatialTriangles.emplace(idx);
    for(auto idx : mutualIndexSet2) spatialTriangles.emplace(idx);
    for(auto idx : mutualIndexSet3) spatialTriangles.emplace(idx);

    std::vector<int> ret;

    for(auto idx : spatialTriangles){
        ret.emplace_back(idx);
    }

    return ret;
}

/**
 * @fn void TriangleDivision::constructPreviousCodingTree(std::vector<CodingTreeUnit*> trees, int pic_num)
 * @brief 過去の動きベクトルを参照するためのTreeを構築する
 * @param trees 分割形状
 * @param pic_num 何枚目のPピクチャか
 */
void TriangleDivision::constructPreviousCodingTree(std::vector<CodingTreeUnit*> trees, int pic_num) {

    for(int i = 0 ; i < triangles.size() ; i++) {
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
 * @fn void TriangleDivision::constructPreviousCodingTree(std::vector<CollocatedMvTree*> trees)
 * @brief 木を再帰的に呼び出し構築する
 * @param codingTree 分割結果を表す木
 * @param constructedTree 構築するための木
 */
void TriangleDivision::constructPreviousCodingTree(CodingTreeUnit* codingTree, CollocatedMvTree* constructedTree) {
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
bool TriangleDivision::isMvExists(const std::vector<std::pair<cv::Point2f, MV_CODE_METHOD>> &vectors, const cv::Point2f &mv){
    for(auto vector : vectors) {
        if(vector.first == mv) {
            return true;
        }
    }
    return false;
}

/**
 * @fn std::tuple<cv::Point2f, int, MV_CODE_METHOD> RD(int triangle_idx, CodingTreeUnit* ctu)
 * @brief RDを行い，最適な差分ベクトルを返す
 * @param[in] mv 動きベクトル
 * @param[in] triangle_idx 三角パッチの番号
 * @param[in] residual そのパッチの残差
 * @param[in] ctu CodingTreeUnit 符号木
 * @param[in] area_flag 含まれる画素のフラグ
 * @return 差分ベクトル，参照したパッチ，空間or時間のフラグのtuple
 */
std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD> TriangleDivision::getMVD(std::vector<cv::Point2f> mv, double residual, int triangle_idx, cv::Point2f &collocated_mv, const std::vector<std::vector<int>> &area_flag, CodingTreeUnit* ctu, bool translation_flag, std::vector<cv::Point2f> &pixels){
    // 空間予測と時間予測の候補を取り出す
    std::vector<int> spatial_triangles = getSpatialTriangleList(triangle_idx);
    int spatial_triangle_size = static_cast<int>(spatial_triangles.size());
    std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> vectors; // ベクトルとモードを表すフラグのペア
    std::vector<std::vector<cv::Point2f>> warping_vectors;

    // すべてのベクトルを格納する．
    for(int i = 0 ; i < spatial_triangle_size ; i++) {
        int spatial_triangle_index = spatial_triangles[i];
        GaussResult spatial_triangle = triangle_gauss_results[spatial_triangle_index];

        if(spatial_triangle.translation_flag){
            if(!isMvExists(vectors, spatial_triangle.mv_translation) && vectors.size() <= MV_LIST_MAX_NUM) {
                vectors.emplace_back(spatial_triangle.mv_translation, SPATIAL);
                warping_vectors.emplace_back();
            }
        }else{
            // 隣接パッチがワーピングで予想されている場合、そのパッチの0番の動きベクトルを候補とする
            cv::Point2f p1 = spatial_triangle.mv_warping[0];
            cv::Point2f p2 = spatial_triangle.mv_warping[1];
            cv::Point2f p3 = spatial_triangle.mv_warping[2];
#if MVD_DEBUG_LOG
            std::cout << "target_triangle_coordinate:";
            std::cout << corners[triangles[triangle_idx].first.p1_idx] << " ";
            std::cout << corners[triangles[triangle_idx].first.p2_idx] << " ";
            std::cout << corners[triangles[triangle_idx].first.p3_idx] << std::endl;
            std::cout << "ref_triangle_coordinate:";
            std::cout << corners[triangles[spatial_triangle_index].first.p1_idx] << " ";
            std::cout << corners[triangles[spatial_triangle_index].first.p2_idx] << " ";
            std::cout << corners[triangles[spatial_triangle_index].first.p3_idx] << std::endl;
            std::cout << "ref_triangle_mvs:";
            std::cout << p1 << " " << p2 << " " << p3 << std::endl;
#endif
            cv::Point2f mv_average;
            std::vector<cv::Point2f> ref_mvs{p1, p2, p3};
            std::pair<Triangle, int> target_triangle = triangles[triangle_idx];
            cv::Point2f pp1 = corners[target_triangle.first.p1_idx], pp2 = corners[target_triangle.first.p2_idx], pp3 = corners[target_triangle.first.p3_idx];
            std::pair<Triangle, int> ref_triangle = triangles[spatial_triangle_index];
            std::vector<cv::Point2f> ref_triangle_coordinates{corners[ref_triangle.first.p1_idx], corners[ref_triangle.first.p2_idx], corners[ref_triangle.first.p3_idx]};
            std::vector<cv::Point2f> target_triangle_coordinates{cv::Point2f((pp1.x + pp2.x + pp3.x) / 3.0, (pp1.y + pp2.y + pp3.y) / 3.0)};
            std::vector<cv::Point2f> mvs = getPredictedWarpingMv(ref_triangle_coordinates, ref_mvs, target_triangle_coordinates);
            mv_average = mvs[0];

            if (!translation_flag) {
                target_triangle_coordinates.clear();
                target_triangle_coordinates.emplace_back(pp1);
                target_triangle_coordinates.emplace_back(pp2);
                target_triangle_coordinates.emplace_back(pp3);
                mvs = getPredictedWarpingMv(ref_triangle_coordinates, ref_mvs, target_triangle_coordinates);
                std::vector<cv::Point2f> v{mvs[0], mvs[1], mvs[2]};
                warping_vectors.emplace_back(v);

            }else{
                warping_vectors.emplace_back();
            }

            mv_average = roundVecQuarter(mv_average);
            if(!isMvExists(vectors, mv_average) && vectors.size() <= MV_LIST_MAX_NUM){
                vectors.emplace_back(mv_average, SPATIAL);
            }
        }
    }

#if MVD_DEBUG_LOG
    std::cout << corners[triangles[triangle_idx].first.p1_idx] << " " << corners[triangles[triangle_idx].first.p2_idx] << " " << corners[triangles[triangle_idx].first.p3_idx] << std::endl;
    #endif

    if(!isMvExists(vectors, collocated_mv)) {
        vectors.emplace_back(collocated_mv, Collocated);
        warping_vectors.emplace_back();
    }

    if(vectors.size() < 2) {
        vectors.emplace_back(cv::Point2f(0.0, 0.0), SPATIAL);
        warping_vectors.emplace_back();
    }

    double lambda = getLambdaPred(qp, (translation_flag ? 1.0 : 1.0));

    int flags_code = 1;
    if (INTRA_MODE) flags_code++;
    if (MERGE_MODE) flags_code++;

    //                      コスト, 差分ベクトル, 番号, タイプ
    std::vector<std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum, Flags> > results;
    for(int i = 0 ; i < vectors.size() ; i++) {
        std::pair<cv::Point2f, MV_CODE_METHOD> vector = vectors[i];
        cv::Point2f current_mv = vector.first;
        // TODO: ワーピング対応
        if(translation_flag) { // 平行移動成分に関してはこれまで通りにやる
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
                    mvd_code_length += getExponentialGolombCodeLength((int) mvd_y_minus_2, 0);
                    flag_code_sum.addMvdCodeLength(getExponentialGolombCodeLength((int) mvd_y_minus_2, 0));
                }

                flag_code_sum.countGreater1Code();
                flag_code_sum.setYGreater1Flag(is_y_greater_than_one);
                flag_code_sum.countSignFlagCode();
            }

            // 参照箇所符号化
            int reference_index = i;
            int reference_index_code_length = getUnaryCodeLength(reference_index);

            // 各種フラグ分を(3*2)bit足してます
            double rd = residual + lambda * (mvd_code_length + reference_index_code_length + flags_code);

            std::vector<cv::Point2f> mvds{mvd};
            // 結果に入れる
            results.emplace_back(rd, mvd_code_length + reference_index_code_length + flags_code, mvds, i, vector.second, flag_code_sum, flags);
        }else{
            std::vector<cv::Point2f> mvds;
            if(!warping_vectors[i].empty()){
                mvds.emplace_back(warping_vectors[i][0] - mv[0]);
                mvds.emplace_back(warping_vectors[i][1] - mv[1]);
                mvds.emplace_back(warping_vectors[i][2] - mv[2]);
            }else {
                mvds.emplace_back(current_mv - mv[0]);
                mvds.emplace_back(current_mv - mv[1]);
                mvds.emplace_back(current_mv - mv[2]);
            }

            int mvd_code_length = 6;
            FlagsCodeSum flag_code_sum(0, 0, 0, 0);
            Flags flags;
            for(int j = 0 ; j < mvds.size() ; j++){

#if MVD_DEBUG_LOG
                std::cout << "target_vector_idx       :" << j << std::endl;
                std::cout << "diff_target_mv(warping) :" << current_mv << std::endl;
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
            int reference_index = i;
            int reference_index_code_length = getUnaryCodeLength(reference_index);

            // 各種フラグ分を(3*2)bit足してます
            double rd = residual + lambda * (mvd_code_length + reference_index_code_length + flags_code);

            // 結果に入れる
            results.emplace_back(rd, mvd_code_length + reference_index_code_length + flags_code, mvds, i, vector.second, flag_code_sum, flags);
        }
    }

    // マージ符号化
    // マージで参照する動きベクトルを使って残差を求め直す
    Triangle current_triangle_coordinate = triangles[triangle_idx].first;
    cv::Point2f p1 = corners[current_triangle_coordinate.p1_idx];
    cv::Point2f p2 = corners[current_triangle_coordinate.p2_idx];
    cv::Point2f p3 = corners[current_triangle_coordinate.p3_idx];
    Point3Vec coordinate = Point3Vec(p1, p2, p3);
    vectors.clear();

    std::vector<cv::Point2f> pixels_in_triangle;
    std::vector<std::pair<cv::Point2f, MV_CODE_METHOD>> merge_vectors;
    if(pixels.empty()) {
         pixels_in_triangle = getPixelsInTriangle(coordinate, area_flag, triangle_idx, ctu,
                                                                          block_size_x, block_size_y);
    }else{
        pixels_in_triangle = pixels;
    }

    double sx = std::min({coordinate.p1.x, coordinate.p2.x, coordinate.p3.x});
    double sy = std::min({coordinate.p1.y, coordinate.p2.y, coordinate.p3.y});
    double lx = std::max({coordinate.p1.x, coordinate.p2.x, coordinate.p3.x});
    double ly = std::max({coordinate.p1.y, coordinate.p2.y, coordinate.p3.y});

    int merge_count = 0;
    int merge2_count = 0;

    std::vector<std::vector<bool>> share_flags;
#if MERGE_MODE
    if(translation_flag) {
        for (int i = 0; i < spatial_triangle_size; i++) {
            int spatial_triangle_index = spatial_triangles[i];
            GaussResult spatial_triangle = triangle_gauss_results[spatial_triangle_index];
            std::vector<cv::Point2f> mvds;
            cv::Rect rect(-SEARCH_RANGE * 4, -SEARCH_RANGE * 4, 4 * (target_image.cols + 2 * SEARCH_RANGE), 4 * (target_image.rows + 2 * SEARCH_RANGE));
            std::vector<cv::Point2f> mvs;

            if (spatial_triangle.translation_flag) {
                if(spatial_triangle.mv_translation.x + sx < -SEARCH_RANGE || spatial_triangle.mv_translation.y + sy < -SEARCH_RANGE || spatial_triangle.mv_translation.x + lx >= target_image.cols + SEARCH_RANGE || spatial_triangle.mv_translation.y + ly >= target_image.rows + SEARCH_RANGE) continue;
                if (!isMvExists(merge_vectors, spatial_triangle.mv_translation) && merge_count < MV_LIST_MAX_NUM) {
                    merge_vectors.emplace_back(spatial_triangle.mv_translation, MERGE);
                    mvs.emplace_back(spatial_triangle.mv_translation);
                    mvs.emplace_back(spatial_triangle.mv_translation);
                    mvs.emplace_back(spatial_triangle.mv_translation);
                    double ret_residual = getTriangleResidual(ref_hevc, target_image, coordinate, mvs, pixels_in_triangle, rect);
                    double rd = ret_residual + lambda * (getUnaryCodeLength(merge_count) + flags_code);
                    results.emplace_back(rd, getUnaryCodeLength(merge_count) + flags_code, mvs, merge_count, MERGE, FlagsCodeSum(0, 0, 0, 0), Flags());
                    merge_count++;
                }
            } else {
                if(spatial_triangle.mv_warping[0].x + sx < -SEARCH_RANGE || spatial_triangle.mv_warping[0].y + sy < -SEARCH_RANGE || spatial_triangle.mv_warping[0].x + lx >= target_image.cols + SEARCH_RANGE || spatial_triangle.mv_warping[0].y + ly >= target_image.rows + SEARCH_RANGE) continue;
                if (!isMvExists(merge_vectors, spatial_triangle.mv_warping[0]) && merge_count < MV_LIST_MAX_NUM) {
                    merge_vectors.emplace_back(spatial_triangle.mv_warping[0], MERGE);
                    mvs.emplace_back(spatial_triangle.mv_warping[0]);
                    mvs.emplace_back(spatial_triangle.mv_warping[0]);
                    mvs.emplace_back(spatial_triangle.mv_warping[0]);
                    double ret_residual = getTriangleResidual(ref_hevc, target_image, coordinate, mvs,
                                                              pixels_in_triangle, rect);
                    double rd = ret_residual + lambda * (getUnaryCodeLength(merge_count) + flags_code);
                    results.emplace_back(rd, getUnaryCodeLength(merge_count)  + flags_code, mvs, merge_count, MERGE, FlagsCodeSum(0, 0, 0, 0), Flags());
                    merge_count++;
                }
            }

        }
    }else{
        std::vector<Point3Vec> warping_vector_history;
        std::vector<Point3Vec> warping2_vector_history;
        for(int i = 0 ; i < warping_vectors.size() ; i++){
            cv::Rect rect(-SEARCH_RANGE * 4, -SEARCH_RANGE * 4, 4 * (target_image.cols + 2 * SEARCH_RANGE), 4 * (target_image.rows + 2 * SEARCH_RANGE));
            std::vector<cv::Point2f> mvs;
            std::vector<cv::Point2f> mvds;

            if(!warping_vectors[i].empty()) {
                mvs.emplace_back(warping_vectors[i][0]);
                mvs.emplace_back(warping_vectors[i][1]);
                mvs.emplace_back(warping_vectors[i][2]);

                if(mvs[0].x + sx < -SEARCH_RANGE || mvs[0].y + sy < -SEARCH_RANGE || mvs[0].x + lx >= target_image.cols + SEARCH_RANGE  || mvs[0].y + ly>=target_image.rows + SEARCH_RANGE ) continue;
                if(mvs[1].x + sx < -SEARCH_RANGE || mvs[1].y + sy < -SEARCH_RANGE || mvs[1].x + lx >= target_image.cols + SEARCH_RANGE  || mvs[1].y + ly>=target_image.rows + SEARCH_RANGE ) continue;
                if(mvs[2].x + sx < -SEARCH_RANGE || mvs[2].y + sy < -SEARCH_RANGE || mvs[2].x + lx >= target_image.cols + SEARCH_RANGE  || mvs[2].y + ly>=target_image.rows + SEARCH_RANGE ) continue;

                if (!isMvExists(warping_vector_history, mvs) && warping_vector_history.size() <= MV_LIST_MAX_NUM) {
                    double ret_residual = getTriangleResidual(ref_hevc, target_image, coordinate, mvs, pixels_in_triangle, rect);
                    double rd = ret_residual + lambda * (getUnaryCodeLength(merge_count));
                    results.emplace_back(rd, getUnaryCodeLength(merge_count), mvs, merge_count, MERGE, FlagsCodeSum(0, 0, 0, 0), Flags());
                    merge_count++;
                    warping_vector_history.emplace_back(mvs[0], mvs[1], mvs[2]);
                }

                if(MERGE2_ENABLE){
                    // 共有してる頂点を探す
                    Triangle tmp_target_t = triangles[triangle_idx].first;
                    std::set<int> p1_set = same_corner_list[tmp_target_t.p1_idx];
                    std::set<int> p2_set = same_corner_list[tmp_target_t.p2_idx];
                    std::set<int> p3_set = same_corner_list[tmp_target_t.p3_idx];

                    Triangle tmp_ref_t = triangles[spatial_triangles[i]].first;

                    share_flags.emplace_back();
                    share_flags[merge2_count].resize(3, false);
                    if(p1_set.find(tmp_ref_t.p1_idx) != p1_set.end() || p1_set.find(tmp_ref_t.p2_idx) != p1_set.end() || p1_set.find(tmp_ref_t.p3_idx) != p1_set.end()){
                        share_flags[merge2_count][0] = true;
                    }

                    if(p2_set.find(tmp_ref_t.p1_idx) != p2_set.end() || p2_set.find(tmp_ref_t.p2_idx) != p2_set.end() || p2_set.find(tmp_ref_t.p3_idx) != p2_set.end()){
                        share_flags[merge2_count][1] = true;
                    }

                    if(p3_set.find(tmp_ref_t.p1_idx) != p3_set.end() || p3_set.find(tmp_ref_t.p2_idx) != p3_set.end() || p3_set.find(tmp_ref_t.p3_idx) != p3_set.end()){
                        share_flags[merge2_count][2] = true;
                    }

                    mvs.clear();
                    std::vector<cv::Point2f> warping2_mvds;
                    for(int j = 0 ; j < 3 ; j++) {
                        if(share_flags[merge2_count][j]){
                            mvs.emplace_back(warping_vectors[i][j]);
                        }else{
                            mvs.emplace_back(mv[j]);
                            warping2_mvds.emplace_back(warping_vectors[i][j] - mv[j]);
                        }
                    }

                    if(mvs[0].x + sx < -SEARCH_RANGE || mvs[0].y + sy < -SEARCH_RANGE || mvs[0].x + lx >= target_image.cols + SEARCH_RANGE  || mvs[0].y + ly>=target_image.rows + SEARCH_RANGE ) continue;
                    if(mvs[1].x + sx < -SEARCH_RANGE || mvs[1].y + sy < -SEARCH_RANGE || mvs[1].x + lx >= target_image.cols + SEARCH_RANGE  || mvs[1].y + ly>=target_image.rows + SEARCH_RANGE ) continue;
                    if(mvs[2].x + sx < -SEARCH_RANGE || mvs[2].y + sy < -SEARCH_RANGE || mvs[2].x + lx >= target_image.cols + SEARCH_RANGE  || mvs[2].y + ly>=target_image.rows + SEARCH_RANGE ) continue;

                    if (!isMvExists(warping2_vector_history, mvs) && warping_vector_history.size() <= MV_LIST_MAX_NUM) {
                        double ret_residual = getTriangleResidual(ref_hevc, target_image, coordinate, mvs,
                                                                  pixels_in_triangle, rect);

                        int mvd_code_length = 6;
                        FlagsCodeSum flag_code_sum(0, 0, 0, 0);
                        Flags flags;
                        for (int j = 0; j < warping2_mvds.size(); j++) {

    #if MVD_DEBUG_LOG
                            std::cout << "target_vector_idx       :" << j << std::endl;
                    std::cout << "diff_target_mv(warping) :" << current_mv << std::endl;
                    std::cout << "encode_mv(warping)      :" << mv[j] << std::endl;
    #endif

                            cv::Point2f mvd = getQuantizedMv(warping2_mvds[j], 4);

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
                            warping2_mvds[j] = mvd;

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
                            warping2_mvds[j].x = mvd.x;
                            warping2_mvds[j].y = mvd.y;
                        }

                        double rd = ret_residual + lambda * (getUnaryCodeLength(merge2_count) + mvd_code_length + flags_code);
                        results.emplace_back(rd, getUnaryCodeLength(merge2_count) + mvd_code_length + flags_code, mvs, merge2_count,
                                             MERGE2, flag_code_sum, flags);
                        merge2_count++;
                        warping2_vector_history.emplace_back(mvs[0], mvs[1], mvs[2]);
                    }
                }
            }
        }
    }
#endif

    Triangle t = triangles[triangle_idx].first;

    int width = std::max({corners[t.p1_idx].x, corners[t.p2_idx].x, corners[t.p3_idx].x}) - std::min({corners[t.p1_idx].x, corners[t.p2_idx].x, corners[t.p3_idx].x});
    // イントラ
    if(INTRA_MODE){
        double sad = 0.0;
        setIntraImage(pixels_in_triangle, Point3Vec(corners[t.p1_idx], corners[t.p2_idx], corners[t.p3_idx]));
        for(const auto &p : pixels_in_triangle){
            sad += std::abs(R(intra_tmp_image, (int)p.x, (int)p.y) - R(target_image, (int)p.x, (int)p.y));
        }

        std::vector<cv::Point2f> mvs;
        results.emplace_back(sad, flags_code, mvs, 0, MV_CODE_METHOD::INTRA, FlagsCodeSum(0, 0, 0, 0), Flags());
    }

    // RDしたスコアが小さい順にソート
    std::sort(results.begin(), results.end(), [](const std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum, Flags >& a, const std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum, Flags>& b){
        return std::get<0>(a) < std::get<0>(b);
    });
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
    ctu->mvds = mvds;
    ctu->ref_triangle_idx = selected_idx;

#if MVD_DEBUG_LOG
    puts("Result ===========================================");
    std::cout << "code_length:" << code_length << std::endl;
    std::cout << "cost       :" << cost << std::endl;
    if(method != MERGE && method != INTRA){
        if(translation_flag) {
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
    if(method != MERGE && method != INTRA) {
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

    if(method == MERGE2){
        (ctu->mvds_x).clear();
        (ctu->mvds_y).clear();
        (ctu->original_mvds_x).clear();
        (ctu->original_mvds_y).clear();

        for (int i = 0; i < 3; i++) {
            if(share_flags[selected_idx][i]) {
                (ctu->mvds_x).emplace_back(mvds[i].x);
                (ctu->mvds_y).emplace_back(mvds[i].y);
            }
        }
        ctu->share_flag[0] = share_flags[selected_idx][0];
        ctu->share_flag[1] = share_flags[selected_idx][1];
        ctu->share_flag[2] = share_flags[selected_idx][2];
    }

    return {cost, code_length, mvds, selected_idx, method};
}

/**
 * @fn cv::Point2f TriangleDivision::getQuantizedMv(cv::Point2f mv, int quantize_step)
 * @param mv 動きベクトル
 * @param quantize_step 量子化ステップ幅
 * @return 量子化済みの動きベクトル
 */
cv::Point2f TriangleDivision::getQuantizedMv(cv::Point2f &mv, double quantize_step){
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

cv::Mat TriangleDivision::getPredictedDiagonalImageFromCtu(std::vector<CodingTreeUnit*> ctus, std::vector<std::vector<std::vector<int>>> &area_flag){
    cv::Mat out = cv::Mat::zeros(ref_image.size(), CV_8UC3);
    for(int i = 0 ; i < ctus.size() ; i++) {
        getPredictedDiagonalImageFromCtu(ctus[i], area_flag[(int)i/2], out);
    }

    return out;
}

void TriangleDivision::getPredictedDiagonalImageFromCtu(CodingTreeUnit* ctu, std::vector<std::vector<int>> &area_flag, const cv::Mat &out){

    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int triangle_index = ctu->triangle_index;
        Triangle triangle_corner_idx = triangles[triangle_index].first;
        Point3Vec triangle(corners[triangle_corner_idx.p1_idx], corners[triangle_corner_idx.p2_idx], corners[triangle_corner_idx.p3_idx]);
        std::vector<cv::Point2f> pixels = getPixelsInTriangle(triangle, area_flag, triangle_index, ctu, block_size_x, block_size_y);
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

cv::Mat TriangleDivision::getPredictedImageFromCtu(std::vector<CodingTreeUnit*> ctus, std::vector<std::vector<std::vector<int>>> &area_flag){
    cv::Mat out = cv::Mat::zeros(ref_image.size(), CV_8UC3);

#pragma omp parallel for
    for(int i = 0 ; i < ctus.size() ; i++) {
        getPredictedImageFromCtu(ctus[i], out, area_flag[i/2]);
    }

    return out;
}

void TriangleDivision::getPredictedImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out, std::vector<std::vector<int>> &area_flag){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int triangle_index = ctu->triangle_index;
        cv::Point2f mv = ctu->mv1;
        Triangle triangle_corner_idx = triangles[triangle_index].first;
        Point3Vec triangle(corners[triangle_corner_idx.p1_idx], corners[triangle_corner_idx.p2_idx], corners[triangle_corner_idx.p3_idx]);

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

        if(ctu->method == MV_CODE_METHOD::INTRA){
            std::cout << "INTRA!!!!!!!!!!!!!" << std::endl;
            Triangle t = triangles[ctu->triangle_index].first;
            auto pixels = getPixelsInTriangle(Point3Vec(corners[t.p1_idx], corners[t.p2_idx], corners[t.p3_idx]), area_flag, ctu->triangle_index, ctu, block_size_x, block_size_y);
            for(const auto& pixel : pixels) {
                R(out, (int)pixel.x, (int)pixel.y) = R(intra_tmp_image, (int)pixel.x, (int)pixel.y);
                G(out, (int)pixel.x, (int)pixel.y) = G(intra_tmp_image, (int)pixel.x, (int)pixel.y);
                B(out, (int)pixel.x, (int)pixel.y) = B(intra_tmp_image, (int)pixel.x, (int)pixel.y);
            }

        }else {
            getPredictedImage(expansion_ref_uchar, target_image, out, triangle, mvs, SEARCH_RANGE, area_flag, ctu->triangle_index,
                              ctu, cv::Rect(0, 0, block_size_x, block_size_y), ref_hevc);
        }
        return;
    }

    if(ctu->node1 != nullptr) getPredictedImageFromCtu(ctu->node1, out, area_flag);
    if(ctu->node2 != nullptr) getPredictedImageFromCtu(ctu->node2, out, area_flag);
    if(ctu->node3 != nullptr) getPredictedImageFromCtu(ctu->node3, out, area_flag);
    if(ctu->node4 != nullptr) getPredictedImageFromCtu(ctu->node4, out, area_flag);
}


cv::Mat TriangleDivision::getMergeModeColorImageFromCtu(std::vector<CodingTreeUnit*> ctus, std::vector<std::vector<std::vector<int>>> &area_flag){
    cv::Mat out = cv::Mat::zeros(ref_image.size(), CV_8UC3);

    for(int i = 0 ; i < ctus.size() ; i++) {
        getMergeModeColorImageFromCtu(ctus[i], out, area_flag[i/2]);
    }

    std::vector<Point3Vec> ts = getTriangleCoordinateList();
    for(const auto &t : ts) {
        drawTriangle(out, t.p1, t.p2, t.p3, WHITE);
    }

    return out;
}

void TriangleDivision::getMergeModeColorImageFromCtu(CodingTreeUnit* ctu, cv::Mat &out, std::vector<std::vector<int>> &area_flag){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int triangle_index = ctu->triangle_index;
        cv::Point2f mv = ctu->mv1;
        Triangle triangle_corner_idx = triangles[triangle_index].first;
        Point3Vec triangle(corners[triangle_corner_idx.p1_idx], corners[triangle_corner_idx.p2_idx], corners[triangle_corner_idx.p3_idx]);

        std::vector<cv::Point2f> mvs{mv, mv, mv};
        std::vector<cv::Point2f> pixels = getPixelsInTriangle(triangle, area_flag, triangle_index, ctu, block_size_x, block_size_y);

        if(ctu->method == MERGE || ctu->method == MERGE2){
            int r, g, b;
            switch(ctu->ref_triangle_idx){
                case 0:
                    r = 255; g = 0; b = 0;
                    break;
                case 1:
                    r = 0;   g = 255; b = 0;
                    break;
                case 2:
                    r = 0;   g = 0; b = 255;
                    break;
                case 3:
                    r = 255; g = 255; b = 0;
                    break;
                case 4:
                    r = 255; g = 0; b = 255;
                    break;
            }

            for(auto pixel : pixels) {
                R(out, (int)pixel.x, (int)pixel.y) = r;
                G(out, (int)pixel.x, (int)pixel.y) = g;
                B(out, (int)pixel.x, (int)pixel.y) = b;
            }
        }

        return;
    }

    if(ctu->node1 != nullptr) getMergeModeColorImageFromCtu(ctu->node1, out, area_flag);
    if(ctu->node2 != nullptr) getMergeModeColorImageFromCtu(ctu->node2, out, area_flag);
    if(ctu->node3 != nullptr) getMergeModeColorImageFromCtu(ctu->node3, out, area_flag);
    if(ctu->node4 != nullptr) getMergeModeColorImageFromCtu(ctu->node4, out, area_flag);
}

cv::Mat TriangleDivision::getPredictedColorImageFromCtu(std::vector<CodingTreeUnit*> ctus, std::vector<std::vector<std::vector<int>>> &area_flag, double original_psnr){
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

    std::vector<Point3Vec> ts = getTriangleCoordinateList();
    for(const auto &t : ts) {
        drawTriangle(out, t.p1, t.p2, t.p3, WHITE);
    }

    return out;
}

void TriangleDivision::getPredictedColorImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out, std::vector<std::vector<int>> &area_flag, double original_psnr, std::vector<cv::Scalar> &colors){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int triangle_index = ctu->triangle_index;
        cv::Point2f mv = ctu->mv1;
        Triangle triangle_corner_idx = triangles[triangle_index].first;
        Point3Vec triangle(corners[triangle_corner_idx.p1_idx], corners[triangle_corner_idx.p2_idx], corners[triangle_corner_idx.p3_idx]);

        std::vector<cv::Point2f> mvs{mv, mv, mv};
        std::vector<cv::Point2f> pixels = getPixelsInTriangle(triangle, area_flag, triangle_index, ctu, block_size_x, block_size_y);

        if(ctu->translation_flag) {
            if(ctu->method == MV_CODE_METHOD::MERGE){
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = 0;
                    G(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    B(out, (int)pixel.x, (int)pixel.y) = 0;
                }
            }else if(ctu->method == MV_CODE_METHOD::MERGE2){

                int share_count = 0;
                for(auto f : ctu->share_flag) {
                    if(f) share_count++;
                }

                if(share_count == 1){
                    for(auto pixel : pixels) {
                        R(out, (int)pixel.x, (int)pixel.y) = 54;
                        G(out, (int)pixel.x, (int)pixel.y) = 115;
                        B(out, (int)pixel.x, (int)pixel.y) = 255;
                    }
                }else if(share_count == 2){
                    for(auto pixel : pixels) {
                        R(out, (int)pixel.x, (int)pixel.y) = 212;
                        G(out, (int)pixel.x, (int)pixel.y) = 61;
                        B(out, (int)pixel.x, (int)pixel.y) = 0;
                    }
                }

                std::cout << "------------------- MERGE2 -------------------" << std::endl;
            }else if(ctu->method == MV_CODE_METHOD::SPATIAL){
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    G(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    B(out, (int)pixel.x, (int)pixel.y) = 0;
                }
            }else if(ctu->method == MV_CODE_METHOD::INTRA){
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    G(out, (int)pixel.x, (int)pixel.y) = 0;
                    B(out, (int)pixel.x, (int)pixel.y) = 0;
                }
            }

        }else{
            if(ctu->method == MV_CODE_METHOD::MERGE){
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = 0;
                    G(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    B(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                }
            }else if(ctu->method == MV_CODE_METHOD::MERGE2){

                int share_count = 0;
                for(auto f : ctu->share_flag) {
                    if(f) share_count++;
                }

                if(share_count == 1){
                    for(auto pixel : pixels) {
                        R(out, (int)pixel.x, (int)pixel.y) = 54;
                        G(out, (int)pixel.x, (int)pixel.y) = 115;
                        B(out, (int)pixel.x, (int)pixel.y) = 255;
                    }
                }else if(share_count == 2){
                    for(auto pixel : pixels) {
                        R(out, (int)pixel.x, (int)pixel.y) = 212;
                        G(out, (int)pixel.x, (int)pixel.y) = 61;
                        B(out, (int)pixel.x, (int)pixel.y) = 0;
                    }
                }
                std::cout << "------------------- MERGE2 -------------------" << std::endl;
            }else if(ctu->method == MV_CODE_METHOD::SPATIAL){
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = 0;
                    G(out, (int)pixel.x, (int)pixel.y) = 0;
                    B(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                }
            }else if(ctu->method == MV_CODE_METHOD::INTRA){
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    G(out, (int)pixel.x, (int)pixel.y) = 0;
                    B(out, (int)pixel.x, (int)pixel.y) = 0;
                }
            }

//            getPredictedImage(expansion_ref_uchar, target_image, out, triangle, mvs, SEARCH_RANGE, area_flag, ctu->triangle_index, ctu, cv::Rect(0, 0, block_size_x, block_size_y), ref_hevc);
        }

//        Triangle t = triangles[ctu->triangle_index].first;
//        cv::Point2f p1 = corners[t.p1_idx];
//        cv::Point2f p2 = corners[t.p2_idx];
//        cv::Point2f p3 = corners[t.p3_idx];
//
//        if(ctu->translation_flag) {
//            cv::Point2f g = (p1 + p2 + p3) / 3.0;
//
//            cv::arrowedLine(out, g, g + 10 * ctu->mv1, RED, 1);
//
//            if(ctu->method == MV_CODE_METHOD::MERGE || ctu->method == MV_CODE_METHOD::MERGE2){
//                cv::arrowedLine(out, g, g + 10 * ctu->original_mv1, GREEN, 1);
//                cv::arrowedLine(out, g, g + ctu->merge_triangle_ref_vector, WHITE);
//            }
//        }else{
//            cv::Point2f g = (p1 + p2 + p3) / 3.0;
//            cv::arrowedLine(out, p1, p1 + 10 * ctu->mv1, RED, 1);
//            cv::arrowedLine(out, p2, p2 + 10 * ctu->mv2, RED, 1);
//            cv::arrowedLine(out, p3, p3 + 10 * ctu->mv3, RED, 1);
//
//            if(ctu->method == MV_CODE_METHOD::MERGE || ctu->method == MV_CODE_METHOD::MERGE2){
//                cv::arrowedLine(out, g, g + ctu->merge_triangle_ref_vector, WHITE);
//                cv::arrowedLine(out, p1, p1 + 10 * ctu->original_mv1, GREEN, 1);
//                cv::arrowedLine(out, p2, p2 + 10 * ctu->original_mv2, GREEN, 1);
//                cv::arrowedLine(out, p3, p3 + 10 * ctu->original_mv3, GREEN, 1);
//            }
//        }

        return;
    }

    if(ctu->node1 != nullptr) getPredictedColorImageFromCtu(ctu->node1, out, area_flag, original_psnr, colors);
    if(ctu->node2 != nullptr) getPredictedColorImageFromCtu(ctu->node2, out, area_flag, original_psnr, colors);
    if(ctu->node3 != nullptr) getPredictedColorImageFromCtu(ctu->node3, out, area_flag, original_psnr, colors);
    if(ctu->node4 != nullptr) getPredictedColorImageFromCtu(ctu->node4, out, area_flag, original_psnr, colors);
}

int TriangleDivision::getCtuCodeLength(std::vector<CodingTreeUnit*> ctus) {
    int code_length_sum = 0;
    for(auto & ctu : ctus){
        code_length_sum += getCtuCodeLength(ctu);
    }
    return code_length_sum;
}

int TriangleDivision::getCtuCodeLength(CodingTreeUnit *ctu){

    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
//        // この1bitは手法フラグ(translation/warping)，もう1bitはマージフラグ
//        int flags_code = 1;
//        if (INTRA_MODE) flags_code++;
//        if (MERGE_MODE) flags_code++;
//
        return ctu->code_length;
    }

    // ここで足している1はsplit_cu_flag分です
    return 1 + getCtuCodeLength(ctu->node1) + getCtuCodeLength(ctu->node2) + getCtuCodeLength(ctu->node3) + getCtuCodeLength(ctu->node4);
}


cv::Mat TriangleDivision::getMvImage(std::vector<CodingTreeUnit*> ctus){
    cv::Mat out = target_image.clone();

    for(const auto& triangle : getTriangleCoordinateList()){
        drawTriangle(out, triangle.p1, triangle.p2, triangle.p3, cv::Scalar(255, 255, 255));
    }

    for(int i = 0 ; i < ctus.size() ; i++){
        drawMvImage(out, ctus[i]);
    }

    return out;
}

void TriangleDivision::drawMvImage(cv::Mat &out, CodingTreeUnit *ctu){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        Triangle t = triangles[ctu->triangle_index].first;
        cv::Point2f p1 = corners[t.p1_idx];
        cv::Point2f p2 = corners[t.p2_idx];
        cv::Point2f p3 = corners[t.p3_idx];

        if(ctu->translation_flag) {
            cv::Point2f g = (p1 + p2 + p3) / 3.0;

            cv::arrowedLine(out, g, g + 10 * ctu->mv1, GREEN);

            if(ctu->method == MV_CODE_METHOD::MERGE || ctu->method == MV_CODE_METHOD::MERGE2){
                cv::arrowedLine(out, g, g + 10 * ctu->original_mv1, BLUE);
            }
        }else{
            cv::arrowedLine(out, p1, p1 + 10 * ctu->mv1, GREEN);
            cv::arrowedLine(out, p2, p2 + 10 * ctu->mv2, GREEN);
            cv::arrowedLine(out, p3, p3 + 10 * ctu->mv3, GREEN);

            if(ctu->method == MV_CODE_METHOD::MERGE || ctu->method == MV_CODE_METHOD::MERGE2){
                cv::arrowedLine(out, p1, p1 + 10 * ctu->original_mv1, BLUE);
                cv::arrowedLine(out, p2, p2 + 10 * ctu->original_mv2, BLUE);
                cv::arrowedLine(out, p3, p3 + 10 * ctu->original_mv3, BLUE);
            }
        }
    }

    if(ctu->node1 != nullptr) drawMvImage(out, ctu->node1);
    if(ctu->node2 != nullptr) drawMvImage(out, ctu->node2);
    if(ctu->node3 != nullptr) drawMvImage(out, ctu->node3);
    if(ctu->node4 != nullptr) drawMvImage(out, ctu->node4);
}

TriangleDivision::TriangleDivision() {}

TriangleDivision::SplitResult::SplitResult(const Point3Vec &t1, const Point3Vec &t2, int t1Type, int t2Type) : t1(t1),
                                                                                                               t2(t2),
                                                                                                               t1_type(t1Type),
                                                                                                               t2_type(t2Type) {}

std::tuple<std::vector<cv::Point2f>, std::vector<double>> TriangleDivision::fullpellBlockMatching(Point3Vec triangle, const cv::Mat& target_image, cv::Mat expansion_ref_image, std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, cv::Point2f fullpell_initial_vector) {
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
    int SX = SEARCH_RANGE; // ブロックマッチングの探索範囲(X)
    int SY = SEARCH_RANGE; // ブロックマッチングの探索範囲(Y)

    double error_min = 1e9, rd_min = 1e9;
    int e_count;
    cv::Point2f mv_min;
    int spread_quarter = SEARCH_RANGE * 4;
    int s = 4;                   //4 : Full-pel, 2 : Half-pel, 1 : Quarter-pel
    std::vector<cv::Point2f> pixels = getPixelsInTriangle(triangle, area_flag, triangle_index, ctu, block_size_x, block_size_y);

    if(fullpell_initial_vector.x == -10000 && fullpell_initial_vector.y == -10000){
//#pragma omp parallel for
        for(int j = -SY * 4 ; j <= SY * 4 ; j += s) {            //j : y方向のMV
            for(int i = -SX * 4 ; i <= SX * 4 ; i += s) {        //i : x方向のMV
                double rd = 1e9, e = 1e9;
                //探索範囲が画像上かどうか判定
                if(-spread_quarter <= round(sx) + i && round(lx) + i < expansion_ref_image.cols - spread_quarter
                   && -spread_quarter <= round(sy) + j && round(ly) + j < expansion_ref_image.rows - spread_quarter) {
                    e = 0.0;
                    for(auto &pixel : pixels) {
                        int ref_x = std::max((int)(4 * pixel.x), 0);
                        ref_x = (i + ref_x + spread_quarter);
                        int ref_y = std::max((int)((4 * pixel.y)), 0);
                        ref_y = (j + ref_y + spread_quarter);
                        e += fabs(R(expansion_ref_image, ref_x, ref_y) - R(target_image, (int)pixel.x, (int)pixel.y));
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


bool TriangleDivision::isMvExists(const std::vector<Point3Vec> &vectors, const std::vector<cv::Point2f> &mvs) {
    for(const auto& vector : vectors){
        if(vector.p1 == mvs[0] && vector.p2 == mvs[1] && vector.p3 == mvs[2]) return true;
    }

    return false;
}

TriangleDivision::~TriangleDivision() {
    std::vector<cv::Point2f>().swap(corners);
    std::vector<std::pair<Triangle, int> >().swap(triangles);
    std::vector<std::set<int> >().swap(neighbor_vtx);
    std::vector<std::set<int> >().swap(covered_triangle);
    std::vector<std::set<int> >().swap(same_corner_list);
    std::vector<std::vector<int> >().swap(corner_flag);
    std::vector<bool>().swap(delete_flag);
    std::vector<bool>().swap(isCodedTriangle);
    std::vector<std::vector<CollocatedMvTree*>>().swap(previousMvList);
//    std::vector<cv::Mat>().swap(predicted_buf);
    for(auto i : predicted_buf) i.release();
    std::vector<GaussResult>().swap(triangle_gauss_results);
//    std::vector<std::vector<cv::Mat>>().swap(ref_images);
    for(int i = 0 ; i < ref_images.size() ; i++){
        for(int j = 0 ; j < ref_images[i].size(); j++){
            ref_images[i][j].release();
        }
    }
//    std::vector<std::vector<cv::Mat>>().swap(target_images);
    for(int i = 0 ; i < target_images.size() ; i++){
        for(int j = 0 ; j < target_images[i].size(); j++){
            target_images[i][j].release();
        }
    }

    int scaled_expansion_size = SEARCH_RANGE + 2;
    for(int i = -scaled_expansion_size ; i < target_image.cols + scaled_expansion_size ; i++){
        expansion_ref_uchar[i] -= scaled_expansion_size;
        free(expansion_ref_uchar[i]);
    }
    expansion_ref_uchar -= scaled_expansion_size;
    free(expansion_ref_uchar);

    for(int i = 4 * (SEARCH_RANGE + 4) ; i < 4 * (ref_image.cols + (SEARCH_RANGE + 4)) ; i++) {
        ref_hevc[i] -= 4 * (SEARCH_RANGE + 4);
        free(ref_hevc[i]);
    }
    ref_hevc -= 4 * (SEARCH_RANGE + 4);
    free(ref_hevc);

    expansion_ref.release();
}

/**
 * @fn void TriangleDivision::setIntraImage(std::vector<cv::Point2f> pixels)
 * @attention 呼び出す前にかならずisIntraAvailableでチェックをすること
 * @param[in] pixels 画素値
 */
void TriangleDivision::setIntraImage(std::vector<cv::Point2f> pixels, Point3Vec triangle) {

    // まずは参照画素を列挙する
    int sx = std::min({(int)triangle.p1.x, (int)triangle.p2.x, (int)triangle.p3.x});
    int sy = std::min({(int)triangle.p1.y, (int)triangle.p2.y, (int)triangle.p3.y});
    int lx = std::max({(int)triangle.p1.x, (int)triangle.p2.x, (int)triangle.p3.x});
    int ly = std::max({(int)triangle.p1.y, (int)triangle.p2.y, (int)triangle.p3.y});

    std::vector<int> y_axis_luminance(ly - sy + 1);
    std::vector<int> x_axis_luminance(lx - sx + 1);


    if(ly == target_image.cols - 1) {
        // 下端が外周上に乗っている場合は，一番最後だけ127にする
        for (int y = sy; y <= ly; y++) {
            bool flag = false;
            for (int x = lx; 0 <= x; x--) {
                if (intra_flag[x][y]) {
                    y_axis_luminance[y - sy] = R(intra_tmp_image, x, y);
                    flag = true;
                    break;
                }
            }
            if(!flag) y_axis_luminance[y - sy] = 127;
        }
        y_axis_luminance[(ly + 1)- sy] = 127;
    }else{
        for (int y = sy; y <= ly + 1; y++) {
            bool flag = false;
            for (int x = lx; 0 <= x; x--) {
                if (intra_flag[x][y]) {
                    y_axis_luminance[y - sy] = R(intra_tmp_image, x, y);
                    flag = true;
                    break;
                }
            }
            if(!flag) y_axis_luminance[y - sy] = 127;
        }
    }

    if(lx == target_image.cols - 1){
        for (int x = sx; x <= lx; x++) {
            bool flag = false;
            for (int y = ly; 0 <= y; y--) {
                if (intra_flag[x][y]) {
                    x_axis_luminance[x - sx] = R(intra_tmp_image, x, y);
                    flag = true;
                    break;
                }
            }
            if (!flag) x_axis_luminance[x - sx] = 127;
        }
        x_axis_luminance[lx + 1 - sx] = 127;
    }else {
        for (int x = sx; x <= lx + 1; x++) {
            bool flag = false;
            for (int y = ly; 0 <= y; y--) {
                if (intra_flag[x][y]) {
                    x_axis_luminance[x - sx] = R(intra_tmp_image, x, y);
                    flag = true;
                    break;
                }
            }
            if (!flag) x_axis_luminance[x - sx] = 127;
        }
    }


    // 線形補間して補間画素を求める
    int pixel_nums = (int)pixels.size();
    for(const auto pixel : pixels) {
        int x = pixel.x;
        int y = pixel.y;

        //   *************
        //   *   |β      *
        //   *---+       *
        //   * α         *
        //   *           *
        //   *           *
        //   *************
        //
        double alpha = (double)x / (lx - sx);
        double beta  = (double)y / (ly - sy);

        // ref:
        int N = lx - sx + 1;
        int luminance = (int)((double)(lx - sx - x + sx) * x_axis_luminance[x - sx]
                            + (double)(x + 1 - sx      ) * x_axis_luminance[N     ]
                            + (double)(ly - sy - y + sy) * y_axis_luminance[y - sy]
                            + (double)(y + 1 - sy      ) * y_axis_luminance[N     ]
                            + N
                        ) / (2 * N);

        luminance = (luminance > 255 ? 255 : (luminance < 0 ? 0 : luminance));

        R(intra_tmp_image, x, y) = luminance;
        G(intra_tmp_image, x, y) = luminance;
        B(intra_tmp_image, x, y) = luminance;
    }

}

/**
 * @fn イントラ符号化できるかチェックする
 * @param _x イントラ符号化するx座標
 * @param _y イントラ符号化するy座標
 * @return 符号化できるならtrue, そうでないならfalse
 */
bool TriangleDivision::isIntraAvailable(Point3Vec triangle){
    int sx = std::min({(int)triangle.p1.x, (int)triangle.p2.x, (int)triangle.p3.x});
    int sy = std::min({(int)triangle.p1.y, (int)triangle.p2.y, (int)triangle.p3.y});
    int lx = std::max({(int)triangle.p1.x, (int)triangle.p2.x, (int)triangle.p3.x});
    int ly = std::max({(int)triangle.p1.y, (int)triangle.p2.y, (int)triangle.p3.y});

    if(sx == 0 || sy == 0 || lx == target_image.cols - 1 || ly == target_image.rows - 1) return false;

    for(int y = sy ; y <= ly ; y++) {
        bool check_flag_x = false;
        for (int x = sx; 0 <= x; x--) {
            if (intra_flag[x][y]) {
                check_flag_x = true;
                break;
            }
        }
        if(!check_flag_x) return false;
    }

    for(int x = sx ; x <= lx ; x++) {
        bool check_flag_y = false;
        for(int y = sy ; 0 <= y ; y--){
            if (intra_flag[sx][y]) {
                check_flag_y = true;
                break;
            }
        }
        if(!check_flag_y) return false;
    }

    return true;
}