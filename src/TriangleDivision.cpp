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
#include "../includes/ImageUtil.h"

TriangleDivision::TriangleDivision(const cv::Mat &refImage, const cv::Mat &targetImage) : target_image(targetImage),
                                                                                          ref_image(refImage) {}



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
    int block_num_x = target_image.cols / block_size_x;
    int block_num_y = target_image.rows / block_size_y;
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
    for(int block_y = 0 ; block_y <= block_num_y ; block_y++) {
        for (int block_x = 0; block_x <= block_num_x; block_x++) {
            int nx = block_x * block_size_x;
            int ny = block_y * block_size_y;

            if(nx < 0) nx = 0;
            if(target_image.cols <= nx) nx = target_image.cols - 1;
            if(ny < 0) ny = 0;
            if(target_image.rows <= ny) ny = target_image.rows - 1;
            corners.emplace_back(nx, ny);
            corner_flag[ny][nx] = static_cast<int>(corners.size() - 1);
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
        }
    }

    // 過去のMVを残すやつを初期化
    for(auto node : previousMvList[coded_picture_num]) {
        node->leftNode = node->rightNode = nullptr;
        node->mv1 = cv::Point2f(0.0, 0.0);
        node->mv2 = cv::Point2f(0.0, 0.0);
        node->mv3 = cv::Point2f(0.0, 0.0);
    }

    std::cout << "block_num_y:" << block_num_y << std::endl;
    std::cout << "block_num_x:" << block_num_x << std::endl;

    covered_triangle.resize(static_cast<unsigned long>((block_num_x + 1) * (block_num_y + 1)));

    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {
        for(int block_x = 0 ; block_x < block_num_x ; block_x++) {
            int p1_idx = block_x + block_y * (block_num_x + 1);
            int p2_idx = p1_idx + 1;
            int p3_idx = p1_idx + block_num_x + 1;
            int p4_idx = p3_idx + 1;
            if(divide_flag == LEFT_DIVIDE) {
                int triangleIndex = insertTriangle(p1_idx, p2_idx, p3_idx, TYPE1);
                addNeighborVertex(p1_idx, p2_idx, p3_idx);
                addCoveredTriangle(p1_idx, p2_idx, p3_idx, triangleIndex); // p1/p2/p3はtriangleIndex番目の三角形に含まれている

                triangleIndex = insertTriangle(p2_idx, p3_idx, p4_idx, TYPE2);
                addNeighborVertex(p2_idx, p3_idx, p4_idx);
                addCoveredTriangle(p2_idx, p3_idx, p4_idx, triangleIndex);
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
}

/**
 * @fn std::vector<Point3Vec> getTriangleCoordinateList()
 * @brief 現在存在する三角形の集合(座標)を返す
 * @return 三角形の集合（座標）
 */
std::vector<Point3Vec> TriangleDivision::getTriangleCoordinateList() {
    std::vector<Point3Vec> vec;

    for(int i = 0 ; i < triangles.size() ; i++) {
        if(delete_flag[i]) continue;
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
    covered_triangle.emplace_back();
    isCodedTriangle.emplace_back(false);
    triangle_gauss_results.emplace_back();
    triangle_gauss_results[triangle_gauss_results.size() - 1].residual = -1.0;

    return static_cast<int>(triangles.size() - 1);
}

/**
 * @fn void TriangleDivision::eraseTriangle(int t_idx)
 * @brief 三角パッチに関わる情報を削除する
 * @param t_idx 三角パッチの番号
 */
void TriangleDivision::eraseTriangle(int t_idx){
    isCodedTriangle.erase(isCodedTriangle.begin() + t_idx);
    triangles.erase(triangles.begin() + t_idx);
    covered_triangle.erase(covered_triangle.begin() + t_idx);
    isCodedTriangle.erase(isCodedTriangle.begin() + t_idx);
    triangle_gauss_results.erase(triangle_gauss_results.begin() + t_idx);
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
    if(p1_idx == 329 || p2_idx == 329 || p3_idx == 329) {
        std::cout << p1_idx << " " << p2_idx << " " << p3_idx << std::endl;
    }
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
    std::set<int> s = covered_triangle[target_vertex_idx];
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
    std::set<int> s = covered_triangle[target_vertex_idx];
    std::vector<int> v(s.size());

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
 * @fn int TriangleDivision::addCorner()
 * @param[in] p 追加する頂点の座標
 * @return 頂点番号を返す
 */
int TriangleDivision::addCorner(cv::Point2f p) {
    if(corner_flag[(int)p.y][(int)p.x] != -1) return corner_flag[(int)p.y][(int)p.x];
    corners.emplace_back(p);
    neighbor_vtx.emplace_back();
    corner_flag[(int)p.y][(int)p.x] = static_cast<int>(corners.size() - 1);
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

            int c_idx = addCorner(c);

            int a_idx = triangle.p1_idx;
            int b_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, c_idx, TYPE5);
            int t2_idx = insertTriangle(a_idx, c_idx, d_idx, TYPE6);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, c_idx);
            addNeighborVertex(a_idx, c_idx, d_idx);

            covered_triangle.emplace_back();
            covered_triangle.emplace_back();
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

            int b_idx = addCorner(b);

            int a_idx = triangle.p1_idx;
            int c_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, d_idx, TYPE8);
            int t2_idx = insertTriangle(b_idx, c_idx, d_idx, TYPE7);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, d_idx);
            addNeighborVertex(b_idx, c_idx, d_idx);

            covered_triangle.emplace_back();
            covered_triangle.emplace_back();
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

            int c_idx = addCorner(c);

            int a_idx = triangle.p1_idx;
            int b_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, c_idx, TYPE5);
            int t2_idx = insertTriangle(b_idx, c_idx, d_idx, TYPE8);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, c_idx);
            addNeighborVertex(b_idx, c_idx, d_idx);

            covered_triangle.emplace_back();
            covered_triangle.emplace_back();
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

            int b_idx = addCorner(b);

            int a_idx = triangle.p1_idx;
            int c_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, c_idx, TYPE6);
            int t2_idx = insertTriangle(b_idx, c_idx, d_idx, TYPE7);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, c_idx);
            addNeighborVertex(b_idx, c_idx, d_idx);

            covered_triangle.emplace_back();
            covered_triangle.emplace_back();
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

            cv::Point2f b = p1 + x;

            int b_idx = addCorner(b);

            int a_idx = triangle.p1_idx;
            int c_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, d_idx, TYPE3);
            int t2_idx = insertTriangle(b_idx, c_idx, d_idx, TYPE1);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, d_idx);
            addNeighborVertex(b_idx, c_idx, d_idx);

            covered_triangle.emplace_back();
            covered_triangle.emplace_back();
            addCoveredTriangle(a_idx, b_idx, d_idx, t1_idx);
            addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);

        }
            break;
        case DIVIDE::TYPE6:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f y = (p3 - p1) / 2.0;

            cv::Point2f b = p1 + y;

            int b_idx = addCorner(b);

            int a_idx = triangle.p1_idx;
            int c_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, c_idx, TYPE4);
            int t2_idx = insertTriangle(b_idx, c_idx, d_idx, TYPE1);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, c_idx);
            addNeighborVertex(b_idx, c_idx, d_idx);

            covered_triangle.emplace_back();
            covered_triangle.emplace_back();
            addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
            addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);

        }
            break;
        case DIVIDE::TYPE7:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p3 - p2) / 2.0;

            cv::Point2f c = p2 + x;

            int c_idx = addCorner(c);

            int a_idx = triangle.p1_idx;
            int b_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, c_idx, TYPE2);
            int t2_idx = insertTriangle(a_idx, c_idx, d_idx, TYPE4);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, c_idx);
            addNeighborVertex(a_idx, c_idx, d_idx);

            covered_triangle.emplace_back();
            covered_triangle.emplace_back();
            addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
            addCoveredTriangle(a_idx, c_idx, d_idx, t2_idx);

        }
            break;
        case DIVIDE::TYPE8:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f y = (p3 - p1) / 2.0;

            cv::Point2f c = p1 + y;

            int c_idx = addCorner(c);

            int a_idx = triangle.p2_idx;
            int b_idx = triangle.p1_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, c_idx, TYPE2);
            int t2_idx = insertTriangle(a_idx, c_idx, d_idx, TYPE3);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, c_idx);
            addNeighborVertex(a_idx, c_idx, d_idx);

            covered_triangle.emplace_back();
            covered_triangle.emplace_back();
            addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
            addCoveredTriangle(a_idx, c_idx, d_idx, t2_idx);

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
bool TriangleDivision::split(cv::Mat &gaussRefImage, CodingTreeUnit* ctu, CollocatedMvTree* cmt, Point3Vec triangle, int triangle_index, int type, int steps) {
    if(steps == 0) return false;

    double RMSE_before_subdiv = 0.0;
    cv::Point2f p1 = triangle.p1;
    cv::Point2f p2 = triangle.p2;
    cv::Point2f p3 = triangle.p3;

    Point3Vec refTriangle(p1, p2, p3);
    Point3Vec targetTriangle(p1, p2, p3);
    int triangle_size = 0;
    double error = 0.0;
    bool parallel_flag;
    int num;
    cv::Mat warp_p_image, residual_ref, parallel_p_image;

    std::vector<cv::Point2f> gauss_result_warping;
    cv::Point2f gauss_result_parallel;

    if(triangle_gauss_results[triangle_index].residual > 0) {
        std::cout << "cache hit! triangle_index:" << triangle_index << std::endl;
        GaussResult result_before = triangle_gauss_results[triangle_index];
        gauss_result_warping = result_before.mv_warping;
        gauss_result_parallel = result_before.mv_parallel;
        RMSE_before_subdiv = result_before.residual;
        triangle_size = result_before.triangle_size;
        parallel_flag = result_before.parallel_flag;
    }else {
        std::tie(gauss_result_warping, gauss_result_parallel, RMSE_before_subdiv, triangle_size,
                 parallel_flag) = GaussNewton(ref_image, target_image, gaussRefImage, targetTriangle);

        triangle_gauss_results[triangle_index].mv_warping = gauss_result_warping;
        triangle_gauss_results[triangle_index].mv_parallel = gauss_result_parallel;
        triangle_gauss_results[triangle_index].triangle_size = triangle_size;
        triangle_gauss_results[triangle_index].residual = RMSE_before_subdiv;
        triangle_gauss_results[triangle_index].parallel_flag = parallel_flag;
    }

    cv::Point2f mvd;
    int selected_index;
    MV_CODE_METHOD method_flag;

    if(cmt == nullptr) {
        cmt = previousMvList[0][triangle_index];
    }

//    std::cout << "gauss_result_parallel:" << gauss_result_parallel << std::endl;

    double cost_before_subdiv;
    std::tie(cost_before_subdiv, mvd, selected_index, method_flag) = getMVD({gauss_result_parallel,gauss_result_parallel,gauss_result_parallel}, RMSE_before_subdiv, triangle_index, cmt->mv1);

//    std::cout << "mvd result:" << mvd << std::endl;

    warp_p_image = ref_image.clone();
    residual_ref = cv::Mat::zeros(1920, 1024, CV_8UC1);

    std::vector<cv::Point2i> ret_gauss2;

    RMSE_before_subdiv = RMSE_before_subdiv / triangle_size;

    std::vector<cv::Point2f> mv;
    if(parallel_flag){
        mv.emplace_back(gauss_result_parallel);
        mv.emplace_back(gauss_result_parallel);
        mv.emplace_back(gauss_result_parallel);
    }else{
        mv = gauss_result_warping;
    }
    ctu->mv1 = mv[0];
    ctu->mv2 = mv[1];
    ctu->mv3 = mv[2];

    ctu->collocated_mv = cmt->mv1;
//    std::cout << ctu->collocated_mv << std::endl;

    SplitResult split_triangles = getSplitTriangle(p1, p2, p3, type);

    std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
    subdiv_ref_triangles.push_back(split_triangles.t1);
    subdiv_target_triangles.push_back(split_triangles.t1);
    subdiv_ref_triangles.push_back(split_triangles.t2);
    subdiv_target_triangles.push_back(split_triangles.t2);

    double RMSE_after_subdiv = 0.0;
    std::vector<GaussResult> split_mv_result(2);

    addCornerAndTriangle(Triangle(corner_flag[(int) p1.y][(int) p1.x], corner_flag[(int) p2.y][(int) p2.x],
                                  corner_flag[(int) p3.y][(int) p3.x]), triangle_index, type);

    #pragma omp parallel for
    for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
        double error_tmp;
        bool flag_tmp;
        int triangle_size_tmp;
        cv::Point2f mv_parallel_tmp;
        std::vector<cv::Point2f> mv_warping_tmp;
        std::tie(mv_warping_tmp, mv_parallel_tmp, error_tmp, triangle_size_tmp, flag_tmp) = GaussNewton(ref_image, target_image, gaussRefImage, subdiv_target_triangles[j]);
        split_mv_result[j] = GaussResult(mv_warping_tmp, mv_parallel_tmp, error_tmp, triangle_size_tmp, flag_tmp);
        RMSE_after_subdiv += error_tmp;
    }

    int triangle_indexes[] = {(int)triangles.size() - 2, (int)triangles.size() - 1};

    double cost_after_subdiv1;
    std::tie(cost_after_subdiv1, mvd, selected_index, method_flag) = getMVD(
            {split_mv_result[0].mv_parallel, split_mv_result[0].mv_parallel, split_mv_result[0].mv_parallel}, split_mv_result[0].residual,
            triangle_indexes[0], (cmt->leftNode != nullptr ? cmt->leftNode->mv1 : cmt->mv1));

    double cost_after_subdiv2;
    std::tie(cost_after_subdiv2, mvd, selected_index, method_flag) = getMVD(
            {split_mv_result[1].mv_parallel, split_mv_result[1].mv_parallel, split_mv_result[1].mv_parallel}, split_mv_result[1].residual,
            triangle_indexes[1], (cmt->rightNode != nullptr ? cmt->rightNode->mv1 : cmt->mv1));

    RMSE_after_subdiv /= (double) triangle_size;

    std::cout << "before:" << cost_before_subdiv << " after:" << (cost_after_subdiv1 + cost_after_subdiv2) << std::endl;
    if(cost_before_subdiv > (cost_after_subdiv1 + cost_after_subdiv2)) {
//        addCornerAndTriangle(Triangle(corner_flag[(int) p1.y][(int) p1.x], corner_flag[(int) p2.y][(int) p2.x],
//                                      corner_flag[(int) p3.y][(int) p3.x]), triangle_index, type);
        ctu->split_cu_flag1 = true;
        ctu->split_cu_flag2 = true;

        ctu->leftNode = new CodingTreeUnit();
        ctu->leftNode->parentNode = ctu;
        int t1_idx = triangles.size() - 2;
        triangle_gauss_results[t1_idx] = split_mv_result[0]; // TODO: warping対応

        isCodedTriangle[t1_idx] = true;
        bool result = split(gaussRefImage, ctu->leftNode, (cmt->leftNode != nullptr ? cmt->leftNode : cmt), split_triangles.t1, t1_idx,split_triangles.t1_type, steps - 1);
        if(result) {
            ctu->leftNode->split_cu_flag1 = true;
            ctu->leftNode->parentNode = ctu;
            ctu->leftNode->triangle_index = t1_idx;
            ctu->depth = divide_steps - steps;
        }

        ctu->rightNode = new CodingTreeUnit();
        ctu->rightNode->parentNode = ctu;
        int t2_idx = triangles.size() - 1;
        triangle_gauss_results[t2_idx] = split_mv_result[1];
        isCodedTriangle[t2_idx] = true;
        result = split(gaussRefImage, ctu->rightNode, (cmt->rightNode != nullptr ? cmt->rightNode : cmt), split_triangles.t2, t2_idx, split_triangles.t2_type, steps - 1);
        if(result) {
            ctu->rightNode->split_cu_flag2 = true;
            ctu->rightNode->parentNode = ctu;
            ctu->rightNode->triangle_index = t2_idx;
            ctu->depth = divide_steps - steps;
        }

        return true;
    }else{
        isCodedTriangle[triangle_index] = true;
        eraseTriangle(triangles.size() - 1);
        eraseTriangle(triangles.size() - 1);
        return false;
    }

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
            cv::Point2f x = (p2 - p1) / 2.0;
            cv::Point2f y = (p3 - p1) / 2.0;

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

            a = p1;
            b = p1 + x;
            c = p2;
            d = p3;

            return {Point3Vec(a, b, d), Point3Vec(b, c, d), TYPE3, TYPE1};
        }
        case DIVIDE::TYPE6:
        {
            cv::Point2f y = (p3 - p1) / 2.0;

            a = p1;
            b = p1 + y;
            c = p2;
            d = p3;

            return {Point3Vec(a, b, c), Point3Vec(b, c, d), TYPE4, TYPE1};
        }
        case DIVIDE::TYPE7:
        {
            cv::Point2f x = (p3 - p2) / 2.0;

            a = p1;
            b = p2;
            c = p2 + x;
            d = p3;

            return {Point3Vec(a, b, c), Point3Vec(a, c, d), TYPE2, TYPE4};
        }
        case DIVIDE::TYPE8:
        {
            cv::Point2f y = (p3 - p1) / 2.0;

            a = p2;
            b = p1;
            c = p1 + y;
            d = p3;

            return {Point3Vec(a, b, c), Point3Vec(a, c, d), TYPE2, TYPE3};
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
 * @fn cv::Point2f TriangleDivision::getCollocatedTriangleList(int t_idx)
 * @brief 時間予測したベクトル候補を返す
 * @param t_idx 三角パッチのインデックス
 * @return 整数の動きベクトルと小数部の動きベクトルのペア
 */
cv::Point2f TriangleDivision::getCollocatedTriangleList(CodingTreeUnit* unit) {
    CodingTreeUnit* tmp_unit = unit;

    if(tmp_unit == nullptr) {
        std::cout << "nullptr" << std::endl;
    }
    while(tmp_unit->parentNode != nullptr) tmp_unit = tmp_unit->parentNode;

    int root_triangle_idx = tmp_unit->triangle_index;

    std::vector<int> route = getDivideOrder(unit);
    std::cout << "root_idx:" << root_triangle_idx << std::endl;
    CollocatedMvTree* currentNode = previousMvList[0][root_triangle_idx];
    CollocatedMvTree* previousNode = currentNode;

    int depth = 2;

    if(route.empty()) return currentNode->mv1;

    for(int i = 0 ; i < depth || currentNode != nullptr ; i++){
        int direction = route[i];

        previousNode = currentNode;
        if(direction == 1) {
            currentNode = currentNode->rightNode;
        }else{
            currentNode = currentNode->leftNode;
        }
    }

    return previousNode->mv1;
}

/**
 * @fn bool TriangleDivision::isCTU(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3)
 * @brief CTU相当のパッチ（つまりは1回も分割されていないパッチ）であるかどうか調べる
 * @param[in] p1 頂点１の座標
 * @param[in] p2 頂点２の座標
 * @param[in] p3 頂点３の座標
 * @return CTUであるばあいtrue, それ以外はfalse
 */
bool TriangleDivision::isCTU(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3) {
    double x1 = fabs(p1.x - p2.x);
    double x2 = fabs(p2.x - p3.x);
    double x3 = fabs(p3.x - p1.x);
    double y1 = fabs(p1.y - p2.y);
    double y2 = fabs(p2.y - p3.y);
    double y3 = fabs(p3.y - p1.y);

    return (x1 == block_size_x || x2 == block_size_x || x3 == block_size_x) && (y1 == block_size_y || y2 == block_size_y || y3 == block_size_y);
}

/**
 * @fn std::vector<int> TriangleDivision::getDivideOrder(CodingTreeUnit* currentNode)
 * @brief 木をたどって分割の順番を調べて返す
 * @param currentNode 調べたいノード
 * @return 1なら右、0なら左を表すvectorを返す
 */
std::vector<int> TriangleDivision::getDivideOrder(CodingTreeUnit* currentNode){
    if(currentNode->parentNode == nullptr) return std::vector<int>(); // ルートノード

    std::vector<int> route;
    while(currentNode->parentNode != nullptr) {
        if(currentNode->parentNode->rightNode != nullptr) {
            route.emplace_back(1); // right-node
        }else{
            route.emplace_back(0); // left-node
        }
        currentNode = currentNode->parentNode;
    }

    std::reverse(route.begin(), route.end());
    for(auto idx : route) std::cout << idx << " ";
    puts("");

    return route;
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

        auto* left = new CollocatedMvTree();
        left->mv1 = cv::Point2f(0, 0);
        left->mv2 = cv::Point2f(0, 0);
        left->mv3 = cv::Point2f(0, 0);

        left->leftNode = new CollocatedMvTree();
        left->leftNode->mv1 = cv::Point2f(0, 0);
        left->leftNode->mv2 = cv::Point2f(0, 0);
        left->leftNode->mv3 = cv::Point2f(0, 0);
//        left->leftNode->leftNode = new CollocatedMvTree();
//        left->leftNode->leftNode->mv1 = cv::Point2f(10, 10);
//        left->leftNode->leftNode->mv2 = cv::Point2f(10, 10);
//        left->leftNode->leftNode->mv3 = cv::Point2f(10, 10);
//        left->leftNode->rightNode = new CollocatedMvTree();
//        left->leftNode->rightNode->mv1 = cv::Point2f(1, 1);
//        left->leftNode->rightNode->mv2 = cv::Point2f(1, 1);
//        left->leftNode->rightNode->mv3 = cv::Point2f(1, 1);

        left->rightNode = new CollocatedMvTree();
        left->rightNode->mv1 = cv::Point2f(0, 0);
        left->rightNode->mv2 = cv::Point2f(0, 0);
        left->rightNode->mv3 = cv::Point2f(0, 0);
//        left->rightNode->leftNode = new CollocatedMvTree();
//        left->rightNode->leftNode->mv1 = cv::Point2f(10, 10);
//        left->rightNode->leftNode->mv2 = cv::Point2f(10, 10);
//        left->rightNode->leftNode->mv3 = cv::Point2f(10, 10);
//        left->rightNode->rightNode = new CollocatedMvTree();
//        left->rightNode->rightNode->mv1 = cv::Point2f(1, 1);
//        left->rightNode->rightNode->mv2 = cv::Point2f(1, 1);
//        left->rightNode->rightNode->mv3 = cv::Point2f(1, 1);
        previousMvList[pic_num][i]->leftNode = left;

        auto* right = new CollocatedMvTree();
        right->mv1 = cv::Point2f(0, 0);
        right->mv2 = cv::Point2f(0, 0);
        right->mv3 = cv::Point2f(0, 0);

        right->leftNode = new CollocatedMvTree();
        right->leftNode->mv1 = cv::Point2f(0, 0);
        right->leftNode->mv2 = cv::Point2f(0, 0);
        right->leftNode->mv3 = cv::Point2f(0, 0);
//        right->leftNode->leftNode = new CollocatedMvTree();
//        right->leftNode->leftNode->mv1 = cv::Point2f(10, 10);
//        right->leftNode->leftNode->mv2 = cv::Point2f(10, 10);
//        right->leftNode->leftNode->mv3 = cv::Point2f(10, 10);
//        right->leftNode->rightNode = new CollocatedMvTree();
//        right->leftNode->rightNode->mv1 = cv::Point2f(1, 1);
//        right->leftNode->rightNode->mv2 = cv::Point2f(1, 1);
//        right->leftNode->rightNode->mv3 = cv::Point2f(1, 1);

        right->rightNode = new CollocatedMvTree();
        right->rightNode->mv1 = cv::Point2f(0, 0);
        right->rightNode->mv2 = cv::Point2f(0, 0);
        right->rightNode->mv3 = cv::Point2f(0, 0);
//        right->rightNode->leftNode = new CollocatedMvTree();
//        right->rightNode->leftNode->mv1 = cv::Point2f(10, 10);
//        right->rightNode->leftNode->mv2 = cv::Point2f(10, 10);
//        right->rightNode->leftNode->mv3 = cv::Point2f(10, 10);
//        right->rightNode->rightNode = new CollocatedMvTree();
//        right->rightNode->rightNode->mv1 = cv::Point2f(1, 1);
//        right->rightNode->rightNode->mv2 = cv::Point2f(1, 1);
//        right->rightNode->rightNode->mv3 = cv::Point2f(1, 1);
        previousMvList[pic_num][i]->rightNode = right;
    }


//    for(int i = 0 ; i < 10 ; i++) {
//        constructPreviousCodingTree(trees[i], previousMvList[pic_num][i]);
//    }
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

    if(codingTree->rightNode != nullptr) {
        constructedTree->rightNode = new CollocatedMvTree();
        constructPreviousCodingTree(codingTree->rightNode, constructedTree->rightNode);
    }
    if(codingTree->leftNode != nullptr) {
        constructedTree->leftNode = new CollocatedMvTree();
        constructPreviousCodingTree(codingTree->leftNode, constructedTree->leftNode);
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
        if(std::get<0>(vector) == mv) {
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
 * @return 差分ベクトル，参照したパッチ，空間or時間のフラグのtuple
 */
std::tuple<double, cv::Point2f, int, MV_CODE_METHOD> TriangleDivision::getMVD(std::vector<cv::Point2f> mv, double residual, int triangle_idx, cv::Point2f &collocated_mv){
    std::cout << "triangle_index(getMVD):" << triangle_idx << std::endl;
    // 空間予測と時間予測の候補を取り出す
    std::vector<int> spatial_triangles = getSpatialTriangleList(triangle_idx);

    int spatial_triangle_size = static_cast<int>(spatial_triangles.size());
    std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> vectors;

    // すべてのベクトルを格納する．
    for(int i = 0 ; i < spatial_triangle_size ; i++) {
        // TODO: これ平行移動のみしか対応してないがどうする…？
        if(!isMvExists(vectors, triangle_gauss_results[spatial_triangles[i]].mv_parallel)) {
            // とりあえず平行移動のみ考慮
            vectors.emplace_back(triangle_gauss_results[spatial_triangles[i]].mv_parallel, SPATIAL);
        }
    }

    if(!isMvExists(vectors, collocated_mv)) vectors.emplace_back(collocated_mv, SPATIAL);

    if(vectors.size() < 2) vectors.emplace_back(cv::Point2f(0.0, 0.0), Collocated);

    double lambda = getLambdaPred(qp);

    //                      コスト, 差分ベクトル, 番号, タイプ
    std::vector<std::tuple<double, cv::Point2f, int, MV_CODE_METHOD> > results;
    for(int i = 0 ; i < vectors.size() ; i++) {
        std::pair<cv::Point2f, MV_CODE_METHOD> vector = vectors[i];
        cv::Point2f current_mv = vector.first;
        std::cout << "current_mv:" << current_mv << std::endl;
        cv::Point2f mvd = current_mv - mv[0];

        mvd = getQuantizedMv(mvd, 4);
        mvd *= 4;

        /* 動きベクトル符号化 */

        // 動きベクトル差分の絶対値が0より大きいのか？
        bool is_x_greater_than_zero = mvd.x > 0 ? true : false;
        bool is_y_greater_than_zero = mvd.y > 0 ? true : false;

        // 動きベクトル差分の絶対値が1より大きいのか？
        bool is_x_greater_than_one = mvd.x > 1 ? true : false;
        bool is_y_greater_than_one = mvd.y > 1 ? true : false;

        // 正負の判定
        bool is_x_minus = mvd.x < 0 ? true : false;
        bool is_y_minus = mvd.y < 0 ? true : false;

        // 動きベクトル差分から2を引いたろ！
        int mvd_x_minus_2 = mvd.x - 2;
        int mvd_y_minus_2 = mvd.y - 2;

        int mvd_code_length = getExponentialGolombCodeLength((int)mvd_x_minus_2, 0) + getExponentialGolombCodeLength((int)mvd_y_minus_2, 0);

        // 参照箇所符号化
        int reference_index = std::get<1>(vector);
        int reference_index_code_length = getUnaryCodeLength(reference_index);

        // 各種フラグ分を(3*2)bit足してます
        double rd = residual + lambda * (mvd_code_length + reference_index_code_length + 6);

        // 結果に入れる
        results.emplace_back(rd, mvd, i, vector.second);
    }

    // マージ符号化
    // マージで参照する動きベクトルを使って残差を求め直す
    Triangle current_triangle_coordinate = triangles[triangle_idx].first;
    cv::Point2f p1 = corners[current_triangle_coordinate.p1_idx];
    cv::Point2f p2 = corners[current_triangle_coordinate.p2_idx];
    cv::Point2f p3 = corners[current_triangle_coordinate.p3_idx];
    Point3Vec coordinate = Point3Vec(p1, p2, p3);

    vectors.clear();
    for(int i = 0 ; i < spatial_triangle_size ; i++) {
        // TODO: これ平行移動のみしか対応してないがどうする…？
        if(!isMvExists(vectors, mv[0])) {
            vectors.emplace_back(mv[0], MERGE);
            double ret_residual = getTriangleResidual(ref_image, target_image, coordinate, mv);
            double rd = ret_residual + lambda * (getUnaryCodeLength(i));
            results.emplace_back(rd, cv::Point2f(0, 0), results.size(), MERGE);
        }
    }

    // RDしたスコアが小さい順にソート
    std::sort(results.begin(), results.end(), [](const std::tuple<double, cv::Point2f, int, MV_CODE_METHOD >& a, const std::tuple<double, cv::Point2f, int, MV_CODE_METHOD>& b){
        return std::get<0>(a) < std::get<0>(b);
    });

    double cost = std::get<0>(results[0]);
    cv::Point2f mvd = std::get<1>(results[0]);
    int selected_idx = std::get<2>(results[0]);
    MV_CODE_METHOD method = std::get<3>(results[0]);

    return {cost, mvd, selected_idx, method};
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

TriangleDivision::SplitResult::SplitResult(const Point3Vec &t1, const Point3Vec &t2, int t1Type, int t2Type) : t1(t1),
                                                                                                               t2(t2),
                                                                                                               t1_type(t1Type),
                                                                                                               t2_type(t2Type) {}

TriangleDivision::GaussResult::GaussResult(const std::vector<cv::Point2f> &mvWarping, const cv::Point2f &mvParallel,
                                           double residual, int triangleSize, bool parallelFlag) : mv_warping(
        mvWarping), mv_parallel(mvParallel), residual(residual), triangle_size(triangleSize), parallel_flag(
        parallelFlag) {}

TriangleDivision::GaussResult::GaussResult() {}
