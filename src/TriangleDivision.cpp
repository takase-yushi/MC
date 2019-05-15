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
#include <set>
#include <vector>
#include <utility>
#include <algorithm>
#include <queue>
#include <opencv2/imgcodecs.hpp>

TriangleDivision::TriangleDivision(const cv::Mat &refImage, const cv::Mat &targetImage) : target_image(targetImage),
                                                                                          ref_image(refImage) {}



TriangleDivision::GaussResult::GaussResult(int triangleIndex, const Triangle &triangle, int type,
                                           double rmseBeforeSubdiv, double rmseAfterSubdiv) : triangle_index(
        triangleIndex), triangle(triangle), type(type), RMSE_before_subdiv(rmseBeforeSubdiv), RMSE_after_subdiv(
        rmseAfterSubdiv) {}

TriangleDivision::GaussResult::GaussResult(): triangle(Triangle(-1, -1, -1)) {}


/**
 * @fn void TriangleDivision::initTriangle(int block_size_x, int block_size_y, int _divide_steps, int divide_flag)
 * @brief 三角形を初期化する
 * @param[in] block_size_x
 * @param[in] block_size_y
 * @param[in] divide_flag
 */
void TriangleDivision::initTriangle(int _block_size_x, int _block_size_y, int _divide_steps, int divide_flag) {
    block_size_x = _block_size_x;
    block_size_y = _block_size_y;
    int block_num_x = target_image.cols / block_size_x;
    int block_num_y = target_image.rows / block_size_y;
    divide_steps = _divide_steps;

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

    corner_flag.resize(ref_image.rows);
    for(int i = 0 ; i < ref_image.rows ; i++) {
        corner_flag[i].resize(ref_image.cols);
    }

    for(int y = 0 ; y < ref_image.rows ; y++) {
        for(int x = 0 ; x < ref_image.cols ; x++) {
            corner_flag[y][x] = -1;
        }
    }

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
            corner_flag[ny][nx] = corners.size() - 1;
            neighbor_vtx.emplace_back();
        }
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

    Triangle triangle(v[0].second, v[1].second, v[2].second, triangles.size());

    triangles.emplace_back(triangle, type);
    covered_triangle.emplace_back();
    isCodedTriangle.emplace_back(true);

    return static_cast<int>(triangles.size() - 1);
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
    std::vector<int> v;

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
    std::vector<cv::Point2f> v;

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
    std::vector<Point3Vec> v;

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
 * @fn void TriangleDivision::subdivision()
 * @brief 三角パッチの再分割を行う
 * @param[in] gaussRefImage ガウスニュートン法の第1層目の画像
 * @param[in] steps 何回分割するかを示す値
 */
void TriangleDivision::subdivision(cv::Mat gaussRefImage, int steps) {

    if(steps == 0) return;

    // 一つ前に分割されている場合、更に分割すればよいが
    // 分割されていないのであればこれ以上分割する必要はない
    std::vector<bool> previousDivideFlag(triangles.size(), true);

    std::vector<CodingTreeUnit> ctu(triangles.size());

    // ctuIndexMapper[index] := index番目の三角形のCTUを表す
    std::vector<int> ctuIndexMapper(triangles.size());

    for(int i = 0 ; i < triangles.size() ; i++) {
        ctuIndexMapper[i] = i;
    }

    for(int step = 0 ; step < steps ; step++) {
        std::vector<GaussResult> results(triangles.size());
        previousDivideFlag.resize(triangles.size());
        delete_flag.resize(triangles.size());

        int denominator = triangles.size();
        int numerator = 0;

#pragma omp parallel for
        for (int i = 0; i < (int) triangles.size(); i++) {
            // 一つ前のステップで分割されてないならばやる必要はない
            if(!previousDivideFlag[i]) continue;

            std::pair<Triangle, int> triangle = triangles[i];

            double RMSE_before_subdiv = 0.0;
            cv::Point2f p1 = corners[triangle.first.p1_idx];
            cv::Point2f p2 = corners[triangle.first.p2_idx];
            cv::Point2f p3 = corners[triangle.first.p3_idx];

            Point3Vec refTriangle(p1, p2, p3);
            Point3Vec targetTriangle(p1, p2, p3);
            int triangle_size = 0;

            RMSE_before_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, targetTriangle, refTriangle,
                                               triangle_size);
            RMSE_before_subdiv /= triangle_size;

            double RMSE_after_subdiv = 0.0;
            triangle_size = 0;
            int triangle_size_sum = 0;

            switch(triangle.second) {
                case DIVIDE::TYPE1:
                {
                    cv::Point2f p1 = corners[triangle.first.p1_idx];
                    cv::Point2f p2 = corners[triangle.first.p2_idx];
                    cv::Point2f p3 = corners[triangle.first.p3_idx];

                    cv::Point2f x = (p2 - p1) / 2.0;
                    cv::Point2f y = (p3 - p1) / 2.0;

                    cv::Point2f a = p1;
                    cv::Point2f b = p2;
                    cv::Point2f c = a + x + y;
                    cv::Point2f d = p3;

                    std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
                    subdiv_ref_triangles.emplace_back(a, b, c);
                    subdiv_target_triangles.emplace_back(a, b, c);
                    subdiv_ref_triangles.emplace_back(a, c, d);
                    subdiv_target_triangles.emplace_back(a, c, d);

                    for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
                        RMSE_after_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, subdiv_target_triangles[j],
                                                          subdiv_ref_triangles[j], triangle_size);
                        triangle_size_sum += triangle_size;
                    }

                    RMSE_after_subdiv /= (double) triangle_size_sum;
                    results[i] = GaussResult(i, triangle.first, triangle.second, RMSE_before_subdiv, RMSE_after_subdiv);
                }
                    break;
                case TYPE2:
                {
                    cv::Point2f p1 = corners[triangle.first.p1_idx];
                    cv::Point2f p2 = corners[triangle.first.p2_idx];
                    cv::Point2f p3 = corners[triangle.first.p3_idx];

                    cv::Point2f x = (p2 - p3) / 2.0;
                    cv::Point2f y = (p1 - p3) / 2.0;

                    cv::Point2f a = p1;
                    cv::Point2f b = p3 + x + y;
                    cv::Point2f c = p2;
                    cv::Point2f d = p3;

                    std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
                    subdiv_ref_triangles.emplace_back(a, b, d);
                    subdiv_target_triangles.emplace_back(a, b, d);
                    subdiv_ref_triangles.emplace_back(b, c, d);
                    subdiv_target_triangles.emplace_back(b, c, d);

                    for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
                        RMSE_after_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, subdiv_target_triangles[j],
                                                          subdiv_ref_triangles[j], triangle_size);
                        triangle_size_sum += triangle_size;
                    }

                    RMSE_after_subdiv /= (double) triangle_size_sum;
                    results[i] = GaussResult(i, triangle.first, triangle.second, RMSE_before_subdiv, RMSE_after_subdiv);
                }
                    break;
                case TYPE3:
                {
                    cv::Point2f p1 = corners[triangle.first.p1_idx];
                    cv::Point2f p2 = corners[triangle.first.p2_idx];
                    cv::Point2f p3 = corners[triangle.first.p3_idx];

                    cv::Point2f x = (p1 - p2) / 2.0;
                    cv::Point2f y = (p3 - p2) / 2.0;

                    cv::Point2f a = p1;
                    cv::Point2f b = p2;
                    cv::Point2f c = p2 + x + y;
                    cv::Point2f d = p3;

                    std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
                    subdiv_ref_triangles.emplace_back(a, b, c);
                    subdiv_target_triangles.emplace_back(a, b, c);
                    subdiv_ref_triangles.emplace_back(b, c, d);
                    subdiv_target_triangles.emplace_back(b, c, d);

                    for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
                        RMSE_after_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, subdiv_target_triangles[j],
                                                          subdiv_ref_triangles[j], triangle_size);
                        triangle_size_sum += triangle_size;
                    }

                    RMSE_after_subdiv /= (double) triangle_size_sum;
                    results[i] = GaussResult(i, triangle.first, triangle.second, RMSE_before_subdiv, RMSE_after_subdiv);
                }
                    break;
                case TYPE4:
                {
                    cv::Point2f p1 = corners[triangle.first.p1_idx];
                    cv::Point2f p2 = corners[triangle.first.p2_idx];
                    cv::Point2f p3 = corners[triangle.first.p3_idx];

                    cv::Point2f x = (p3 - p2) / 2.0;
                    cv::Point2f y = (p1 - p2) / 2.0;

                    cv::Point2f a = p1;
                    cv::Point2f b = p2 + x + y;
                    cv::Point2f c = p2;
                    cv::Point2f d = p3;

                    std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
                    subdiv_ref_triangles.emplace_back(a, b, c);
                    subdiv_target_triangles.emplace_back(a, b, c);
                    subdiv_ref_triangles.emplace_back(b, c, d);
                    subdiv_target_triangles.emplace_back(b, c, d);

                    for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
                        RMSE_after_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, subdiv_target_triangles[j],
                                                          subdiv_ref_triangles[j], triangle_size);
                        triangle_size_sum += triangle_size;
                    }

                    RMSE_after_subdiv /= (double) triangle_size_sum;
                    results[i] = GaussResult(i, triangle.first, triangle.second, RMSE_before_subdiv, RMSE_after_subdiv);
                }
                    break;
                case DIVIDE::TYPE5:
                {
                    cv::Point2f p1 = corners[triangle.first.p1_idx];
                    cv::Point2f p2 = corners[triangle.first.p2_idx];
                    cv::Point2f p3 = corners[triangle.first.p3_idx];

                    cv::Point2f x = (p2 - p1) / 2.0;

                    cv::Point2f a = p1;
                    cv::Point2f b = p1 + x;
                    cv::Point2f c = p2;
                    cv::Point2f d = p3;

                    std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
                    subdiv_ref_triangles.emplace_back(a, b, d);
                    subdiv_target_triangles.emplace_back(a, b, d);
                    subdiv_ref_triangles.emplace_back(b, c, d);
                    subdiv_target_triangles.emplace_back(b, c, d);

                    for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
                        RMSE_after_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, subdiv_target_triangles[j],
                                                          subdiv_ref_triangles[j], triangle_size);
                        triangle_size_sum += triangle_size;
                    }

                    RMSE_after_subdiv /= (double) triangle_size_sum;
                    results[i] = GaussResult(i, triangle.first, triangle.second, RMSE_before_subdiv, RMSE_after_subdiv);
                }
                    break;
                case DIVIDE::TYPE6:
                {
                    cv::Point2f p1 = corners[triangle.first.p1_idx];
                    cv::Point2f p2 = corners[triangle.first.p2_idx];
                    cv::Point2f p3 = corners[triangle.first.p3_idx];

                    cv::Point2f y = (p3 - p1) / 2.0;

                    cv::Point2f a = p1;
                    cv::Point2f b = p1 + y;
                    cv::Point2f c = p2;
                    cv::Point2f d = p3;

                    std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
                    subdiv_ref_triangles.emplace_back(a, b, c);
                    subdiv_target_triangles.emplace_back(a, b, c);
                    subdiv_ref_triangles.emplace_back(b, c, d);
                    subdiv_target_triangles.emplace_back(b, c, d);

                    for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
                        RMSE_after_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, subdiv_target_triangles[j],
                                                          subdiv_ref_triangles[j], triangle_size);
                        triangle_size_sum += triangle_size;
                    }

                    RMSE_after_subdiv /= (double) triangle_size_sum;
                    results[i] = GaussResult(i, triangle.first, triangle.second, RMSE_before_subdiv, RMSE_after_subdiv);
                }
                    break;
                case DIVIDE::TYPE7:
                {
                    cv::Point2f p1 = corners[triangle.first.p1_idx];
                    cv::Point2f p2 = corners[triangle.first.p2_idx];
                    cv::Point2f p3 = corners[triangle.first.p3_idx];

                    cv::Point2f x = (p3 - p2) / 2.0;

                    cv::Point2f a = p1;
                    cv::Point2f b = p2;
                    cv::Point2f c = p2 + x;
                    cv::Point2f d = p3;

                    std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
                    subdiv_ref_triangles.emplace_back(a, b, c);
                    subdiv_target_triangles.emplace_back(a, b, c);
                    subdiv_ref_triangles.emplace_back(a, c, d);
                    subdiv_target_triangles.emplace_back(a, c, d);

                    for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
                        RMSE_after_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, subdiv_target_triangles[j],
                                                          subdiv_ref_triangles[j], triangle_size);
                        triangle_size_sum += triangle_size;
                    }

                    RMSE_after_subdiv /= (double) triangle_size_sum;
                    results[i] = GaussResult(i, triangle.first, triangle.second, RMSE_before_subdiv, RMSE_after_subdiv);
                }
                    break;
                case DIVIDE::TYPE8:
                {
                    cv::Point2f p1 = corners[triangle.first.p1_idx];
                    cv::Point2f p2 = corners[triangle.first.p2_idx];
                    cv::Point2f p3 = corners[triangle.first.p3_idx];

                    cv::Point2f y = (p3 - p1) / 2.0;

                    cv::Point2f a = p2;
                    cv::Point2f b = p1;
                    cv::Point2f c = p1 + y;
                    cv::Point2f d = p3;

                    std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
                    subdiv_ref_triangles.emplace_back(a, b, c);
                    subdiv_target_triangles.emplace_back(a, b, c);
                    subdiv_ref_triangles.emplace_back(a, c, d);
                    subdiv_target_triangles.emplace_back(a, c, d);

                    for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
                        RMSE_after_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, subdiv_target_triangles[j],
                                                          subdiv_ref_triangles[j], triangle_size);
                        triangle_size_sum += triangle_size;
                    }

                    RMSE_after_subdiv /= (double) triangle_size_sum;
                    results[i] = GaussResult(i, triangle.first, triangle.second, RMSE_before_subdiv, RMSE_after_subdiv);
                }
                    break;
                default:
                    break;
            }

            numerator++;
            std::cout << numerator << "/" << denominator << std::endl;
        }

        numerator = 0;
        const double divide_th = 0.04;

        // Queueが空になるまで続ける
        for (int i = 0; i < (int) results.size(); i++) {
            // 一つ前のステップで分割されてないならばやる必要はない
            if(!previousDivideFlag[i]) continue;

            double diff = (results[i].RMSE_before_subdiv - results[i].RMSE_after_subdiv) / results[i].RMSE_before_subdiv;

            // caseがなにの形に相当しているかはTriangleDivision.hを見るとわかります
            if(diff > divide_th) {
                switch(results[i].type) {
                    case DIVIDE::TYPE1:
                    {
                        Triangle triangle = results[i].triangle;
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
                        removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, results[i].triangle_index);

                        addNeighborVertex(a_idx, b_idx, c_idx);
                        addNeighborVertex(a_idx, c_idx, d_idx);

                        covered_triangle.emplace_back();
                        covered_triangle.emplace_back();
                        addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
                        addCoveredTriangle(a_idx, c_idx, d_idx, t2_idx);

                        previousDivideFlag[results[i].triangle_index] = false;
                        previousDivideFlag.emplace_back(true);
                        previousDivideFlag.emplace_back(true);

                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper[t1_idx] = ctuIndexMapper[t2_idx] = ctuIndexMapper[results[i].triangle_index];

                    }
                        break;
                    case DIVIDE::TYPE2:
                    {
                        Triangle triangle = results[i].triangle;
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
                        removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, results[i].triangle_index);

                        addNeighborVertex(a_idx, b_idx, d_idx);
                        addNeighborVertex(b_idx, c_idx, d_idx);

                        covered_triangle.emplace_back();
                        covered_triangle.emplace_back();
                        addCoveredTriangle(a_idx, b_idx, d_idx, t1_idx);
                        addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);

                        previousDivideFlag[results[i].triangle_index] = false;
                        previousDivideFlag.emplace_back(true);
                        previousDivideFlag.emplace_back(true);

                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper[t1_idx] = ctuIndexMapper[t2_idx] = ctuIndexMapper[results[i].triangle_index];

                    }
                        break;
                    case DIVIDE::TYPE3:
                    {
                        Triangle triangle = results[i].triangle;
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
                        removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, results[i].triangle_index);

                        addNeighborVertex(a_idx, b_idx, c_idx);
                        addNeighborVertex(b_idx, c_idx, d_idx);

                        covered_triangle.emplace_back();
                        covered_triangle.emplace_back();
                        addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
                        addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);

                        previousDivideFlag[results[i].triangle_index] = false;
                        previousDivideFlag.emplace_back(true);
                        previousDivideFlag.emplace_back(true);

                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper[t1_idx] = ctuIndexMapper[t2_idx] = ctuIndexMapper[results[i].triangle_index];

                    }
                        break;
                    case DIVIDE::TYPE4:
                    {
                        Triangle triangle = results[i].triangle;
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
                        removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, results[i].triangle_index);

                        addNeighborVertex(a_idx, b_idx, c_idx);
                        addNeighborVertex(b_idx, c_idx, d_idx);

                        covered_triangle.emplace_back();
                        covered_triangle.emplace_back();
                        addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
                        addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);

                        previousDivideFlag[results[i].triangle_index] = false;
                        previousDivideFlag.emplace_back(true);
                        previousDivideFlag.emplace_back(true);

                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper[t1_idx] = ctuIndexMapper[t2_idx] = ctuIndexMapper[results[i].triangle_index];

                    }
                        break;
                    case DIVIDE::TYPE5:
                    {
                        Triangle triangle = results[i].triangle;
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
                        removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, results[i].triangle_index);

                        addNeighborVertex(a_idx, b_idx, d_idx);
                        addNeighborVertex(b_idx, c_idx, d_idx);

                        covered_triangle.emplace_back();
                        covered_triangle.emplace_back();
                        addCoveredTriangle(a_idx, b_idx, d_idx, t1_idx);
                        addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);

                        previousDivideFlag[results[i].triangle_index] = false;
                        previousDivideFlag.emplace_back(true);
                        previousDivideFlag.emplace_back(true);

                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper[t1_idx] = ctuIndexMapper[t2_idx] = ctuIndexMapper[results[i].triangle_index];

                    }
                        break;
                    case DIVIDE::TYPE6:
                    {
                        Triangle triangle = results[i].triangle;
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
                        removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, results[i].triangle_index);

                        addNeighborVertex(a_idx, b_idx, c_idx);
                        addNeighborVertex(b_idx, c_idx, d_idx);

                        covered_triangle.emplace_back();
                        covered_triangle.emplace_back();
                        addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
                        addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);

                        previousDivideFlag[results[i].triangle_index] = false;
                        previousDivideFlag.emplace_back(true);
                        previousDivideFlag.emplace_back(true);

                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper[t1_idx] = ctuIndexMapper[t2_idx] = ctuIndexMapper[results[i].triangle_index];

                    }
                        break;
                    case DIVIDE::TYPE7:
                    {
                        Triangle triangle = results[i].triangle;
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
                        removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, results[i].triangle_index);

                        addNeighborVertex(a_idx, b_idx, c_idx);
                        addNeighborVertex(a_idx, c_idx, d_idx);

                        covered_triangle.emplace_back();
                        covered_triangle.emplace_back();
                        addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
                        addCoveredTriangle(a_idx, c_idx, d_idx, t2_idx);

                        previousDivideFlag[results[i].triangle_index] = false;
                        previousDivideFlag.emplace_back(true);
                        previousDivideFlag.emplace_back(true);

                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper[t1_idx] = ctuIndexMapper[t2_idx] = ctuIndexMapper[results[i].triangle_index];

                    }
                        break;
                    case DIVIDE::TYPE8:
                    {
                        Triangle triangle = results[i].triangle;
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
                        removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, results[i].triangle_index);

                        addNeighborVertex(a_idx, b_idx, c_idx);
                        addNeighborVertex(a_idx, c_idx, d_idx);

                        covered_triangle.emplace_back();
                        covered_triangle.emplace_back();
                        addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
                        addCoveredTriangle(a_idx, c_idx, d_idx, t2_idx);

                        previousDivideFlag[results[i].triangle_index] = false;
                        previousDivideFlag.emplace_back(true);
                        previousDivideFlag.emplace_back(true);

                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper.emplace_back();
                        ctuIndexMapper[t1_idx] = ctuIndexMapper[t2_idx] = ctuIndexMapper[results[i].triangle_index];

                    }
                        break;
                    default:
                        break;
                }

                delete_flag[results[i].triangle_index] = true;
            }else{
                previousDivideFlag[results[i].triangle_index] = false;
            }

            numerator++;
            std::cout << "subdiv: "<< numerator << "/" << denominator << std::endl;
        }
    }

}

bool TriangleDivision::split(cv::Mat &gaussRefImage, CodingTreeUnit* ctu, Point3Vec triangle, int triangle_index, int type, int steps) {
    if(steps == 0) return false;

    double RMSE_before_subdiv = 0.0;
    cv::Point2f p1 = triangle.p1;
    cv::Point2f p2 = triangle.p2;
    cv::Point2f p3 = triangle.p3;

    Point3Vec refTriangle(p1, p2, p3);
    Point3Vec targetTriangle(p1, p2, p3);
    int triangle_size = 0;
    std::vector<cv::Mat> predicted_buf; //
    double error = 0.0;
    bool parallel_flag;
    int num;
    cv::Mat warp_p_image, residual_ref;

    std::vector<cv::Point2i> mv_parallel = Gauss_Newton2(ref_image, target_image, gaussRefImage, predicted_buf, warp_p_image, warp_p_image, error, targetTriangle, refTriangle, &parallel_flag, num, residual_ref, triangle_size, 0.4);

    RMSE_before_subdiv = error / triangle_size;

    SplitResult split_triangles = getSplitTriangle(p1, p2, p3, type);

    std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
    subdiv_ref_triangles.push_back(split_triangles.t1);
    subdiv_target_triangles.push_back(split_triangles.t1);
    subdiv_ref_triangles.push_back(split_triangles.t2);
    subdiv_target_triangles.push_back(split_triangles.t2);

    double RMSE_after_subdiv = 0.0;
    int triangle_size_sum = 0;

    std::vector<std::vector<cv::Point2i> > split_mv_result;
#pragma omp parallel for
    for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
        mv_parallel = Gauss_Newton2(ref_image, target_image, gaussRefImage, predicted_buf, warp_p_image, warp_p_image, error, subdiv_target_triangles[j], subdiv_ref_triangles[j], &parallel_flag, num, residual_ref, triangle_size, 0.4);
        split_mv_result.emplace_back(mv_parallel);
        RMSE_after_subdiv += error;
        triangle_size_sum += triangle_size;
    }

    RMSE_after_subdiv /= (double) triangle_size_sum;

    if((RMSE_before_subdiv - RMSE_after_subdiv) / RMSE_before_subdiv >= 0.04) {
        addCornerAndTriangle(Triangle(corner_flag[(int) p1.y][(int) p1.x], corner_flag[(int) p2.y][(int) p2.x],
                                      corner_flag[(int) p3.y][(int) p3.x]), triangle_index, type);

        ctu->split_cu_flag1 = true;
        ctu->split_cu_flag2 = true;
        ctu->parentNode = ctu;

        ctu->leftNode = new CodingTreeUnit();
        int t1_idx = triangles.size() - 2;
        bool ret = split(gaussRefImage, ctu->leftNode, split_triangles.t1, t1_idx,split_triangles.t1_type, steps - 1);
        if(ret) {
            ctu->leftNode->split_cu_flag1 = true;
        }

        ctu->rightNode = new CodingTreeUnit();
        int t2_idx = triangles.size() - 1;
        ret = split(gaussRefImage, ctu->rightNode, split_triangles.t2, t2_idx, split_triangles.t2_type, steps - 1);
        if(ret) {
            ctu->rightNode->split_cu_flag2 = true;
        }

        return true;
    }else{
        isCodedTriangle[triangle_index] = true;
        return false;
    }

}

TriangleDivision::SplitResult TriangleDivision::getSplitTriangle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, int type){
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

            return SplitResult(Point3Vec(a, b, c), Point3Vec(a, c, d), TYPE5, TYPE6);
        }
        case DIVIDE::TYPE2:
        {
            cv::Point2f x = (p2 - p3) / 2.0;
            cv::Point2f y = (p1 - p3) / 2.0;

            a = p1;
            b = p3 + x + y;
            c = p2;
            d = p3;

            return SplitResult(Point3Vec(a, b, d), Point3Vec(b, c, d), TYPE8, TYPE7);
        }
        case DIVIDE::TYPE3:
        {
            cv::Point2f x = (p1 - p2) / 2.0;
            cv::Point2f y = (p3 - p2) / 2.0;

            a = p1;
            b = p2;
            c = p2 + x + y;
            d = p3;

            return SplitResult(Point3Vec(a, b, c), Point3Vec(b, c, d), TYPE5, TYPE8);
        }
        case DIVIDE::TYPE4:
        {
            cv::Point2f x = (p3 - p2) / 2.0;
            cv::Point2f y = (p1 - p2) / 2.0;

            a = p1;
            b = p2 + x + y;
            c = p2;
            d = p3;

            return SplitResult(Point3Vec(a, b, c), Point3Vec(b, c, d), TYPE6, TYPE7);
        }
        case DIVIDE::TYPE5:
        {
            cv::Point2f x = (p2 - p1) / 2.0;

            a = p1;
            b = p1 + x;
            c = p2;
            d = p3;

            return SplitResult(Point3Vec(a, b, d), Point3Vec(b, c, d), TYPE3, TYPE1);
        }
        case DIVIDE::TYPE6:
        {
            cv::Point2f y = (p3 - p1) / 2.0;

            a = p1;
            b = p1 + y;
            c = p2;
            d = p3;

            return SplitResult(Point3Vec(a, b, c), Point3Vec(b, c, d), TYPE4, TYPE1);
        }
        case DIVIDE::TYPE7:
        {
            cv::Point2f x = (p3 - p2) / 2.0;

            a = p1;
            b = p2;
            c = p2 + x;
            d = p3;

            return SplitResult(Point3Vec(a, b, c), Point3Vec(a, c, d), TYPE2, TYPE4);
        }
        case DIVIDE::TYPE8:
        {
            cv::Point2f y = (p3 - p1) / 2.0;

            a = p2;
            b = p1;
            c = p1 + y;
            d = p3;

            return SplitResult(Point3Vec(a, b, c), Point3Vec(a, c, d), TYPE2, TYPE3);
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

    for(auto idx : list1) if(isCodedTriangle[idx]) mutualIndexSet1.emplace(idx);
    for(auto idx : list2) if(isCodedTriangle[idx]) mutualIndexSet2.emplace(idx);
    for(auto idx : list3) if(isCodedTriangle[idx]) mutualIndexSet3.emplace(idx);

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
 * @fn std::vector<int> TriangleDivision::getCollocatedTriangleList(int t_idx)
 * @brief 時間予測したベクトル候補を返す
 * @param t_idx 三角パッチのインデックス
 * @return
 */
std::vector<int> TriangleDivision::getCollocatedTriangleList(int t_idx) {

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
    }

    std::reverse(route.begin(), route.end());
    return route;
}

/**
 * @fn void TriangleDivision::constructPreviousCodingTree(std::vector<CollocatedMvTree*> trees)
 * @param trees
 */
void TriangleDivision::constructPreviousCodingTree(std::vector<CollocatedMvTree*> trees) {

}

TriangleDivision::SplitResult::SplitResult(const Point3Vec &t1, const Point3Vec &t2, int t1Type, int t2Type) : t1(t1),
                                                                                                               t2(t2),
                                                                                                               t1_type(t1Type),
                                                                                                               t2_type(t2Type) {}