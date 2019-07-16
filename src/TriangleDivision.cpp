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
    for(int block_y = 0 ; block_y <= block_num_y ; block_y++) {
        for (int block_x = 0 ; block_x <= block_num_x; block_x++) {
            int nx = block_x * (block_size_x);
            int ny = block_y * (block_size_y);

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

            nx = (block_x + 1) * (block_size_x) - 1;
            ny = (block_y) * (block_size_y);

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

        if(block_y == block_num_y) continue;

        for (int block_x = 0 ; block_x <= block_num_x; block_x++) {
            int nx = block_x * (block_size_x);
            int ny = (block_y + 1) * (block_size_y) - 1;

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

            nx = (block_x + 1) * (block_size_x) - 1;
            ny = (block_y + 1) * (block_size_y) - 1;

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
        node->leftNode = node->rightNode = nullptr;
        node->mv1 = cv::Point2f(0.0, 0.0);
        node->mv2 = cv::Point2f(0.0, 0.0);
        node->mv3 = cv::Point2f(0.0, 0.0);
    }

    std::cout << "block_num_y:" << block_num_y << std::endl;
    std::cout << "block_num_x:" << block_num_x << std::endl;

    covered_triangle.resize(static_cast<unsigned long>((block_num_x * 2 + 1) * (block_num_y * 2 + 1)));

    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {
        for(int block_x = 0 ; block_x < block_num_x ; block_x++) {
            int p1_idx;
            int p2_idx;
            int p3_idx;
            int p4_idx;
            if(divide_flag == LEFT_DIVIDE) {
                p1_idx = 2 * block_x + (2 * block_y) * ((block_num_x) * 2 + 1);
                p2_idx = p1_idx + 1;
                p3_idx = p1_idx + ((block_num_x) * 2 + 1 );

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
    covered_triangle.emplace_back();
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
    covered_triangle.erase(covered_triangle.begin() + t_idx);
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
    if(corner_flag[(int)(p.y * 2)][(int)(p.x * 2)] != -1) return corner_flag[(int)(p.y * 2)][(int)(p.x * 2)];
    corners.emplace_back(p);
    neighbor_vtx.emplace_back();
    corner_flag[(int)(p.y * 2)][(int)(p.x * 2)] = static_cast<int>(corners.size() - 1);
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

            int b1_idx = addCorner(b1);
            int b2_idx = addCorner(b2);
            int d1_idx = addCorner(d1);
            int d2_idx = addCorner(d2);

            int a_idx = triangle.p1_idx;
            int c_idx = triangle.p2_idx;

            int t1_idx = insertTriangle(a_idx, b1_idx, d1_idx, TYPE3);
            int t2_idx = insertTriangle(b2_idx, c_idx, d2_idx, TYPE1);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b1_idx, d1_idx);
            addNeighborVertex(b2_idx, c_idx, d2_idx);

            covered_triangle.emplace_back();
            covered_triangle.emplace_back();
            addCoveredTriangle(a_idx, b1_idx, d1_idx, t1_idx);
            addCoveredTriangle(b2_idx, c_idx, d2_idx, t2_idx);

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

            int b1_idx = addCorner(b1);
            int b2_idx = addCorner(b2);
            int c1_idx = addCorner(c1);
            int c2_idx = addCorner(c2);

            int a_idx = triangle.p1_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b1_idx, c1_idx, TYPE4);
            int t2_idx = insertTriangle(b2_idx, c2_idx, d_idx, TYPE1);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b1_idx, c1_idx);
            addNeighborVertex(b2_idx, c2_idx, d_idx);

            covered_triangle.emplace_back();
            covered_triangle.emplace_back();
            addCoveredTriangle(a_idx, b1_idx, c1_idx, t1_idx);
            addCoveredTriangle(b2_idx, c2_idx, d_idx, t2_idx);
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

            int a1_idx = addCorner(a1);
            int a2_idx = addCorner(a2);
            int c1_idx = addCorner(c1);
            int c2_idx = addCorner(c2);

            int b_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a1_idx, b_idx, c1_idx, TYPE2);
            int t2_idx = insertTriangle(a2_idx, c2_idx, d_idx, TYPE4);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a1_idx, b_idx, c1_idx);
            addNeighborVertex(a2_idx, c2_idx, d_idx);

            covered_triangle.emplace_back();
            covered_triangle.emplace_back();
            addCoveredTriangle(a1_idx, b_idx, c1_idx, t1_idx);
            addCoveredTriangle(a2_idx, c2_idx, d_idx, t2_idx);

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

            int a1_idx = addCorner(a1);
            int a2_idx = addCorner(a2);
            int c1_idx = addCorner(c1);
            int c2_idx = addCorner(c2);

            int b_idx = triangle.p1_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(b_idx, a1_idx, c1_idx, TYPE2);
            int t2_idx = insertTriangle(a2_idx, c2_idx, d_idx, TYPE3);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(b_idx, a1_idx, c1_idx);
            addNeighborVertex(a2_idx, c2_idx, d_idx);

            covered_triangle.emplace_back();
            covered_triangle.emplace_back();
            addCoveredTriangle(b_idx, a1_idx, c1_idx, t1_idx);
            addCoveredTriangle(a2_idx, c2_idx, d_idx, t2_idx);

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
    if(steps <= 0) return false;

    double RMSE_before_subdiv = 0.0;
    double error_warping, error_parallel;
    cv::Point2f p1 = triangle.p1;
    cv::Point2f p2 = triangle.p2;
    cv::Point2f p3 = triangle.p3;

    Point3Vec targetTriangle(p1, p2, p3);
    int triangle_size = 0;
    bool parallel_flag;
    int num;

    std::vector<cv::Point2f> gauss_result_warping;
    cv::Point2f gauss_result_parallel;


    if(cmt == nullptr) {
        cmt = previousMvList[0][triangle_index];
    }

    if(triangle_gauss_results[triangle_index].residual > 0) {
        GaussResult result_before = triangle_gauss_results[triangle_index];
        gauss_result_warping = result_before.mv_warping;
        gauss_result_parallel = result_before.mv_parallel;
        RMSE_before_subdiv = result_before.residual;
        triangle_size = result_before.triangle_size;
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
                std::tie(tmp_bm_mv, tmp_bm_errors) = blockMatching(triangle, target_image, expansion_ref,
                                                                   diagonal_line_area_flag, triangle_index, ctu);
                std::tie(gauss_result_warping, gauss_result_parallel, error_warping, error_parallel, triangle_size) = GaussNewton(ref_images, target_images, expand_images, targetTriangle,
                                                      diagonal_line_area_flag, triangle_index, ctu, block_size_x,
                                                      block_size_y, tmp_bm_mv[2], ref_hevc);
            }else{
                std::tie(gauss_result_warping, gauss_result_parallel, error_warping, error_parallel, triangle_size) = GaussNewton(ref_images, target_images, expand_images, targetTriangle,
                                                      diagonal_line_area_flag, triangle_index, ctu, block_size_x,
                                                      block_size_y, cv::Point2f(-1000, -1000), ref_hevc);
            }

            triangle_gauss_results[triangle_index].mv_warping = gauss_result_warping;
            triangle_gauss_results[triangle_index].mv_parallel = gauss_result_parallel;
            triangle_gauss_results[triangle_index].triangle_size = triangle_size;
            triangle_gauss_results[triangle_index].residual = RMSE_before_subdiv;

            int cost_warping, cost_parallel;

            std::tie(cost_parallel, std::ignore, std::ignore, std::ignore, std::ignore) = getMVD(
                    {gauss_result_parallel, gauss_result_parallel, gauss_result_parallel}, error_parallel,
                    triangle_index, cmt->mv1, diagonal_line_area_flag, ctu);

            std::tie(cost_warping, std::ignore, std::ignore, std::ignore, std::ignore) = getMVD(
                    triangle_gauss_results[triangle_index].mv_warping, error_warping,
                    triangle_index, cmt->mv1, diagonal_line_area_flag, ctu);

            if(cost_parallel < cost_warping){
                triangle_gauss_results[triangle_index].parallel_flag = true;
                triangle_gauss_results[triangle_index].residual = error_parallel;
            }else{
                triangle_gauss_results[triangle_index].parallel_flag = false;
                triangle_gauss_results[triangle_index].residual = error_warping;
//                std::cout << "warping!" << std::endl;
            }

        }else if(PRED_MODE == BM) {
            std::vector<cv::Point2f> tmp_bm_mv;
            std::vector<double> tmp_bm_errors;
            std::tie(tmp_bm_mv, tmp_bm_errors) = blockMatching(triangle, target_image, expansion_ref,
                                                               diagonal_line_area_flag, triangle_index, ctu);
            triangle_gauss_results[triangle_index].residual_bm = tmp_bm_errors[2];
            triangle_gauss_results[triangle_index].residual_newton = RMSE_before_subdiv;
            ctu->error_newton = RMSE_before_subdiv;
            ctu->error_bm = tmp_bm_errors[2];
            gauss_result_warping = tmp_bm_mv;
            gauss_result_parallel = tmp_bm_mv[2];
            RMSE_before_subdiv = tmp_bm_errors[2];
            triangle_gauss_results[triangle_index].mv_warping = gauss_result_warping;
            triangle_gauss_results[triangle_index].mv_parallel = gauss_result_parallel;
            triangle_gauss_results[triangle_index].triangle_size = triangle_size;
            triangle_gauss_results[triangle_index].residual = RMSE_before_subdiv;
            triangle_gauss_results[triangle_index].parallel_flag = true;
        }
    }

    cv::Point2f mvd;
    int selected_index;
    MV_CODE_METHOD method_flag;
    double cost_before_subdiv;
    int code_length;

    if(triangle_gauss_results[triangle_index].parallel_flag) {
        std::tie(cost_before_subdiv, code_length, mvd, selected_index, method_flag) = getMVD(
                {gauss_result_parallel, gauss_result_parallel, gauss_result_parallel}, error_parallel,
                triangle_index, cmt->mv1, diagonal_line_area_flag, ctu);
    }else{
        std::tie(cost_before_subdiv, code_length, mvd, selected_index, method_flag) = getMVD(
                triangle_gauss_results[triangle_index].mv_warping, error_warping,
                triangle_index, cmt->mv1, diagonal_line_area_flag, ctu);
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
    ctu->mv2 = mv[1];
    ctu->mv3 = mv[2];
    ctu->triangle_index = triangle_index;
    ctu->code_length = code_length;
    ctu->collocated_mv = cmt->mv1;
    ctu->parallel_flag = parallel_flag;

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

    int p1_idx = addCorner(p1);
    int p2_idx = addCorner(p2);
    int p3_idx = addCorner(p3);
    addCornerAndTriangle(Triangle(p1_idx, p2_idx, p3_idx), triangle_index, type);

    int t1_idx = (int)triangles.size() - 2;
    int t2_idx = (int)triangles.size() - 1;
    int t1_p1_idx = addCorner(split_triangles.t1.p1);
    int t1_p2_idx = addCorner(split_triangles.t1.p2);
    int t1_p3_idx = addCorner(split_triangles.t1.p3);
    addCornerAndTriangle(Triangle(t1_p1_idx, t1_p2_idx, t1_p3_idx), t1_idx, split_triangles.t1_type);

    int t2_p1_idx = addCorner(split_triangles.t2.p1);
    int t2_p2_idx = addCorner(split_triangles.t2.p2);
    int t2_p3_idx = addCorner(split_triangles.t2.p3);
    addCornerAndTriangle(Triangle(t2_p1_idx, t2_p2_idx, t2_p3_idx), t2_idx, split_triangles.t2_type);

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

    ctu->leftNode = new CodingTreeUnit();
    ctu->leftNode->triangle_index = triangles.size() - 6;
    ctu->leftNode->parentNode = ctu;
    ctu->leftNode->leftNode = new CodingTreeUnit();
    ctu->leftNode->leftNode->parentNode = ctu->leftNode;
    ctu->leftNode->rightNode = new CodingTreeUnit();
    ctu->leftNode->rightNode->parentNode = ctu->leftNode;

    ctu->rightNode = new CodingTreeUnit();
    ctu->rightNode->triangle_index = triangles.size() - 5;
    ctu->rightNode->parentNode = ctu;
    ctu->rightNode->leftNode = new CodingTreeUnit();
    ctu->rightNode->leftNode->parentNode = ctu->rightNode;
    ctu->rightNode->rightNode = new CodingTreeUnit();
    ctu->rightNode->rightNode->parentNode = ctu->rightNode;

#pragma omp parallel for
    for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
        double error_warping_tmp, error_parallel_tmp;
        int triangle_size_tmp;
        cv::Point2f mv_parallel_tmp;
        std::vector<cv::Point2f> mv_warping_tmp;
        std::vector<cv::Point2f> tmp_bm_mv;
        std::vector<double> tmp_bm_errors;
        double cost_warping_tmp, cost_parallel_tmp;
        double tmp_error_newton;
        if(PRED_MODE == NEWTON){
            if(GAUSS_NEWTON_INIT_VECTOR) {
                std::tie(tmp_bm_mv, tmp_bm_errors) = blockMatching(subdiv_target_triangles[j], target_image,
                                                                   expansion_ref, diagonal_line_area_flag,
                                                                   triangle_indexes[j], ctu);
                std::tie(mv_warping_tmp, mv_parallel_tmp, error_warping_tmp, error_parallel_tmp,triangle_size_tmp) = GaussNewton(
                        ref_images, target_images, expand_images, subdiv_target_triangles[j], diagonal_line_area_flag,
                        triangle_indexes[j], (j == 0 ? ctu->leftNode : ctu->rightNode), block_size_x, block_size_y,
                        tmp_bm_mv[2], ref_hevc);


            }else{
                std::tie(mv_warping_tmp, mv_parallel_tmp, error_warping_tmp, error_parallel_tmp, triangle_size_tmp) = GaussNewton(
                        ref_images, target_images, expand_images, subdiv_target_triangles[j], diagonal_line_area_flag,
                        triangle_indexes[j], (j == 0 ? ctu->leftNode : ctu->rightNode), block_size_x, block_size_y,
                        cv::Point2f(-1000, -1000), ref_hevc);
            }

            std::tie(cost_parallel_tmp,std::ignore, std::ignore, std::ignore, std::ignore) = getMVD(
                    {mv_parallel_tmp, mv_parallel_tmp, mv_parallel_tmp}, error_parallel_tmp,
                    triangle_indexes[j], cmt->mv1, diagonal_line_area_flag, ctu);

            std::tie(cost_warping_tmp, std::ignore, std::ignore, std::ignore, std::ignore) = getMVD(
                    mv_warping_tmp, error_warping_tmp,
                    triangle_indexes[j], cmt->mv1, diagonal_line_area_flag, ctu);

            if(cost_parallel_tmp < cost_warping_tmp){
                triangle_gauss_results[triangle_indexes[j]].parallel_flag = true;
                split_mv_result[j] = GaussResult(mv_warping_tmp, mv_parallel_tmp, error_parallel_tmp, triangle_size_tmp, true, error_parallel_tmp, error_warping_tmp);
            }else{
                 triangle_gauss_results[triangle_indexes[j]].parallel_flag = false;
                split_mv_result[j] = GaussResult(mv_warping_tmp, mv_parallel_tmp, error_warping_tmp, triangle_size_tmp, false, error_parallel_tmp, error_warping_tmp);
            }

        }else if(PRED_MODE == BM){
            std::tie(tmp_bm_mv, tmp_bm_errors) = blockMatching(subdiv_target_triangles[j], target_image, expansion_ref, diagonal_line_area_flag, triangle_indexes[j], ctu);
            mv_warping_tmp = tmp_bm_mv;
            mv_parallel_tmp = tmp_bm_mv[2];
            error_parallel_tmp = tmp_bm_errors[2];
            triangle_size_tmp = (double)1e6;

            split_mv_result[j] = GaussResult(mv_warping_tmp, mv_parallel_tmp, error_parallel_tmp, triangle_size_tmp, true, tmp_bm_errors[2], tmp_error_newton);
        }

    }

    double cost_after_subdiv1;
    int code_length1;
    CollocatedMvTree *cmt_left_left, *cmt_left_right, *cmt_right_left, *cmt_right_right;

    cmt_left_left  = (cmt->leftNode == nullptr ? cmt : (cmt->leftNode->leftNode == nullptr ? cmt->leftNode : cmt->leftNode->leftNode));
    cmt_left_right  = (cmt->leftNode == nullptr ? cmt : (cmt->leftNode->rightNode == nullptr ? cmt->rightNode : cmt->leftNode->rightNode));
    cmt_right_left  = (cmt->rightNode == nullptr ? cmt : (cmt->rightNode->leftNode == nullptr ? cmt->rightNode : cmt->rightNode->leftNode));
    cmt_right_right  = (cmt->rightNode == nullptr ? cmt : (cmt->rightNode->rightNode == nullptr ? cmt->rightNode : cmt->rightNode->rightNode));

    if(split_mv_result[0].parallel_flag) {
        std::tie(cost_after_subdiv1, code_length1, mvd, selected_index, method_flag) = getMVD(
                {split_mv_result[0].mv_parallel, split_mv_result[0].mv_parallel, split_mv_result[0].mv_parallel},
                split_mv_result[0].residual,
                triangle_indexes[0], cmt_left_left->mv1, diagonal_line_area_flag, ctu->leftNode);
    }else{
        std::tie(cost_after_subdiv1, code_length1, mvd, selected_index, method_flag) = getMVD(
                split_mv_result[0].mv_warping, split_mv_result[0].residual,
                triangle_indexes[0], cmt_left_left->mv1, diagonal_line_area_flag, ctu->leftNode);
    }

    double cost_after_subdiv2;
    int code_length2;
    if(split_mv_result[1].parallel_flag){
        std::tie(cost_after_subdiv2, code_length2, mvd, selected_index, method_flag) = getMVD(
                {split_mv_result[1].mv_parallel, split_mv_result[1].mv_parallel, split_mv_result[1].mv_parallel}, split_mv_result[1].residual,
                triangle_indexes[1], cmt_left_right->mv1, diagonal_line_area_flag, ctu->leftNode);

    }else{
        std::tie(cost_after_subdiv2, code_length2, mvd, selected_index, method_flag) = getMVD(
                split_mv_result[1].mv_warping, split_mv_result[1].residual,
                triangle_indexes[1], cmt_left_right->mv1, diagonal_line_area_flag, ctu->leftNode);
    }

    double cost_after_subdiv3;
    int code_length3;
    if(split_mv_result[2].parallel_flag) {
        std::tie(cost_after_subdiv3, code_length3, mvd, selected_index, method_flag) = getMVD(
                {split_mv_result[2].mv_parallel, split_mv_result[2].mv_parallel, split_mv_result[2].mv_parallel},
                split_mv_result[2].residual,
                triangle_indexes[2], cmt_right_left->mv1, diagonal_line_area_flag, ctu->rightNode);
    }else{
        std::tie(cost_after_subdiv3, code_length3, mvd, selected_index, method_flag) = getMVD(
                split_mv_result[2].mv_warping, split_mv_result[2].residual,
                triangle_indexes[2], cmt_right_left->mv1, diagonal_line_area_flag, ctu->rightNode);
    }

    double cost_after_subdiv4;
    int code_length4;
    if(split_mv_result[3].parallel_flag){
        std::tie(cost_after_subdiv4, code_length4, mvd, selected_index, method_flag) = getMVD(
                {split_mv_result[3].mv_parallel, split_mv_result[3].mv_parallel, split_mv_result[3].mv_parallel}, split_mv_result[3].residual,
                triangle_indexes[3], cmt_right_right->mv1, diagonal_line_area_flag, ctu->rightNode);
    }else{
        std::tie(cost_after_subdiv4, code_length4, mvd, selected_index, method_flag) = getMVD(
                split_mv_result[3].mv_warping, split_mv_result[3].residual,
                triangle_indexes[3], cmt_right_right->mv1, diagonal_line_area_flag, ctu->rightNode);
    }

    double alpha = 1;
    std::cout << "before:" << cost_before_subdiv << " after:" << alpha * (cost_after_subdiv1 + cost_after_subdiv2 + cost_after_subdiv3 + cost_after_subdiv4) << std::endl;
    if(cost_before_subdiv >= alpha * (cost_after_subdiv1 + cost_after_subdiv2 + cost_after_subdiv3 + cost_after_subdiv4)) {
        ctu->split_cu_flag = true;

        int t1_idx = triangles.size() - 4;
        int t2_idx = triangles.size() - 3;
        int t3_idx = triangles.size() - 2;
        int t4_idx = triangles.size() - 1;

        // 1つ目の頂点追加
        ctu->leftNode->leftNode->triangle_index = t1_idx;
        if(split_mv_result[0].parallel_flag) {
            ctu->leftNode->leftNode->mv1 = split_mv_result[0].mv_parallel;
            ctu->leftNode->leftNode->mv2 = split_mv_result[0].mv_parallel;
            ctu->leftNode->leftNode->mv3 = split_mv_result[0].mv_parallel;
        }else{
            ctu->leftNode->leftNode->mv1 = split_mv_result[0].mv_warping[0];
            ctu->leftNode->leftNode->mv2 = split_mv_result[0].mv_warping[1];
            ctu->leftNode->leftNode->mv3 = split_mv_result[0].mv_warping[2];
        }
        ctu->leftNode->leftNode->code_length = code_length1;
        ctu->leftNode->leftNode->parallel_flag = split_mv_result[0].parallel_flag;
        triangle_gauss_results[t1_idx] = split_mv_result[0];
        isCodedTriangle[t1_idx] = true;
        bool result = split(expand_images, ctu->leftNode->leftNode, cmt_left_left, split_sub_triangles1.t1, t1_idx,split_sub_triangles1.t1_type, steps - 2, diagonal_line_area_flag);

        // 2つ目の三角形
        ctu->leftNode->rightNode->triangle_index = t2_idx;
        if(split_mv_result[1].parallel_flag){
            ctu->leftNode->rightNode->mv1 = split_mv_result[1].mv_parallel;
            ctu->leftNode->rightNode->mv2 = split_mv_result[1].mv_parallel;
            ctu->leftNode->rightNode->mv3 = split_mv_result[1].mv_parallel;
        }else{
            ctu->leftNode->rightNode->mv1 = split_mv_result[1].mv_warping[0];
            ctu->leftNode->rightNode->mv2 = split_mv_result[1].mv_warping[1];
            ctu->leftNode->rightNode->mv3 = split_mv_result[1].mv_warping[2];
        }
        ctu->rightNode->rightNode->code_length = code_length2;
        ctu->rightNode->rightNode->parallel_flag = split_mv_result[1].parallel_flag;

        triangle_gauss_results[t2_idx] = split_mv_result[1];
        isCodedTriangle[t2_idx] = true;
        result = split(expand_images, ctu->leftNode->rightNode, cmt_left_right, split_sub_triangles1.t2, t2_idx, split_sub_triangles1.t2_type, steps - 2, diagonal_line_area_flag);

        // 3つ目の三角形
        ctu->rightNode->leftNode->triangle_index = t3_idx;
        if(split_mv_result[2].parallel_flag) {
            ctu->rightNode->leftNode->mv1 = split_mv_result[2].mv_parallel;
            ctu->rightNode->leftNode->mv2 = split_mv_result[2].mv_parallel;
            ctu->rightNode->leftNode->mv3 = split_mv_result[2].mv_parallel;
        }else{
            ctu->rightNode->leftNode->mv1 = split_mv_result[2].mv_warping[0];
            ctu->rightNode->leftNode->mv2 = split_mv_result[2].mv_warping[1];
            ctu->rightNode->leftNode->mv3 = split_mv_result[2].mv_warping[2];
        }
        ctu->rightNode->leftNode->code_length = code_length3;
        ctu->rightNode->leftNode->parallel_flag = split_mv_result[2].parallel_flag;

        triangle_gauss_results[t3_idx] = split_mv_result[2];
        isCodedTriangle[t3_idx] = true;
        result = split(expand_images, ctu->rightNode->leftNode, cmt_right_left, split_sub_triangles2.t1, t3_idx, split_sub_triangles2.t1_type, steps - 2, diagonal_line_area_flag);

        // 4つ目の三角形
        ctu->rightNode->rightNode->triangle_index = t4_idx;
        if(split_mv_result[3].parallel_flag) {
            ctu->rightNode->rightNode->mv1 = split_mv_result[3].mv_parallel;
            ctu->rightNode->rightNode->mv2 = split_mv_result[3].mv_parallel;
            ctu->rightNode->rightNode->mv3 = split_mv_result[3].mv_parallel;
        }else{
            ctu->rightNode->rightNode->mv1 = split_mv_result[3].mv_warping[0];
            ctu->rightNode->rightNode->mv2 = split_mv_result[3].mv_warping[1];
            ctu->rightNode->rightNode->mv3 = split_mv_result[3].mv_warping[2];
        }
        ctu->rightNode->rightNode->code_length = code_length4;
        ctu->rightNode->rightNode->parallel_flag = split_mv_result[3].parallel_flag;

        triangle_gauss_results[t4_idx] = split_mv_result[3];
        isCodedTriangle[t4_idx] = true;
        result = split(expand_images, ctu->rightNode->rightNode, cmt_right_right, split_sub_triangles2.t2, t4_idx, split_sub_triangles2.t2_type, steps - 2, diagonal_line_area_flag);

        return true;
    }else{
        isCodedTriangle[triangle_index] = true;
        delete_flag[triangle_index] = false;
        ctu->leftNode = ctu->rightNode = nullptr;
        diagonal_line_area_flag = prev_area_flag;
        eraseTriangle(triangles.size() - 1);
        eraseTriangle(triangles.size() - 1);
        eraseTriangle(triangles.size() - 1);
        eraseTriangle(triangles.size() - 1);
        eraseTriangle(triangles.size() - 1);
        eraseTriangle(triangles.size() - 1);
        addNeighborVertex(triangles[triangle_index].first.p1_idx,triangles[triangle_index].first.p2_idx,triangles[triangle_index].first.p3_idx);
        addCoveredTriangle(triangles[triangle_index].first.p1_idx,triangles[triangle_index].first.p2_idx,triangles[triangle_index].first.p3_idx, triangle_index);
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

        left->rightNode = new CollocatedMvTree();
        left->rightNode->mv1 = cv::Point2f(0, 0);
        left->rightNode->mv2 = cv::Point2f(0, 0);
        left->rightNode->mv3 = cv::Point2f(0, 0);
        previousMvList[pic_num][i]->leftNode = left;

        auto* right = new CollocatedMvTree();
        right->mv1 = cv::Point2f(0, 0);
        right->mv2 = cv::Point2f(0, 0);
        right->mv3 = cv::Point2f(0, 0);

        right->leftNode = new CollocatedMvTree();
        right->leftNode->mv1 = cv::Point2f(0, 0);
        right->leftNode->mv2 = cv::Point2f(0, 0);
        right->leftNode->mv3 = cv::Point2f(0, 0);
        right->rightNode = new CollocatedMvTree();
        right->rightNode->mv1 = cv::Point2f(0, 0);
        right->rightNode->mv2 = cv::Point2f(0, 0);
        right->rightNode->mv3 = cv::Point2f(0, 0);
        previousMvList[pic_num][i]->rightNode = right;
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
 * @param[in] area_flag 含まれる画素のフラグ
 * @return 差分ベクトル，参照したパッチ，空間or時間のフラグのtuple
 */
std::tuple<double, int, cv::Point2f, int, MV_CODE_METHOD> TriangleDivision::getMVD(std::vector<cv::Point2f> mv, double residual, int triangle_idx, cv::Point2f &collocated_mv, const std::vector<std::vector<int>> &area_flag, CodingTreeUnit* ctu){
//    std::cout << "triangle_index(getMVD):" << triangle_idx << std::endl;
    // 空間予測と時間予測の候補を取り出す
    std::vector<int> spatial_triangles = getSpatialTriangleList(triangle_idx);
    int spatial_triangle_size = static_cast<int>(spatial_triangles.size());
    std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> vectors; // ベクトルとモードを表すフラグのペア

    // すべてのベクトルを格納する．
    for(int i = 0 ; i < spatial_triangle_size ; i++) {
        int spatial_triangle_index = spatial_triangles[i];
        GaussResult spatial_triangle = triangle_gauss_results[spatial_triangle_index];

        if(spatial_triangle.parallel_flag){
            if(!isMvExists(vectors, spatial_triangle.mv_parallel)) {
                vectors.emplace_back(spatial_triangle.mv_parallel, SPATIAL);
            }
        }else{
            // 隣接パッチがワーピングで予想されている場合、そのパッチの0番の動きベクトルを候補とする
            if(!isMvExists(vectors, spatial_triangle.mv_warping[0])){
                vectors.emplace_back(spatial_triangle.mv_warping[0], SPATIAL);
            }
        }
    }

    if(!isMvExists(vectors, collocated_mv)) vectors.emplace_back(collocated_mv, SPATIAL);

    if(vectors.size() < 2) vectors.emplace_back(cv::Point2f(0.0, 0.0), Collocated);

    double lambda = getLambdaPred(qp);

    //                      コスト, 差分ベクトル, 番号, タイプ
    std::vector<std::tuple<double, int, cv::Point2f, int, MV_CODE_METHOD> > results;
    for(int i = 0 ; i < vectors.size() ; i++) {
        std::pair<cv::Point2f, MV_CODE_METHOD> vector = vectors[i];
        cv::Point2f current_mv = vector.first;
        // TODO: ワーピング対応

        if(triangle_gauss_results[triangle_idx].parallel_flag) { // 平行移動成分に関してはこれまで通りにやる
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
            int mvd_x_minus_2 = (mvd.x - 2.0) * 4;
            int mvd_y_minus_2 = (mvd.y - 2.0) * 4;

            int mvd_code_length = getExponentialGolombCodeLength((int) mvd_x_minus_2, 0) +
                                  getExponentialGolombCodeLength((int) mvd_y_minus_2, 0);

            // 参照箇所符号化
            int reference_index = std::get<1>(vector);
            int reference_index_code_length = getUnaryCodeLength(reference_index);

            // 各種フラグ分を(3*2)bit足してます
            double rd = residual + lambda * (mvd_code_length + reference_index_code_length + 6  );

            // 結果に入れる
            results.emplace_back(rd, mvd_code_length + reference_index_code_length + 6 + 1, mvd, i, vector.second);
        }else{
            std::vector<cv::Point2f> mvds;
            mvds.emplace_back(current_mv - mv[0]);
            mvds.emplace_back(current_mv - mv[1]);
            mvds.emplace_back(current_mv - mv[2]);

            for(auto &mvd : mvds){
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
                int mvd_x_minus_2 = (mvd.x - 2.0) * 4;
                int mvd_y_minus_2 = (mvd.y - 2.0) * 4;

                int mvd_code_length = getExponentialGolombCodeLength((int) mvd_x_minus_2, 0) +
                                      getExponentialGolombCodeLength((int) mvd_y_minus_2, 0);

                // 参照箇所符号化
                int reference_index = std::get<1>(vector);
                int reference_index_code_length = getUnaryCodeLength(reference_index);

                // 各種フラグ分を(3*2)bit足してます
                double rd = residual + lambda * (mvd_code_length + reference_index_code_length + 6);

                // 結果に入れる
                results.emplace_back(rd, mvd_code_length + reference_index_code_length + 6 + 1, mvd, i, vector.second);
            }
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

    std::vector<std::pair<cv::Point2f, MV_CODE_METHOD>> merge_vectors;
    std::vector<cv::Point2f> pixels_in_triangle = getPixelsInTriangle(coordinate, area_flag, triangle_idx, ctu, block_size_x, block_size_y);
    for(int i = 0 ; i < spatial_triangle_size ; i++) {
        int spatial_triangle_index = spatial_triangles[i];
        GaussResult spatial_triangle = triangle_gauss_results[spatial_triangle_index];

        if(spatial_triangle.parallel_flag){
            if(!isMvExists(merge_vectors, spatial_triangle.mv_parallel)) {
                merge_vectors.emplace_back(spatial_triangle.mv_parallel, MERGE);
                double ret_residual = getTriangleResidual(ref_image, target_image, coordinate, mv, pixels_in_triangle);
                double rd = ret_residual + lambda * (getUnaryCodeLength(i) + 1);
                results.emplace_back(rd, getUnaryCodeLength(i) + 1, cv::Point2f(0, 0), results.size(), MERGE);
            }
        }else{
            if(!isMvExists(merge_vectors, spatial_triangle.mv_warping[0])) {
                merge_vectors.emplace_back(spatial_triangle.mv_warping[0], MERGE);
                double ret_residual = getTriangleResidual(ref_image, target_image, coordinate, mv, pixels_in_triangle);
                double rd = ret_residual + lambda * (getUnaryCodeLength(i) + 1);
                results.emplace_back(rd, getUnaryCodeLength(i) + 1, cv::Point2f(0, 0), results.size(), MERGE);
            }
        }

    }

    // RDしたスコアが小さい順にソート
    std::sort(results.begin(), results.end(), [](const std::tuple<double, int, cv::Point2f, int, MV_CODE_METHOD >& a, const std::tuple<double, int, cv::Point2f, int, MV_CODE_METHOD>& b){
        return std::get<0>(a) < std::get<0>(b);
    });
    double cost = std::get<0>(results[0]);
    int code_length = std::get<1>(results[0]);
    cv::Point2f mvd = std::get<2>(results[0]);
    int selected_idx = std::get<3>(results[0]);
    MV_CODE_METHOD method = std::get<4>(results[0]);

    return {cost, code_length, mvd, selected_idx, method};
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

    if(ctu->leftNode == nullptr && ctu->rightNode == nullptr) {
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

    if(ctu->leftNode != nullptr) getPredictedDiagonalImageFromCtu(ctu->leftNode, area_flag, out);
    if(ctu->leftNode != nullptr) getPredictedDiagonalImageFromCtu(ctu->rightNode, area_flag, out);
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
    if(ctu->leftNode == nullptr && ctu->rightNode == nullptr) {
        int triangle_index = ctu->triangle_index;
        cv::Point2f mv = ctu->mv1;
        Triangle triangle_corner_idx = triangles[triangle_index].first;
        Point3Vec triangle(corners[triangle_corner_idx.p1_idx], corners[triangle_corner_idx.p2_idx], corners[triangle_corner_idx.p3_idx]);

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

        getPredictedImage(expansion_ref_uchar, target_image, out, triangle, mvs, 16, area_flag, ctu->triangle_index, ctu, cv::Rect(0, 0, block_size_x, block_size_y), ref_hevc);
        return;
    }

    if(ctu->leftNode != nullptr) getPredictedImageFromCtu(ctu->leftNode, out, area_flag);
    if(ctu->rightNode != nullptr) getPredictedImageFromCtu(ctu->rightNode, out, area_flag);
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

    return out;
}

void TriangleDivision::getPredictedColorImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out, std::vector<std::vector<int>> &area_flag, double original_psnr, std::vector<cv::Scalar> &colors){
    if(ctu->leftNode == nullptr && ctu->rightNode == nullptr) {
        int triangle_index = ctu->triangle_index;
        cv::Point2f mv = ctu->mv1;
        Triangle triangle_corner_idx = triangles[triangle_index].first;
        Point3Vec triangle(corners[triangle_corner_idx.p1_idx], corners[triangle_corner_idx.p2_idx], corners[triangle_corner_idx.p3_idx]);

        std::vector<cv::Point2f> mvs{mv, mv, mv};
        std::vector<cv::Point2f> pixels = getPixelsInTriangle(triangle, area_flag, triangle_index, ctu, block_size_x, block_size_y);
//        double residual = getTriangleResidual(expansion_ref_uchar, target_image, triangle, mvs, pixels, cv::Rect(-16, -16, target_image.cols + 2 * 16, target_image.rows + 2 * 16));
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
            getPredictedImage(expansion_ref_uchar, target_image, out, triangle, mvs, 16, area_flag, ctu->triangle_index, ctu, cv::Rect(0, 0, block_size_x, block_size_y), ref_hevc);
        }
        return;
    }

    if(ctu->leftNode != nullptr) getPredictedColorImageFromCtu(ctu->leftNode, out, area_flag, original_psnr, colors);
    if(ctu->leftNode != nullptr) getPredictedColorImageFromCtu(ctu->rightNode, out, area_flag, original_psnr, colors);
}

int TriangleDivision::getCtuCodeLength(std::vector<CodingTreeUnit*> ctus) {
    int code_length_sum = 0;
    for(int i = 0 ; i < ctus.size() ; i++){
        code_length_sum += getCtuCodeLength(ctus[i]);
    }
    return code_length_sum;
}

int TriangleDivision::getCtuCodeLength(CodingTreeUnit *ctu){

    if(ctu->leftNode == nullptr && ctu->rightNode == nullptr) {
        return 1+ctu->code_length;
    }

    // ここで足している1はsplit_cu_flag分です
    return 1 + getCtuCodeLength(ctu->leftNode) + getCtuCodeLength(ctu->rightNode);
}


cv::Mat TriangleDivision::getMvImage(std::vector<CodingTreeUnit*> ctus){
    cv::Mat out = target_image.clone();

    for(auto triangle : getTriangleCoordinateList()){
        drawTriangle(out, triangle.p1, triangle.p2, triangle.p3, cv::Scalar(255, 255, 255));
    }

    for(int i = 0 ; i < ctus.size() ; i++){
        drawMvImage(out, ctus[i]);
    }

    return out;
}

void TriangleDivision::drawMvImage(cv::Mat &out, CodingTreeUnit *ctu){
    if(ctu->leftNode == nullptr && ctu->rightNode == nullptr) {
        Triangle t = triangles[ctu->triangle_index].first;
        cv::Point2f p1 = corners[t.p1_idx];
        cv::Point2f p2 = corners[t.p2_idx];
        cv::Point2f p3 = corners[t.p3_idx];

        cv::Point2f g = (p1 + p2 + p3) / 3.0;

        cv::line(out, g, g+ctu->mv1, GREEN);
    }

    if(ctu->leftNode != nullptr) drawMvImage(out, ctu->leftNode);
    if(ctu->rightNode != nullptr) drawMvImage(out, ctu->rightNode);
}

TriangleDivision::SplitResult::SplitResult(const Point3Vec &t1, const Point3Vec &t2, int t1Type, int t2Type) : t1(t1),
                                                                                                               t2(t2),
                                                                                                               t1_type(t1Type),
                                                                                                               t2_type(t2Type) {}

TriangleDivision::GaussResult::GaussResult(const std::vector<cv::Point2f> &mvWarping, const cv::Point2f &mvParallel,
                                           double residual, int triangleSize, bool parallelFlag, double residualBm, double residualNewton) : mv_warping(
        mvWarping), mv_parallel(mvParallel), residual(residual), triangle_size(triangleSize), parallel_flag(parallelFlag), residual_bm(residualBm), residual_newton(residualNewton) {}

TriangleDivision::GaussResult::GaussResult() {}
