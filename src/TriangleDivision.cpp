#include <cmath>

//
// Created by kasph on 2019/04/08.
//

#include "../includes/TriangleDivision.h"
#include <opencv2/core.hpp>
#include <iostream>
#include "../includes/Utils.h"
#include <set>
#include <vector>
#include <utility>
#include <algorithm>

TriangleDivision::TriangleDivision(const cv::Mat &refImage, const cv::Mat &targetImage) : target_image(targetImage),
                                                                                          ref_image(refImage) {}

void TriangleDivision::initTriangle(int block_size_x, int block_size_y, int divide_flag) {

    int block_num_x = target_image.cols / block_size_x;
    int block_num_y = target_image.rows / block_size_y;

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
            neighbor_vtx.emplace_back();
        }
    }

    std::cout << "block_num_y:" << block_num_y << std::endl;
    std::cout << "block_num_x:" << block_num_x << std::endl;

    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {
        for(int block_x = 0 ; block_x < block_num_x ; block_x++) {
            int p1_idx = block_x + block_y * (block_num_x + 1);
            int p2_idx = p1_idx + 1;
            int p3_idx = p1_idx + block_num_x + 1;
            int p4_idx = p3_idx + 1;
            if(divide_flag == LEFT_DIVIDE) {
                insertTriangle(p1_idx, p2_idx, p3_idx);
                insertTriangle(p2_idx, p3_idx, p4_idx);
                addNeighborVertex(p1_idx, p2_idx, p3_idx, LEFT_DIVIDE);
                addNeighborVertex(p2_idx, p3_idx, p4_idx, LEFT_DIVIDE);
            }else{
                insertTriangle(p1_idx, p2_idx, p4_idx);
                insertTriangle(p1_idx, p3_idx, p4_idx);
                addNeighborVertex(p1_idx, p2_idx, p4_idx, RIGHT_DIVIDE);
                addNeighborVertex(p1_idx, p3_idx, p4_idx, RIGHT_DIVIDE);
            }
        }
    }

}

/**
 * @fn std::vector<Point3Vec> getTriangleCoordinateList()
 * @brief
 * @return
 */
std::vector<Point3Vec> TriangleDivision::getTriangleCoordinateList() {
    std::vector<Point3Vec> vec;

    for(const Triangle triangle : triangles) {
        vec.emplace_back(corners[triangle.p1_idx], corners[triangle.p2_idx], corners[triangle.p3_idx]);
    }

    return vec;
}

/**
 *
 * @return
 */
std::vector<Triangle> TriangleDivision::getTriangleIndexList() {
    return triangles;
}

std::vector<cv::Point2f> TriangleDivision::getCorners() {
    return corners;
}


/**
 * @fn void insertTriangle(Point3Vec triangle)
 * @param[in] triangle 三角形を表すアレ
 */
void TriangleDivision::insertTriangle(Point3Vec triangle) {
    insertTriangle(triangle.p1, triangle.p2, triangle.p3);
}

/**
 * @fn void insertTriangle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3)
 * @param[in] p1 頂点1
 * @param[in] p2 頂点2
 * @param[in] p3 頂点3
 */
void TriangleDivision::insertTriangle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3) {
    std::vector<cv::Point2f> v;
    v.push_back(p1);
    v.push_back(p2);
    v.push_back(p3);

    // ラスタスキャン順でソート
    std::sort(v.begin(), v.end(), [](const cv::Point2f &a1, const cv::Point2f &a2) {
        if (a1.y != a2.y) {
            return a1.y < a2.y;
        } else {
            return a1.x < a2.x;
        }
    });

//    triangles.emplace_back(p1, p2, p3);
}

void TriangleDivision::insertTriangle(int p1_idx, int p2_idx, int p3_idx) {
    std::vector<int> v;
    v.push_back(p1_idx);
    v.push_back(p2_idx);
    v.push_back(p3_idx);

    // ラスタスキャン順でソート
    std::sort(v.begin(), v.end());

    triangles.emplace_back(p1_idx, p2_idx, p3_idx);
}

/**
 * @fn void insertTriangle(float x1, float y1, float x2, float y2, float x3, float y3)
 * @param[in] x1 頂点1のx座標
 * @param[in] y1 頂点1のy座標
 * @param[in] x2 頂点2のx座標
 * @param[in] y2 頂点2のy座標
 * @param[in] x3 頂点3のx座標
 * @param[in] y3 頂点3のy座標
 */
void TriangleDivision::insertTriangle(float x1, float y1, float x2, float y2, float x3, float y3) {
//    triangles.emplace_back(cv::Point2f(x1, y1), cv::Point2f(x2, y2), cv::Point2f(x3, y3));
}

/**
 *
 * @param p1_idx
 * @param p2_idx
 * @param p3_idx
 * @param divide_flag
 */
void TriangleDivision::addNeighborVertex(int p1_idx, int p2_idx, int p3_idx, int divide_flag) {

    neighbor_vtx[p1_idx].emplace(p2_idx);
    neighbor_vtx[p2_idx].emplace(p1_idx);

    neighbor_vtx[p1_idx].emplace(p3_idx);
    neighbor_vtx[p3_idx].emplace(p1_idx);

    neighbor_vtx[p2_idx].emplace(p3_idx);
    neighbor_vtx[p3_idx].emplace(p2_idx);

}

double TriangleDivision::getDistance(const cv::Point2f &a, const cv::Point2f &b){
    cv::Point2f v = a - b;
    return std::sqrt(v.x * v.x + v.y * v.y);
}

std::vector<int> TriangleDivision::getNeighborVertexIndexList(int idx) {
    std::set<int> s = neighbor_vtx[idx];
    std::vector<int> v;

    for(const auto e : s) {
        v.emplace_back(e);
    }

    return v;
}

std::vector<cv::Point2f> TriangleDivision::getNeighborVertexCoordinateList(int idx) {
    std::set<int> s = neighbor_vtx[idx];
    std::vector<cv::Point2f> v;

    for(const auto e : s) {
        v.emplace_back(corners[e]);
    }

    return v;
}