#include <cmath>

//
// Created by kasph on 2019/04/08.
//

#include "../includes/TriangleDivision.h"
#include <opencv2/core.hpp>
#include <iostream>
#include "../includes/Utils.h"
#include "../includes/ME.hpp"
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

    covered_triangle.resize(static_cast<unsigned long>((block_num_x + 1) * (block_num_y + 1)));

    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {
        for(int block_x = 0 ; block_x < block_num_x ; block_x++) {
            int p1_idx = block_x + block_y * (block_num_x + 1);
            int p2_idx = p1_idx + 1;
            int p3_idx = p1_idx + block_num_x + 1;
            int p4_idx = p3_idx + 1;
            if(divide_flag == LEFT_DIVIDE) {
                int triangleIndex = insertTriangle(p1_idx, p2_idx, p3_idx, TYPE1);
                addNeighborVertex(p1_idx, p2_idx, p3_idx, LEFT_DIVIDE);
                addCoveredTriangle(p1_idx, p2_idx, p3_idx, triangleIndex); // p1/p2/p3はtriangleIndex番目の三角形に含まれている

                triangleIndex = insertTriangle(p2_idx, p3_idx, p4_idx, TYPE2);
                addNeighborVertex(p2_idx, p3_idx, p4_idx, LEFT_DIVIDE);
                addCoveredTriangle(p2_idx, p3_idx, p4_idx, triangleIndex);
            }else{
                int triangleIndex = insertTriangle(p1_idx, p2_idx, p4_idx, TYPE1);
                addNeighborVertex(p1_idx, p2_idx, p4_idx, RIGHT_DIVIDE);
                addCoveredTriangle(p1_idx, p2_idx, p4_idx, triangleIndex);

                triangleIndex = insertTriangle(p1_idx, p3_idx, p4_idx, TYPE2);
                addNeighborVertex(p1_idx, p3_idx, p4_idx, RIGHT_DIVIDE);
                addCoveredTriangle(p1_idx, p3_idx, p4_idx, triangleIndex);
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

    for(const auto t : triangles) {
        Triangle triangle = t.first;
        vec.emplace_back(corners[triangle.p1_idx], corners[triangle.p2_idx], corners[triangle.p3_idx]);
    }

    return vec;
}

/**
 *
 * @return
 */
std::vector<Triangle> TriangleDivision::getTriangleIndexList() {
    std::vector<Triangle> v;
    for(const auto t : triangles) {
        v.emplace_back(t.first);
    }
    return v;
}

std::vector<cv::Point2f> TriangleDivision::getCorners() {
    return corners;
}


/**
 * @fn void insertTriangle(Point3Vec triangle)
 * @param[in] triangle 三角形を表すアレ
 */
void TriangleDivision::insertTriangle(Point3Vec triangle, int type) {
    insertTriangle(triangle.p1, triangle.p2, triangle.p3, type);
}

/**
 * @fn void insertTriangle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3)
 * @param[in] p1 頂点1
 * @param[in] p2 頂点2
 * @param[in] p3 頂点3
 * @param[in] type 三角形分割のタイプ
 */
void TriangleDivision::insertTriangle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, int type) {
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

/**
 *
 * @param p1_idx
 * @param p2_idx
 * @param p3_idx
 * @param type
 * @return
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

    Triangle triangle(v[0].second, v[1].second, v[2].second);

    triangles.emplace_back(triangle, type);

    return static_cast<int>(triangles.size() - 1);
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
void TriangleDivision::insertTriangle(float x1, float y1, float x2, float y2, float x3, float y3, int type) {
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

void TriangleDivision::addCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_no) {
    if(p1_idx == 329 || p2_idx == 329 || p3_idx == 329) {
      std::cout << p1_idx << " " << p2_idx << " " << p3_idx << std::endl;
    }
    covered_triangle[p1_idx].emplace(triangle_no);
    covered_triangle[p2_idx].emplace(triangle_no);
    covered_triangle[p3_idx].emplace(triangle_no);
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

std::vector<Point3Vec> TriangleDivision::getIdxCoveredTriangleCoordinateList(int idx) {
  std::cout << "idx:" << idx << std::endl;
  std::cout << "covered_triangle.size(): " << covered_triangle.size() << std::endl;
  std::set<int> s = covered_triangle[idx];
  std::vector<Point3Vec> v;
  std::cout << corners[idx] << std::endl;
  for(auto idx : s) {
    Triangle triangle = triangles[idx].first;
    v.emplace_back(corners[triangle.p1_idx], corners[triangle.p2_idx], corners[triangle.p3_idx]);
  }

  return v;
}

std::vector<Triangle> TriangleDivision::getIdxCoveredTriangleIndexList(int idx) {
    return std::vector<Triangle>();
}

/**
 * @fn void TriangleDivision::subdivision()
 * @brief 三角パッチの再分割を行う
 * @param[in] gaussRefImage ガウスニュートン法の第1層目の画像
 */
void TriangleDivision::subdivision(cv::Mat gaussRefImage) {
  std::queue<std::pair<Triangle, int> > targetTriangleQueue;

  // initで分割された三角パッチはすべて入れる
  for(const auto triangle : triangles) targetTriangleQueue.emplace(triangle);

  int denominator = targetTriangleQueue.size();
  int numerator = 0;

  std::vector<GaussResult> results(triangles.size());

#pragma omp parallel for
  for(int i = 0 ; i < (int)triangles.size() ; i++) {
    std::pair<Triangle, int> triangle = triangles[i];

    double RMSE_before_subdiv = 0.0;
    cv::Point2f p1 = corners[triangle.first.p1_idx];
    cv::Point2f p2 = corners[triangle.first.p2_idx];
    cv::Point2f p3 = corners[triangle.first.p3_idx];

    Point3Vec refTriangle(p1, p2, p3);
    Point3Vec targetTriangle(p1, p2, p3);
    int triangle_size = 0;

    RMSE_before_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, targetTriangle, refTriangle, triangle_size);
    RMSE_before_subdiv /= triangle_size;

    double RMSE_after_subdiv = 0.0;
    triangle_size = 0;
    int triangle_size_sum = 0;

    // 三角形を分割し，その結果でRMSEを見る
    if(triangle.second == TYPE1) {
      // 縦横それぞれ1/2したものを作る
      cv::Point2f x = (p2 - p1) / 2; // x-axis
      cv::Point2f y = (p3 - p1) / 2; // y-axis

      cv::Point2f a = corners[triangle.first.p1_idx];
      cv::Point2f b = a + x;
      cv::Point2f c = corners[triangle.first.p2_idx];
      cv::Point2f d = a + y;
      cv::Point2f e = d + x;
      cv::Point2f f = corners[triangle.first.p3_idx];

      std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
      subdiv_ref_triangles.emplace_back(a, b, d);
      subdiv_target_triangles.emplace_back(a, b, d);
      subdiv_ref_triangles.emplace_back(b, c, e);
      subdiv_target_triangles.emplace_back(b, c, e);
      subdiv_ref_triangles.emplace_back(b, d, e);
      subdiv_target_triangles.emplace_back(b, d, e);
      subdiv_ref_triangles.emplace_back(d, e, f);
      subdiv_target_triangles.emplace_back(d, e, f);

      for(int j = 0 ; j < (int)subdiv_ref_triangles.size() ; j++) {
        RMSE_after_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, subdiv_target_triangles[j], subdiv_ref_triangles[j], triangle_size);
        triangle_size_sum += triangle_size;
      }

      RMSE_after_subdiv /= (double)triangle_size_sum;
      std::cout << "before:" << RMSE_before_subdiv << " after:" << RMSE_after_subdiv << std::endl;
      results[i] = GaussResult(i, triangle.first, triangle.second, RMSE_before_subdiv, RMSE_after_subdiv);
    }else if(triangle.second == TYPE2){
      cv::Point2f x = (p2 - p3) / 2.0;
      cv::Point2f y = (p1 - p3) / 2.0;

      std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
      cv::Point2f a = corners[triangle.first.p1_idx];
      cv::Point2f b = corners[triangle.first.p3_idx] + x + y;
      cv::Point2f c = corners[triangle.first.p3_idx] + y;
      cv::Point2f d = corners[triangle.first.p2_idx];
      cv::Point2f e = corners[triangle.first.p3_idx] + x;
      cv::Point2f f = corners[triangle.first.p3_idx];

      subdiv_ref_triangles.emplace_back(a, b, c);
      subdiv_ref_triangles.emplace_back(b, d, e);
      subdiv_ref_triangles.emplace_back(b, c, e);
      subdiv_ref_triangles.emplace_back(c, e, f);

      subdiv_target_triangles.emplace_back(b, d, e);
      subdiv_target_triangles.emplace_back(a, b, c);
      subdiv_target_triangles.emplace_back(b, c, e);
      subdiv_target_triangles.emplace_back(c, e, f);

      for(int j = 0 ; j < (int)subdiv_ref_triangles.size() ; j++) {
        RMSE_after_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, subdiv_target_triangles[j],
                                          subdiv_ref_triangles[j], triangle_size);
        triangle_size_sum += triangle_size;
      }

      RMSE_after_subdiv /= triangle_size_sum;

      results[i] = GaussResult(i, triangle.first, triangle.second, RMSE_before_subdiv, RMSE_after_subdiv);
    }
    numerator++;
    std::cout << numerator << "/" << denominator << std::endl;
  }

  numerator = 0;
  // Queueが空になるまで続ける
  for(int i = 0 ; i < (int)results.size() ; i++) {
    double diff = (results[i].RMSE_before_subdiv - results[i].RMSE_after_subdiv) / results[i].RMSE_before_subdiv;
    if(results[i].type == TYPE1) {
      if(diff > 0.06) {
        std::pair<Triangle, int> triangle = std::make_pair(results[i].triangle, results[i].type);

        cv::Point2f p1 = corners[triangle.first.p1_idx];
        cv::Point2f p2 = corners[triangle.first.p2_idx];
        cv::Point2f p3 = corners[triangle.first.p3_idx];
        cv::Point2f x = (p2 - p1) / 2; // x-axis
        cv::Point2f y = (p3 - p1) / 2; // y-axis

        cv::Point2f a = corners[triangle.first.p1_idx];
        cv::Point2f b = a + x;
        cv::Point2f c = corners[triangle.first.p2_idx];
        cv::Point2f d = a + y;
        cv::Point2f e = d + x;
        cv::Point2f f = corners[triangle.first.p3_idx];

        corners.push_back(b);
        int b_idx = static_cast<int>(corners.size() - 1);
        corners.push_back(d);
        int d_idx = static_cast<int>(corners.size() - 1);
        corners.push_back(e);
        int e_idx = static_cast<int>(corners.size() - 1);
        int t1_idx = insertTriangle(triangle.first.p1_idx, b_idx, d_idx, TYPE1);
        int t2_idx = insertTriangle(b_idx, d_idx, e_idx, TYPE2);
        int t3_idx = insertTriangle(b_idx, triangle.first.p2_idx, e_idx, TYPE1);
        int t4_idx = insertTriangle(d_idx, e_idx, triangle.first.p3_idx, TYPE1);

        std::cout << triangle.first.p1_idx << " " << triangle.first.p2_idx << " " << triangle.first.p3_idx << std::endl;
        neighbor_vtx[triangle.first.p1_idx].erase(triangle.first.p2_idx);
        neighbor_vtx[triangle.first.p1_idx].erase(triangle.first.p3_idx);
        neighbor_vtx[triangle.first.p2_idx].erase(triangle.first.p1_idx);
        neighbor_vtx[triangle.first.p2_idx].erase(triangle.first.p3_idx);
        neighbor_vtx[triangle.first.p3_idx].erase(triangle.first.p1_idx);
        neighbor_vtx[triangle.first.p3_idx].erase(triangle.first.p2_idx);
//
        covered_triangle[triangle.first.p1_idx].erase(results[i].triangle_index);
        covered_triangle[triangle.first.p2_idx].erase(results[i].triangle_index);
        covered_triangle[triangle.first.p3_idx].erase(results[i].triangle_index);

        neighbor_vtx.emplace_back();
        neighbor_vtx.emplace_back();
        neighbor_vtx.emplace_back();
        addNeighborVertex(triangle.first.p1_idx, b_idx, d_idx, TYPE1);
        addNeighborVertex(b_idx, d_idx, e_idx, TYPE2);
        addNeighborVertex(b_idx, triangle.first.p2_idx, e_idx, TYPE1);
        addNeighborVertex(d_idx, e_idx, triangle.first.p3_idx, TYPE1);

        covered_triangle.emplace_back();
        covered_triangle.emplace_back();
        covered_triangle.emplace_back();
        addCoveredTriangle(triangle.first.p1_idx, b_idx, d_idx, t1_idx);
        addCoveredTriangle(b_idx, d_idx, e_idx, t2_idx);
        addCoveredTriangle(b_idx, triangle.first.p2_idx, e_idx, t3_idx);
        addCoveredTriangle(d_idx, e_idx, triangle.first.p3_idx, t4_idx);

//          targetTriangleQueue.emplace(Triangle(triangle.first.p1_idx, b_idx, d_idx), TYPE1);
//          targetTriangleQueue.emplace(b_idx, d_idx, e_idx, TYPE2);
//          targetTriangleQueue.emplace(b_idx, triangle.first.p2_idx, e_idx, TYPE1);
//          targetTriangleQueue.emplace(d_idx, e_idx, triangle.first.p3_idx, TYPE1);
      }
    }else if(results[i].type == TYPE2){
      if(diff > 0.06) {
        std::pair<Triangle, int> triangle = std::make_pair(results[i].triangle, results[i].type);

        cv::Point2f p1 = corners[triangle.first.p1_idx];
        cv::Point2f p2 = corners[triangle.first.p2_idx];
        cv::Point2f p3 = corners[triangle.first.p3_idx];

        cv::Point2f x = (p2 - p3) / 2.0;
        cv::Point2f y = (p1 - p3) / 2.0;

        std::vector<Point3Vec> subdiv_ref_triangles, subdiv_target_triangles;
        cv::Point2f a = corners[triangle.first.p1_idx];
        cv::Point2f b = corners[triangle.first.p3_idx] + x + y;
        cv::Point2f c = corners[triangle.first.p3_idx] + y;
        cv::Point2f d = corners[triangle.first.p2_idx];
        cv::Point2f e = corners[triangle.first.p3_idx] + x;
        cv::Point2f f = corners[triangle.first.p3_idx];

        corners.push_back(b);
        int b_idx = static_cast<int>(corners.size() - 1);
        corners.push_back(c);
        int c_idx = static_cast<int>(corners.size() - 1);
        corners.push_back(e);
        int e_idx = static_cast<int>(corners.size() - 1);

        int t1_idx = insertTriangle(triangle.first.p1_idx, b_idx, c_idx, TYPE2);
        int t2_idx = insertTriangle(b_idx, triangle.first.p2_idx, e_idx, TYPE2);
        int t3_idx = insertTriangle(b_idx, c_idx, e_idx, TYPE1);
        int t4_idx = insertTriangle(c_idx, e_idx, triangle.first.p3_idx, TYPE2);

        neighbor_vtx[triangle.first.p1_idx].erase(triangle.first.p2_idx);
        neighbor_vtx[triangle.first.p1_idx].erase(triangle.first.p3_idx);
        neighbor_vtx[triangle.first.p2_idx].erase(triangle.first.p1_idx);
        neighbor_vtx[triangle.first.p2_idx].erase(triangle.first.p3_idx);
        neighbor_vtx[triangle.first.p3_idx].erase(triangle.first.p1_idx);
        neighbor_vtx[triangle.first.p3_idx].erase(triangle.first.p2_idx);

        covered_triangle[triangle.first.p1_idx].erase(results[i].triangle_index);
        covered_triangle[triangle.first.p2_idx].erase(results[i].triangle_index);
        covered_triangle[triangle.first.p3_idx].erase(results[i].triangle_index);

        neighbor_vtx.emplace_back();
        neighbor_vtx.emplace_back();
        neighbor_vtx.emplace_back();
        addNeighborVertex(triangle.first.p1_idx, b_idx, c_idx, TYPE2);
        addNeighborVertex(b_idx, triangle.first.p2_idx, e_idx, TYPE2);
        addNeighborVertex(b_idx, c_idx, e_idx, TYPE1);
        addNeighborVertex(c_idx, e_idx, triangle.first.p3_idx, TYPE2);

        covered_triangle.emplace_back();
        covered_triangle.emplace_back();
        covered_triangle.emplace_back();
        addCoveredTriangle(triangle.first.p1_idx, b_idx, c_idx, t1_idx);
        addCoveredTriangle(b_idx, triangle.first.p2_idx, e_idx, t2_idx);
        addCoveredTriangle(b_idx, c_idx, e_idx, t3_idx);
        addCoveredTriangle(c_idx, e_idx, triangle.first.p3_idx, t4_idx);

//            targetTriangleQueue.emplace(triangle.first.p1_idx, b_idx, c_idx, TYPE2);
//            targetTriangleQueue.emplace(b_idx, triangle.first.p2_idx, e_idx, TYPE2);
//            targetTriangleQueue.emplace(b_idx, c_idx, e_idx, TYPE1);
//            targetTriangleQueue.emplace(c_idx, e_idx, triangle.first.p3_idx, TYPE2);
      }
    }else if(results[i].type == TYPE3) {
        // TODO: あとで書く
    }else if(results[i].type == TYPE4){
        // TODO: あとで書く
    }

    numerator++;
    std::cout << numerator << "/" << denominator << std::endl;
  }


}
