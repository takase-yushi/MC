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


/**
 * @fn void TriangleDivision::initTriangle(int block_size_x, int block_size_y, int divide_flag)
 * @brief 三角形を初期化する
 * @param[in] block_size_x
 * @param[in] block_size_y
 * @param[in] divide_flag
 */
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

}

/**
 * @fn std::vector<Point3Vec> getTriangleCoordinateList()
 * @brief 現在存在する三角形の集合(座標)を返す
 * @return 三角形の集合（座標）
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
 * @fn std::vector<Triangle> TriangleDivision::getTriangleIndexList()
 * @brief 現在存在する三角形の集合(インデックス)を返す
 * @return 三角形の集合（インデックス）
 */
std::vector<Triangle> TriangleDivision::getTriangleIndexList() {
    std::vector<Triangle> v;
    for(const auto t : triangles) {
        v.emplace_back(t.first);
    }
    return v;
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

    Triangle triangle(v[0].second, v[1].second, v[2].second);

    triangles.emplace_back(triangle, type);
    covered_triangle.emplace_back();

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
 * @param[in] idx 頂点のインデックス
 * @return 三角形の集合(座標で返される)
 */
std::vector<Point3Vec> TriangleDivision::getIdxCoveredTriangleCoordinateList(int idx) {
  std::set<int> s = covered_triangle[idx];
  std::vector<Point3Vec> v;
  std::cout << corners[idx] << std::endl;
  for(auto idx : s) {
    Triangle triangle = triangles[idx].first;
    v.emplace_back(corners[triangle.p1_idx], corners[triangle.p2_idx], corners[triangle.p3_idx]);
  }

  return v;
}

/**
 * @fn std::vector<Triangle> TriangleDivision::getIdxCoveredTriangleIndexList(int idx)
 * @brief 指定の頂点を含む三角形の集合（座標）を返す
 * @param[in] idx 頂点のインデックス
 * @return 三角形の集合（座標）
 */
std::vector<Triangle> TriangleDivision::getIdxCoveredTriangleIndexList(int idx) {
    return std::vector<Triangle>(); // TODO: 実装
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
    corners.emplace_back(p);
    neighbor_vtx.emplace_back();
}

/**
 * @fn void TriangleDivision::subdivision()
 * @brief 三角パッチの再分割を行う
 * @param[in] gaussRefImage ガウスニュートン法の第1層目の画像
 * @param[in] steps 何回分割するかを示す値
 */
void TriangleDivision::subdivision(cv::Mat gaussRefImage, int steps) {
  // 一つ前に分割されている場合、更に分割すればよいが
  // 分割されていないのであればこれ以上分割する必要はない
  std::vector<bool> previousDivideFlag(triangles.size(), true);

  for(int step = 0 ; step < steps ; step++) {
    std::vector<GaussResult> results(triangles.size());
    previousDivideFlag.resize(triangles.size());

    int denominator = triangles.size();
    int numerator = 0;

    const double divide_th = 0.05;

    std::cout << "triangles.size(): " << triangles.size() << std::endl;
    std::cout << "results.size():" << results.size() << std::endl;
    std::cout << "previousDivideFlag.size():" << previousDivideFlag.size() << std::endl;

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

      // 三角形を分割し，その結果でRMSEを見る
      if (triangle.second == TYPE1) {
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

        for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
          RMSE_after_subdiv += Gauss_Newton(gaussRefImage, target_image, ref_image, subdiv_target_triangles[j],
                                            subdiv_ref_triangles[j], triangle_size);
          triangle_size_sum += triangle_size;
        }

        RMSE_after_subdiv /= (double) triangle_size_sum;
        std::cout << "before:" << RMSE_before_subdiv << " after:" << RMSE_after_subdiv << std::endl;
        results[i] = GaussResult(i, triangle.first, triangle.second, RMSE_before_subdiv, RMSE_after_subdiv);
      } else if (triangle.second == TYPE2) {
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

        for (int j = 0; j < (int) subdiv_ref_triangles.size(); j++) {
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

                      int c_idx = corner_flag[(int)c.y][(int)c.x];
                      if (c_idx == -1) {
                          c_idx = addCorner(c) - 1;
                          corner_flag[(int)c.y][(int)c.x] = c_idx;
                      }

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

                      int b_idx = corner_flag[(int)b.y][(int)b.x];
                      if (b_idx == -1) {
                          b_idx = addCorner(b) - 1;
                          corner_flag[(int)b.y][(int)b.x] = b_idx;
                      }

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

                      int c_idx = corner_flag[(int)c.y][(int)c.x];
                      if (c_idx == -1) {
                          c_idx = addCorner(c) - 1;
                          corner_flag[(int)c.y][(int)c.x] = c_idx;
                      }

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

                      int b_idx = corner_flag[(int)b.y][(int)b.x];
                      if (b_idx == -1) {
                          b_idx = addCorner(b) - 1;
                          corner_flag[(int)b.y][(int)b.x] = b_idx;
                      }

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

                      int b_idx = corner_flag[(int)b.y][(int)b.x];
                      if (b_idx == -1) {
                          b_idx = addCorner(b) - 1;
                          corner_flag[(int)b.y][(int)b.x] = b_idx;
                      }

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

                      int b_idx = corner_flag[(int)b.y][(int)b.x];
                      if (b_idx == -1) {
                          b_idx = addCorner(b) - 1;
                          corner_flag[(int)b.y][(int)b.x] = b_idx;
                      }

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

                      int c_idx = corner_flag[(int)c.y][(int)c.x];
                      if (c_idx == -1) {
                          c_idx = addCorner(c) - 1;
                          corner_flag[(int)c.y][(int)c.x] = c_idx;
                      }

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

                      int c_idx = corner_flag[(int)c.y][(int)c.x];
                      if (c_idx == -1) {
                          c_idx = addCorner(c) - 1;
                          corner_flag[(int)c.y][(int)c.x] = c_idx;
                      }

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
                  }
                  break;
              default:
                  break;
          }
      }else{
          previousDivideFlag[results[i].triangle_index] = false;
      }

      numerator++;
      std::cout << "subdiv: "<< numerator << "/" << denominator << std::endl;
    }
  }

}

