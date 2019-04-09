//
// Created by kasph on 2019/04/08.
//

#include "../includes/TriangleDivision.h"
#include <opencv2/core.hpp>
#include "../includes/Utils.h"


TriangleDivision::TriangleDivision(const cv::Mat &refImage, const cv::Mat &targetImage) : target_image(targetImage),
                                                                                          ref_image(refImage) {};

void TriangleDivision::initTriangle(int block_size_x, int block_size_y, int divide_flag) {

//        // block_x_size, block_y_sizeで割って, あまりは捨ててしまう処理
//        int width = ((int)(target_image.cols / block_x_size) * block_x_size);
//        int height = ((int)(target_image.rows / block_y_size) * block_y_size);

//        cv::Rect image_size(0, 0, width, height);

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
    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {
        for(int block_x = 0 ; block_x < block_num_x ; block_x++) {
            cv::Point2f p1(      block_x * block_size_x,        block_y * block_size_y);
            cv::Point2f p2((block_x + 1) * block_size_x,        block_y * block_size_y);
            cv::Point2f p3(      block_x * block_size_x,  (block_y + 1) * block_size_y);
            cv::Point2f p4((block_x + 1) * block_size_x,  (block_y + 1) * block_size_y);
            insertTriangle(p1, p2, p3);
            insertTriangle(p2, p3, p4);
        }
    }

}

/**
 * @fn std::vector<Point3Vec> getTriangleCoordinateList()
 * @brief
 * @return
 */
std::vector<Point3Vec> TriangleDivision::getTriangleCoordinateList() {
    return triangles;
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

    triangles.emplace_back(p1, p2, p3);
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
    triangles.emplace_back(cv::Point2f(x1, y1), cv::Point2f(x2, y2), cv::Point2f(x3, y3));
}
