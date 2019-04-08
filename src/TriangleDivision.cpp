//
// Created by kasph on 2019/04/08.
//

#include "TriangleDivision.h"
#include <opencv2/core.hpp>
#include "../includes/Utils.h"

class TriangleDivision {
public:
    TriangleDivision(const cv::Mat &targetImage, const cv::Mat &refImage) : target_image(targetImage),
                                                                            ref_image(refImage) {};

    void initTriangle(int block_size_x, int block_size_y, int devide_flag = LEFT_DEVIDE) {

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
        for(int block_y = 0 ; block_y < block_num_y - 1; block_y++) {
            for(int block_x = 0 ; block_x < block_num_x - 1 ; block_x++) {
                cv::Point2f p1(block_x * block_size_x, block_y * block_size_y);
                cv::Point2f p2((block_x + 1) * block_size_x, block_y * block_size_y);
                cv::Point2f p3(block_x * block_size_x, (block_y  + 1) * block_size_y);
                cv::Point2f p4((block_x + 1) * block_size_x, (block_y + 1) * block_size_y);
                insertTriangle();
            }
        }

    }

private:
    std::vector<Point3Vec> triangles;
    cv::Mat target_image, ref_image;

    /**
     * @fn void insertTriangle(Point3Vec triangle)
     * @param[in] triangle 三角形を表すアレ
     */
    void insertTriangle(Point3Vec triangle) {
        triangles.push_back(triangle);
    }

    /**
     * @fn void insertTriangle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3)
     * @param[in] p1 頂点1
     * @param[in] p2 頂点2
     * @param[in] p3 頂点3
     */
    void insertTriangle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3) {
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
    void insertTriangle(float x1, float y1, float x2, float y2, float x3, float y3) {
        triangles.emplace_back(cv::Point2f(x1, y1), cv::Point2f(x2, y2), cv::Point2f(x3, y3));
    }
};