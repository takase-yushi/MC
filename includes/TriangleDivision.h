//
// Created by kasph on 2019/04/08.
//

#ifndef ENCODER_TRIANGLEDIVISION_H
#define ENCODER_TRIANGLEDIVISION_H

#include "Utils.h"

#define LEFT_DEVIDE 1
#define RIGHT_DEVIDE 2

class TriangleDivision {

public:
    TriangleDivision(const cv::Mat &refImage, const cv::Mat &targetImage);
    void initTriangle(int block_size_x, int block_size_y, int divide_flag = LEFT_DEVIDE);
    std::vector<Point3Vec> getTriangleCoordinateList();

private:
    std::vector<Point3Vec> triangles;
    cv::Mat target_image, ref_image;

    void insertTriangle(Point3Vec triangle);
    void insertTriangle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3);
    void insertTriangle(float x1, float y1, float x2, float y2, float x3, float y3);
};

#endif //ENCODER_TRIANGLEDIVISION_H
