//
// Created by kasph on 2019/04/08.
//

#ifndef ENCODER_TRIANGLEDIVISION_H
#define ENCODER_TRIANGLEDIVISION_H

#include "Utils.h"
#include <set>
#include <vector>

#define LEFT_DEVIDE 1
#define RIGHT_DEVIDE 2

class TriangleDivision {

public:
    TriangleDivision(const cv::Mat &refImage, const cv::Mat &targetImage);
    void initTriangle(int block_size_x, int block_size_y, int divide_flag = LEFT_DEVIDE);
    std::vector<Point3Vec> getTriangleCoordinateList();
    std::vector<Triangle> getTriangleIndexList();
    std::vector<cv::Point2f> getCorners();
    std::vector<int> getNeighborVertexIndexList(int idx);
    std::vector<cv::Point2f> getNeighborVertexCoordinateList(int idx);
    double getDistance(const cv::Point2f &a, const cv::Point2f &b);

private:
    std::vector<cv::Point2f> corners;
    std::vector<Triangle> triangles;
    cv::Mat target_image, ref_image;
    std::vector<std::set<int> > neighbor_vtx;

    void insertTriangle(Point3Vec triangle);
    void insertTriangle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3);
    void insertTriangle(int p1_idx, int p2_idx, int p3_idx);
    void insertTriangle(float x1, float y1, float x2, float y2, float x3, float y3);
    void addNeighborVertex(int p1_idx, int p2_idx, int p3_idx, int divide_flag);
};

#endif //ENCODER_TRIANGLEDIVISION_H
