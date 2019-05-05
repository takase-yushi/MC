//
// Created by kasph on 2019/05/05.
//

#ifndef ENCODER_RECONSTRUCTION_H
#define ENCODER_RECONSTRUCTION_H

#include <vector>
#include "CodingTreeUnit.h"
#include "Utils.h"

class Reconstruction {
public:
    void init(int block_size_x, int block_size_y, int divide_flag);
    int addCorner(cv::Point2f p);
    void reconstructionTriangle(std::vector<CodingTreeUnit*> ctu);

    Reconstruction(const cv::Mat &gaussRefImage);

    int insertTriangle(int p1_idx, int p2_idx, int p3_idx, int type);
    int insertTriangle(std::vector<std::pair<Triangle, int> >& target_triangles, int p1_idx, int p2_idx, int p3_idx, int type);
    std::vector<Point3Vec> getTriangleCoordinateList();

    std::vector<std::pair<Triangle, int> > triangles;
    std::vector<cv::Point2f> corners;
    cv::Mat gaussRefImage;
    std::vector<std::pair<Triangle, int> > init_triangles;

private:
    bool isAdditionalPoint(cv::Point2f p);
    void reconstructionTriangle(CodingTreeUnit* ctu, Point3Vec triangle, int type);
    std::vector<std::pair<cv::Point2f, int>> sortTriangle(std::pair<cv::Point2f, int> a, std::pair<cv::Point2f, int> b, std::pair<cv::Point2f, int> c);

    std::vector<bool> delete_flag;
    std::vector<std::vector<int> > corner_flag;

};


#endif //ENCODER_RECONSTRUCTION_H
