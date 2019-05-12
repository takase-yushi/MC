//
// Created by kasph on 2019/04/08.
//

#ifndef ENCODER_TRIANGLEDIVISION_H
#define ENCODER_TRIANGLEDIVISION_H

#include "Utils.h"
#include "CodingTreeUnit.h"
#include <set>
#include <vector>

#define LEFT_DIVIDE 1
#define RIGHT_DIVIDE 2

/**
 * （初期の分割を，右上から左下に対角線を引くタイプの場合と仮定すると）
 * --------------
 * TYPE1の分割
 *   -----------------
 *   |             *
 *   |          *
 *   |       *
 *   |   *
 *   |*
 *
 * TYPE2の分割
 *
 *                  * |
 *             *      |
 *         *          |
 *     *              |
 *  ------------------|
 *
 * TYPE3の分割
 *    -----------------
 *     *              |
 *         *          |
 *             *      |
 *                *   |
 *                    |
 * TYPE4の分割
 *
 *    | *
 *    |     *
 *    |        *
 *    |            *
 *    ----------------|
 *
 * TYPE5の分割
 *  ------------------
 *   *             *
 *     *         *
 *        *   *
 *          *
 *
 * TYPE6の分割
 *  |*
 *  |  *
 *  |     *
 *  |        *
 *  |     *
 *  |  *
 *  |*
 *
 * TYPE7の分割
 *           *
 *        *     *
 *      *         *
 *   *              *
 *  ------------------
 *
 * TYPE8の分割
 *                 |
 *              *  |
 *           *     |
 *         *       |
 *           *     |
 *              *  |
 *                 |
 */

enum DIVIDE {
    TYPE1,
    TYPE2,
    TYPE3,
    TYPE4,
    TYPE5,
    TYPE6,
    TYPE7,
    TYPE8
};

class TriangleDivision {

public:

    class GaussResult{
    public:
        int triangle_index;

        GaussResult(int triangleIndex, const Triangle &triangle, int type, double rmseBeforeSubdiv,
                    double rmseAfterSubdiv);

        GaussResult();

        Triangle triangle;
        int type;
        double RMSE_before_subdiv;
        double RMSE_after_subdiv;

    };

    struct SplitResult {
    public:
        Point3Vec t1, t2;
        int t1_type, t2_type;

        SplitResult(const Point3Vec &t1, const Point3Vec &t2, int t1Type, int t2Type);
    };

    TriangleDivision(const cv::Mat &refImage, const cv::Mat &targetImage);
    void initTriangle(int block_size_x, int block_size_y, int divide_flag = LEFT_DIVIDE);
    std::vector<std::pair<Point3Vec, int> > getTriangles();
    std::vector<Point3Vec> getTriangleCoordinateList();
    std::vector<Triangle> getTriangleIndexList();
    std::vector<cv::Point2f> getCorners();
    std::vector<int> getNeighborVertexIndexList(int idx);
    std::vector<cv::Point2f> getNeighborVertexCoordinateList(int idx);
    double getDistance(const cv::Point2f &a, const cv::Point2f &b);

    std::vector<Point3Vec> getIdxCoveredTriangleCoordinateList(int target_vertex_idx);
    std::vector<Triangle> getIdxCoveredTriangleIndexList(int idx);

    void subdivision(cv::Mat gaussRefImage, int steps);


    static SplitResult getSplitTriangle(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, int type);
    bool split(cv::Mat& gaussRefImage, CodingTreeUnit* ctu, Point3Vec triangle, int triangle_index, int type, int steps);

    std::vector<Point3Vec> getAllTriangleCoordinateList();
    std::vector<Triangle> getAllTriangleIndexList();

private:
    std::vector<cv::Point2f> corners;
    std::vector<std::pair<Triangle, int> > triangles;
    cv::Mat target_image, ref_image;
    std::vector<std::set<int> > neighbor_vtx;
    std::vector<std::set<int> > covered_triangle;
    std::vector<std::vector<int> > corner_flag;
    std::vector<bool> delete_flag;

    int insertTriangle(int p1_idx, int p2_idx, int p3_idx, int type);
    void addNeighborVertex(int p1_idx, int p2_idx, int p3_idx);
    void addCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_no);
    void removeTriangleNeighborVertex(int p1_idx, int p2_idx, int p3_idx);
    void removeTriangleCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_idx);
    int addCorner(cv::Point2f p);
    int addCorner(Triangle triangle, int triangle_index, int type);

};

#endif //ENCODER_TRIANGLEDIVISION_H
