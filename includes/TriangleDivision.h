//
// Created by kasph on 2019/04/08.
//

#ifndef ENCODER_TRIANGLEDIVISION_H
#define ENCODER_TRIANGLEDIVISION_H

#include "Utils.h"
#include "CodingTreeUnit.h"
#include "CollocatedMvTree.h"
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

enum MV_CODE_METHOD {
  SPATIAL,
  Collocated,
  MERGE
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
    void initTriangle(int block_size_x, int block_size_y, int _divide_steps, int _qp, int divide_flag = LEFT_DIVIDE);
    std::vector<std::pair<Point3Vec, int> > getTriangles();
    std::vector<Point3Vec> getTriangleCoordinateList();
    std::vector<Triangle> getTriangleIndexList();
    std::vector<cv::Point2f> getCorners();
    std::vector<int> getNeighborVertexIndexList(int idx);
    std::vector<cv::Point2f> getNeighborVertexCoordinateList(int idx);
    double getDistance(const cv::Point2f &a, const cv::Point2f &b);

    std::vector<Point3Vec> getIdxCoveredTriangleCoordinateList(int target_vertex_idx);
    std::vector<int> getIdxCoveredTriangleIndexList(int idx);

    void subdivision(cv::Mat gaussRefImage, int steps);
    void constructPreviousCodingTree(std::vector<CodingTreeUnit*> trees, int pic_num = 0);

    static SplitResult getSplitTriangle(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, int type);
    bool split(cv::Mat& gaussRefImage, CodingTreeUnit* ctu, CollocatedMvTree* cmt, Point3Vec triangle, int triangle_index, int type, int steps);
    std::vector<int> getSpatialTriangleList(int t_idx);
    cv::Point2f getCollocatedTriangleList(CodingTreeUnit* unit);

    std::vector<Point3Vec> getAllTriangleCoordinateList();
    std::vector<Triangle> getAllTriangleIndexList();
    int divide_steps; // 分割回数

private:
    std::vector<cv::Point2f> corners;
    std::vector<std::pair<Triangle, int> > triangles;
    cv::Mat target_image, ref_image;
    std::vector<std::set<int> > neighbor_vtx;
    std::vector<std::set<int> > covered_triangle;
    std::vector<std::vector<int> > corner_flag;
    std::vector<bool> delete_flag;
    std::vector<bool> isCodedTriangle;
    int block_size_x, block_size_y;
    std::vector<std::vector<CollocatedMvTree*>> previousMvList;
    int coded_picture_num;
    std::vector<cv::Mat> predicted_buf;
    std::vector<std::vector<cv::Point2f>> triangle_mvs;
    int qp;

    int insertTriangle(int p1_idx, int p2_idx, int p3_idx, int type);
    void addNeighborVertex(int p1_idx, int p2_idx, int p3_idx);
    void addCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_no);
    void removeTriangleNeighborVertex(int p1_idx, int p2_idx, int p3_idx);
    void removeTriangleCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_idx);
    int addCorner(cv::Point2f p);
    void addCornerAndTriangle(Triangle triangle, int triangle_index, int type);
    bool isCTU(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3);
    std::vector<int> getDivideOrder(CodingTreeUnit* currentNode);
    void constructPreviousCodingTree(CodingTreeUnit* codingTree, CollocatedMvTree* constructedTree);
    static cv::Point2f getQuantizedMv(cv::Point2f &mv, double quantize_step);
    std::tuple<cv::Point2f, int, MV_CODE_METHOD> getMVD(std::vector<cv::Point2f> mv, double residual, int triangle_idx, cv::Point2f &collocated_mv);
    bool isMvExists(const std::vector<std::pair<cv::Point2f, MV_CODE_METHOD>> &vectors, const cv::Point2f &mv);

};

#endif //ENCODER_TRIANGLEDIVISION_H
