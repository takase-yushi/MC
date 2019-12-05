//
// Created by kasph on 2019/04/08.
//

#ifndef ENCODER_TRIANGLEDIVISION_H
#define ENCODER_TRIANGLEDIVISION_H

#include "Utils.h"
#include "CodingTreeUnit.h"
#include "CollocatedMvTree.h"
#include "GaussResult.h"
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

    struct SplitResult {
    public:
        Point3Vec t1, t2;
        int t1_type, t2_type;

        SplitResult(const Point3Vec &t1, const Point3Vec &t2, int t1Type, int t2Type);
    };

    TriangleDivision(const cv::Mat &refImage, const cv::Mat &targetImage, const cv::Mat &refGaussImage);

    TriangleDivision();

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

    void constructPreviousCodingTree(std::vector<CodingTreeUnit*> trees, int pic_num = 0);

    static SplitResult getSplitTriangle(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, int type);
    bool split(std::vector<std::vector<std::vector<unsigned char *>>> expand_images, CodingTreeUnit* ctu, CollocatedMvTree* cmt, Point3Vec triangle, int triangle_index, int type, int steps, std::vector<std::vector<int>> &diagonal_line_area_flag);
    std::vector<int> getSpatialTriangleList(int t_idx);
    std::vector<int> getSpatialTriangleListWithParentNode(CodingTreeUnit *ctu);
    cv::Point2f getCollocatedTriangleList(CodingTreeUnit* unit);
    int getCtuCodeLength(std::vector<CodingTreeUnit*> ctus);

    cv::Mat getPredictedImageFromCtu(std::vector<CodingTreeUnit*> ctus, std::vector<std::vector<std::vector<int>>> &area_flag);
    std::vector<Point3Vec> getAllTriangleCoordinateList();
    std::vector<Triangle> getAllTriangleIndexList();
    int divide_steps; // 分割回数

    cv::Mat getMvImage(std::vector<CodingTreeUnit*> ctus);
    std::tuple<std::vector<cv::Point2f>, std::vector<double>> fullpellBlockMatching(Point3Vec triangle, const cv::Mat& target_image, cv::Mat expansion_ref_image, std::vector<std::vector<int>> &area_flag, int triangle_index, CodingTreeUnit *ctu, cv::Point2f fullpell_initial_vector = cv::Point2f(-10000, -10000));

    cv::Mat getPredictedDiagonalImageFromCtu(std::vector<CodingTreeUnit*> ctus, std::vector<std::vector<std::vector<int>>> &area_flag);
    cv::Mat getPredictedColorImageFromCtu(std::vector<CodingTreeUnit*> ctus,std::vector<std::vector<std::vector<int>>> &area_flag, double original_psnr);
    void storeIntraImage();
    std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD> getMVD(std::vector<cv::Point2f> mv, double residual, int triangle_idx, cv::Point2f &collocated_mv, const std::vector<std::vector<int>> &area_flag, CodingTreeUnit* ctu, bool translation_flag, std::vector<cv::Point2f> &pixels);
    cv::Mat getMergeModeColorImageFromCtu(std::vector<CodingTreeUnit*> ctus, std::vector<std::vector<std::vector<int>>> &area_flag);
    void getMergeModeColorImageFromCtu(CodingTreeUnit* ctu, cv::Mat &out, std::vector<std::vector<int>> &area_flag);

    virtual ~TriangleDivision();
//    std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD> getMVD(std::vector<cv::Point2f> mv, double residual, std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> &vectors, cv::Point2f &collocated_mv, const CodingTreeUnit *ctu, bool parallel_flag);

private:
    std::vector<cv::Point2f> corners;
    std::vector<std::pair<Triangle, int> > triangles;
    cv::Mat target_image, ref_image, ref_gauss_image;
    std::vector<std::set<int> > neighbor_vtx;
    std::vector<std::set<int> > covered_triangle;
    std::vector<std::set<int> > same_corner_list; // 実数頂点になった場合、同じ頂点であるとみなす感じの逆引き
    std::vector<std::vector<int> > corner_flag;
    std::vector<bool> delete_flag;
    std::vector<bool> isCodedTriangle;
    int block_size_x, block_size_y;
    std::vector<std::vector<CollocatedMvTree*>> previousMvList;
    int coded_picture_num;
    std::vector<cv::Mat> predicted_buf;
    std::vector<GaussResult> triangle_gauss_results;
    std::vector<std::vector<cv::Mat>> ref_images;
    std::vector<std::vector<cv::Mat>> target_images;
    int qp;
    cv::Mat expansion_ref;
    unsigned char *ref_hevc;
    unsigned char **expansion_ref_uchar;
    std::vector<std::vector<bool>> intra_flag;
    cv::Mat intra_tmp_image;

    void getPredictedColorImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out, std::vector<std::vector<int>> &area_flag, double original_psnr, std::vector<cv::Scalar> &colors);
    int insertTriangle(int p1_idx, int p2_idx, int p3_idx, int type);
    void addNeighborVertex(int p1_idx, int p2_idx, int p3_idx);
    void addCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_no);
    void removeTriangleNeighborVertex(int p1_idx, int p2_idx, int p3_idx);
    void removeTriangleCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_idx);
    int getCornerIndex(cv::Point2f p);
    void addCornerAndTriangle(Triangle triangle, int triangle_index, int type);
    void constructPreviousCodingTree(CodingTreeUnit* codingTree, CollocatedMvTree* constructedTree);
    static cv::Point2f getQuantizedMv(cv::Point2f &mv, double quantize_step);
    bool isMvExists(const std::vector<std::pair<cv::Point2f, MV_CODE_METHOD>> &vectors, const cv::Point2f &mv);
    static bool isMvExists(const std::vector<Point3Vec> &vectors, const std::vector<cv::Point2f> &mvs);
    void eraseTriangle(int t_idx);
    void getPredictedImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out, std::vector<std::vector<int>> &area_Flag);
    int getCtuCodeLength(CodingTreeUnit *ctu);
    void drawMvImage(cv::Mat &out, CodingTreeUnit *ctu);
    void getPredictedDiagonalImageFromCtu(CodingTreeUnit* ctu, std::vector<std::vector<int>> &area_flag, const cv::Mat &out);
    void setIntraImage(std::vector<cv::Point2f> pixels, Point3Vec triangle);
    bool isIntraAvailable(Point3Vec triangle);

};

#endif //ENCODER_TRIANGLEDIVISION_H
