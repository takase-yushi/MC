//
// Created by takahiro on 2019/07/19.
//

#ifndef ENCODER_SQUAREDIVISION_H
#define ENCODER_SQUAREDIVISION_H

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

//enum DIVIDE {
//    TYPE1,
//    TYPE2,
//    TYPE3,
//    TYPE4,
//    TYPE5,
//    TYPE6,
//    TYPE7,
//    TYPE8
//};
//
//enum MV_CODE_METHOD {
//    SPATIAL,
//    Collocated,
//    MERGE
//};

class SquareDivision {

public:

    class GaussResult{
    public:
        GaussResult();

        GaussResult(const std::vector<cv::Point2f> &mvWarping, const cv::Point2f &mvParallel, double residual,
                    int squareSize, bool parallelFlag, double residual_bm, double residual_newton);
        std::vector<cv::Point2f> mv_warping;
        cv::Point2f mv_parallel;
        double residual;
        int square_size;
        bool parallel_flag;
        double residual_bm;
        double residual_newton;
    };

    struct SplitResult {
    public:
        Point3Vec t1, t2;
        int t1_type, t2_type;

        SplitResult(const Point3Vec &t1, const Point3Vec &t2, int t1Type, int t2Type);
    };

    SquareDivision(const cv::Mat &refImage, const cv::Mat &targetImage, const cv::Mat &refGaussImage);
    void initSquare(int block_size_x, int block_size_y, int _divide_steps, int _qp, int divide_flag = LEFT_DIVIDE);
    std::vector<std::pair<Point3Vec, int> > getSquares();
    std::vector<Point3Vec> getSquareCoordinateList();
    std::vector<Square> getSquareIndexList();
    std::vector<cv::Point2f> getCorners();
    std::vector<int> getNeighborVertexIndexList(int idx);
    std::vector<cv::Point2f> getNeighborVertexCoordinateList(int idx);
    double getDistance(const cv::Point2f &a, const cv::Point2f &b);

    std::vector<Point3Vec> getIdxCoveredSquareCoordinateList(int target_vertex_idx);
    std::vector<int> getIdxCoveredSquareIndexList(int idx);

    void constructPreviousCodingTree(std::vector<CodingTreeUnit*> trees, int pic_num = 0);

    static SplitResult getSplitSquare(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, int type);
    bool split(std::vector<std::vector<std::vector<unsigned char **>>> expand_images, CodingTreeUnit* ctu, CollocatedMvTree* cmt, Point3Vec square, int square_index, int type, int steps, std::vector<std::vector<int>> &diagonal_line_area_flag);
    std::vector<int> getSpatialSquareList(int t_idx);
    cv::Point2f getCollocatedSquareList(CodingTreeUnit* unit);
    int getCtuCodeLength(std::vector<CodingTreeUnit*> ctus);

    cv::Mat getPredictedImageFromCtu(std::vector<CodingTreeUnit*> ctus, std::vector<std::vector<std::vector<int>>> &area_flag);
    std::vector<Point3Vec> getAllSquareCoordinateList();
    std::vector<Square> getAllSquareIndexList();
    int divide_steps; // 分割回数

    cv::Mat getMvImage(std::vector<CodingTreeUnit*> ctus);

    cv::Mat getPredictedDiagonalImageFromCtu(std::vector<CodingTreeUnit*> ctus, std::vector<std::vector<std::vector<int>>> &area_flag);
    cv::Mat getPredictedColorImageFromCtu(std::vector<CodingTreeUnit*> ctus,std::vector<std::vector<std::vector<int>>> &area_flag, double original_psnr);

private:
    std::vector<cv::Point2f> corners;
    std::vector<std::pair<Square, int> > squares;
    cv::Mat target_image, ref_image, ref_gauss_image;
    std::vector<std::set<int> > neighbor_vtx;
    std::vector<std::set<int> > covered_square;　　//ある頂点に含まれる四角形のインデックス　　　　添え字 : インデックス(頂点)番号　setは集合なので同じ要素は入らない
    std::vector<std::vector<int> > corner_flag;
    std::vector<bool> delete_flag;
    std::vector<bool> isCodedSquare;
    int block_size_x, block_size_y;
    std::vector<std::vector<CollocatedMvTree*>> previousMvList;
    int coded_picture_num;
    std::vector<cv::Mat> predicted_buf;
    std::vector<GaussResult> square_gauss_results;
    std::vector<std::vector<cv::Mat>> ref_images;
    std::vector<std::vector<cv::Mat>> target_images;
    int qp;
    cv::Mat expansion_ref;
    unsigned char **ref_hevc;
    unsigned char **expansion_ref_uchar;

    void getPredictedColorImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out, std::vector<std::vector<int>> &area_flag, double original_psnr, std::vector<cv::Scalar> &colors);
    int insertSquare(int p1_idx, int p2_idx, int p3_idx, int type);
    void addNeighborVertex(int p1_idx, int p2_idx, int p3_idx);
    void addCoveredSquare(int p1_idx, int p2_idx, int p3_idx, int square_no);
    void removeSquareNeighborVertex(int p1_idx, int p2_idx, int p3_idx);
    void removeSquareCoveredSquare(int p1_idx, int p2_idx, int p3_idx, int square_idx);
    int addCorner(cv::Point2f p);
    void addCornerAndSquare(Square square, int square_index, int type);
    std::vector<int> getDivideOrder(CodingTreeUnit* currentNode);
    void constructPreviousCodingTree(CodingTreeUnit* codingTree, CollocatedMvTree* constructedTree);
    static cv::Point2f getQuantizedMv(cv::Point2f &mv, double quantize_step);
    std::tuple<double, int, cv::Point2f, int, MV_CODE_METHOD> getMVD(std::vector<cv::Point2f> mv, double residual, int square_idx, cv::Point2f &collocated_mv, const std::vector<std::vector<int>> &area_flag, CodingTreeUnit* ctu);
    bool isMvExists(const std::vector<std::pair<cv::Point2f, MV_CODE_METHOD>> &vectors, const cv::Point2f &mv);
    void eraseSquare(int t_idx);
    void getPredictedImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out, std::vector<std::vector<int>> &area_Flag);
    int getCtuCodeLength(CodingTreeUnit *ctu);
    void drawMvImage(cv::Mat &out, CodingTreeUnit *ctu);
    void getPredictedDiagonalImageFromCtu(CodingTreeUnit* ctu, std::vector<std::vector<int>> &area_flag, const cv::Mat &out);


};


#endif //ENCODER_SQUAREDIVISION_H