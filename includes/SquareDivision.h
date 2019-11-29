//
// Created by takahiro on 2019/07/19.
//

#ifndef ENCODER_SQUAREDIVISION_H
#define ENCODER_SQUAREDIVISION_H

#include "Utils.h"
#include "CodingTreeUnit.h"
#include "CollocatedMvTree.h"
#include "Flags.h"
#include <set>
#include <vector>

#define LEFT_DIVIDE 1
#define RIGHT_DIVIDE 2

/*
 *
 * 1の分割
 *   -----------------
 *   |               |
 *   |               |
 *   |               |
 *   |               |
 *   -----------------
 *
 * 2の分割
 *
 *   ---------
 *   |       |
 *   |       |
 *   |       |
 *   |       |
 *   ---------
 *
**/

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

    GaussResult(const std::vector<cv::Point2f> &mvWarping, const cv::Point2f &mvTranslation, double residual_warping, double residual_translation,
                int squareSize, bool translationFlag, double residual_bm);
    std::vector<cv::Point2f> mv_warping, original_mv_warping;
    cv::Point2f mv_translation, original_mv_translation;
    double residual_warping;
    double residual_translation;
    int square_size;
    bool translation_flag;
    double residual_bm;
    MV_CODE_METHOD method;
};

    struct SplitResult {
    public:
        Point4Vec s1, s2;
        int s_type; //, t2_type;

        SplitResult(const Point4Vec &s1, const Point4Vec &s2, int type);
    };

    SquareDivision(const cv::Mat &refImage, const cv::Mat &targetImage, const cv::Mat &refGaussImage);

    SquareDivision();

    void initSquare(int block_size_x, int block_size_y, int _divide_steps, int _qp, int divide_flag = LEFT_DIVIDE);
    std::vector<Point4Vec> getSquares();
    std::vector<Point4Vec> getSquareCoordinateList();
    std::vector<Square> getSquareIndexList();
    std::vector<cv::Point2f> getCorners();
    std::vector<int> getNeighborVertexIndexList(int idx);
    std::vector<cv::Point2f> getNeighborVertexCoordinateList(int idx);
    double getDistance(const cv::Point2f &a, const cv::Point2f &b);

    std::vector<Point4Vec> getIdxCoveredSquareCoordinateList(int target_vertex_idx);
    std::vector<int> getIdxCoveredSquareIndexList(int idx);

    void constructPreviousCodingTree(std::vector<CodingTreeUnit*> trees, int pic_num = 0);

    static SplitResult getSplitSquare(const cv::Point2f& p1, const cv::Point2f& p2, const cv::Point2f& p3, const cv::Point2f& p4, int type);
    void addReferenceBlock(Point4Vec subdiv_target_square, int square_index);
    bool split(std::vector<std::vector<std::vector<unsigned char **>>> expand_images, CodingTreeUnit* ctu, CollocatedMvTree* cmt, Point4Vec square, int square_index, int square_number, int steps);
    std::tuple< std::vector<std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >>>, std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> > getSpatialSquareList(int s_idx, bool translation_flag);
    std::tuple< std::vector<std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >>>, std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> > getMergeSquareList(int s_idx, bool translation_flag, Point4Vec coordinate);
    std::vector<std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >>> getMerge2SquareList(int square_idx, Point4Vec coordinate);
//    cv::Point2f getCollocatedSquareList(CodingTreeUnit* unit);
    int getCtuCodeLength(std::vector<CodingTreeUnit*> ctus);

    cv::Mat getPredictedImageFromCtu(std::vector<CodingTreeUnit*> ctus);
    std::vector<Point4Vec> getAllSquareCoordinateList();
    std::vector<Square> getAllSquareIndexList();
    int divide_steps; // 分割回数

    cv::Mat getMvImage(std::vector<CodingTreeUnit*> ctus);
    std::tuple<std::vector<cv::Point2f>, std::vector<double>> blockMatching(Point4Vec square, const cv::Mat& target_image, cv::Mat expansion_ref_image, int square_index, CodingTreeUnit *ctu);

    cv::Mat getPredictedDiagonalImageFromCtu(std::vector<CodingTreeUnit*> ctus);
    cv::Mat getPredictedColorImageFromCtu(std::vector<CodingTreeUnit*> ctus, double original_psnr, int select);
    std::tuple<double, int, std::vector<cv::Point2f>, int, MV_CODE_METHOD, FlagsCodeSum, Flags> getMVD(std::vector<cv::Point2f> mv, double residual, int square_idx, int square_number, cv::Point2f &collocated_mv, CodingTreeUnit* ctu, bool translation_flag, std::vector<cv::Point2f> &pixels, int steps);
    double  getRDCost(std::vector<cv::Point2f> mv, double residual, int square_idx, cv::Point2f &collocated_mv, CodingTreeUnit* ctu, std::vector<cv::Point2f> &pixels, std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> vectors);

    virtual ~SquareDivision();

private:
    std::vector<cv::Point2f> corners;
    std::vector<Square> squares;
    cv::Mat target_image, ref_image, ref_gauss_image;
    std::vector<int> covered_square;  //ある頂点が含まれる四角形のインデックス　　　　添え字 : インデックス(頂点)番号
    std::vector<std::vector<int> > corner_flag;
    std::vector<std::vector<int> > reference_block_list;
    std::vector<std::vector<int> > merge_reference_block_list;
    std::vector<bool> isCodedSquare;
    int block_size_x, block_size_y;
    std::vector<std::vector<CollocatedMvTree*>> previousMvList;
    int coded_picture_num;
    std::vector<cv::Mat> predicted_buf;
    std::vector<GaussResult> square_gauss_results;
    std::vector<std::vector<cv::Mat>> ref_images;
    std::vector<std::vector<cv::Mat>> target_images;
    int qp;
    int flags_code = 0;
    cv::Mat expansion_ref;
    unsigned char **ref_hevc;
    unsigned char **expansion_ref_uchar;

    void getPredictedColorImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out, double original_psnr, std::vector<cv::Scalar> &colors);
    int insertSquare(int p1_idx, int p2_idx, int p3_idx, int p4_idx);
    void addNeighborVertex(int p1_idx, int p2_idx, int p3_idx, int p4_idx);
    void addCoveredSquare(int p1_idx, int p2_idx, int p3_idx, int p4_idx, int square_no);
    void removeSquareNeighborVertex(int p1_idx, int p2_idx, int p3_idx, int p4_idx);
    int getOrAddCornerIndex(cv::Point2f p);
    int getCornerIndex(cv::Point2f p);
    void addCornerAndSquare(Square square);
    void eraseCornerFlag(Point4Vec s1, Point4Vec s2, Point4Vec s3, Point4Vec s4);
    void constructPreviousCodingTree(CodingTreeUnit* codingTree, CollocatedMvTree* constructedTree);
    static cv::Point2f getQuantizedMv(cv::Point2f &mv, double quantize_step);
    bool isMvExists(const std::vector<std::pair<cv::Point2f, MV_CODE_METHOD>> &vectors, const cv::Point2f &mv);
    bool isMvExists(const std::vector<Point3Vec> &vectors, const std::vector<cv::Point2f> &mvs);
    void eraseSquare(int s_idx);
    void getPredictedImageFromCtu(CodingTreeUnit *ctu, cv::Mat &out);
    int getCtuCodeLength(CodingTreeUnit *ctu);
    void drawMvImage(cv::Mat &out, CodingTreeUnit *ctu);
    void getPredictedDiagonalImageFromCtu(CodingTreeUnit* ctu, const cv::Mat &out);

};


#endif //ENCODER_SQUAREDIVISION_H