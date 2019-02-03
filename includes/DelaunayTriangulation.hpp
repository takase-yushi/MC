//
// Created by keisuke on 2017/12/06.
//

#ifndef ENCODER_MYDELAUNAY_HPP
#define ENCODER_MYDELAUNAY_HPP


#include <opencv2/core/types.hpp>
#include <set>
#include <unordered_set>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/mat.hpp>
#include "Rectangle.hpp"
#include "Utils.h"

class DelaunayTriangulation {
public:


    /**
     * @struct PointCode
     * @brief 頂点の符号化に使うclass
     */
  struct PointCode {
    int prev_id;       ///< 参照する以前のid(通常は0, Endフラグの場合は0以外の正の数, 最初の場合は-1)
    cv::Point2f coord; ///< 差分ベクトル

    PointCode(int prev_id, const cv::Point2d coord) : prev_id(prev_id), coord(coord) {}
  };

  DelaunayTriangulation();

  explicit DelaunayTriangulation(Rectangle rect);

  void init(Rectangle rectangle);

  void insert(const std::vector<cv::Point2f>& corners);

  int insert(cv::Point2f pt);

  void getTriangleList(std::vector<cv::Vec6f> &triangle_list);

  std::vector<PointCode> getPointCoordinateCode(const std::vector<cv::Point2f>& corners, int mode);

  std::vector<cv::Point2f> getPointMotionVectorCode(const std::vector<cv::Point2f>& corners, const std::vector<cv::Point2f>& mv);

  cv::Mat getDecodedCornerImage(std::vector<PointCode>& code, const cv::Mat& target_image, int mode);

  cv::Mat getDecodedMotionVectorImage(std::vector<cv::Point2f>& code, std::vector<cv::Point2f>& corners, cv::Mat target_image);

  std::vector<cv::Point2f> getNeighborVertex(int pt);

  std::vector<int> getNeighborVertexNum(int pt);

  //double getDistance(const cv::Point2d& a, const cv::Point2d& b);

  std::priority_queue<int> getUnnecessaryPoint(const std::vector<cv::Point2f> &mv, double th, const cv::Mat& target_image);

  std::vector<Triangle> Get_triangles_around(int idx,std::vector<cv::Point2f>corners,std::vector<bool> &flag_around );

  std::vector<Triangle> Get_triangles_later(DelaunayTriangulation md,int idx,std::vector<cv::Point2f> corners,std::vector<bool> flag_around );

  std::vector<cv::Point2f> repair_around(std::vector<cv::Point2f> &corners,const cv::Mat target);

  void Sort_Coners(std::vector<cv::Point2f> &corners);

  void serch_wrong(std::vector<cv::Point2f>corners_later,cv::Mat target,bool *skip_flag);

  void inter_div(std::vector<cv::Point2f>&corners,cv::Point2f corner,std::vector<Triangle> triangles, int t);

  double neighbor_distance(std::vector<cv::Point2f>&corners, int idx);

  double getDistance(const cv::Point2d& a, const cv::Point2d& b);

  std::vector<Triangle> Get_triangles(std::vector<cv::Point2f> corners);

private:


  /**
   * @def RASTER_SCAN 1
   * @brief ラスタスキャンを表す定数
   * @details
   *  スキャン順がラスタスキャンであることを示す
   */
  #define RASTER_SCAN 1

  /**
   * @def QUEUE 2
   * @brief キューに入れてスキャンする
   * @details
   *  キューに入れてスキャンするときの定数
   */
  #define QUEUE 2

  enum {
    LOCATION_ERROR = -2,
    LOCATION_OUT_OF_RECTANGLE = -1,
    LOCATION_INSIDE = 0,
    LOCATION_ON_THE_VERTEX = 1,
    LOCATION_ON_THE_EDGE = 2,
  };

//    NEXT_AROUND_ORG   = 0x00, 0000 0000 eOnext           00
//    PREV_AROUND_RIGHT = 0x02  0000 0010 reversed eDnext  10
//    PREV_AROUND_LEFT  = 0x20, 0010 0000 reversed eOnext  00
//    NEXT_AROUND_DST   = 0x22, 0010 0010 eDnext           10

//    PREV_AROUND_ORG   = 0x11, 0001 0001 reversed eRnext  01
//    NEXT_AROUND_LEFT  = 0x13, 0001 0011 eLnext           11
//    NEXT_AROUND_RIGHT = 0x31, 0011 0001 eRnext           01
//    PREV_AROUND_DST   = 0x33, 0011 0011 reversed eLnext  11
  enum {
    NEXT_AROUND_ORG   = 0x00,
    NEXT_AROUND_DST   = 0x22,
    PREV_AROUND_ORG   = 0x11,
    PREV_AROUND_DST   = 0x33,
    NEXT_AROUND_LEFT  = 0x13,
    NEXT_AROUND_RIGHT = 0x31,
    PREV_AROUND_LEFT  = 0x20,
    PREV_AROUND_RIGHT = 0x02
  };

  //
  struct QEdge {
    QEdge();
    QEdge(int edge);

    int next_edge[4];
    int pt[4];
  };

  // 特徴点
  struct Vertex {
    Vertex();

    Vertex(cv::Point2f _pt, bool _isvirtual, int _firstEdge);

    int firstEdge;
    int type;
    cv::Point2f pt;
  };


  std::vector<Vertex> vertex;
  std::vector<QEdge> edges;
  std::vector<int> idx_converer;

  int recent_edge;          // 最近参照したエッジ
  cv::Point2f upper_left;   // 左上
  cv::Point2f bottom_right; // 右下

  int freeQEdge;
  int freePoint;

  std::vector<std::set<int> > neighbor_vtx;

  int newPoint(cv::Point2f pt, bool isvirtual, int firstEdge = 0);

  int newEdge();

  void setEdgePoints(int edge_id, int org_point, int dist_point);

  void splice(int edge_A, int edge_B);

  int symEdge(int edge_id);

  int rotateEdge(int edge_id, int type);

  int getPointLocation(cv::Point2f pt, int& edge_id, int& vertex_id);

  int isRightOf(cv::Point2f _pt, int edge_id);

  int edgeOrg(int edge_id, cv::Point2f *_pt = nullptr);

  int edgeDst(int edge_id, cv::Point2f *_pt = nullptr);

  double triangleArea(cv::Point2f pt, cv::Point2f dist, cv::Point2f org);

  int nextEdge(int edge_id);

  int getEdge(int edge_id, int type);

  void deletedEdge(int edge);

  int connectEdges(int edge_a, int edge_b);

  int isPtInCircle(cv::Point2f pt, cv::Point2f a, cv::Point2f b, cv::Point2f c);

  void swapEdges(int edge);

  void addNeighborVertex(int edge);

  void deleteNeighborVertex(int edge);

};


#endif //ENCODER_MYDELAUNAY_HPP
