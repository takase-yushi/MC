//
// Created by keisuke on 2017/12/06.
//

#include <algorithm>
#include <iostream>
#include <queue>
#include <array>
#include <opencv/cv.hpp>
#include "../includes/DelaunayTriangulation.hpp"
#include "../includes/Utils.h"

//! Default constructor
DelaunayTriangulation::DelaunayTriangulation() {}

/**
 * @fn DelaunayTriangulation::DelaunayTriangulation(Rectangle rect)
 * @brief rectを伴うコンストラクタ
 * @param rect
 */
DelaunayTriangulation::DelaunayTriangulation(Rectangle rect) {
  init(rect);
}

/**
 * Vertex Constructor
 */
DelaunayTriangulation::Vertex::Vertex() {
  firstEdge = 0;
  type = -1;
}

/**
 * @brief 座標を指定してvertexを生成
 * @param _pt
 * @param _isvirtual
 * @param _firstEdge
 */
DelaunayTriangulation::Vertex::Vertex(cv::Point2f _pt, bool _isvirtual, int _firstEdge) {
  firstEdge = _firstEdge;
  type = (int)_isvirtual;
  pt = _pt;
}

/**
 * Qedge Constructor
 */
DelaunayTriangulation::QEdge::QEdge() {
  next_edge[0] = next_edge[1] = next_edge[2] = next_edge[3] = 0;
  pt[0] = pt[1] = pt[2] = pt[3] = 0;
}

/**
 * edgeIDを用いて初期化するコンストラクタ
 * @param edge
 */

DelaunayTriangulation::QEdge::QEdge(int edge) {
  next_edge[0] = edge;
  next_edge[1] = edge + 3;
  next_edge[2] = edge + 2;
  next_edge[3] = edge + 1;

  pt[0] = pt[1] = pt[2] = pt[3] = 0;
}


/**
 * @fn int DelaunayTriangulation::insert(cv::Point2f pt)
 * @brief 点ptを挿入する
 * @param pt 挿入したい点の座標(Point2f型)
 * @return current_point
 */
int DelaunayTriangulation::insert(cv::Point2f pt){

  int current_point = 0;
  int current_edge = 0;
  int deleted_edge = 0;

  // 置いた点がどのような場所か取得する(ex. 三角形の内側, 線上, 頂点)
  int location = getPointLocation(pt, current_edge, current_point);

  // 何らかのエラー
  if(location == LOCATION_ERROR){
    std::cerr << "Error : LOCATION_ERROR" << std::endl;
    exit(1);
  }

  // 指定した範囲外(Rectangle(x, y, width, height)の外
  if(location == LOCATION_OUT_OF_RECTANGLE){
    std::cerr << "Error : pt out of range" << std::endl;
    exit(1);
  }

  // 頂点の上に重なった
  if(location == LOCATION_ON_THE_VERTEX){
    return current_point;
  }

  // 辺の上
  if(location == LOCATION_ON_THE_EDGE){
    deleted_edge = current_edge;
    recent_edge = current_edge = getEdge(current_edge, PREV_AROUND_ORG); // reversed eRnext
    deletedEdge(deleted_edge);
  }else if(location == LOCATION_INSIDE){
    ;
  }else{
    std::cerr << "Error : getPointLocation() returned invalid location" << std::endl;
    exit(1);
  }

  assert(current_edge != 0);

  current_point = newPoint(pt, false);
  int base_edge = newEdge();
  int first_point = edgeOrg(current_edge);

  // インデックスのコンバータ
  idx_converer.emplace_back(vertex.size() - 1);

  setEdgePoints(base_edge, first_point, current_point);
  splice(base_edge, current_edge);

//  addNeighborVertex(base_edge);

  do{
    base_edge = connectEdges(current_edge, symEdge(base_edge));
    current_edge = getEdge(base_edge, PREV_AROUND_ORG); // reversed eRnext
  }while(edgeDst(current_edge) != first_point);

  current_edge = getEdge(base_edge, PREV_AROUND_ORG);

  int max_edge_size = (int)(edges.size() * 4);

  for(int i = 0 ; i < max_edge_size ; i++){
    int tmp_dest = 0;
    int current_org = 0;
    int current_dest = 0;

    int tmp_edge = getEdge(current_edge, PREV_AROUND_ORG);

    tmp_dest = edgeDst(tmp_edge);
    current_org = edgeOrg(current_edge);
    current_dest = edgeDst(current_edge);

    if(isRightOf(vertex[tmp_dest].pt, current_edge) > 0 &&
            isPtInCircle(vertex[current_org].pt, vertex[tmp_dest].pt, vertex[current_dest].pt, vertex[current_point].pt) < 0){
      // 辺をFlipする
      swapEdges(current_edge);
      current_edge = getEdge(current_edge, PREV_AROUND_ORG);
    }else if(current_edge == first_point){
      break;
    }else{
      current_edge = getEdge(nextEdge(current_edge), PREV_AROUND_LEFT);
    }
  }

  return current_point;
}

/**
 * @fn void DelaunayTriangulation::insert(const std::vector<cv::Point2f>& corners)
 * @brief 点を挿入する
 * @param corners cv::Point2fを格納したvector
 */
void DelaunayTriangulation::insert(const std::vector<cv::Point2f>& corners) {
//  idx_converer = std::vector<int>(corners.size(), -1);
  for(int i = 0 ; i < (int)corners.size() ; i++){
    neighbor_vtx.emplace_back();
    insert(corners[i]);
    if(i < static_cast<int>(idx_converer.size()) - 1 - 1){
      idx_converer.emplace_back(-1);
    }
  }
}

/**
 * @fn void MyDelaunay::init(Rectangle rect)
 * @brief ドロネー図の初期化
 * @param rect 分割範囲を表す四角形
 */
void DelaunayTriangulation::init(Rectangle rect) {
  // ドロネー図の最初は、全てを囲むような大きい三角形で初期化する
  float multiply_coordinate = 3 * std::max(rect.width, rect.height);

  // 特徴点と辺を初期化
  vertex.clear();
  edges.clear();

  // ある点に隣接するノード
  neighbor_vtx.clear();

  recent_edge = 0;

  // 左上と右下を設定
  upper_left = cv::Point2f(rect.x, rect.y);
  bottom_right = cv::Point2f(rect.x + rect.width, rect.y + rect.height);

  // 大きな三角形の各点を初期化
  cv::Point2f ppA(rect.x + multiply_coordinate, rect.y);
  cv::Point2f ppB(rect.x, rect.y + multiply_coordinate);
  cv::Point2f ppC(rect.x - multiply_coordinate, rect.y - multiply_coordinate);

  vertex.emplace_back();
  edges.emplace_back();

  neighbor_vtx.emplace_back();
  neighbor_vtx.emplace_back();
  neighbor_vtx.emplace_back();
  neighbor_vtx.emplace_back();

  freeQEdge = 0;
  freePoint = 0;

  // 新たな点を作る
  int pA = newPoint(ppA, false);
  int pB = newPoint(ppB, false);
  int pC = newPoint(ppC, false);

  // 新たなエッジも作る
  int edge_AB = newEdge();
  int edge_BC = newEdge();
  int edge_CA = newEdge();

  // エッジに対して, orgとdistを設定する
  setEdgePoints(edge_AB, pA, pB);
  setEdgePoints(edge_BC, pB, pC);
  setEdgePoints(edge_CA, pC, pA);

  // spliceってなんだ…
  splice(edge_AB, symEdge(edge_CA));
  splice(edge_BC, symEdge(edge_AB));
  splice(edge_CA, symEdge(edge_BC));

  recent_edge = edge_AB;
}

/**
 * @fn int DelaunayTriangulation::newPoint(cv::Point2f pt, bool isvirtual, int firstEdge)
 * @brief 新たな頂点を作成する
 * @param pt 座標
 * @param isvirtual ??
 * @param firstEdge ??
 * @return 新たな頂点のid
 */
int DelaunayTriangulation::newPoint(cv::Point2f pt, bool isvirtual, int firstEdge) {
  if(freePoint == 0){
    vertex.emplace_back();
    freePoint = (int)(vertex.size() - 1);
  }

  int vertex_idx = freePoint;
  freePoint = vertex[vertex_idx].firstEdge;
  vertex[vertex_idx] = Vertex(pt, isvirtual, firstEdge);

  return vertex_idx;
}

/**
 * @fn int DelaunayTriangulation::newEdge()
 * @brief 新しいエッジを作成する
 * @return 新たに作成したエッジのid
 */
int DelaunayTriangulation::newEdge() {
  if(freeQEdge <= 0){
    edges.emplace_back();
    freeQEdge = (int)(edges.size() - 1);
  }

  int edge = freeQEdge * 4;
  freeQEdge = edges[edge >> 2].next_edge[1];
  edges[edge >> 2] = QEdge(edge);
  return edge;
}

/**
 * @fn void DelaunayTriangulation::setEdgePoints(int edge_id, int org_point, int dist_point)
 * @brief edgeに始点と終点をセットする
 * @param edge_id
 * @param org_point
 * @param dist_point
 */
void DelaunayTriangulation::setEdgePoints(int edge_id, int org_point, int dist_point) {
  edges[edge_id >> 2].pt[edge_id & 3] = org_point;
  edges[edge_id >> 2].pt[(edge_id + 2) & 3] = dist_point;
  vertex[org_point].firstEdge = edge_id;
  vertex[dist_point].firstEdge = edge_id ^ 2; // 3の場所
}

// TODO: AからBに付け替えるってことで良い…？
void DelaunayTriangulation::splice( int edgeA, int edgeB ) {
  int& a_next = edges[edgeA >> 2].next_edge[edgeA & 3];
  int& b_next = edges[edgeB >> 2].next_edge[edgeB & 3];
  int a_rot = rotateEdge(a_next, 1);
  int b_rot = rotateEdge(b_next, 1);
  int& a_rot_next = edges[a_rot >> 2].next_edge[a_rot & 3];
  int& b_rot_next = edges[b_rot >> 2].next_edge[b_rot & 3];

  std::swap(a_next, b_next);
  std::swap(a_rot_next, b_rot_next);
}

/**
 * @fn int DelaunayTriangulation::symEdge(int edge_id)
 * @brief 辺を逆転させる
 * @param edge_id
 * @return 反転した辺のid
 */
int DelaunayTriangulation::symEdge(int edge_id) {
  return edge_id ^ 2;
}

/**
 * @fn int MyDelaunay::rotateEdge(int edge_id, int type)
 * @brief 辺を回転させる
 * @param edge_id edgeの番号
 * @param type 回転タイプ
 *  0 : 入力したエッジそのもの
 *  1 : 入力したエッジを90度反時計回りに回転した辺
 *  2 : 入力したエッジを反転した辺
 *  3 : 反転したエッジを90度反時計回りに回転した辺
 * @return
 */
int DelaunayTriangulation::rotateEdge(int edge_id, int type) {
  return (edge_id & ~3) + ((edge_id + type) & 3);
}

int DelaunayTriangulation::getPointLocation(cv::Point2f pt, int& _edge_id, int& _vertex_id) {
  int vertex_id = 0;

  int max_edges_size = static_cast<int>(edges.size()) * 4;

  // 1つもない
  if(edges.size() < 4){
    std::cerr << "Error" << std::endl;
    exit(1);
  }

  if(pt.x < upper_left.x || bottom_right.x <= pt.x || pt.y < upper_left.y || bottom_right.y <= pt.y){
    std::cerr << "Error : out of range" << std::endl;
    exit(1);
  }

  int edge_id = recent_edge;
  assert(edge_id > 0);

  // とりあえず置けないと仮定しておく
  int location = LOCATION_ERROR;

  // ptがedges[edge_id]の右にいるのかどうか
  int right_of_current_edge = isRightOf(pt, edge_id);

  // 右側にいたら反転する
  if(right_of_current_edge > 0) {
    edge_id = symEdge(edge_id);
    right_of_current_edge = -right_of_current_edge;
  }

  for(int i = 0 ; i < max_edges_size ; i++){
    int o_next = nextEdge(edge_id); // eOnext
    int l_next = getEdge(edge_id, PREV_AROUND_DST); // reversed eLnext


    int right_of_oNext = isRightOf(pt, o_next);
    int right_of_eLnext = isRightOf(pt, l_next);

    /**
     *                    * B
     *                 |  |
     * eLnext(CB)    |   |
     *             |    |
     *           |     | current_edge(AB)
     *        C *     |
     *           |   |
     * eOnext(AC) | |
     *             *
     *             A
     */

    // すでにcurrent_edgeの左側に点はあります（辺上にある場合は除く）
    // 辺CBの右側にあったら
    if(right_of_eLnext > 0){
      // 辺ACの右側にある or 頂点Bの場合は内側判定
      if(right_of_oNext > 0 || (right_of_oNext == 0 && right_of_current_edge == 0)){
        location = LOCATION_INSIDE;
        break;
      }else{
        // current_edgeをeOnextにして再探索
        right_of_current_edge = right_of_oNext;
        edge_id = o_next;
      }
    }else{
      if(right_of_oNext > 0){
        // 頂点Bの上にのっている場合
        if(right_of_eLnext == 0 && right_of_current_edge == 0){
          location = LOCATION_INSIDE;
          break;
        }else{
          // eLnextをcurrentにして再探索
          right_of_current_edge = right_of_eLnext;
          edge_id = l_next;
        }
      }else if(right_of_current_edge == 0 && isRightOf(vertex[edgeDst(o_next)].pt, edge_id) >= 0){
        // current_edge上に点があって, eOnextの始点がcurrent_edge上 or 右側にある場合
        edge_id = symEdge(edge_id);
      }else{
        right_of_current_edge = right_of_oNext;
        edge_id = o_next;
      }
    }
  }

  recent_edge = edge_id;

  if(location == LOCATION_INSIDE){
    cv::Point2f org_pt, dist_pt;
    org_pt = vertex[edgeOrg(edge_id)].pt;
    dist_pt = vertex[edgeDst(edge_id)].pt;

    double t1 = std::fabs(pt.x - org_pt.x) + std::fabs(pt.y - org_pt.y);
    double t2 = std::fabs(pt.x - dist_pt.x) + std::fabs(pt.y - dist_pt.y);
    double t3 = std::fabs(dist_pt.x - org_pt.x) + std::fabs(dist_pt.y - org_pt.y);

    if(t1 < FLT_EPSILON){
      location = LOCATION_ON_THE_VERTEX;
      vertex_id = edgeOrg(edge_id);
    }else if(t2 < FLT_EPSILON){
      location = LOCATION_ON_THE_VERTEX;
      vertex_id = edgeDst(edge_id);
    }else if((t1 < t3 || t2 < t3) && fabs(triangleArea(pt, org_pt, dist_pt)) < FLT_EPSILON){
      location = LOCATION_ON_THE_EDGE;
      vertex_id = 0;
    }

  }

  if(location == LOCATION_ERROR){
    edge_id = vertex_id = 0;
  }

  _edge_id = edge_id;
  _vertex_id = vertex_id;

  return location;
}

/**
 * @fn int DelaunayTriangulation::isRightOf(cv::Point2f _pt, int edge_id)
 * @brief ptが辺の右側にあるかどうかを調べる
 * @param _pt 点
 * @param edge_id エッジ
 * @return 1 ptがedgeの右側にある場合
 * @return -1 ptがedgeの左側にある場合
 * @return 0 ptがedge上にある場合
 */
int DelaunayTriangulation::isRightOf(cv::Point2f _pt, int edge_id) {
  cv::Point2f org_pt, dst_pt;
  int v_org_idx = edgeOrg(edge_id);
  int v_dist_idx = edgeDst(edge_id);

  org_pt = vertex[v_org_idx].pt;
  dst_pt = vertex[v_dist_idx].pt;

  double ret = triangleArea(_pt, dst_pt, org_pt);
  return (ret > 0) - (ret < 0);
}

/**
 * @fn int MyDelaunay::edgeOrg(int edge_id, cv::Point2f *_pt)
 * @brief エッジの始点を求める.
 * @param edge_id
 * @param _pt
 * @return vertex_idx 頂点のインデックス
 * @details
 *  第二引数にcv::Point2fの変数を指定した場合, 始点の座標が代入される.
 */
int DelaunayTriangulation::edgeOrg(int edge_id, cv::Point2f *_pt) {
  int vertex_idx = edges[edge_id >> 2].pt[edge_id & 3];
  if(_pt != nullptr){
    *_pt = vertex[vertex_idx].pt;
  }
  return vertex_idx;
}

/**
 * @fn int MyDelaunay::edgeOrg(int edge_id, cv::Point2f *_pt)
 * @brief エッジの始点を求める.
 * @param edge_id
 * @param _pt
 * @return vertex_idx 頂点のインデックス
 * @details
 *  第二引数にcv::Point2fの変数を指定した場合, 始点の座標が代入される.
 */
int DelaunayTriangulation::edgeDst(int edge_id, cv::Point2f *_pt) {
  int vertex_idx = edges[edge_id >> 2].pt[(edge_id + 2) & 3];
  if(_pt != nullptr){
    *_pt = vertex[vertex_idx].pt;
  }
  return vertex_idx;
}

/**
 * @fn double DelaunayTriangulation::triangleArea(cv::Point2f a, cv::Point2f b, cv::Point2f c)
 * @brief 点a, b, cからなる三角形の面積を返す
 * @param a 三角形の頂点1
 * @param b 三角形の頂点2
 * @param c 三角形の頂点3
 * @return 三角形の面積
 */
double DelaunayTriangulation::triangleArea(cv::Point2f a, cv::Point2f b, cv::Point2f c) {
  cv::Point2f ab = cv::Point2f(b.x - a.x, b.y - a.y);
  cv::Point2f ac = cv::Point2f(c.x - a.x, c.y - a.y);
  return (ab.x * ac.y - ab.y * ac.x);
}

/**
 * @fn int DelaunayTriangulation::nextEdge(int edge_id)
 * @brief eOnextを返す
 * @param edge_id
 * @return eOnextのid
 */
int DelaunayTriangulation::nextEdge(int edge_id) {
  return edges[edge_id >> 2].next_edge[edge_id & 3];
}

/**
 * @fn int DelaunayTriangulation::getEdge(int edge_id, int type)
 * @brief edge_idに接続されている辺をtypeを指定して取り出す
 * @param edge_id 現在の辺
 * @param type ほしい辺のtype
 * @return 取り出した辺のid
 */
int DelaunayTriangulation::getEdge(int edge_id, int type) {
  edge_id = edges[edge_id  >> 2].next_edge[(edge_id + type) & 3];
  int edge_no = edge_id & ~3;
  return edge_no + ((edge_id + (type >> 4)) & 3);
}

/**
 * @fn void DelaunayTriangulation::deletedEdge(int edge)
 * @brief 辺を削除する
 * @param edge 削除したいエッジ
 */
void DelaunayTriangulation::deletedEdge(int edge) {

  deleteNeighborVertex(edge);
  splice(edge, getEdge(edge, PREV_AROUND_ORG));
  int sym_edge = symEdge(edge);
  deleteNeighborVertex(sym_edge);
  splice(sym_edge, getEdge(sym_edge, PREV_AROUND_ORG));

  edge >>= 2;
  edges[edge].next_edge[0] = 0;
  edges[edge].next_edge[1] = freeQEdge;
  freeQEdge = edge;
}

/**
 * @fn int DelaunayTriangulation::connectEdges(int edge_a, int edge_b)
 * @brief edge_aのdstからedge_bのorgをつないでいる
 * @param edge_a
 * @param edge_b
 * @return 作成した辺のid
 */
int DelaunayTriangulation::connectEdges(int edge_a, int edge_b) {
  int edge_id = newEdge();

  splice(edge_id, getEdge(edge_a, NEXT_AROUND_LEFT));
  splice(symEdge(edge_id), edge_b);

  setEdgePoints(edge_id, edgeDst(edge_a), edgeOrg(edge_b));
  // addNeighborVertex(edge_id);
  return edge_id;
}

/**
 * @fn inline int MyDelaunay::isPtInCircle(cv::Point2f pt, cv::Point2f a, cv::Point2f b, cv::Point2f c)
 * @brief ptがa, b, cをすべて通る円内に存在するかを調べる
 * @param pt 検索したい点
 * @param a 三角形の頂点1
 * @param b 三角形の頂点2
 * @param c 三角形の頂点3
 * @return 1 存在する
 */
inline int DelaunayTriangulation::isPtInCircle(cv::Point2f pt, cv::Point2f a, cv::Point2f b, cv::Point2f c){
  const double eps = FLT_EPSILON * 0.125;

  double det = ((double)a.x * a.x + (double)a.y * a.y) * triangleArea(b, c, pt);
  det -= ((double)b.x * b.x + (double)b.y * b.y) * triangleArea(a, c, pt);
  det += ((double)c.x * c.x + (double)c.y * c.y) * triangleArea(a, b, pt);
  det -= ((double)pt.x * pt.x + (double)pt.y * pt.y) * triangleArea(a, b, c);

  if(det > eps) return 1;
  else if(det < -eps) return -1;
  else return 0;
}

/**
 * @fn void DelaunayTriangulation::swapEdges(int edge)
 * @brief 辺をflipする
 * @param edge flipしたエッジ
 */
void DelaunayTriangulation::swapEdges(int edge) {
  int sym_edge = symEdge(edge);

  deleteNeighborVertex(edge);

  int a = getEdge(edge, PREV_AROUND_ORG);
  int b = getEdge(sym_edge, PREV_AROUND_ORG);

  splice(edge, a);
  splice(sym_edge, b);

  setEdgePoints(edge, edgeDst(a), edgeDst(b));

  // addNeighborVertex(edge);

  splice(edge, getEdge(a, NEXT_AROUND_LEFT));
  splice(sym_edge, getEdge(b, NEXT_AROUND_LEFT));
}

/**
 * @fn void MyDelaunay::getTriangleList(std::vector<cv::Vec6f>& triangle_list)
 * @brief 三角形の集合を返す
 * @param triangle_list
 */
void DelaunayTriangulation::getTriangleList(std::vector<cv::Vec6f>& triangle_list){
  triangle_list.clear();
  unsigned int total_edge_size = (unsigned int) (edges.size() * 4);
  std::vector<bool> mask(total_edge_size, false);

  // 最初の辺は飛ばす
  for(int i = 4 ; i < (int)total_edge_size ; i+=2){
    if(mask[i]) continue;
    int edge_id = i;
    addNeighborVertex(edge_id);
    cv::Point2f a = vertex[edgeOrg(edge_id)].pt;
    mask[edge_id] = true;
    edge_id = getEdge(edge_id, NEXT_AROUND_LEFT);
    addNeighborVertex(edge_id);
    cv::Point2f b = vertex[edgeOrg(edge_id)].pt;
    mask[edge_id] = true;
    edge_id = getEdge(edge_id, NEXT_AROUND_LEFT);
    addNeighborVertex(edge_id);
    cv::Point2f c = vertex[edgeOrg(edge_id)].pt;
    mask[edge_id] = true;
    triangle_list.emplace_back(a.x, a.y, b.x, b.y, c.x, c.y);
  }
}

/**
 * @fn void DelaunayTriangulation::addNeighborVertex(int edge)
 * @brief 辺edgeの関係からneighbor_vtxに追加する
 * @param edge 追加したい辺
 */
void DelaunayTriangulation::addNeighborVertex(int edge) {
  int base_edge_org = edgeOrg(edge), base_edge_dst = edgeDst(edge);
  neighbor_vtx[base_edge_org].emplace(base_edge_dst);
  neighbor_vtx[base_edge_dst].emplace(base_edge_org);
}

/**
 * @fn void DelaunayTriangulation::deleteNeighborVertex(int edge)
 * @brief 削除する辺edgeに関係のある頂点を消す
 * @param edge 削除された辺
 */
void DelaunayTriangulation::deleteNeighborVertex(int edge) {
  int base_edge_org = edgeOrg(edge), base_edge_dst = edgeDst(edge);
//  std::cout << "delete:" << neighbor_vtx[base_edge_org].erase(base_edge_dst) << std::endl;
//  std::cout << "delete:" << neighbor_vtx[base_edge_dst].erase(base_edge_org) << std::endl;
  neighbor_vtx[base_edge_org].erase(base_edge_dst);
  neighbor_vtx[base_edge_dst].erase(base_edge_org);
}

/**
 * @fn double getDistance(const cv::Point2d& a, const cv::Point2d& b)
 * @brief 2点間のユークリッド距離を返す
 * @param a 点A
 * @param b 点B
 * @return double 2点間のユークリッド距離
 */
double DelaunayTriangulation::getDistance(const cv::Point2d& a, const cv::Point2d& b){
  cv::Point2d v = a - b;
  return sqrt(v.x * v.x + v.y * v.y);
}

/**
 * @fn std::vector<PointCode> getPointCoordinateCode(const std::vector<cv::Point2f>& corners)
 * @brief 頂点の座標の復元
 * @param corners 頂点座標
 * @return 以前のidとそれに対する差分ベクトルを含んだクラスPointCodeのstd::vector
 */
std::vector<DelaunayTriangulation::PointCode> DelaunayTriangulation::getPointCoordinateCode(const std::vector<cv::Point2f>& corners, int mode){
  std::vector<PointCode> code;

  cv::Point2f prev_pt = cv::Point2f(0.0, 0.0); // 1つ前の頂点

  // cornersじゃなくて, DelaunayTriangulationのメンバvertexでやるぞ！
  if(mode == RASTER_SCAN) {
    for (int i = 4; i < (int) vertex.size(); i++) {
      double diff = getDistance(vertex[i].pt, prev_pt);
      if (diff > 100.0) { // 離れている場合
        int current_idx = i - 1;
        int min_pt_idx = i - 1; double min_pt_dist = getDistance(prev_pt, vertex[i].pt);

        while(0 <= current_idx) {
          double current_dist = getDistance(vertex[current_idx].pt, vertex[i].pt);
          if(min_pt_dist > current_dist){
            min_pt_dist = current_dist;
            min_pt_idx = current_idx;
          }
          current_idx--;
        }
        code.emplace_back(i - min_pt_idx, vertex[i].pt - vertex[min_pt_idx].pt);
      } else { // 大きく離れていない場合
        code.emplace_back(0, vertex[i].pt - prev_pt);
      }
      prev_pt = vertex[i].pt; // 1つ前の更新
    }
  }else if(mode == QUEUE){
    std::vector<int> history;
    std::queue<int> queue;
    std::vector<bool> mask(corners.size() + 4, false), inqueue(corners.size() + 4, false);

    // 左上の点に隣接する頂点をインキューします
    for(const auto& v : neighbor_vtx[4]){
      if(v < 4) continue;
      queue.emplace(v);
    }

    mask[4] = true;

    // 一つ目のは0.0, 0.0に存在するのでそれを配列にプッシュ
    code.emplace_back(0, cv::Point2f(0.0, 0.0));
    history.emplace_back(4);

    // すべての点が終わるまで回す
    while(!queue.empty()){
      // キューの先頭に格納されてる頂点をとりだす
      int v_idx = queue.front(); queue.pop();
      history.emplace_back(v_idx);

      // 距離がある程度離れている場合
      if(100.0 < getDistance(vertex[v_idx].pt, prev_pt)){
        int current_idx = static_cast<int>(history.size() - 1 - 1);
        int min_pt_idx = (int)history.size() - 2;
        double min_pt_dist = getDistance(prev_pt, vertex[v_idx].pt);

        // TODO: 遡ればいいとは限らないよね…？
        while(0 <= current_idx) {
          double current_dist = getDistance(vertex[history[current_idx]].pt, vertex[v_idx].pt);
          if(min_pt_dist > current_dist){
            min_pt_dist = current_dist;
            min_pt_idx = current_idx;
          }
          current_idx--;
        }

        code.emplace_back((int)history.size() - min_pt_idx - 1, vertex[v_idx].pt - vertex[history[min_pt_idx]].pt);

      }else{ // ある程度の長さの場合
        code.emplace_back(0, vertex[v_idx].pt - prev_pt);
        if(code[code.size() - 1].coord.x >= 1000){
          std::cout << "corners[v_idx]:" << vertex[v_idx].pt << " prev_pt:" << prev_pt << std::endl;
        }
      }

      // 復号した頂点をマスクする
      mask[v_idx] = true;

      // 1つ前の点を更新
      prev_pt = vertex[v_idx].pt;

      // 今回復号した頂点に接していて, かつインキューorマスクされていない場合はジョブキューに追加
      for(const auto& v : neighbor_vtx[v_idx]){
        if(v < 4) continue; // 頂点が4未満は分割領域外の頂点になるので除外
        if(inqueue[v] || mask[v]) continue; // インキュー or すでに復号済の場合は復元キューにいれない
        queue.emplace(v);
        inqueue[v] = true;
      }
    }
  }

  return code;
}

/**
 * @fn std::vector<PointCode> DelaunayTriangulation::getPointMotionVectorCode(const std::vector<cv::Point2f>& corners)
 * @brief 頂点の動きベクトルの符号化
 * @param corners
 * @return code 差分ベクトル
 */
std::vector<cv::Point2f> DelaunayTriangulation::getPointMotionVectorCode(const std::vector<cv::Point2f>& corners, const std::vector<cv::Point2f>& mv){
  std::vector<cv::Point2f> code;
  std::queue<int> queue;
  std::vector<bool> mask(corners.size(), false), inqueue(corners.size(), false);

  // 最初はそのものを送るしか無いため
  std::set<int> first_vtx = neighbor_vtx[4];
  for(const auto& v : first_vtx){
    if(v < 4) continue;
    queue.emplace(v);
  }
  mask[4] = true;

  code.emplace_back(mv[0]);

  // 基本頂点のインデックスが4未満のものは分割に含まれていないので, すっ飛ばして良い（最初に作成した巨大な三角形の頂点なので）
  while(!queue.empty()){
    int v_idx = queue.front(); queue.pop();
    std::set<int> vertex = neighbor_vtx[v_idx];
    std::priority_queue<std::pair<int, int> > pqueue; // 距離でソートしてほしいためpriority_queueを使う
    for(const auto& v : vertex){
      if(!mask[v] || v < 4) continue; // 未だ復元されていない頂点であればインキューしない（復元済みの頂点のみいれる）
      pqueue.emplace(std::make_pair(-getDistance(corners[v_idx - 4], corners[v - 4]), v));
    }
    std::pair<int, int> min_dist_vertex = pqueue.top();

    cv::Point2f nearest_mv = (mv[min_dist_vertex.second - 4] - 2 * corners[min_dist_vertex.second - 4]);
    cv::Point2f current_mv = (mv[v_idx - 4] - 2 * corners[v_idx - 4]);

    // 差分ベクトル
    cv::Point2f diff_mv = nearest_mv - current_mv;

    // 座標がハーフペルのそれになっているので, 全部2で割ります
    code.emplace_back(diff_mv / 2.0);
    std::cout << "diff_mv / 2.0 : " << diff_mv / 2.0 << std::endl;

    // すでに復号したためマスク
    mask[v_idx] = true;

    // 新しく復元した頂点に隣接するノードを突っ込む
    for(const auto& v : vertex){
      if(inqueue[v] || mask[v] || v < 4) continue; // インキュー or すでに復号済 or 不正な頂点の場合は復元キューにいれない
      queue.emplace(v);
      inqueue[v] = true;
    }
  }

  return code;
}

/**
 * @fn cv::Mat DelaunayTriangulation::getDecodedCornerImage(std::vector<DelaunayTriangulation::PointCode> &code, const cv::Mat& target_image, int mode)
 * @param code 得られた頂点間の差分ベクトルの集合
 * @param target_image 対象画像
 * @param mode キューかラスタスキャンか
 * @return codeから復元された頂点が描画された画像
 */
cv::Mat DelaunayTriangulation::getDecodedCornerImage(std::vector<DelaunayTriangulation::PointCode> &code, const cv::Mat& target_image, int mode) {
  cv::Mat ret_image = target_image.clone();

  if(mode == RASTER_SCAN || mode == QUEUE){
    cv::Point2f prev_pt(0.0, 0.0);
    std::vector<cv::Point2f> history;

    // 頂点復元
    for(int i = 0 ; i < (int)code.size() ; i++){
      cv::Point2f current_pt;
      if(code[i].prev_id == 0){ // 通常通り（前のに足す）
        current_pt = prev_pt + code[i].coord;
        if(current_pt.x < 0) current_pt.x = 0;
        if(current_pt.y < 0) current_pt.y = 0;
        if(target_image.cols <= current_pt.x) current_pt.x = target_image.cols - 1;
        if(target_image.rows <= current_pt.y) current_pt.y = target_image.rows - 1;
        drawPoint(ret_image, current_pt, BLUE, 4);
      }else if(code[i].prev_id > 0){
        current_pt = history[i - code[i].prev_id] + code[i].coord;
        if(current_pt.x < 0) current_pt.x = 0;
        if(current_pt.y < 0) current_pt.y = 0;
        if(target_image.cols <= current_pt.x) current_pt.x = target_image.cols - 1;
        if(target_image.rows <= current_pt.y) current_pt.y = target_image.rows - 1;
        drawPoint(ret_image, current_pt, cv::Scalar(255, 255, 255), 4);
      }
      prev_pt = current_pt;
      history.emplace_back(current_pt);
    }
  }else{
    std::cerr << "Invalid option" << std::endl;
    exit(1);
  }

  return ret_image;
}


/**
 * @fn cv::Mat DelaunayTriangulation::getDecodedMotionVectorImage(std::vector<cv::Point2f>& code, std::vector<cv::Point2f>& corners, cv::Mat target_image)
 * @brief 差分ベクトルをもとに, 動きベクトルを復元してそれを描画した画像を返す
 * @param code 差分ベクトル
 * @param corners 頂点座標
 * @param target_image 対象画像
 * @return 動きベクトルを描画した画像
 */
cv::Mat DelaunayTriangulation::getDecodedMotionVectorImage(std::vector<cv::Point2f>& code, std::vector<cv::Point2f>& corners, cv::Mat target_image) {
  cv::Mat ret;

  ret = target_image.clone();

  std::queue<int>queue;
  std::vector<bool> mask(corners.size() + 4, false), inqueue(corners.size() + 4, false);
  std::vector<cv::Point2f> mv(corners.size() + 4, cv::Point2f(0.0, 0.0));

  // 1つ目はそれはそう
  mv[4] = code[0];
  mask[4] = true;

  drawPoint(ret, corners[0], RED, 3);
  drawPoint(ret, code[0]/2.0, BLUE, 3);
  cv::line(ret, corners[0], code[0]/2.0, GREEN);

  // 1つ目の頂点に隣接する頂点をキューにいれる
  for(const auto& v : neighbor_vtx[4]){
    if(v < 4) continue;
    queue.emplace(v);
  }

  // ターゲット画像上の頂点を描画
  for(const auto& corner : corners){
    drawPoint(ret, corner, RED, 3);
  }

  auto current_mv_diff = code.begin();
  current_mv_diff++;

  while(!queue.empty()){
    int v_idx = queue.front(); queue.pop();
    std::set<int> vertex = neighbor_vtx[v_idx];
    std::priority_queue<std::pair<int, int> > pqueue; // 距離でソートしてほしいためpriority_queueを使う
    for(const auto& v : vertex){
      if(!mask[v] || v < 4) continue; // 未だ復元されていない頂点であればインキューしない（復元済みの頂点のみいれる）
      pqueue.emplace(std::make_pair(-getDistance(corners[v_idx - 4], corners[v - 4]), v));
    }

    std::pair<int, int> min_dist_vertex = pqueue.top();
    cv::Point2f current_mv;

    // mvの復元
    // <MEMO>
    // これ前はcode[v_idx]とかやってたんですが, よくよく考えたら採用した頂点順にvectorにpushしているので, イテレータを使って
    // 順番に取り出さないとおかしくなるよねー、ってバグがありました. つらいね.
    current_mv = mv[v_idx] = mv[min_dist_vertex.second] - *current_mv_diff;

    mask[v_idx] = true;
    current_mv_diff++;

    cv::line(ret, corners[v_idx - 4], current_mv + corners[v_idx - 4], GREEN);
    drawPoint(ret, current_mv + corners[v_idx - 4], BLUE, 3);

    // 復号した頂点に隣接している頂点をインキューする
    vertex = neighbor_vtx[v_idx];
    for(const auto& v : vertex){
      if(inqueue[v] || mask[v] || v < 4) continue; // インキュー or すでに復号済 or 不正な頂点の場合は復元キューにいれない
      queue.emplace(v);
      inqueue[v] = true;
    }
  }

  return ret;
}

/**
 * @fn std::vector<cv::Point2f> DelaunayTriangulation::getNeighborVertex(int pt, cv::Mat target)
 * @brief pt番目の頂点に隣接する点を返す
 * @param pt
 * @param target
 * @return 頂点に隣接する頂点の座標を返す
 */
std::vector<cv::Point2f> DelaunayTriangulation::getNeighborVertex(int pt) {
  std::vector<cv::Point2f> ret_pts;

  for(const auto p : neighbor_vtx[pt + 4]){
    if(p < 4) continue;
    ret_pts.emplace_back(vertex[p].pt);
  }

  return ret_pts;
}

std::vector<int> DelaunayTriangulation::getNeighborVertexNum(int pt) {
    std::vector<int> ret;

    for(const auto p : neighbor_vtx[pt + 4]){
        if(p < 4) continue;
        ret.emplace_back(p - 4);
    }

    return ret;
}
/**
 * @fn std::priority_queue<int> DelaunayTriangulation::getUnnecessaryPoint
 * @brief 閾値th以下の相関のものを取り除く
 * @param mv 動きベクトルの集合
 * @param th 閾値
 * @return 削除対象の頂点のインデックス
 */
std::priority_queue<int> DelaunayTriangulation::getUnnecessaryPoint(
        const std::vector<cv::Point2f> &mv, double th, const cv::Mat& target_img) {
  std::priority_queue<int> ret;
  std::priority_queue<std::pair<double, int>> pqueue;
  int  height = target_img.rows;
  int  width = target_img.cols;
  for(int i = 4 ; i < static_cast<int>(neighbor_vtx.size()) ; i++) {

    //if(vertex[i].pt.x == 0.0 || vertex[i].pt.x == width -1 || vertex[i].pt.y == 0.0 || vertex[i].pt.y == height -1) continue;

    // 周囲に打たれた点は取り除かない
    if((vertex[i].pt.x == 0.0 && vertex[i].pt.y == 0.0) || (vertex[i].pt.x == width -1 && vertex[i].pt.y == 0.0)
            || (vertex[i].pt.x == width -1 && vertex[i].pt.y == height -1) || (vertex[i].pt.x == 0.0 && vertex[i].pt.y == height -1) ){
      continue;
    }

    // 120ごとにうってある点は除かない                                                                  (追加                                                                                          )追加
    if(((int)vertex[i].pt.x % SIDE_X_MIN == 0 && (vertex[i].pt.y == 0.0 || vertex[i].pt.y == height -1)) || (((int)vertex[i].pt.y % SIDE_Y_MIN == 0) && (vertex[i].pt.x == 0.0 || vertex[i].pt.x == width -1))) continue;

    double x = 0, y = 0, diff = 0;

    int denominator = 0;
    double distance = 0.0;

    for (const int v : neighbor_vtx[i]) {
      if (v < 4) continue;
      x += (mv[v - 4].x - 2 * vertex[v].pt.x);
      y += (mv[v - 4].y - 2 * vertex[v].pt.y);
      distance += getDistance(vertex[v].pt, vertex[i].pt);
      denominator++;
    }
    x /= (double) denominator;
    y /= (double) denominator;
    distance /= denominator;

    if (vertex[i].pt.x == 0.0 || vertex[i].pt.x == width -1) {    // 左の辺, 右の辺である場合, y成分だけで見る
      diff = std::fabs(y - (mv[i - 4].y - 2 * vertex[i].pt.y));
    } else if (vertex[i].pt.y == 0.0 || vertex[i].pt.y == height) { // 上下の場合はx成分だけ
      diff = std::fabs(x - (mv[i - 4].x - 2 * vertex[i].pt.x));
    } else {
      diff = std::fabs(x - (mv[i - 4].x - 2 * vertex[i].pt.x)) + std::fabs(y - (mv[i - 4].y - 2 * vertex[i].pt.y));
    }

    if(45 <= distance){
      diff *= 1.5;
    }else{
      diff *= 0.7;
    }
std::cout << "diff = " << diff << std::endl;
    if (diff < th || diff > 80) {
      pqueue.emplace(std::make_pair(diff, i - 4));
    }
  }

  std::vector<bool> flag(mv.size() + 5, false);
  while(!pqueue.empty()){
    std::pair<double, int> top = pqueue.top(); pqueue.pop();
    if(!flag[top.second]){
      ret.emplace(top.second);
      for(const int v : neighbor_vtx[top.second]){
        flag[v] = true;
      }
    }
  }

  return ret;
}

/*std::priority_queue<int> Erase_UnnecessaryPoint(DelaunayTriangulation md, const cv::Mat &ref,const cv::Mat &target){

}*/

/**
 * @fn std::vector<Triangle> DelaunayTriangulation::Get_triangles_around(int idx,std::vector<cv::Point2f> corners,std::vector<bool> &flag_around )
 * @brief idx番目の頂点の周りの三角形を返す
 * @param[in] idx 頂点番号
 * @param[in] corners 頂点群
 * @param[out] flag_around
 * @return idx番目の頂点を含む三角形
 */
std::vector<Triangle> DelaunayTriangulation::Get_triangles_around(int idx, std::vector<cv::Point2f> corners, std::vector<bool> &flag_around ){
    std::vector<Triangle> triangles_around;
    std::vector<cv::Vec6f> triangleList;
    this->getTriangleList(triangleList);
    for (auto t:triangleList) {
        cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
        int i1 = -1, i2 = -1, i3 = -1;
        for (int i = 0; i < (int) corners.size(); i++) {
            if (corners[i] == p1) i1 = i;
            else if (corners[i] == p2) i2 = i;
            else if (corners[i] == p3) i3 = i;
        }
        if ((0 <= i1 && 0 <= i2 && 0 <= i3)&&(i1 == idx||i2 == idx || i3 == idx)) {
            triangles_around.emplace_back(i1, i2, i3);
            for(const int v : neighbor_vtx[idx + 4]){
                if(v < 4) continue;
                flag_around[v-4] = true;
            }
        }
    }
    return triangles_around;
}

std::vector<Triangle> DelaunayTriangulation::Get_triangles_later(DelaunayTriangulation md_later,int idx,std::vector<cv::Point2f> corners,std::vector<bool> flag_around ){
    std::vector<Triangle> triangles_later;
    triangles_later.clear();
    std::vector<cv::Vec6f> triangles_mydelaunay;
    this->getTriangleList(triangles_mydelaunay);
    //std::cout << "triangle_list_later_size = " << triangles_mydelaunay.size() << std::endl;
    for (auto t:triangles_mydelaunay) {
        cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
        //std::cout << "p1 =" << p1 << "p2 =" << p2 << "p3 =" << p3<<std::endl;
        int i1 = -1, i2 = -1, i3 = -1;
        for (int i = 0; i < (int) corners.size(); i++) {
            if (corners[i] == p1) i1 = i;
            else if (corners[i] == p2) i2 = i;
            else if (corners[i] == p3) i3 = i;
        }
        if ((0 <= i1 && 0 <= i2 && 0 <= i3)&&(flag_around[i1] == true && flag_around[i2] == true && flag_around[i3] == true)) {
            //std::cout << "add_later_list" << std::endl;
            triangles_later.emplace_back(i1, i2, i3);
        }
    }
    return triangles_later;
}
/*
void bubbleSort(std::vector<std::pair<cv::Point2f,double>> &sort_cornes, int array_size) {
  int i, j;
  std::pair<cv::Point2f,double> tmp;

  for (i = 0; i < (array_size - 1); i++) {
    for (j = (array_size - 1); j > i; j--) {
      if (sort_cornes[j - 1].second > sort_cornes[j].second) {
        tmp = sort_cornes[j - 1];
        sort_cornes[j - 1] = sort_cornes[j];
        sort_cornes[j] = tmp;
      }
    }
  }
}*/

/***
 * @fn void DelaunayTriangulation::Sort_Coners(std::vector<cv::Point2f> &corners)
 * @brief 周囲の点との平均距離が短い頂点順にソート
 * @param[in,out] corners ソートされた頂点
 */
void DelaunayTriangulation::Sort_Coners(std::vector<cv::Point2f> &corners){
    std::vector<std::pair<cv::Point2f,double>> sort_coners(corners.size());
    for(int idx = 0;idx < (int)corners.size();idx++) {
        sort_coners[idx].first = corners[idx];
        std::vector<cv::Point2f> around = getNeighborVertex(idx);
        sort_coners[idx].second = 0;
        for(int i = 0;i < (int)around.size();i++){
            sort_coners[idx].second += getDistance(sort_coners[idx].first,around[i]);
        }
        sort_coners[idx].second /= around.size();
        //std::cout << "Distance =" << sort_coners[idx].second << std::endl;
    }
    bubbleSort(sort_coners,sort_coners.size());
    corners.clear();
    for(int idx = 0;idx < (int)sort_coners.size();idx++){
        corners.emplace_back(sort_coners[idx].first);
    }
}

std::vector<cv::Point2f> DelaunayTriangulation::repair_around(std::vector<cv::Point2f> &corners,const cv::Mat target){
    std::vector<cv::Point2f> ret(corners);
    for(int idx = 0;idx < (int)corners.size();idx++){
        if(corners[idx].x <= 0 || corners[idx].x >= target.cols - 1||corners[idx].y <= 0 || corners[idx].y >= target.rows - 1){
            continue;
        }
        for(const int v : neighbor_vtx[idx + 4]){
            //std::cout << "neighbor_vtx[" << idx << "] = " << v-4 << std::endl;
            if(v < 4) {
                if(v == 0) {//右辺
                    ret.emplace_back(cv::Point2f(target.cols - 1, corners[idx].y));
                }
                else if(v == 1){//下辺
                    ret.emplace_back(cv::Point2f(corners[idx].x,target.rows));
                }
                else if(v == 2){
                    if(corners[idx].y <= corners[idx].x){//上辺
                        ret.emplace_back(cv::Point2f(corners[idx].x,0));
                    }
                    else if(corners[idx].y > corners[idx].x){//左辺
                        ret.emplace_back(cv::Point2f(0,corners[idx].y));
                    }
                }
            }
        }
    }
    return ret;
}

/***
 * @fn void DelaunayTriangulation::serch_wrong(std::vector<cv::Point2f>corners,cv::Mat target,bool *skip_flag)
 * @brief 画像外の頂点を含む三角形があるかどうか判定
 * @details いわゆる黒い三角形があるかどうか判定
 * @param[in] corners
 * @param[in] target
 * @param[out] skip_flag
 */
void DelaunayTriangulation::serch_wrong(std::vector<cv::Point2f>corners,cv::Mat target,bool *skip_flag) {
  for (int idx = 0; idx < (int) corners.size(); idx++) {
    if (corners[idx].x <= 0 || corners[idx].x >= target.cols - 1 || corners[idx].y <= 0 ||
        corners[idx].y >= target.rows - 1) {
      continue;
    }
    for (const int v : neighbor_vtx[idx + 4]) {
      if (v < 4) {
        *skip_flag = true;
      }
    }
  }
}

void DelaunayTriangulation::inter_div(std::vector<cv::Point2f>&corners,cv::Point2f corner,std::vector<Triangle> triangles, int t){
    Triangle triangle = triangles[t];
    int new_id =  newPoint(corner,false);
    neighbor_vtx.emplace_back();
    idx_converer.emplace_back(vertex.size() - 1);
    //corners.emplace_back(corner);
    int edge_p1 = newEdge();
    int edge_p2 = newEdge();
    int edge_p3 = newEdge();
    setEdgePoints(edge_p1, new_id, triangle.p1_idx);
    setEdgePoints(edge_p2, new_id, triangle.p2_idx);
    setEdgePoints(edge_p3, new_id, triangle.p3_idx);
    /*
    Triangle triangle_p1(new_id,triangle.p1_idx,triangle.p2_idx);
    Triangle triangle_p2(new_id,triangle.p2_idx,triangle.p3_idx);
    Triangle triangle_p3(new_id,triangle.p3_idx,triangle.p1_idx);
    triangles.erase(triangles.begin() + t);
    triangles.insert()
     */
}

double DelaunayTriangulation::neighbor_distance(std::vector<cv::Point2f>&corners, int idx ){
    double min_distance = 100;
    double distance;
    for(const int v : neighbor_vtx[idx + 4]){
       distance = getDistance(corners[idx],corners[v]);
       if(min_distance > distance){
           min_distance = distance;
       }
    }
    return  min_distance;
}

/**
 * @fn std::vector<Triangle> DelaunayTriangulation::Get_triangles(std::vector<cv::Point2f> corners)
 * @brief 三角形の情報を、インデックスの組で取得する
 * @param corners[in]
 * @return cornersのインデックスで構成された三角パッチの頂点群
 */
std::vector<Triangle> DelaunayTriangulation::Get_triangles(std::vector<cv::Point2f> corners){
    std::vector<Triangle> triangles_later;
    triangles_later.clear();
    std::vector<cv::Vec6f> triangles_mydelaunay;
    this->getTriangleList(triangles_mydelaunay);
    for (auto t:triangles_mydelaunay) {
        cv::Point2f p1(t[0], t[1]), p2(t[2], t[3]), p3(t[4], t[5]);
        int i1 = -1, i2 = -1, i3 = -1;
        for (int i = 0; i < (int) corners.size(); i++) {
            if (corners[i] == p1) i1 = i;
            else if (corners[i] == p2) i2 = i;
            else if (corners[i] == p3) i3 = i;
        }
        if (0 <= i1 && 0 <= i2 && 0 <= i3) {
            triangles_later.emplace_back(i1, i2, i3);
        }
    }
    return triangles_later;
}

