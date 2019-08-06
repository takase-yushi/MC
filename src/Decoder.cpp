//
// Created by kasph on 2019/08/05.
//

#include <cmath>
#include "../includes/Decoder.h"
#include "../includes/CollocatedMvTree.h"
#include "../includes/TriangleDivision.h"
#include "../includes/ImageUtil.h"

void Decoder::initTriangle(int _block_size_x, int _block_size_y, int _divide_steps, int _qp, int divide_flag) {
    block_size_x = _block_size_x;
    block_size_y = _block_size_y;
    qp = _qp;
    int block_num_x = ceil((double)image_width / (block_size_x));
    int block_num_y = ceil((double)image_height / (block_size_y));
    divide_steps = _divide_steps;
    coded_picture_num = 0;

    image_width = ref_image.cols;
    image_height = ref_image.rows;

    corners.clear();
    neighbor_vtx.clear();
    covered_triangle.clear();
    triangles.clear();

    /*
     *  p1                     p2
     *   *---------------------*
     *   |                     |
     *   |                     |
     *   |                     |
     *   |                     |
     *   |                     |
     *   *---------------------*
     *  p3                     p4
     *
     */

    corner_flag.resize(static_cast<unsigned long>(image_height * 2));
    for(int i = 0 ; i < image_height * 2 ; i++) {
        corner_flag[i].resize(static_cast<unsigned long>(image_width * 2));
    }

    for(int y = 0 ; y < image_height * 2; y++) {
        for(int x = 0 ; x < image_width * 2; x++) {
            corner_flag[y][x] = -1;
        }
    }

    previousMvList.emplace_back();
    // すべての頂点を入れる
    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {
        for (int block_x = 0 ; block_x < block_num_x; block_x++) {
            int nx = block_x * (block_size_x);
            int ny = block_y * (block_size_y);

            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            same_corner_list.emplace_back();
            same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1); // 他と共有している頂点は、自分の番号だけ入れる
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());

            nx = (block_x + 1) * (block_size_x) - 1;
            ny = (block_y) * (block_size_y);

            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            same_corner_list.emplace_back();
            same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1);
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
        }

        for (int block_x = 0 ; block_x < block_num_x; block_x++) {
            int nx = block_x * (block_size_x);
            int ny = (block_y + 1) * (block_size_y) - 1;

            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            same_corner_list.emplace_back();
            same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1);;
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());

            nx = (block_x + 1) * (block_size_x) - 1;
            ny = (block_y + 1) * (block_size_y) - 1;

            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
            same_corner_list.emplace_back();
            same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1);
            neighbor_vtx.emplace_back();

            // 前の動きベクトルを保持しておくやつ
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
            previousMvList[coded_picture_num].emplace_back(new CollocatedMvTree());
        }

    }

    // 過去のMVを残すやつを初期化
    for(auto node : previousMvList[coded_picture_num]) {
        node->node1 = node->node2 = node->node3 = node->node4 = nullptr;
        node->mv1 = cv::Point2f(0.0, 0.0);
        node->mv2 = cv::Point2f(0.0, 0.0);
        node->mv3 = cv::Point2f(0.0, 0.0);
    }

    covered_triangle.resize(static_cast<unsigned long>((block_num_x * 2) * (block_num_y * 2)));

    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {
        for(int block_x = 0 ; block_x < block_num_x ; block_x++) {
            int p1_idx;
            int p2_idx;
            int p3_idx;
            int p4_idx;
            if(divide_flag == LEFT_DIVIDE) {
                p1_idx = 2 * block_x + (2 * block_y) * ((block_num_x) * 2);
                p2_idx = p1_idx + 1;
                p3_idx = p1_idx + ((block_num_x) * 2);

                int triangleIndex = insertTriangle(p1_idx, p2_idx, p3_idx, TYPE1);
                addNeighborVertex(p1_idx, p2_idx, p3_idx);
                addCoveredTriangle(p1_idx, p2_idx, p3_idx, triangleIndex); // p1/p2/p3はtriangleIndex番目の三角形に含まれている

                int p4_idx = p2_idx;
                int p5_idx = p3_idx;
                int p6_idx = p3_idx + 1;

                triangleIndex = insertTriangle(p4_idx, p5_idx, p6_idx, TYPE2);
                addNeighborVertex(p4_idx, p5_idx, p6_idx);
                addCoveredTriangle(p4_idx, p5_idx, p6_idx, triangleIndex);
            }else{
                int triangleIndex = insertTriangle(p1_idx, p2_idx, p4_idx, TYPE1);
                addNeighborVertex(p1_idx, p2_idx, p4_idx);
                addCoveredTriangle(p1_idx, p2_idx, p4_idx, triangleIndex);

                triangleIndex = insertTriangle(p1_idx, p3_idx, p4_idx, TYPE2);
                addNeighborVertex(p1_idx, p3_idx, p4_idx);
                addCoveredTriangle(p1_idx, p3_idx, p4_idx, triangleIndex);
            }
        }
    }

    for(int i = 0 ; i < isCodedTriangle.size() ; i++) {
        isCodedTriangle[i] = false;
    }

    delete_flag.resize(triangles.size());
    for(int i = 0 ; i < delete_flag.size() ; i++) {
        delete_flag[i] = false;
    }

    int expansion_size = 16;
    int scaled_expansion_size = expansion_size + 2;
    hevc_expansion_ref = getExpansionMatHEVCImage(ref_image, 4, scaled_expansion_size);

    // 0行目
    for(int block_x = 1 ; block_x < (block_num_x * 2) - 1; block_x+=2){
        int p1_idx = block_x;
        int p2_idx = block_x + 1;
        same_corner_list[p1_idx].emplace(p2_idx);
        same_corner_list[p2_idx].emplace(p1_idx);
    }

    for(int block_y = 0 ; block_y < 2 * block_num_y ; block_y+=2){
        for(int block_x = 1 ; block_x < (block_num_x * 2) - 1; block_x+=2){
            int p1_idx = block_x +     2 * block_num_x * block_y;
            int p2_idx = block_x + 1 + 2 * block_num_x * block_y;
            same_corner_list[p1_idx].emplace(p2_idx);
            same_corner_list[p2_idx].emplace(p1_idx);

            if(block_y == 0 || block_y == (block_num_y - 1)) continue;

            int p3_idx = p1_idx + 2 * block_num_x;
            int p4_idx = p3_idx + 1;

            same_corner_list[p1_idx].emplace(p3_idx);
            same_corner_list[p1_idx].emplace(p4_idx);
            same_corner_list[p2_idx].emplace(p3_idx);
            same_corner_list[p2_idx].emplace(p4_idx);
            same_corner_list[p3_idx].emplace(p1_idx);
            same_corner_list[p3_idx].emplace(p2_idx);
            same_corner_list[p3_idx].emplace(p4_idx);
            same_corner_list[p4_idx].emplace(p1_idx);
            same_corner_list[p4_idx].emplace(p2_idx);
            same_corner_list[p4_idx].emplace(p3_idx);
        }
    }

    // 0行目
    for(int block_x = 1 ; block_x < (block_num_x * 2) - 1; block_x+=2){
        int p1_idx = block_x + 2 * block_num_x * (2 * block_num_y - 1);
        int p2_idx = block_x + 1;
        same_corner_list[p1_idx].emplace(p2_idx);
        same_corner_list[p2_idx].emplace(p1_idx);
    }
}


/**
 * @fn int Decoder::insertTriangle(int p1_idx, int p2_idx, int p3_idx, int type)
 * @brief 三角形を追加する
 * @param[in] p1_idx 頂点1の座標のインデックス
 * @param[in] p2_idx 頂点2の座標のインデックス
 * @param[in] p3_idx 頂点3の座標のインデックス
 * @param[in] type 分割タイプ
 * @return 挿入した三角形が格納されているインデックス
 */
int Decoder::insertTriangle(int p1_idx, int p2_idx, int p3_idx, int type) {
    std::vector<std::pair<cv::Point2f, int> > v;
    v.emplace_back(corners[p1_idx], p1_idx);
    v.emplace_back(corners[p2_idx], p2_idx);
    v.emplace_back(corners[p3_idx], p3_idx);

    // ラスタスキャン順でソート
    sort(v.begin(), v.end(), [](const std::pair<cv::Point2f, int> &a1, const std::pair<cv::Point2f, int> &a2) {
        if (a1.first.y != a2.first.y) {
            return a1.first.y < a2.first.y;
        } else {
            return a1.first.x < a2.first.x;
        }
    });

    Triangle triangle(v[0].second, v[1].second, v[2].second, static_cast<int>(triangles.size()));

    triangles.emplace_back(triangle, type);
    isCodedTriangle.emplace_back(false);
    triangle_gauss_results.emplace_back();
    triangle_gauss_results[triangle_gauss_results.size() - 1].residual = -1.0;
    delete_flag.emplace_back(false);

    return static_cast<int>(triangles.size() - 1);
}

Decoder::Decoder(const cv::Mat &refImage, const cv::Mat &targetImage) : ref_image(refImage),
                                                                            target_image(targetImage) {}

void Decoder::addNeighborVertex(int p1_idx, int p2_idx, int p3_idx) {
    neighbor_vtx[p1_idx].emplace(p2_idx);
    neighbor_vtx[p2_idx].emplace(p1_idx);

    neighbor_vtx[p1_idx].emplace(p3_idx);
    neighbor_vtx[p3_idx].emplace(p1_idx);

    neighbor_vtx[p2_idx].emplace(p3_idx);
    neighbor_vtx[p3_idx].emplace(p2_idx);
}

void Decoder::addCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_no) {
    covered_triangle[p1_idx].emplace(triangle_no);
    covered_triangle[p2_idx].emplace(triangle_no);
    covered_triangle[p3_idx].emplace(triangle_no);
}

void Decoder::reconstructionTriangle(std::vector<CodingTreeUnit*> ctu) {
    for(int i = 0 ; i < ctu.size() ; i++) {
        CodingTreeUnit* cu = ctu[i];
        int type = (i % 2 == 0 ? DIVIDE::TYPE1 : DIVIDE::TYPE2);
        Triangle t = triangles[i].first;
        reconstructionTriangle(cu, Point3Vec(corners[t.p1_idx], corners[t.p2_idx], corners[t.p3_idx]),type);
    }
}

void Decoder::reconstructionTriangle(CodingTreeUnit *ctu, Point3Vec triangle, int type) {

    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int p1_idx, p2_idx, p3_idx;
        p1_idx = getCornerIndex(triangle.p1);
        p2_idx = getCornerIndex(triangle.p2);
        p3_idx = getCornerIndex(triangle.p3);

        insertTriangle(p1_idx, p2_idx, p3_idx, type);
        return;
    }

    TriangleDivision::SplitResult result = TriangleDivision::getSplitTriangle(triangle.p1, triangle.p2, triangle.p3, type);

    TriangleDivision::SplitResult result_subdiv_1 = TriangleDivision::getSplitTriangle(result.t1.p1, result.t1.p2, result.t1.p3, result.t1_type);
    TriangleDivision::SplitResult result_subdiv_2 = TriangleDivision::getSplitTriangle(result.t2.p1, result.t2.p2, result.t2.p3, result.t2_type);

    if(ctu->node1 != nullptr) reconstructionTriangle(ctu->node1, result_subdiv_1.t1, result_subdiv_1.t1_type);
    if(ctu->node2 != nullptr) reconstructionTriangle(ctu->node2, result_subdiv_1.t2, result_subdiv_1.t2_type);
    if(ctu->node3 != nullptr) reconstructionTriangle(ctu->node3, result_subdiv_2.t1, result_subdiv_2.t1_type);
    if(ctu->node4 != nullptr) reconstructionTriangle(ctu->node4, result_subdiv_2.t2, result_subdiv_2.t2_type);
}

int Decoder::getCornerIndex(cv::Point2f p) {
    if(corner_flag[(int)(p.y * 2)][(int)(p.x * 2)] != -1) return corner_flag[(int)(p.y * 2)][(int)(p.x * 2)];
    corners.emplace_back(p);
    neighbor_vtx.emplace_back();
    covered_triangle.emplace_back();
    corner_flag[(int)(p.y * 2)][(int)(p.x * 2)] = static_cast<int>(corners.size() - 1);
    same_corner_list.emplace_back();
    same_corner_list[(int)corners.size() - 1].emplace(corners.size() - 1);
    return static_cast<int>(corners.size() - 1);
}
