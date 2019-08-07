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

    image_width = ref_image.cols;
    image_height = ref_image.rows;

    int block_num_x = ceil((double)image_width / (block_size_x));
    int block_num_y = ceil((double)image_height / (block_size_y));
    divide_steps = _divide_steps;
    coded_picture_num = 0;


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

    // 最終行
    for(int block_x = 1 ; block_x < (block_num_x * 2) - 1; block_x+=2){
        int p1_idx = block_x + 2 * block_num_x * (2 * block_num_y - 1);
        int p2_idx = block_x + 1;
        same_corner_list[p1_idx].emplace(p2_idx);
        same_corner_list[p2_idx].emplace(p1_idx);
    }

    triangle_index_counter = 0;

    for (int i = 0; i < triangles.size(); i++) {
        decode_ctus[i] = new CodingTreeUnit();
        decode_ctus[i]->split_cu_flag = false;
        decode_ctus[i]->node1 = decode_ctus[i]->node2 = decode_ctus[i]->node3 = decode_ctus[i]->node4 = nullptr;
        decode_ctus[i]->triangle_index = i;
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

    Triangle triangle(v[0].second, v[1].second, v[2].second, static_cast<int>(triangles.size()));

    triangles.emplace_back(triangle, type);
    isCodedTriangle.emplace_back(false);
    triangle_info.emplace_back();
    triangle_info[triangle_info.size() - 1].residual = -1.0;
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
        if(i % 2 == 0){
            bool flag = false;
            for (int x = 0; x < block_size_x; x++) {
                // diagonal line
                area_flag[i/2][x][block_size_y - x - 1] = (flag ? i : i + 1);
                flag = !flag;
            }
        }

        CodingTreeUnit* cu = ctu[i];
        int type = (i % 2 == 0 ? DIVIDE::TYPE1 : DIVIDE::TYPE2);
        Triangle t = triangles[i].first;
        reconstructionTriangle(cu, decode_ctus[i], Point3Vec(corners[t.p1_idx], corners[t.p2_idx], corners[t.p3_idx]),type);
    }
}

void Decoder::reconstructionTriangle(CodingTreeUnit *ctu, CodingTreeUnit *decode_ctu, Point3Vec triangle, int type) {

    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int p1_idx, p2_idx, p3_idx;
        p1_idx = getCornerIndex(triangle.p1);
        p2_idx = getCornerIndex(triangle.p2);
        p3_idx = getCornerIndex(triangle.p3);

        int t_idx = insertTriangle(p1_idx, p2_idx, p3_idx, type);
        decode_ctu->method = ctu->method;
        decode_ctu->triangle_index = t_idx;

        return;
    }

    TriangleDivision::SplitResult result = TriangleDivision::getSplitTriangle(triangle.p1, triangle.p2, triangle.p3, type);

    int p1_idx = getCornerIndex(result.t1.p1);
    int p2_idx = getCornerIndex(result.t1.p2);
    int p3_idx = getCornerIndex(result.t1.p3);
    insertTriangle(p1_idx, p2_idx, p3_idx, result.t1_type);

    p1_idx = getCornerIndex(result.t2.p1);
    p2_idx = getCornerIndex(result.t2.p2);
    p3_idx = getCornerIndex(result.t2.p3);
    insertTriangle(p1_idx, p2_idx, p3_idx, result.t2_type);

    TriangleDivision::SplitResult result_subdiv_1 = TriangleDivision::getSplitTriangle(result.t1.p1, result.t1.p2, result.t1.p3, result.t1_type);
    TriangleDivision::SplitResult result_subdiv_2 = TriangleDivision::getSplitTriangle(result.t2.p1, result.t2.p2, result.t2.p3, result.t2_type);

    p1_idx = getCornerIndex(result_subdiv_1.t1.p1);
    p2_idx = getCornerIndex(result_subdiv_1.t1.p2);
    p3_idx = getCornerIndex(result_subdiv_1.t1.p3);

    insertTriangle(p1_idx, p2_idx, p3_idx, result_subdiv_1.t1_type);

    p1_idx = getCornerIndex(result_subdiv_1.t2.p1);
    p2_idx = getCornerIndex(result_subdiv_1.t2.p2);
    p3_idx = getCornerIndex(result_subdiv_1.t2.p3);

    insertTriangle(p1_idx, p2_idx, p3_idx, result_subdiv_1.t2_type);

    p1_idx = getCornerIndex(result_subdiv_2.t1.p1);
    p2_idx = getCornerIndex(result_subdiv_2.t1.p2);
    p3_idx = getCornerIndex(result_subdiv_2.t1.p3);

    insertTriangle(p1_idx, p2_idx, p3_idx, result_subdiv_2.t1_type);

    p1_idx = getCornerIndex(result_subdiv_2.t2.p1);
    p2_idx = getCornerIndex(result_subdiv_2.t2.p2);
    p3_idx = getCornerIndex(result_subdiv_2.t2.p3);

    insertTriangle(p1_idx, p2_idx, p3_idx, result_subdiv_2.t2_type);

    // 分割回数が偶数回目のとき斜線の更新を行う
    int triangle_indexes[] = {(int)triangles.size() - 4, (int)triangles.size() - 3, (int)triangles.size() - 2, (int)triangles.size() - 1};

    int sx = ceil( std::min({triangle.p1.x, triangle.p2.x, triangle.p3.x}));
    int lx = floor(std::max({triangle.p1.x, triangle.p2.x, triangle.p3.x}));
    int sy = ceil( std::min({triangle.p1.y, triangle.p2.y, triangle.p3.y}));
    int ly = floor(std::max({triangle.p1.y, triangle.p2.y, triangle.p3.y}));

    int width =  (lx - sx) / 2 + 1;
    int height = (ly - sy) / 2 + 1;

    if(type == TYPE1) {
        for (int x = 0 ; x < width  ; x++) {
            area_flag[(x + sx) % block_size_x][(x + sy) % block_size_y] = (x % 2 == 0 ? triangle_indexes[0] : triangle_indexes[2]);
        }
    }else if(type == TYPE2) {
        for (int x = 0 ; x < width ; x++) {
            area_flag[(sx + width + x) % block_size_x][(sy + height + x) % block_size_y] = (x % 2 == 0 ? triangle_indexes[1] : triangle_indexes[3]);
        }

    }else if(type == TYPE3){
        for(int x = 0 ; x < width ; x++){
            area_flag[(sx + width + x) % block_size_x][(sy + height - x - 1) % block_size_y] = (x % 2 == 0 ? triangle_indexes[1] : triangle_indexes[2]);
        }
    }else if(type == TYPE4){
        for(int x = 0 ; x < width ; x++){
            area_flag[(x + sx) % block_size_x][(ly - x) % block_size_y] = (x % 2 == 0 ? triangle_indexes[1] : triangle_indexes[2]);
        }
    }


    decode_ctu->node1 = new CodingTreeUnit();
    decode_ctu->node1->node1 = decode_ctu->node1->node2 = decode_ctu->node1->node3 = decode_ctu->node1->node4 = nullptr;
    if(ctu->node1 != nullptr) reconstructionTriangle(ctu->node1, decode_ctu->node1, result_subdiv_1.t1, result_subdiv_1.t1_type);

    decode_ctu->node2 = new CodingTreeUnit();
    decode_ctu->node2->node1 = decode_ctu->node2->node2 = decode_ctu->node2->node3 = decode_ctu->node2->node4 = nullptr;
    if(ctu->node2 != nullptr) reconstructionTriangle(ctu->node2, decode_ctu->node2, result_subdiv_1.t2, result_subdiv_1.t2_type);

    decode_ctu->node3 = new CodingTreeUnit();
    decode_ctu->node3->node1 = decode_ctu->node3->node2 = decode_ctu->node3->node3 = decode_ctu->node3->node4 = nullptr;
    if(ctu->node3 != nullptr) reconstructionTriangle(ctu->node3, decode_ctu->node3, result_subdiv_2.t1, result_subdiv_2.t1_type);

    decode_ctu->node4 = new CodingTreeUnit();
    decode_ctu->node4->node1 = decode_ctu->node4->node2 = decode_ctu->node4->node3 = decode_ctu->node4->node4 = nullptr;
    if(ctu->node4 != nullptr) reconstructionTriangle(ctu->node4, decode_ctu->node4, result_subdiv_2.t2, result_subdiv_2.t2_type);
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

cv::Mat Decoder::getReconstructionTriangleImage() {
    cv::Mat out = target_image.clone();

    for(int i = 0 ; i < triangles.size() ; i++){
        cv::Point2f p1 = corners[triangles[i].first.p1_idx];
        cv::Point2f p2 = corners[triangles[i].first.p2_idx];
        cv::Point2f p3 = corners[triangles[i].first.p3_idx];
        drawTriangle(out, p1, p2, p3, WHITE);
    }

    return out;
}

cv::Mat Decoder::getModeImage(std::vector<CodingTreeUnit*> ctus, const std::vector<std::vector<std::vector<int>>> &area_flag) {
    cv::Mat out = cv::Mat::zeros(ref_image.size(), CV_8UC3);

    for(int i = 0 ; i < decode_ctus.size() ; i++){
        getModeImage(decode_ctus[i], out, area_flag[i]);
    }

    for(const auto& triangle : triangles){
        drawTriangle(out, corners[triangle.first.p1_idx], corners[triangle.first.p2_idx], corners[triangle.first.p3_idx], WHITE);
    }
    return out;
}

void Decoder::getModeImage(CodingTreeUnit* ctu, cv::Mat &out, const std::vector<std::vector<int>> &area_flag){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int triangle_index = triangle_index_counter;
        Triangle triangle_corner_idx = triangles[triangle_index].first;
        Point3Vec triangle(corners[triangle_corner_idx.p1_idx], corners[triangle_corner_idx.p2_idx], corners[triangle_corner_idx.p3_idx]);

        std::vector<cv::Point2f> pixels = getPixelsInTriangle(triangle, area_flag, triangle_index, ctu, block_size_x, block_size_y);

        if(ctu->parallel_flag) {
            if(ctu->method == MV_CODE_METHOD::MERGE){
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = 0;
                    G(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    B(out, (int)pixel.x, (int)pixel.y) = 0;
                }
            }else{
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    G(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    B(out, (int)pixel.x, (int)pixel.y) = 0;
                }
            }

        }else{
            if(ctu->method == MV_CODE_METHOD::MERGE){
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = 0;
                    G(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                    B(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                }
            }else{
                for(auto pixel : pixels) {
                    R(out, (int)pixel.x, (int)pixel.y) = 0;
                    G(out, (int)pixel.x, (int)pixel.y) = 0;
                    B(out, (int)pixel.x, (int)pixel.y) = M(target_image, (int)pixel.x, (int)pixel.y);
                }
            }
        }

        triangle_index_counter++;
        return;
    }

    if(ctu->node1 != nullptr) getModeImage(ctu->node1, out, area_flag);
    if(ctu->node2 != nullptr) getModeImage(ctu->node2, out, area_flag);
    if(ctu->node3 != nullptr) getModeImage(ctu->node3, out, area_flag);
    if(ctu->node4 != nullptr) getModeImage(ctu->node4, out, area_flag);
}
