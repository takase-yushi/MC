//
// Created by kasph on 2019/08/05.
//

#include <cmath>
#include <iostream>
#include <vector>
#include "../includes/Decoder.h"
#include "../includes/CollocatedMvTree.h"
#include "../includes/TriangleDivision.h"
#include "../includes/ImageUtil.h"
#include "../includes/ME.hpp"

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

    for(int block_y = 1 ; block_y < (2 * block_num_y) - 1 ; block_y+=2){
        for(int block_x = 1 ; block_x < (block_num_x * 2) - 1; block_x+=2){
            int p1_idx = block_x +     2 * block_num_x * block_y;
            int p2_idx = block_x + 1 + 2 * block_num_x * block_y;
            int p3_idx = p1_idx + 2 * block_num_x;
            int p4_idx = p3_idx + 1;

            same_corner_list[p1_idx].emplace(p2_idx);
            same_corner_list[p1_idx].emplace(p3_idx);
            same_corner_list[p1_idx].emplace(p4_idx);
            same_corner_list[p2_idx].emplace(p1_idx);
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
        int p2_idx = p1_idx + 1;
        same_corner_list[p1_idx].emplace(p2_idx);
        same_corner_list[p2_idx].emplace(p1_idx);
    }

    triangle_index_counter = 0;

    decode_ctus.resize((int)triangles.size());
    for (int i = 0; i < triangles.size(); i++) {
        decode_ctus[i] = new CodingTreeUnit();
        decode_ctus[i]->split_cu_flag = false;
        decode_ctus[i]->node1 = decode_ctus[i]->node2 = decode_ctus[i]->node3 = decode_ctus[i]->node4 = nullptr;
        decode_ctus[i]->triangle_index = i;
    }

    area_flag.assign(triangles.size(), std::vector< std::vector<int> >(block_size_x, std::vector<int>(block_size_y, -1)));

    std::cout << triangles.size() << std::endl;
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
        reconstructionTriangle(cu, decode_ctus[i], area_flag[i], Point3Vec(corners[t.p1_idx], corners[t.p2_idx], corners[t.p3_idx]),i, type);
    }
}

void Decoder::reconstructionTriangle(CodingTreeUnit *ctu, CodingTreeUnit *decode_ctu, std::vector<std::vector<int>> &diagonal_area_flag, Point3Vec triangle, int triangle_index, int type) {

    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        decode_ctu->method = ctu->method;
        decode_ctu->triangle_index = triangle_index;
        decode_ctu->parallel_flag = ctu->parallel_flag;
        decode_ctu->ref_triangle_idx = ctu->ref_triangle_idx;
        isCodedTriangle[triangle_index] = true;

        std::vector<int> spatial_triangle_list = getSpatialTriangleList(triangle_index);
        std::vector<std::pair<cv::Point2f, MV_CODE_METHOD>> vector_list;
        std::vector<std::vector<cv::Point2f>> warping_vector_list;

        puts("");
        std::cout << "spatial_triangle_list.size():" << spatial_triangle_list.size() << std::endl;
        for(const auto& t_list : spatial_triangle_list){
            std::cout << t_list << std::endl;
        }


        for(int i = 0 ; i < spatial_triangle_list.size() ; i++){
            std::pair<Triangle, int> t = triangles[spatial_triangle_list[i]];
            GaussResult spatial_triangle_info = triangle_info[spatial_triangle_list[i]];
            if(spatial_triangle_info.parallel_flag){
                if(!isMvExists(vector_list, spatial_triangle_info.mv_parallel) && vector_list.size() < MV_LIST_MAX_NUM) {
                    vector_list.emplace_back(spatial_triangle_info.mv_parallel, SPATIAL);
                    warping_vector_list.emplace_back();
                }
            }else{
                cv::Point2f p1 = spatial_triangle_info.mv_warping[0];
                cv::Point2f p2 = spatial_triangle_info.mv_warping[1];
                cv::Point2f p3 = spatial_triangle_info.mv_warping[2];
#if MVD_DEBUG_LOG
                std::cout << "target_triangle_coordinate:";
            std::cout << corners[triangles[triangle_idx].first.p1_idx] << " ";
            std::cout << corners[triangles[triangle_idx].first.p2_idx] << " ";
            std::cout << corners[triangles[triangle_idx].first.p3_idx] << std::endl;
            std::cout << "ref_triangle_coordinate:";
            std::cout << corners[triangles[spatial_triangle_index].first.p1_idx] << " ";
            std::cout << corners[triangles[spatial_triangle_index].first.p2_idx] << " ";
            std::cout << corners[triangles[spatial_triangle_index].first.p3_idx] << std::endl;
            std::cout << "ref_triangle_mvs:";
            std::cout << p1 << " " << p2 << " " << p3 << std::endl;
#endif
                cv::Point2f mv_average;
                std::vector<cv::Point2f> ref_mvs{p1, p2, p3};
                std::pair<Triangle, int> target_triangle = triangles[triangle_index];
                cv::Point2f pp1 = corners[target_triangle.first.p1_idx], pp2 = corners[target_triangle.first.p2_idx], pp3 = corners[target_triangle.first.p3_idx];
                std::pair<Triangle, int> ref_triangle = triangles[spatial_triangle_list[i]];
                std::vector<cv::Point2f> ref_triangle_coordinates{corners[ref_triangle.first.p1_idx], corners[ref_triangle.first.p2_idx], corners[ref_triangle.first.p3_idx]};
                std::vector<cv::Point2f> target_triangle_coordinates{cv::Point2f((pp1.x + pp2.x + pp3.x) / 3.0, (pp1.y + pp2.y + pp3.y) / 3.0)};
                std::vector<cv::Point2f> mvs = getPredictedWarpingMv(ref_triangle_coordinates, ref_mvs, target_triangle_coordinates);
                mv_average = mvs[0];

                if (!decode_ctu->parallel_flag) {
                    target_triangle_coordinates.clear();
                    target_triangle_coordinates.emplace_back(pp1);
                    target_triangle_coordinates.emplace_back(pp2);
                    target_triangle_coordinates.emplace_back(pp3);
                    mvs = getPredictedWarpingMv(ref_triangle_coordinates, ref_mvs, target_triangle_coordinates);
                    std::vector<cv::Point2f> v(3);
                    v[0] = mvs[0];
                    v[1] = mvs[1];
                    v[2] = mvs[2];
                    warping_vector_list.emplace_back(v);

                }else{
                    warping_vector_list.emplace_back();
                }

                mv_average = roundVecQuarter(mv_average);
                if(!isMvExists(vector_list, mv_average) && vector_list.size() < MV_LIST_MAX_NUM){
                    vector_list.emplace_back(mv_average, SPATIAL);
                }
            }
        }

        std::cout << "vector_list.size() :" << vector_list.size() << std::endl;
        std::cout << "warping_list.size():" << warping_vector_list.size() << std::endl;
        cv::Point2f collocated_mv(0.0, 0.0);
        if(!isMvExists(vector_list, collocated_mv)) {
            vector_list.emplace_back(collocated_mv, SPATIAL);
            warping_vector_list.emplace_back();
        }

        if(vector_list.size() < 2) {
            vector_list.emplace_back(cv::Point2f(0.0, 0.0), Collocated);
            warping_vector_list.emplace_back();
        }

        double sx = std::min({triangle.p1.x, triangle.p2.x, triangle.p3.x});
        double sy = std::min({triangle.p1.y, triangle.p2.y, triangle.p3.y});
        double lx = std::max({triangle.p1.x, triangle.p2.x, triangle.p3.x});
        double ly = std::max({triangle.p1.y, triangle.p2.y, triangle.p3.y});

        if(decode_ctu->method == SPATIAL){
            std::pair<cv::Point2f, MV_CODE_METHOD> ref_mv = vector_list[decode_ctu->ref_triangle_idx];
            std::vector<cv::Point2f> ref_warping_mv = warping_vector_list[decode_ctu->ref_triangle_idx];
            if(decode_ctu->parallel_flag){

                cv::Point2f mvd = ctu->mvds[0];
                if(!ctu->x_greater_0_flag[0]) mvd.x = 0;
                if(!ctu->y_greater_0_flag[0]) mvd.y = 0;

                if(ctu->x_greater_0_flag[0]) {
                    if(ctu->x_greater_1_flag[0]){
                        mvd.x += 2.0;
                    }else{
                        mvd.x = 1.0;
                    }
                }

                if(ctu->x_sign_flag[0]){
                    mvd.x *= -1.0;
                }

                if(ctu->y_greater_0_flag[0]) {
                    if(ctu->y_greater_1_flag[0]){
                        mvd.y += 2.0;
                    }else{
                        mvd.y = 1.0;
                    }
                }

                if(ctu->y_sign_flag[0]){
                    mvd.y *= -1.0;
                }

                mvd /= 4.0;

                decode_ctu->mv1 = ref_mv.first - mvd;
                decode_ctu->mv2 = ref_mv.first - mvd;
                decode_ctu->mv3 = ref_mv.first - mvd;
                triangle_info[triangle_index].mv_parallel = decode_ctu->mv1;
                triangle_info[triangle_index].parallel_flag = true;
            }else {
                std::vector<cv::Point2f> mvds;
                for(int i = 0 ; i < 3 ; i++){
                    cv::Point2f mvd = ctu->mvds[i];
                    if(!ctu->x_greater_0_flag[i]) mvd.x = 0;

                    if(ctu->x_greater_0_flag[i]) {
                        if(ctu->x_greater_1_flag[i]){
                            mvd.x += 2.0;
                        }else{
                            mvd.x = 1.0;
                        }
                    }

                    if(ctu->x_sign_flag[i]){
                        mvd.x *= -1.0;
                    }

                    if(!ctu->y_greater_0_flag[i]) mvd.y = 0;

                    if(ctu->y_greater_0_flag[i]) {
                        if(ctu->y_greater_1_flag[i]){
                            mvd.y += 2.0;
                        }else{
                            mvd.y = 1.0;
                        }
                    }

                    if(ctu->y_sign_flag[i]){
                        mvd.y *= -1.0;
                    }

                    mvd /= 4.0;
                    mvds.emplace_back(mvd);
                }

                if(ref_warping_mv.empty()){
                    decode_ctu->mv1 = ref_mv.first - mvds[0];
                    decode_ctu->mv2 = ref_mv.first - mvds[1];
                    decode_ctu->mv3 = ref_mv.first - mvds[2];
                }else {
                    decode_ctu->mv1 = ref_warping_mv[0] - mvds[0];
                    decode_ctu->mv2 = ref_warping_mv[1] - mvds[1];
                    decode_ctu->mv3 = ref_warping_mv[2] - mvds[2];
                }
                triangle_info[triangle_index].mv_warping.clear();
                triangle_info[triangle_index].mv_warping.emplace_back(decode_ctu->mv1);
                triangle_info[triangle_index].mv_warping.emplace_back(decode_ctu->mv2);
                triangle_info[triangle_index].mv_warping.emplace_back(decode_ctu->mv3);
                triangle_info[triangle_index].parallel_flag = false;
            }
        }else{
            if(decode_ctu->parallel_flag){
                std::vector<std::pair<cv::Point2f, MV_CODE_METHOD >> merge_list;

                for(int i = 0 ; i < vector_list.size() ; i++){
                    std::pair<cv::Point2f, MV_CODE_METHOD> spatial_triangle_info = vector_list[i];
                    GaussResult spatial_triangle = triangle_info[spatial_triangle_list[i]];

                    if(spatial_triangle_info.second == SPATIAL){
                        if(spatial_triangle.mv_parallel.x + sx < -16 || spatial_triangle.mv_parallel.y + sy < -16 || spatial_triangle.mv_parallel.x + lx >= target_image.cols + 16 || spatial_triangle.mv_parallel.y + ly >= target_image.rows + 16) continue;

                        if(!isMvExists(merge_list, spatial_triangle.mv_parallel) && merge_list.size() < MV_LIST_MAX_NUM) {
                            std::pair<cv::Point2f, MV_CODE_METHOD> ref_mv = vector_list[i];
                            merge_list.emplace_back(ref_mv.first, MERGE);
                        }

                    }else{
                        if (spatial_triangle.mv_warping[0].x + sx < -16 || spatial_triangle.mv_warping[0].y + sy < -16 || spatial_triangle.mv_warping[0].x + lx >= target_image.cols + 16 || spatial_triangle.mv_warping[0].y + ly >= target_image.rows + 16) continue;

                        if(!isMvExists(merge_list, spatial_triangle.mv_warping[0]) && merge_list.size() < MV_LIST_MAX_NUM) {
                            merge_list.emplace_back(spatial_triangle.mv_warping[0], MERGE);
                        }
                    }
                }

                std::pair<cv::Point2f, MV_CODE_METHOD> ref_mv = merge_list[decode_ctu->ref_triangle_idx];
                decode_ctu->mv1 = ref_mv.first;
                decode_ctu->mv2 = ref_mv.first;
                decode_ctu->mv3 = ref_mv.first;

                triangle_info[triangle_index].mv_parallel = ref_mv.first;
                triangle_info[triangle_index].parallel_flag = true;

            }else{
                int merge_idx = decode_ctu->ref_triangle_idx;
                std::vector<std::vector<cv::Point2f>> warping_vector_history;
                std::vector<std::vector<cv::Point2f>> merge_mv_list;

                for(int i = 0 ; i < warping_vector_list.size() ; i++){
                    if(warping_vector_list[i].empty()) continue;
                    std::vector<cv::Point2f> mvs = warping_vector_list[i];

                    if(mvs[0].x + sx < -16 || mvs[0].y + sy < -16 || mvs[0].x + lx >= target_image.cols + 16  || mvs[0].y + ly>=target_image.rows + 16 ) continue;
                    if(mvs[1].x + sx < -16 || mvs[1].y + sy < -16 || mvs[1].x + lx >= target_image.cols + 16  || mvs[1].y + ly>=target_image.rows + 16 ) continue;
                    if(mvs[2].x + sx < -16 || mvs[2].y + sy < -16 || mvs[2].x + lx >= target_image.cols + 16  || mvs[2].y + ly>=target_image.rows + 16 ) continue;

                    if(!isMvExists(merge_mv_list, warping_vector_list[i]) && merge_mv_list.size() < MV_LIST_MAX_NUM){
                        merge_mv_list.emplace_back(warping_vector_list[i]);
                    }
                }

                std::vector<cv::Point2f> ref_warping_mv = merge_mv_list[merge_idx];

                decode_ctu->mv1 = ref_warping_mv[0];
                decode_ctu->mv2 = ref_warping_mv[1];
                decode_ctu->mv3 = ref_warping_mv[2];

                triangle_info[triangle_index].mv_warping = ref_warping_mv;

            }
        }

        triangle_info[triangle_index].parallel_flag = ctu->parallel_flag;
        triangle_info[triangle_index].method = ctu->method;
        return;
    }

    TriangleDivision::SplitResult result = TriangleDivision::getSplitTriangle(triangle.p1, triangle.p2, triangle.p3, type);

    int t1_p1_idx = getCornerIndex(result.t1.p1);
    int t1_p2_idx = getCornerIndex(result.t1.p2);
    int t1_p3_idx = getCornerIndex(result.t1.p3);
    addCornerAndTriangle({t1_p1_idx, t1_p2_idx, t1_p3_idx}, triangle_index, result.t1_type);

    int t2_p1_idx = getCornerIndex(result.t2.p1);
    int t2_p2_idx = getCornerIndex(result.t2.p2);
    int t2_p3_idx = getCornerIndex(result.t2.p3);
    addCornerAndTriangle({t2_p1_idx, t2_p2_idx, t2_p3_idx}, triangle_index, result.t2_type);
//
    TriangleDivision::SplitResult result_subdiv_1 = TriangleDivision::getSplitTriangle(result.t1.p1, result.t1.p2, result.t1.p3, result.t1_type);
    TriangleDivision::SplitResult result_subdiv_2 = TriangleDivision::getSplitTriangle(result.t2.p1, result.t2.p2, result.t2.p3, result.t2_type);


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
            diagonal_area_flag[(x + sx) % block_size_x][(x + sy) % block_size_y] = (x % 2 == 0 ? triangle_indexes[0] : triangle_indexes[2]);
        }
    }else if(type == TYPE2) {
        for (int x = 0 ; x < width ; x++) {
            diagonal_area_flag[(sx + width + x) % block_size_x][(sy + height + x) % block_size_y] = (x % 2 == 0 ? triangle_indexes[1] : triangle_indexes[3]);
        }

    }else if(type == TYPE3){
        for(int x = 0 ; x < width ; x++){
            diagonal_area_flag[(sx + width + x) % block_size_x][(sy + height - x - 1) % block_size_y] = (x % 2 == 0 ? triangle_indexes[1] : triangle_indexes[2]);
        }
    }else if(type == TYPE4){
        for(int x = 0 ; x < width ; x++){
            diagonal_area_flag[(x + sx) % block_size_x][(ly - x) % block_size_y] = (x % 2 == 0 ? triangle_indexes[1] : triangle_indexes[2]);
        }
    }


    decode_ctu->node1 = new CodingTreeUnit();
    decode_ctu->node1->node1 = decode_ctu->node1->node2 = decode_ctu->node1->node3 = decode_ctu->node1->node4 = nullptr;
    if(ctu->node1 != nullptr) reconstructionTriangle(ctu->node1, decode_ctu->node1, diagonal_area_flag, result_subdiv_1.t1, triangle_indexes[0], result_subdiv_1.t1_type);

    decode_ctu->node2 = new CodingTreeUnit();
    decode_ctu->node2->node1 = decode_ctu->node2->node2 = decode_ctu->node2->node3 = decode_ctu->node2->node4 = nullptr;
    if(ctu->node2 != nullptr) reconstructionTriangle(ctu->node2, decode_ctu->node2, diagonal_area_flag, result_subdiv_1.t2, triangle_indexes[1], result_subdiv_1.t2_type);

    decode_ctu->node3 = new CodingTreeUnit();
    decode_ctu->node3->node1 = decode_ctu->node3->node2 = decode_ctu->node3->node3 = decode_ctu->node3->node4 = nullptr;
    if(ctu->node3 != nullptr) reconstructionTriangle(ctu->node3, decode_ctu->node3, diagonal_area_flag, result_subdiv_2.t1, triangle_indexes[2], result_subdiv_2.t1_type);

    decode_ctu->node4 = new CodingTreeUnit();
    decode_ctu->node4->node1 = decode_ctu->node4->node2 = decode_ctu->node4->node3 = decode_ctu->node4->node4 = nullptr;
    if(ctu->node4 != nullptr) reconstructionTriangle(ctu->node4, decode_ctu->node4, diagonal_area_flag, result_subdiv_2.t2, triangle_indexes[3], result_subdiv_2.t2_type);
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

cv::Mat Decoder::getModeImage(std::vector<CodingTreeUnit*> ctus, const std::vector<std::vector<std::vector<int>>> &diagonal_area_flag) {
    cv::Mat out = cv::Mat::zeros(ref_image.size(), CV_8UC3);

    std::cout << "decode_triangles_size():" << triangles.size() << std::endl;
    for(int i = 0 ; i < decode_ctus.size() ; i++){
        getModeImage(decode_ctus[i], out, area_flag[i]);
    }

    for(const auto& triangle : triangles){
        drawTriangle(out, corners[triangle.first.p1_idx], corners[triangle.first.p2_idx], corners[triangle.first.p3_idx], WHITE);
    }
    return out;
}

void Decoder::getModeImage(CodingTreeUnit* ctu, cv::Mat &out, const std::vector<std::vector<int>> &diagonal_area_flag){
    if(ctu->node1 == nullptr && ctu->node2 == nullptr && ctu->node3 == nullptr && ctu->node4 == nullptr) {
        int triangle_index = ctu->triangle_index;
        Triangle triangle_corner_idx = triangles[triangle_index].first;
        Point3Vec triangle(corners[triangle_corner_idx.p1_idx], corners[triangle_corner_idx.p2_idx], corners[triangle_corner_idx.p3_idx]);

        std::vector<cv::Point2f> pixels = getPixelsInTriangle(triangle, diagonal_area_flag, triangle_index, ctu, block_size_x, block_size_y);

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

    if(ctu->node1 != nullptr) getModeImage(ctu->node1, out, diagonal_area_flag);
    if(ctu->node2 != nullptr) getModeImage(ctu->node2, out, diagonal_area_flag);
    if(ctu->node3 != nullptr) getModeImage(ctu->node3, out, diagonal_area_flag);
    if(ctu->node4 != nullptr) getModeImage(ctu->node4, out, diagonal_area_flag);
}

std::vector<int> Decoder::getSpatialTriangleList(int triangle_index) {
    std::pair<Triangle, int> triangle = triangles[triangle_index];
    std::set<int> spatialTriangles;
    std::vector<int> list1 = getIdxCoveredTriangleIndexList(triangle.first.p1_idx);
    std::vector<int> list2 = getIdxCoveredTriangleIndexList(triangle.first.p2_idx);
    std::vector<int> list3 = getIdxCoveredTriangleIndexList(triangle.first.p3_idx);

    std::set<int> mutualIndexSet1, mutualIndexSet2, mutualIndexSet3;

#if MVD_DEBUG_LOG
    std::cout << "p1:" << triangles[t_idx].first.p1_idx << std::endl;
    for(auto item : list1){
        std::cout << item << std::endl;
    }
    puts("");

    std::cout << "p2:" << triangles[t_idx].first.p2_idx << std::endl;
    for(auto item : list2){
        std::cout << item << std::endl;
    }
    puts("");
    std::cout << "p3:" << triangles[t_idx].first.p3_idx << std::endl;

    for(auto item : list3){
        std::cout << item << std::endl;
    }
    std::cout << "t_idx:" << t_idx << std::endl;
    puts("");

#endif

    for(auto idx : list1) if(isCodedTriangle[idx] && idx != triangle_index) mutualIndexSet1.emplace(idx);
    for(auto idx : list2) if(isCodedTriangle[idx] && idx != triangle_index) mutualIndexSet2.emplace(idx);
    for(auto idx : list3) if(isCodedTriangle[idx] && idx != triangle_index) mutualIndexSet3.emplace(idx);

    for(auto idx : mutualIndexSet1) spatialTriangles.emplace(idx);
    for(auto idx : mutualIndexSet2) spatialTriangles.emplace(idx);
    for(auto idx : mutualIndexSet3) spatialTriangles.emplace(idx);

    std::vector<int> ret;

    for(auto idx : spatialTriangles){
        ret.emplace_back(idx);
    }

    if(triangle_index == 250){
        puts("triangle_250 covered triangle");
        for(const auto item : ret){
            std::cout << item << std::endl;
        }
    }
    return ret;
}

std::vector<int> Decoder::getIdxCoveredTriangleIndexList(int vertex_index) {
    std::set<int> same_corners = same_corner_list[vertex_index];

    std::set<int> s;
    std::set<int> tmp_s;
    for(auto same_corner : same_corners){
        tmp_s = covered_triangle[same_corner];
        for(auto idx : tmp_s) s.emplace(idx);
    }
    std::vector<int> v;

    for(auto triangle_idx : s) {
        v.emplace_back(triangle_idx);
    }

    std::sort(v.begin(), v.end());

    return v;
}

bool Decoder::isMvExists(const std::vector<std::pair<cv::Point2f, MV_CODE_METHOD>> &vectors, const cv::Point2f &mv) {
    for(const auto &vector : vectors){
        if(vector.first == mv) return true;
    }

    return false;
}

bool Decoder::isMvExists(const std::vector<std::vector<cv::Point2f>> &vectors, const std::vector<cv::Point2f> &mvs) {
    for(const auto &vector : vectors){
        if(vector[0] == mvs[0] && vector[1] == mvs[1] && vector[2] == mvs[2]) return true;
    }

    return false;
}

void Decoder::addCornerAndTriangle(Triangle triangle, int triangle_index, int type){
    switch(type) {
        case DIVIDE::TYPE1:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p2 - p1) / 2.0;
            cv::Point2f y = (p3 - p1) / 2.0;

            cv::Point2f a = p1;
            cv::Point2f b = p2;
            cv::Point2f c = a + x + y;
            cv::Point2f d = p3;

            int c_idx = getCornerIndex(c);

            int a_idx = triangle.p1_idx;
            int b_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, c_idx, TYPE5);
            int t2_idx = insertTriangle(a_idx, c_idx, d_idx, TYPE6);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, c_idx);
            addNeighborVertex(a_idx, c_idx, d_idx);

            addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
            addCoveredTriangle(a_idx, c_idx, d_idx, t2_idx);

        }
            break;
        case DIVIDE::TYPE2:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p2 - p3) / 2.0;
            cv::Point2f y = (p1 - p3) / 2.0;

            cv::Point2f a = p1;
            cv::Point2f b = p3 + x + y;
            cv::Point2f c = p2;
            cv::Point2f d = p3;

            int b_idx = getCornerIndex(b);

            int a_idx = triangle.p1_idx;
            int c_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, d_idx, TYPE8);
            int t2_idx = insertTriangle(b_idx, c_idx, d_idx, TYPE7);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, d_idx);
            addNeighborVertex(b_idx, c_idx, d_idx);

            addCoveredTriangle(a_idx, b_idx, d_idx, t1_idx);
            addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);
        }
            break;
        case DIVIDE::TYPE3:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p1 - p2) / 2.0;
            cv::Point2f y = (p3 - p2) / 2.0;

            cv::Point2f a = p1;
            cv::Point2f b = p2;
            cv::Point2f c = p2 + x + y;
            cv::Point2f d = p3;

            int c_idx = getCornerIndex(c);

            int a_idx = triangle.p1_idx;
            int b_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, c_idx, TYPE5);
            int t2_idx = insertTriangle(b_idx, c_idx, d_idx, TYPE8);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, c_idx);
            addNeighborVertex(b_idx, c_idx, d_idx);

            addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
            addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);

        }
            break;
        case DIVIDE::TYPE4:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p3 - p2) / 2.0;
            cv::Point2f y = (p1 - p2) / 2.0;

            cv::Point2f a = p1;
            cv::Point2f b = p2 + x + y;
            cv::Point2f c = p2;
            cv::Point2f d = p3;

            int b_idx = getCornerIndex(b);

            int a_idx = triangle.p1_idx;
            int c_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b_idx, c_idx, TYPE6);
            int t2_idx = insertTriangle(b_idx, c_idx, d_idx, TYPE7);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b_idx, c_idx);
            addNeighborVertex(b_idx, c_idx, d_idx);

            addCoveredTriangle(a_idx, b_idx, c_idx, t1_idx);
            addCoveredTriangle(b_idx, c_idx, d_idx, t2_idx);

        }
            break;
        case DIVIDE::TYPE5:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p2 - p1) / 2.0;
            x.x = (int)x.x;

            cv::Point2f b1 = p1 + x;
            cv::Point2f b2 = p1 + x;
            b2.x += 1;
            b1.y = (int)b1.y;
            b2.y = (int)b2.y;

            cv::Point2f d1 = p3;
            cv::Point2f d2 = p3;
            d1.x = (int)d1.x;
            d1.y = (int)d1.y;
            d2.x = ceil(d2.x);
            d2.y = (int)(d2.y);

            int b1_idx = getCornerIndex(b1);
            int b2_idx = getCornerIndex(b2);
            int d1_idx = getCornerIndex(d1);
            int d2_idx = getCornerIndex(d2);

            int a_idx = triangle.p1_idx;
            int c_idx = triangle.p2_idx;

            int t1_idx = insertTriangle(a_idx, b1_idx, d1_idx, TYPE3);
            int t2_idx = insertTriangle(b2_idx, c_idx, d2_idx, TYPE1);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b1_idx, d1_idx);
            addNeighborVertex(b2_idx, c_idx, d2_idx);

            addCoveredTriangle(a_idx, b1_idx, d1_idx, t1_idx);
            addCoveredTriangle(b2_idx, c_idx, d2_idx, t2_idx);

            same_corner_list[b1_idx].emplace(b2_idx);
            same_corner_list[b2_idx].emplace(b1_idx);

            same_corner_list[d1_idx].emplace(d2_idx);
            same_corner_list[d2_idx].emplace(d1_idx);

        }
            break;
        case DIVIDE::TYPE6:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f y = (p3 - p1) / 2.0;
            y.y = (int)y.y;

            cv::Point2f b1 = p1 + y;
            cv::Point2f b2 = p1 + y;
            b2.y += 1;

            cv::Point2f c1 = p2;
            cv::Point2f c2 = p2;
            c1.x = (int)c1.x;
            c1.y = (int)c1.y;
            c2.x = (int)(c2.x);
            c2.y = ceil(c2.y);

            int b1_idx = getCornerIndex(b1);
            int b2_idx = getCornerIndex(b2);
            int c1_idx = getCornerIndex(c1);
            int c2_idx = getCornerIndex(c2);

            int a_idx = triangle.p1_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a_idx, b1_idx, c1_idx, TYPE4);
            int t2_idx = insertTriangle(b2_idx, c2_idx, d_idx, TYPE1);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a_idx, b1_idx, c1_idx);
            addNeighborVertex(b2_idx, c2_idx, d_idx);

            addCoveredTriangle(a_idx, b1_idx, c1_idx, t1_idx);
            addCoveredTriangle(b2_idx, c2_idx, d_idx, t2_idx);

            same_corner_list[b1_idx].emplace(b2_idx);
            same_corner_list[b2_idx].emplace(b1_idx);

            same_corner_list[c1_idx].emplace(c2_idx);
            same_corner_list[c2_idx].emplace(c1_idx);
        }
            break;
        case DIVIDE::TYPE7:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f x = (p3 - p2) / 2.0;
            x.x = (int)x.x;

            cv::Point2f a1 = p1, a2 = p1;
            a1.x = (int)a1.x;
            a1.y = ceil(a1.y);
            a2.x = ceil(a2.x);
            a2.y = ceil(a2.y);

            cv::Point2f c1 = p2 + x;
            cv::Point2f c2 = p2 + x;
            c2.x += 1;

            int a1_idx = getCornerIndex(a1);
            int a2_idx = getCornerIndex(a2);
            int c1_idx = getCornerIndex(c1);
            int c2_idx = getCornerIndex(c2);

            int b_idx = triangle.p2_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(a1_idx, b_idx, c1_idx, TYPE2);
            int t2_idx = insertTriangle(a2_idx, c2_idx, d_idx, TYPE4);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(a1_idx, b_idx, c1_idx);
            addNeighborVertex(a2_idx, c2_idx, d_idx);

            addCoveredTriangle(a1_idx, b_idx, c1_idx, t1_idx);
            addCoveredTriangle(a2_idx, c2_idx, d_idx, t2_idx);

            same_corner_list[a1_idx].emplace(a2_idx);
            same_corner_list[a2_idx].emplace(a1_idx);

            same_corner_list[c1_idx].emplace(c2_idx);
            same_corner_list[c2_idx].emplace(c1_idx);

        }
            break;
        case DIVIDE::TYPE8:
        {
            cv::Point2f p1 = corners[triangle.p1_idx];
            cv::Point2f p2 = corners[triangle.p2_idx];
            cv::Point2f p3 = corners[triangle.p3_idx];

            cv::Point2f y = (p3 - p1) / 2.0;

            cv::Point2f a1 = p2;
            cv::Point2f a2 = p2;
            a1.x = ceil(a1.x);
            a1.y = (int)a1.y;
            a2.x = ceil(a2.x);
            a2.y = ceil(a2.y);

            cv::Point2f c1 = p1 + y;
            cv::Point2f c2 = p1 + y;
            c1.y = (int)c1.y;
            c2.y = ceil(c2.y);

            int a1_idx = getCornerIndex(a1);
            int a2_idx = getCornerIndex(a2);
            int c1_idx = getCornerIndex(c1);
            int c2_idx = getCornerIndex(c2);

            int b_idx = triangle.p1_idx;
            int d_idx = triangle.p3_idx;

            int t1_idx = insertTriangle(b_idx, a1_idx, c1_idx, TYPE2);
            int t2_idx = insertTriangle(a2_idx, c2_idx, d_idx, TYPE3);

            removeTriangleNeighborVertex(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx);
            removeTriangleCoveredTriangle(triangle.p1_idx, triangle.p2_idx, triangle.p3_idx, triangle_index);

            addNeighborVertex(b_idx, a1_idx, c1_idx);
            addNeighborVertex(a2_idx, c2_idx, d_idx);

            addCoveredTriangle(b_idx, a1_idx, c1_idx, t1_idx);
            addCoveredTriangle(a2_idx, c2_idx, d_idx, t2_idx);

            same_corner_list[a1_idx].emplace(a2_idx);
            same_corner_list[a2_idx].emplace(a1_idx);

            same_corner_list[c1_idx].emplace(c2_idx);
            same_corner_list[c2_idx].emplace(c1_idx);
        }
            break;
        default:
            break;
    }

    isCodedTriangle[triangle_index] = false;
    delete_flag[triangle_index] = true;
}

/**
 * @fn void TriangleDivision::removeTriangleNeighborVertex(int p1_idx, int p2_idx, int p3_idx)
 * @brief 指定された三角形に含まれる頂点隣接ノード集合から、自分以外のノードを消す
 * @details 日本語が難しいからコードで理解して
 * @param p1_idx
 * @param p2_idx
 * @param p3_idx
 */
void Decoder::removeTriangleNeighborVertex(int p1_idx, int p2_idx, int p3_idx) {
    neighbor_vtx[p1_idx].erase(p2_idx);
    neighbor_vtx[p1_idx].erase(p3_idx);
    neighbor_vtx[p2_idx].erase(p1_idx);
    neighbor_vtx[p2_idx].erase(p3_idx);
    neighbor_vtx[p3_idx].erase(p1_idx);
    neighbor_vtx[p3_idx].erase(p2_idx);
}

/**
 * @fn void TriangleDivision::removeTriangleCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_idx)
 * @brief p1, p2, p3を含む三角形の集合から, triangle_idx番目の三角形を消す
 * @param p1_idx 頂点1のインデックス
 * @param p2_idx 頂点2のインデックス
 * @param p3_idx 頂点3のインデックス
 * @param triangle_idx 削除対象の三角形のインデックス
 */
void Decoder::removeTriangleCoveredTriangle(int p1_idx, int p2_idx, int p3_idx, int triangle_idx) {
    covered_triangle[p1_idx].erase(triangle_idx);
    covered_triangle[p2_idx].erase(triangle_idx);
    covered_triangle[p3_idx].erase(triangle_idx);
}
