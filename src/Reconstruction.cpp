//
// Created by kasph on 2019/05/05.
//

#include <iostream>
#include "../includes/Reconstruction.h"
#include "../includes/TriangleDivision.h"

int Reconstruction::insertTriangle(int p1_idx, int p2_idx, int p3_idx, int type) {
    std::vector<std::pair<cv::Point2f, int> > v;
    v.emplace_back(corners[p1_idx], p1_idx);
    v.emplace_back(corners[p2_idx], p2_idx);
    v.emplace_back(corners[p3_idx], p3_idx);
    // std::cout  << "before-sort:" << v[0].first << " " << v[1].first << " " << v[2].first << std::endl;

    // ラスタスキャン順でソート
    sort(v.begin(), v.end(), [](const std::pair<cv::Point2f, int> &a1, const std::pair<cv::Point2f, int> &a2) {
        if (a1.first.y != a2.first.y) {
            return a1.first.y < a2.first.y;
        } else {
            return a1.first.x < a2.first.x;
        }
    });

    Triangle triangle(v[0].second, v[1].second, v[2].second);
    // std::cout  << "after-sort:" <<v[0].first << " " << v[1].first << " " << v[2].first << std::endl;

    triangles.emplace_back(triangle, type);

    return static_cast<int>(triangles.size() - 1);
}

int Reconstruction::insertTriangle(std::vector<std::pair<Triangle, int> >& target_triangles, int p1_idx, int p2_idx, int p3_idx, int type) {
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

    Triangle triangle(v[0].second, v[1].second, v[2].second);

    target_triangles.emplace_back(triangle, type);

    return static_cast<int>(triangles.size() - 1);
}


void Reconstruction::reconstructionTriangle(std::vector<CodingTreeUnit*> ctu) {
    for(int i = 0 ; i < ctu.size() ; i++) {
        CodingTreeUnit* cu = ctu[i];
        int type = (i % 2 == 0 ? DIVIDE::TYPE1 : DIVIDE::TYPE2);
        Triangle t = init_triangles[i].first;
        reconstructionTriangle(cu, Point3Vec(corners[t.p1_idx], corners[t.p2_idx], corners[t.p3_idx]),type);
    }
}

void Reconstruction::reconstructionTriangle(CodingTreeUnit* ctu, Point3Vec triangle, int type){

    if(ctu->leftNode == nullptr && ctu->rightNode == nullptr) {
        int p1_idx, p2_idx, p3_idx;
        if(isAdditionalPoint(triangle.p1)) {
            p1_idx = addCorner(triangle.p1) - 1;
        }else {
            p1_idx = corner_flag[(int)(triangle.p1.y * 2)][(int)(triangle.p1.x * 2)];
        }

        if(isAdditionalPoint(triangle.p2)) {
            p2_idx = addCorner(triangle.p2) - 1;
        }else {
            p2_idx = corner_flag[(int)(triangle.p2.y * 2)][(int)(triangle.p2.x * 2)];
        }

        if(isAdditionalPoint(triangle.p3)) {
            p3_idx = addCorner(triangle.p3) - 1;
        }else {
            p3_idx = corner_flag[(int)(triangle.p3.y * 2)][(int)(triangle.p3.x * 2)];
        }

        insertTriangle(p1_idx, p2_idx, p3_idx, type);

        return;
    }

    std::vector<std::pair<cv::Point2f, int> > ret = sortTriangle(std::make_pair(triangle.p1, corner_flag[(int)(triangle.p1.y * 2)][(int)(triangle.p1.x * 2)]), std::make_pair(triangle.p2, corner_flag[(int)(triangle.p2.y * 2)][(int)(triangle.p2.x * 2)]), std::make_pair(triangle.p3, corner_flag[(int)(triangle.p3.y * 2)][(int)(triangle.p3.x * 2)]));
    triangle.p1 = ret[0].first;
    triangle.p2 = ret[1].first;
    triangle.p3 = ret[2].first;
    TriangleDivision::SplitResult result = TriangleDivision::getSplitTriangle(triangle.p1, triangle.p2, triangle.p3, type);

    if(ctu->leftNode != nullptr) reconstructionTriangle(ctu->leftNode, result.t1, result.t1_type);
    if(ctu->rightNode != nullptr) reconstructionTriangle(ctu->rightNode, result.t2, result.t2_type);
}

bool Reconstruction::isAdditionalPoint(cv::Point2f p){
    return corner_flag[(int)(p.y * 2)][(int)(p.x * 2)] == -1;
}

Reconstruction::Reconstruction(const cv::Mat &gaussRefImage): gaussRefImage(gaussRefImage){}

void Reconstruction::init(int block_size_x, int block_size_y, int divide_flag) {
    int block_num_x = gaussRefImage.cols / block_size_x;
    int block_num_y = gaussRefImage.rows / block_size_y;

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

    corner_flag.resize(gaussRefImage.rows * 2);
    for(int i = 0 ; i < gaussRefImage.rows * 2; i++) {
        corner_flag[i].resize(gaussRefImage.cols * 2);
    }

    for(int y = 0 ; y < gaussRefImage.rows * 2 ; y++) {
        for(int x = 0 ; x < gaussRefImage.cols * 2 ; x++) {
            corner_flag[y][x] = -1;
        }
    }

    // すべての頂点を入れる
    for(int block_y = 0 ; block_y <= block_num_y ; block_y++) {
        for (int block_x = 0 ; block_x <= block_num_x; block_x++) {
            int nx = block_x * (block_size_x);
            int ny = block_y * (block_size_y);

            if (nx < 0) nx = 0;
            if (gaussRefImage.cols <= nx) nx = gaussRefImage.cols - 1;
            if (ny < 0) ny = 0;
            if (gaussRefImage.rows <= ny) ny = gaussRefImage.rows - 1;
            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);


            if (block_x == block_num_x) continue;

            nx = (block_x + 1) * (block_size_x) - 1;
            ny = (block_y) * (block_size_y);

            if (nx < 0) nx = 0;
            if (gaussRefImage.cols <= nx) nx = gaussRefImage.cols - 1;
            if (ny < 0) ny = 0;
            if (gaussRefImage.rows <= ny) ny = gaussRefImage.rows - 1;
            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);
        }

        if(block_y == block_num_y) continue;

        for (int block_x = 0 ; block_x <= block_num_x; block_x++) {
            int nx = block_x * (block_size_x);
            int ny = (block_y + 1) * (block_size_y) - 1;

            if(nx < 0) nx = 0;
            if(gaussRefImage.cols <= nx) nx = gaussRefImage.cols - 1;
            if(ny < 0) ny = 0;
            if(gaussRefImage.rows <= ny) ny = gaussRefImage.rows - 1;
            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);

            if(block_x == block_num_x) continue;

            nx = (block_x + 1) * (block_size_x) - 1;
            ny = (block_y + 1) * (block_size_y) - 1;

            if(nx < 0) nx = 0;
            if(gaussRefImage.cols <= nx) nx = gaussRefImage.cols - 1;
            if(ny < 0) ny = 0;
            if(gaussRefImage.rows <= ny) ny = gaussRefImage.rows - 1;
            corners.emplace_back(nx, ny);
            corner_flag[ny * 2][nx * 2] = static_cast<int>(corners.size() - 1);

        }

    }
    for(int block_y = 0 ; block_y < block_num_y ; block_y++) {
        for(int block_x = 0 ; block_x < block_num_x ; block_x++) {
            int p1_idx;
            int p2_idx;
            int p3_idx;
            int p4_idx;
            if(divide_flag == LEFT_DIVIDE) {
                p1_idx = 2 * block_x + (2 * block_y) * ((block_num_x) * 2 + 1);
                p2_idx = p1_idx + 1;
                p3_idx = p1_idx + ((block_num_x) * 2 + 1 );

                int triangleIndex = insertTriangle(init_triangles, p1_idx, p2_idx, p3_idx, TYPE1);

                int p4_idx = p2_idx;
                int p5_idx = p3_idx;
                int p6_idx = p3_idx + 1;

                triangleIndex = insertTriangle(init_triangles, p4_idx, p5_idx, p6_idx, TYPE2);

            }else{
                int triangleIndex = insertTriangle(init_triangles, p1_idx, p2_idx, p4_idx, TYPE1);
                triangleIndex = insertTriangle(init_triangles, p1_idx, p3_idx, p4_idx, TYPE2);
            }
        }
    }

}

/**
 * @fn int Reconstruction::addCorner()
 * @param[in] p 追加する頂点の座標
 * @return 頂点番号を返す
 */
int Reconstruction::addCorner(cv::Point2f p) {
    corners.emplace_back(p);
    corner_flag[(int)(p.y * 2)][(int)(p.x * 2)] = corners.size() - 1;
    return corners.size();
}

std::vector<Point3Vec> Reconstruction::getTriangleCoordinateList() {
    std::vector<Point3Vec> v;

    for(const auto t : triangles) {
        v.emplace_back(corners[t.first.p1_idx], corners[t.first.p2_idx], corners[t.first.p3_idx]);
    }

    return v;
}

std::vector<std::pair<cv::Point2f, int>> Reconstruction::sortTriangle(std::pair<cv::Point2f, int> a, std::pair<cv::Point2f, int> b, std::pair<cv::Point2f, int> c) {
    std::vector<std::pair<cv::Point2f, int> > v;
    v.emplace_back(a);
    v.emplace_back(b);
    v.emplace_back(c);

    // ラスタスキャン順でソート
    sort(v.begin(), v.end(), [](const std::pair<cv::Point2f, int> &a1, const std::pair<cv::Point2f, int> &a2) {
        if (a1.first.y != a2.first.y) {
            return a1.first.y < a2.first.y;
        } else {
            return a1.first.x < a2.first.x;
        }
    });

    return v;
}