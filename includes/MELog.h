//
// Created by kasph on 2019/11/28.
//

#ifndef ENCODER_MELOG_H
#define ENCODER_MELOG_H

#include <vector>
#include <opencv2/core/types.hpp>

class MELog {
public:


    double percentage;

    std::vector<double> residual;
    std::vector<cv::Point2f> mv_newton_translation;
    std::vector<std::vector<cv::Point2f>> mv_newton_warping;

    // ここでのtranslationは動いたあとって意味で使ってますよ
    std::vector<cv::Point2f> coordinate_after_move1;
    std::vector<cv::Point2f> coordinate_after_move2;
    std::vector<cv::Point2f> coordinate_after_move3;
};


#endif //ENCODER_MELOG_H
