//
// Created by kasph on 2019/04/08.
//

#ifndef ENCODER_TRIANGLEDIVISION_H
#define ENCODER_TRIANGLEDIVISION_H

#include "../includes/Utils.h"

#define LEFT_DEVIDE 1
#define RIGHT_DEVIDE 2

void insertTriangle(Point3Vec triangle);
void insertTriangle(cv::Point2f, cv::Point2f, cv::Point2f);
void insertTriangle(float, float, float, float, float, float);

#endif //ENCODER_TRIANGLEDIVISION_H
