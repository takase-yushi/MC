//
// Created by Kamiya Keisuke on 2019/05/24.
//

#ifndef ENCODER_ENCODE_H
#define ENCODER_ENCODE_H
//
// Created by Kamiya Keisuke on 2019/05/24.
//

#include <cassert>
#include <cmath>
#include "../includes/Encode.h"

int mapping(int data);
int getUnaryCodeLength(int data);
bool isPowerOfTwo(int data);
int getGolombCodeLength(int data, int m);

#endif //ENCODER_ENCODE_H
