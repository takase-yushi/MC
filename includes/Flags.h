//
// Created by kasph on 2019/08/08.
//

#ifndef ENCODER_FLAGS_H
#define ENCODER_FLAGS_H

#include <vector>

class Flags{
public:
    std::vector<bool> x_greater_0_flag, y_greater_0_flag;
    std::vector<bool> x_greater_1_flag, y_greater_1_flag;
    std::vector<bool> x_sign_flag, y_sign_flag;
};
#endif //ENCODER_FLAGS_H
