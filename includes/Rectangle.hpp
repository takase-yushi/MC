//
// Created by keisuke on 2017/12/06.
//

#ifndef ENCODER_RECTANGLE_HPP
#define ENCODER_RECTANGLE_HPP


class Rectangle {
public:
    Rectangle();
    Rectangle(int _x, int _y, int _width, int _height);

    int x;      // 四角形の左上のx座標
    int y;      // 四角形の左上のy座標
    int width;  // 四角形の横幅
    int height; // 四角形の縦幅

private:

    // 四角形の縦幅
};


#endif //ENCODER_RECTANGLE_HPP
