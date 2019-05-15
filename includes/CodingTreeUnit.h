//
// Created by kasph on 2019/05/05.
//

#ifndef ENCODER_CODINGTREEUNIT_H
#define ENCODER_CODINGTREEUNIT_H


class CodingTreeUnit {
public:
    virtual ~CodingTreeUnit();

    bool split_cu_flag1;
    bool split_cu_flag2;
    CodingTreeUnit *leftNode;
    CodingTreeUnit *rightNode;
    CodingTreeUnit *parentNode;
    int depth;
    int position; // left or right
};


#endif //ENCODER_CODINGTREEUNIT_H
