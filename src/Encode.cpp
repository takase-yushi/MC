//
// Created by Kamiya Keisuke on 2019/05/24.
//

#include <cassert>
#include <cmath>
#include <iostream>
#include "../includes/Encode.h"

/**
 * @fn int mapping(int x)
 * @brief すべての値を正の値にマッピングする
 * @details 負の値も含めた順序付けをおこなう．順番としては，0→-1→1→-2→2→…となる
 * @param data 入力データ
 * @return マッピングした値
 */
int mapping(int data){
    return data >= 0 ? 2 * data : 2 * std::abs(data) - 1;
}

/**
 * @fn int getUnaryCodeLength(int data)
 * @brief ユーナリー符号の符号長を返す
 * @param data 入力データ
 * @return 符号長
 */
int getUnaryCodeLength(int data){
    return data + 1;
}

/**
 * @fn bool isPowerOfTwo(int data)
 * @brief 入力データが2のべき乗であるかどうか判定する
 * @see https://www.madopro.net/entry/2016/09/13/011136
 * @param data 判定したいデータ
 * @return 2のべき乗であればtrue, それ以外はfalseが返る
 */
bool isPowerOfTwo(int data){
    int msb = data & (data - 1);
    return msb == 0;
}

/**
 * @fn int getGolombCodeLength(int data, int m)
 * @brief dataをパラメタmでゴロム符号化にした際の符号長を返す
 * @return ゴロム付号の符号長
 */
int getGolombCodeLength(int data, int m){
    assert(m > 0);

    // 負の値を正の値にマッピングする
    data = mapping(data);

    int p = data / m;
    int p_length = getUnaryCodeLength(p);

    if(m == 1) return p_length;

    int q = data % m;
    int q_length;

    if (isPowerOfTwo(m)) {
        q_length = static_cast<int>(log2(m));
    } else {
        double b = ceil(log2(m));
        if (q < std::pow(2, b) - m) {
            q_length = static_cast<int>(b-1);
        } else {
            q_length = static_cast<int>(b);
        }
    }

    std::cout << "input:" << data << " length:" << p_length + q_length << std::endl;
    return p_length + q_length;
}

