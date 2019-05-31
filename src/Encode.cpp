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
 * @fn int getBinaryLength(int x)
 * @brief xのビット長を返す
 * @param x 整数の入力値（非負整数）
 * @return ビット長
 */
int getBinaryLength(int x){
    int cnt = 0;
    while(x > 0){
        cnt++;
        x >>= 1;
    }
    return cnt;
}

/**
<<<<<<< HEAD
 * @fn double getLambdaMode(int qp)
 * @brief CUモード選択に使われるラムダをQPの値に応じて計算し返す
 * @see H.265/HEVC教科書 P234
 * @param qp QP
 * @return ラムダの値
 */
double getLambdaMode(int qp){
    double alpha = 1.0; // P picture
    double wk = 0.578;  // Low-Delay P
    return alpha * wk * std::pow(2, (qp - 12)/3.0);
}

/**
 * @fn double getLambdaPred(int qp)
 * @brief 予測モードのRDで使うQPに応じたラムダの値を返す
 * @see H.265/HEVC教科書 P234
 * @param qp 参照画像のQP
 * @return lambdaの値
 */
double getLambdaPred(int qp){
    return sqrt(getLambdaMode(qp));
}

/**
=======
>>>>>>> test/adaptive
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

/**
 * @fn int getExponentialGolombCodeLength(int data, int k)
 * @brief dataをk次指数ゴロム符号で符号化し，符号長を返す
 * @see https://wikivisually.com/wiki/Exponential-Golomb_coding
 * @see https://en.wikipedia.org/wiki/Elias_gamma_coding
 * @param data 入力データ
 * @param k 符号化パラメタ（非負整数）
 * @return 符号長
 */
int getExponentialGolombCodeLength(int data, int k){
    int p_length = 0;
    int q_length = 0;

    data = mapping(data);
    if(k > 0) {
        int bits = static_cast<int>(std::floor(data / std::pow(2, k)));
        p_length += getBinaryLength(bits + 1) * 2 - 1; // Elias Gamma Coding
        int mod = (data % (int)std::pow(2, k));
        if (mod == 0) {
            p_length += 1;
        } else {
            p_length += getBinaryLength(mod);
        }
    }else{
        int bits = getBinaryLength(data + 1);
        p_length += bits * 2 - 1;
    }

    return p_length + q_length;
}