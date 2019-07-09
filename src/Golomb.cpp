//
// Created by Kamiya Keisuke on 2018/02/01.
//

#include <cmath>
#include <iostream>
#include "../includes/Golomb.hpp"


double ozi::C(double theta, double d) {
  return (1.0 - theta) / (std::pow(theta, 1 - d) + std::pow(theta, d));
}

double ozi::P(double theta, double d, double x) {
  return C(theta, d) * std::pow(theta, std::fabs(x + d));
}

int ozi::mapping(int x) {
  return x >= 0 ? 2 * x : 2 * std::abs(x) - 1;
}

int ozi::chi(int x) {
  return std::abs(x);
}

// パラメータmのゴロム符号の符号長だけとりあえず返す
int ozi::getGolombCode(int l, int x, int region, int type, int k) {
  int q_length = 0, p_length = 0;

  if(region == REGION1){
    x = mapping(x);
  }else if(region == REGION2){
    x = chi(x);
    if(x != 0) p_length = 1;
  }

  if(type == ozi::GOLOMB) {
    int p = x / l;
    p_length += getUnaryCode(p);
    int q = x % l;

    if (l > 1) { // m>1のときはこっち
      if (isPowerOfTwo(l)) {
        q_length = static_cast<int>(log2(l));
      } else {
        double b = ceil(log2(l));
        if (q < std::pow(2, b) - l) {
          q_length += static_cast<int>(b-1);
        } else {
          q_length += static_cast<int>(b);
        }
      }
    } else { // m=1のときは普通にUnaryなのでqは0です
      q_length = 0;
    }
  }else if(type == KTH_GOLOMB){
    if(k > 0) {
      int bits = static_cast<int>(std::floor(x / std::pow(2, k)));
      p_length += ozi::getBinaryLength(bits + 1) * 2 - 1;
      int mod = (x % (int)std::pow(2, k));
      if (mod == 0) {
        p_length += 1;
      } else {
        p_length += ozi::getBinaryLength(mod);
      }
    }else{
      int bits = ozi::getBinaryLength(x + 1);
      p_length += bits * 2 - 1;
    }
  }
  return p_length + q_length;
}

int ozi::getUnaryCode(int x) {
  return x + 1;
}

bool ozi::isPowerOfTwo(int x) {
  if(x == 0) return false;

  int msb = x & (x - 1);
  return msb == 0;
}

int ozi::getGolombParam(double p) {
  int m = static_cast<int>(std::ceil(-std::log2(1 + p) / std::log2(p)));

  if((m & (m-1)) == 0) return m;

  int ret = 1;
  while(m > 0){
    ret<<=1;
    m>>=1;
  }

  return ret;
}

int ozi::getUnaryCode(const std::vector<int> &code) {

  return 0;
}

int ozi::getBinaryLength(int x){

  int cnt = 0;
  while(x > 0){
    cnt++;
    x >>= 1;
  }
  return cnt;
}


