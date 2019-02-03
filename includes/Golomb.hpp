//
// Created by Kamiya Keisuke on 2018/02/01.
//

#ifndef ENCODER_GOLOMB_H
#define ENCODER_GOLOMB_H

#include <vector>

namespace ozi {

  enum REGION {
    REGION1,
    REGION2,
    REGION3,
    REGION4,
  };

  enum GOLOMB_TYPE {
    GOLOMB,
    KTH_GOLOMB,
  };

  double P(double theta, double d, double x);
  double C(double theta, double d);
  int mapping(int x);
  int chi(int x);
  int getGolombCode(int l, int x, int region, int type, int k = 1);
  int getUnaryCode(int x);
  int getUnaryCode(const std::vector<int>& code);
  int getGolombParam(double p);
  bool isPowerOfTwo(int x);
  int getBinaryLength(int x);

}


#endif //ENCODER_GOLOMB_H
