//
// Created by keisuke on 2017/12/19.
//

#ifndef ENCODER_VECTOR_HPP
#define ENCODER_VECTOR_HPP

#include <array>

namespace ozi{

    // こいつを使っていい感じにコンストラクタのほうをね, やっていき
    struct AddOp{};
    struct SubOp{};
    struct MulOp{};
    struct DivOp{};

    template <typename T, int _row, int _col> class Mat_ {
    public:
        // Default Constructor
        Mat_();

        // Constructor
        explicit Mat_(T v1);
        Mat_(T v1, T v2);
        Mat_(T v1, T v2, T v3);
        Mat_(T v1, T v2, T v3, T v4);
        Mat_(T v1, T v2, T v3, T v4, T v5);
        Mat_(T v1, T v2, T v3, T v4, T v5, T v6);

        std::array<T, _row*_col> val;
        int channels = _row * _col;

        Mat_(const Mat_<T, _row, _col>& a, const Mat_<T, _row, _col>& b, AddOp);
        Mat_(const Mat_<T, _row, _col>& a, const Mat_<T, _row, _col>& b, SubOp);
        Mat_(const Mat_<T, _row, _col>& a, const Mat_<T, _row, _col>& b, DivOp);
        Mat_(const Mat_<T, _row, _col>& a, const Mat_<T, _row, _col>& b, MulOp);
    };

    template<typename T, int _row, int _col>
    Mat_<T, _row, _col>::Mat_() {
      for(auto& elem : val) elem = T(0);
    }

    template<typename T, int _row, int _col>
    Mat_<T, _row, _col>::Mat_(T v1) {
      val[0] = v1;
      for(int i = 1 ; i < channels ; i++) val[i] = T(0);
    }

    template<typename T, int _row, int _col>
    Mat_<T, _row, _col>::Mat_(T v1, T v2) {
      val[0] = v1; val[1] = v2;
      for(int i = 2;  i < channels ; i++) val[i] = T(0);
    }

    template<typename T, int _row, int _col>
    Mat_<T, _row, _col>::Mat_(T v1, T v2, T v3) {
      val[0] = v1; val[1] = v2; val[2] = v3;
      for(int i = 3 ; i < channels ; i++) val[i] = T(0);
    }

    template<typename T, int _row, int _col>
    Mat_<T, _row, _col>::Mat_(T v1, T v2, T v3, T v4) {
      val[0] = v1; val[1] = v2; val[2] = v3; val[3] = v4;
      for(int i = 4 ; i < channels ; i++) val[i] = T(0);
    }

    template<typename T, int _row, int _col>
    Mat_<T, _row, _col>::Mat_(T v1, T v2, T v3, T v4, T v5) {
      val[0] = v1; val[1] = v2; val[2] = v3;
      val[3] = v4; val[4] = v5;
      for(int i = 5 ; i < channels ; i++) val[i] = T(0);
    }

    template<typename T, int _row, int _col>
    Mat_<T, _row, _col>::Mat_(T v1, T v2, T v3, T v4, T v5, T v6) {
      val[0] = v1; val[1] = v2; val[2] = v3;
      val[3] = v4; val[4] = v5; val[5] = v6;
      for(int i = 6 ; i < channels ; i++) val[i] = T(0);
    }

    template<typename T, int _row, int _col>
    Mat_<T, _row, _col>::Mat_(const Mat_<T, _row, _col>& a, const Mat_<T, _row, _col>& b, AddOp){
      for(int i = 0 ; i < channels ; i++){
        val[i] = a.val[i] + b.val[i];
      }
    };

    template<typename T, int _row, int _col>
    Mat_<T, _row, _col>::Mat_(const Mat_<T, _row, _col>& a, const Mat_<T, _row, _col>& b, SubOp){
      for(int i = 0 ; i < channels ; i++){
        val[i] = a.val[i] - b.val[i];
      }
    };

    template<typename T, int _row, int _col>
    Mat_<T, _row, _col>::Mat_(const Mat_<T, _row, _col>& a, const Mat_<T, _row, _col>& b, MulOp){
      for(int i = 0 ; i < channels ; i++){
        val[i] = a.val[i] * b.val[i];
      }
    };

    template<typename T, int _row, int _col>
    Mat_<T, _row, _col>::Mat_(const Mat_<T, _row, _col>& a, const Mat_<T, _row, _col>& b, DivOp){
      for(int i = 0 ; i < channels ; i++){
        val[i] = a.val[i] / b.val[i];
      }
    };

/**
 * 等しいか判定
 * @tparam T
 * @tparam _row
 * @tparam _col
 * @param a
 * @param b
 * @return
 */
    template<typename T, int _row, int _col> static inline
    bool operator ==(const Mat_<T, _row, _col>& a, const Mat_<T, _row, _col>& b){
      for(int i = 0 ; i < _row * _col ; i++){
        if(a.val[i] != b.val[i]) return false;
      }
      return true;
    }

/**
 * 等しくないか判定
 * @tparam T
 * @tparam _row
 * @tparam _col
 * @param a
 * @param b
 * @return
 */
    template<typename T, int _row, int _col> static inline
    bool operator !=(const Mat_<T, _row, _col>& a, const Mat_<T, _row, _col>& b){
      return !(a == b);
    }


// =====================================================================================================
// Vecの実装 ============================================================================================
// =====================================================================================================

    template <typename T, int n> class Vector : public Mat_ <T, n, 1> {
    public:
        Vector();
        explicit Vector(T v1);
        Vector(T v1, T v2);
        Vector(T v1, T v2, T v3);
        Vector(T v1, T v2, T v3, T v4);
        Vector(T v1, T v2, T v3, T v4, T v5);
        Vector(T v1, T v2, T v3, T v4, T v5, T v6);

        Vector(const Vector<T, n>& a, const Vector<T, n>& b, AddOp);
        Vector(const Vector<T, n>& a, const Vector<T, n>& b, SubOp);
    };

// 委譲コンストラクタでえいっ
    template<typename T, int n>
    Vector<T, n>::Vector() {}

    template<typename T, int n>
    Vector<T, n>::Vector(T v1):Mat_<T, n, 1>(v1) {}

    template<typename T, int n>
    Vector<T, n>::Vector(T v1, T v2):Mat_<T, n, 1>(v1, v2) {}

    template<typename T, int n>
    Vector<T, n>::Vector(T v1, T v2, T v3):Mat_<T, n, 1>(v1, v2, v3) {}

    template<typename T, int n>
    Vector<T, n>::Vector(T v1, T v2, T v3, T v4):Mat_<T, n, 1>(v1, v2, v3, v4) {}

    template<typename T, int n>
    Vector<T, n>::Vector(T v1, T v2, T v3, T v4, T v5):Mat_<T, n, 1>(v1, v2, v3, v4, v5) {}

    template<typename T, int n>
    Vector<T, n>::Vector(T v1, T v2, T v3, T v4, T v5, T v6):Mat_<T, n, 1>(v1, v2, v3, v4, v5, v6) {}

    template<typename T, int n>
    Vector<T, n>::Vector(const Vector<T, n>& a, const Vector<T, n>& b, AddOp):Mat_<T, n, 1>(a, b, AddOp()){}

    template<typename T, int n>
    Vector<T, n>::Vector(const Vector<T, n>& a, const Vector<T, n>& b, SubOp):Mat_<T, n, 1>(a, b, SubOp()){}

    template<typename T, int n>
    Vector<T,n> operator +(const Vector<T, n>& a, const Vector<T, n>& b){
      return Vector<T, n>(a, b, AddOp());
    }

    template<typename T, int n>
    Vector<T,n> operator -(const Vector<T, n>& a, const Vector<T, n>& b){
      return Vector<T, n>(a, b, SubOp());
    }

    typedef Vector<unsigned char, 2> Vec2uc;
    typedef Vector<unsigned char, 3> Vec3uc;
    typedef Vector<unsigned char, 4> Vec4uc;
    typedef Vector<unsigned char, 6> Vec6uc;
    typedef Vector<unsigned char, 8> Vec8uc;

    typedef Vector<int, 2> Vec2i;
    typedef Vector<int, 3> Vec3i;
    typedef Vector<int, 4> Vec4i;
    typedef Vector<int, 6> Vec6i;
    typedef Vector<int, 8> Vec8i;

    typedef Vector<float, 2> Vec2f;
    typedef Vector<float, 3> Vec3f;
    typedef Vector<float, 4> Vec4f;
    typedef Vector<float, 6> Vec6f;
    typedef Vector<float, 8> Vec8f;

    typedef Vector<double, 2> Vec2d;
    typedef Vector<double, 3> Vec3d;
    typedef Vector<double, 4> Vec4d;
    typedef Vector<double, 6> Vec6d;
    typedef Vector<double, 8> Vec8d;
}


#endif //ENCODER_VECTOR_HPP
