//
// Created by keisuke on 2017/12/23.
//

#ifndef ENCODER_SATURATE_HPP
#define ENCODER_SATURATE_HPP


#include <climits>
#include <cmath>

namespace ozi {
    template<typename T>
    static inline T saturate_cast(int num) {
      return static_cast<T>(num);
    }

    template<typename T>
    static inline T saturate_cast(float num) {
      return static_cast<T>(num);
    }

    template<typename T>
    static inline T saturate_cast(double num) {
      return static_cast<T>(num);
    }

    // 完全特殊化テンプレート
    template<>
    inline unsigned char saturate_cast<unsigned char>(int num) {
      return (unsigned char) ((unsigned) num <= UCHAR_MAX ? num : num > 0 ? UCHAR_MAX : 0);
    }

    template<>
    inline unsigned char saturate_cast<unsigned char>(float num) {
      return saturate_cast<unsigned char>((int) std::round(num));
    }

    template<>
    inline unsigned char saturate_cast<unsigned char>(double num) {
      return saturate_cast<unsigned char>((int) std::round(num));
    }

    template<>
    inline signed char saturate_cast<signed char>(int num) {
      return (signed char) ((unsigned) (num - SCHAR_MIN) <= (unsigned) UCHAR_MAX ? num : num > 0 ? SCHAR_MAX
                                                                                                 : SCHAR_MIN);
    }

    template<>
    inline signed char saturate_cast<signed char>(float num) {
      return saturate_cast<signed char>((int) std::round(num));
    }

    template<>
    inline signed char saturate_cast<signed char>(double num) {
      return saturate_cast<signed char>((int) std::round(num));
    }

    template<>
    inline unsigned short saturate_cast<unsigned short>(int num) {
      return (unsigned short) ((unsigned short) num <= USHRT_MAX ? num : num > 0 ? USHRT_MAX : 0);
    }

    template<>
    inline unsigned short saturate_cast<unsigned short>(float num) {
      return saturate_cast<unsigned short>((int) std::round(num));
    }

    template<>
    inline unsigned short saturate_cast<unsigned short>(double num) {
      return saturate_cast<unsigned short>((int) std::round(num));
    }

    template<>
    inline short saturate_cast<short>(int num) {
      return (short) ((unsigned) (num - SHRT_MIN) <= (unsigned) USHRT_MAX ? num : num > 0 ? SHRT_MAX : SHRT_MIN);
    }

    template<>
    inline short saturate_cast<short>(float num) {
      return saturate_cast<short>((int) std::round(num));
    }

    template<>
    inline short saturate_cast<short>(double num) {
      return saturate_cast<short>((int) std::round(num));
    }
}

#endif //ENCODER_SATURATE_HPP
