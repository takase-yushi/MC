//
// Created by keisuke on 2017/12/23.
//

#ifndef ENCODER_POINT_HPP
#define ENCODER_POINT_HPP

#include "Vector.hpp"
#include "saturate.hpp"

namespace ozi {

    template<typename T> class Point_ {
    public:
        Point_();

        Point_(T _x, T _y);

        T x, y;

        template<typename T>
        Point_<T>::Point_() : x(T(0)), y(T(0)) {}

        template<typename T>
        Point_<T>::Point_(T _x, T _y) : x(_x), y(_y) {}
    };

    template<typename T>
    static inline
    Point_<T> operator+(const Point_<T> &a, const Point_<T> &b) {
      return Point_(saturate_cast(a.x + b.x), saturate_cast(a.y + b.y));
    }

    template<typename T>
    static inline
    Point_<T> operator-(const Point_<T> &a, const Point_<T> &b) {
      return Point_(saturate_cast(a.x - b.x), saturate_cast(a.y - b.y));
    }

    template<typename T>
    static inline
    Point_<T> operator*(const Point_<T> &a, const int n) {
      return Point_(saturate_cast(a.x * n), saturate_cast(a.y * n));
    }

    template<typename T>
    static inline
    Point_<T> operator*(const int n, const Point_<T> &a) {
      return Point_(saturate_cast(a.x * n), saturate_cast(a.y * n));
    }

    template<typename T>
    static inline
    Point_<T> operator*(const Point_<T> &a, const float n) {
      return Point_(saturate_cast(a.x * n), saturate_cast(a.y * n));
    }

    template<typename T>
    static inline
    Point_<T> operator*(const float n, const Point_<T> &a) {
      return Point_(saturate_cast(a.x * n), saturate_cast(a.y * n));    }

    template<typename T>
    static inline
    Point_<T> operator*(const Point_<T> &a, const double n) {
      return Point_(saturate_cast(a.x * n), saturate_cast(a.y * n));
    }

    template<typename T>
    static inline
    Point_<T> operator*(const double n, const Point_<T> &a) {
      return Point_(saturate_cast(a.x * n), saturate_cast(a.y * n));
    }

    template<typename T>
    static inline
    Point_<T> operator/(const Point_<T> &a, const int n) {
      return Point_(saturate_cast(a.x / n), saturate_cast(a.y/n));
    }

    template<typename T>
    static inline
    Point_<T> operator/(const Point_<T> &a, const float n) {
      return Point_(saturate_cast(a.x / n), saturate_cast(a.y/n));
    }


    template<typename T>
    static inline
    Point_<T> operator/(const Point_<T> &a, const double n) {
      return Point_(saturate_cast(a.x / n), saturate_cast(a.y/n));
    }


    template<typename T>
    static inline
    bool operator!=(const Point_<T> &a, const Point_<T> &b) const {
      return !(a == b);
    }

    template<typename T>
    static inline
    bool operator==(const Point_<T> &a, const Point_<T> &b) const {
      return a.x == b.x && a.y == b.y;
    }

    typedef Point_<int> Point2i;
    typedef Point_<float> Point2f;
    typedef Point_<double> Point2d;
}
#endif //ENCODER_POINT_HPP
