cmake_minimum_required(VERSION 3.10)
project(includes)

set(CMAKE_CXX_STANDARD 11)

include_directories(.)

add_executable(includes
        config.h
        DelaunayTriangulation.hpp
        GnuplotFileGenerator.hpp
        Golomb.hpp
        ME.hpp
        Point.hpp
        psnr.h
        Rectangle.hpp
        saturate.hpp
        Utils.h
        Vector.hpp)
