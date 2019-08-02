//
// Created by Kamiya Keisuke on 2019-07-25.
//

#ifndef ENCODER_CONFIG_H
#define ENCODER_CONFIG_H


#include <string>

class Config {


private:
    std::string img_directory;
    std::string gauss_ref_image;
    std::string ref_image;
    std::string target_image;
    int qp;
    int ctu_width;
    int ctu_height;
    int division_step;
    bool enable;
public:
    bool isEnable() const;

public:
    const std::string &getImgDirectory() const;

    const std::string &getGaussRefImage() const;

    const std::string &getRefImage() const;

    const std::string &getTargetImage() const;

    int getQp() const;

    int getCtuWidth() const;

    int getCtuHeight() const;

    int getDivisionStep() const;

public:
    Config(bool enableFlag, const std::string &imgDirectory, const std::string &gaussRefImage, const std::string &refImage,
           const std::string &targetImage, int qp, int ctuWidth, int ctuHeight, int divisionStep);
};


#endif //ENCODER_CONFIG_H
