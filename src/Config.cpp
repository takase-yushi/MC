//
// Created by Kamiya Keisuke on 2019-07-25.
//

#include "../includes/Config.h"

Config::Config(const std::string &imgDirectory, const std::string &gaussRefImage, const std::string &refImage,
               const std::string &targetImage, int qp, int ctuWidth, int ctuHeight, int divisionStep) : img_directory(
        imgDirectory), gauss_ref_image(gaussRefImage), ref_image(refImage), target_image(targetImage), qp(qp),
        ctu_width(ctuWidth), ctu_height(ctuHeight), division_step(divisionStep) {}

void Config::setImgDirectory(const std::string &imgDirectory) {
    img_directory = imgDirectory;
}

void Config::setGaussRefImage(const std::string &gaussRefImage) {
    gauss_ref_image = gaussRefImage;
}

void Config::setRefImage(const std::string &refImage) {
    ref_image = refImage;
}

void Config::setTargetImage(const std::string &targetImage) {
    target_image = targetImage;
}

void Config::setQp(int qp) {
    Config::qp = qp;
}

void Config::setCtuWidth(int ctuWidth) {
    ctu_width = ctuWidth;
}

void Config::setCtuHeight(int ctuHeight) {
    ctu_height = ctuHeight;
}

void Config::setDivisionStep(int divisionStep) {
    division_step = divisionStep;
}

const std::string &Config::getImgDirectory() const {
    return img_directory;
}

const std::string &Config::getGaussRefImage() const {
    return gauss_ref_image;
}

const std::string &Config::getRefImage() const {
    return ref_image;
}

const std::string &Config::getTargetImage() const {
    return target_image;
}

int Config::getQp() const {
    return qp;
}

int Config::getCtuWidth() const {
    return ctu_width;
}

int Config::getCtuHeight() const {
    return ctu_height;
}

int Config::getDivisionStep() const {
    return division_step;
}
