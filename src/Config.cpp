//
// Created by Kamiya Keisuke on 2019-07-25.
//

#include "../includes/Config.h"



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

bool Config::isEnable() const {
    return enable;
}

const std::string &Config::getLogDirectory() const {
    return log_directory;
}

bool Config::isLambdaEnable() const {
    return lambda_enable;
}

double Config::getLambda() const {
    return lambda;
}

Config::Config(bool enable, const std::string &imgDirectory, const std::string &logDirectory,
               const std::string &gaussRefImage, const std::string &refImage, const std::string &targetImage, int qp, int qpOffset,
               int ctuWidth, int ctuHeight, int divisionStep, bool lambdaEnable, double lambda) : enable(enable),
                                                                                                  img_directory(
                                                                                                          imgDirectory),
                                                                                                  log_directory(
                                                                                                          logDirectory),
                                                                                                  gauss_ref_image(
                                                                                                          gaussRefImage),
                                                                                                  ref_image(refImage),
                                                                                                  target_image(
                                                                                                          targetImage),
                                                                                                  qp(qp),
                                                                                                  qp_offset(qpOffset),
                                                                                                  ctu_width(ctuWidth),
                                                                                                  ctu_height(ctuHeight),
                                                                                                  division_step(
                                                                                                          divisionStep),
                                                                                                  lambda_enable(
                                                                                                          lambdaEnable),
                                                                                                  lambda(lambda) {}

int Config::getQpOffset() const {
    return qp_offset;
}
