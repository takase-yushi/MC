//
// Created by Kamiya Keisuke on 2019-07-25.
//

#ifndef ENCODER_CONFIG_H
#define ENCODER_CONFIG_H


#include <string>

class Config {

private:
  bool enable;
  std::string img_directory;
  std::string log_directory;
public:
  Config(bool enable, const std::string &imgDirectory, const std::string &logDirectory,
         const std::string &gaussRefImage, const std::string &refImage, const std::string &targetImage, int qp,
         int ctuWidth, int ctuHeight, int divisionStep, bool lambdaEnable, double lambda);

private:

  std::string gauss_ref_image;
  std::string ref_image;
  std::string target_image;
  int qp;
  int ctu_width;
  int ctu_height;
  int division_step;
  bool lambda_enable;
  double lambda;

public:
  bool isLambdaEnable() const;

  double getLambda() const;

public:
    const std::string &getImgDirectory() const;

    const std::string &getGaussRefImage() const;

    const std::string &getRefImage() const;

    const std::string &getTargetImage() const;

    int getQp() const;

    int getCtuWidth() const;

    int getCtuHeight() const;

    int getDivisionStep() const;

    bool isEnable() const;

    const std::string &getLogDirectory() const;

};


#endif //ENCODER_CONFIG_H
