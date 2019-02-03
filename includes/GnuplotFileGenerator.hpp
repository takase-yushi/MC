//
// Created by kamiya on 2018/01/29.
//

#ifndef ENCODER_GNUPLOTFILEGENERATOR_H
#define ENCODER_GNUPLOTFILEGENERATOR_H


#include <fstream>

namespace ozi {

  enum GRAPH_TYPE {
    LINES,
    POINTS,
    LINEPOINTS,
    IMPULSES,
    BOXES,
    STEPS,
  };

  class GnuplotFileGenerator {

  public:
    explicit GnuplotFileGenerator(const std::string &_out_file_path);

    void addXRange(int start, int end);

    void addYRange(int start, int end);

    void clearGraph();

    void unsetKey();

    void plotData(const std::string &file_path, const std::string &color, GRAPH_TYPE type);

    void replotData(const std::string &file_path, const std::string &color, GRAPH_TYPE type);

    void setXLabel(const std::string& label_name);

    void setYLabel(const std::string& label_name);

    void close();

  private:
    std::ofstream os;
    std::string out_file_path;

    std::string getGraphTypeString(int graph_type);
  };
}

#endif //ENCODER_GNUPLOTFILEGENERATOR_H
