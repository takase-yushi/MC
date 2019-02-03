//
// Created by kamiya on 2018/01/29.
//

#include "../includes/GnuplotFileGenerator.hpp"

ozi::GnuplotFileGenerator::GnuplotFileGenerator(const std::string& _out_file_path) {
  os = std::ofstream(_out_file_path);
  out_file_path = _out_file_path;
}

void ozi::GnuplotFileGenerator::addXRange(int start, int end) {
  os << "set xrange [" << start << ":" << end << "]" << std::endl;
}

void ozi::GnuplotFileGenerator::addYRange(int start, int end) {
  os << "set yrange [" << start << ":" << end << "]" << std::endl;
}

void ozi::GnuplotFileGenerator::clearGraph() {
  os << "clear" << std::endl;
}

void ozi::GnuplotFileGenerator::unsetKey() {
  os << "unset key" << std::endl;
}

void
ozi::GnuplotFileGenerator::plotData(const std::string &file_path, const std::string &color, const GRAPH_TYPE type) {
  os << "plot '" << file_path << "' " << getGraphTypeString(type)  << " linecolor rgb \""<< color << "\"" << std::endl;
}

void
ozi::GnuplotFileGenerator::replotData(const std::string &file_path, const std::string &color, GRAPH_TYPE type) {
  os << "replot '" << file_path << "' " << getGraphTypeString(type)  << " linecolor rgb \""<< color << "\"" << std::endl;
}

std::string ozi::GnuplotFileGenerator::getGraphTypeString(int graph_type) {
  switch(graph_type){
    case LINES:
      return "with lines";
    case POINTS:
      return "with points";
    case LINEPOINTS:
      return "with linepoints";
    case IMPULSES:
      return  "with impulses";
    case BOXES:
      return "with boxes";
    case STEPS:
      return "with steps";
    default:
      break;
  }
  return "";
}

void ozi::GnuplotFileGenerator::setXLabel(const std::string &label_name) {
  os << "set xlabel '" << label_name << "'" << std::endl;
}

void ozi::GnuplotFileGenerator::setYLabel(const std::string &label_name) {
  os << "set ylabel '" << label_name << "'" << std::endl;
}

void ozi::GnuplotFileGenerator::close() {
  os.close();
}

