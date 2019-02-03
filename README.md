# Encoder
My research for motion compensation

## environment
- Windows7 Professional 64bit
- MinGW 5.0
- OpenCV 3.2.0
- CLion
- Cmake3.8.0
- OpenMP

## File details（ここは追記します）
- main.cp  
execute motion compensation(MC)
```cpp
void addSideCorners(Mat img, vector<Point2f> &corners) 

void cornersQuantization(vector<Point2f> &corners) 

pair<double, double> getEntropyOfRunlength(Mat img, vector<Point2f> &corners, int maximum_range)
```
- ME.cpp  
execute Block matching and warping
```cpp
void block_matching(cv::Mat &prev, cv::Mat &current, double &error, cv::Point2f &mv, Point3Vec tr);  

std::vector\<cv::Point2f> warping(cv::Mat &prev_gray, cv::Mat &current_gray, cv::Mat &prev_color, cv::Mat &current_color, double &error_warp,
                              double sx, double sy, double lx, double ly, Point3Vec vec);
```

- psnr.cpp  
calculate PSNR
```cpp
double getMSE(cv::Mat in1, cv::Mat in2);

double getPSNR(cv::Mat in1, cv::Mat in2);
```
- Utils.cpp  
Util methods
```cpp
double log_2(double num);

void drawPoint(cv::Mat &img, const cv::Point2f p, const cv::Scalar color, int size);

bool check_coordinate(cv::Point2f coordinate, cv::Vec4f range);

double intersectM(cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Point2f p4);

void interpolation(cv::Mat &in, double x, double y, unsigned char &rr1, unsigned char &gg1, unsigned char &bb1);

void drawTriangle(cv::Mat &img, cv::Point2f p1, cv::Point2f p2, cv::Point2f p3, cv::Scalar color);

inline int RR(cv::Mat &img, int i, int j);

inline int GG(cv::Mat &img, int i, int j);

inline int BB(cv::Mat &img, int i, int j);

inline double MM(cv::Mat &img, int i, int j);

inline bool isInTriangle(Point3Vec trig, cv::Point2d p);
```
