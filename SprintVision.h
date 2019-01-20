#ifndef SPRINT_VISION_H
#define SPRINT_VISION_H

// 正负样本的 lable
#define POS_LABLE 1
#define NEG_LABLE 0

// 调参开关
#define ADJUST_PARAMETER

#include <opencv2/opencv.hpp>
#include <fstream> // 推荐在基类加...
#include <iostream>
#include <iomanip>
using namespace std;
using namespace cv;

#ifdef ADJUST_PARAMETER

// showing image in debugging 
#define SHOW_IMAGE(imgName) \
    namedWindow("imgName", WINDOW_AUTOSIZE); \
    moveWindow("imgName", 300, 300); \
    imshow("imgName", imgName); \
    waitKey(0); \
    destroyWindow("imgName"); \

class ImgProcResult{

public:
	ImgProcResult(){};
	~ImgProcResult(){};
	 virtual void operator=(ImgProcResult &res) = 0;
private:
protected:

};
class ImgProc{

public:
	ImgProc(){};
	~ImgProc(){};
	virtual void imageProcess(cv::Mat img, ImgProcResult *Result) =0;
private:
protected:
	ImgProcResult *res;

};
#else

#include "imgproc.h"
#define SHOW_IMAGE(imgName) ;

#endif 

// adjust_parameter
class SprintResult : public ImgProcResult
{
public:
    cv::Point center_;
    bool valid_;
public:
    SprintResult() : center_(),
                     valid_(false) {}

    // adjust_parameter    
    virtual void operator=(ImgProcResult &res) {
        SprintResult *tmp = dynamic_cast<SprintResult *>(&res);
        center_ = tmp->center_;
        valid_ = tmp->valid_;
    }

    void operator=(SprintResult &res) {
        center_ = res.center_;
        valid_ = res.valid_;
    }
};

struct AllParameters {
    int c_min_thre;
    int c_max_thre;
    int c_direc_forw;
    int c_erode_times;
    int c_dilat_times;
    int c_s_thre;
};

class SprintVision : public ImgProc {
public:
    SprintVision();
    ~SprintVision();

public: // 假装是接口的函数
    void imageProcess(cv::Mat input_image, ImgProcResult* output_result);  // 对外接口

    cv::Mat Pretreat(cv::Mat raw_image);                                    // 所有图像进行目标定位前的预处理

    cv::Mat ProcessColor(cv::Mat pretread_image);                           // 颜色操作

    std::vector<cv::Rect> GetPossibleRect(cv::Mat binary_image);            // 从二值图获得所有可能区域

    cv::Mat GetHogVec(cv::Rect roi);                                        // 获得src_img 上 roi 的 HOG 特征向量

public: // 真实的接口函数
    void LoadParameters();                                                  // 从文件加载参数

    void StoreParameters();                                                 // 将参数存到文件

    void set_all_parameters(AllParameters);                                 // 调参时候传入参数

    void WriteImg(cv::Mat src, string folder_name, int num);                // 写图片

public: // 数据成员
    cv::Mat src_image_;
    cv::Mat src_hsv_channels_[3];
    cv::Mat used_channel;
    cv::Mat pretreaded_image_;
    cv::Mat thresholded_image_;
    std::vector<cv::Rect> possible_rects_;
    CvSVM svm_classifier_;
    SprintResult final_result_;

    // 阈值化相关成员
    int color_min_thre_;
    int color_max_thre_;
    int color_direction_forward_;
    int color_erode_times_;
    int color_dilate_times_;
    int color_s_thre_;

    // 用于获得多个可能结果时候检验
    bool init_former_rect_;
    cv::Rect former_result_rect_;

    // 存图相关
    int start_file_num_;
    int max_file_num_;

    // SVM model path
    string svm_model_name_;

    // 所有在GetPossibleRect得到的待选Rect
    cv::Rect nearest_rect_;
};





#endif