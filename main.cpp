// 调参开关
#define ADJUST_PARAMETER

#include "SprintVision.h"

cv::VideoCapture cp(0);
cv::Mat frame;
SprintVision sprint_vision;
SprintResult gabage;
AllParameters all_p;    


// #define RUN_ON_DARWIN 

int main(int argc, char const *argv[]) {
    if (!cp.isOpened()) {
        cerr<<"open camera fail"<<endl;
        return -1;
    }

    all_p.c_min_thre = sprint_vision.color_min_thre_;
    all_p.c_max_thre = sprint_vision.color_max_thre_;
    all_p.c_direc_forw = sprint_vision.color_direction_forward_;
    all_p.c_erode_times = sprint_vision.color_erode_times_;
    all_p.c_dilat_times = sprint_vision.color_dilate_times_;
    all_p.c_s_thre = sprint_vision.color_s_thre_;

    cv::namedWindow("set_params", CV_WINDOW_NORMAL);
    
    cv::createTrackbar("h_min", "set_params", &all_p.c_min_thre, 255);
    cv::createTrackbar("h_max", "set_params", &all_p.c_max_thre, 255);
    cv::createTrackbar("thre_direc", "set_params", &all_p.c_direc_forw, 1);
    cv::createTrackbar("erode_times", "set_params", &all_p.c_erode_times, 6);
    cv::createTrackbar("dilat_times", "set_params", &all_p.c_dilat_times, 6);
    cv::createTrackbar("s_thre", "set_params", &all_p.c_s_thre, 255);


    while (1) {
        cp >> frame;
        if (frame.empty()) {
            cerr<<"frame empty"<<endl;
            return -1;
        }



#ifdef RUN_ON_DARWIN
        cv::flip(frame, frame, -1);
        cv::resize(frame, frame, cv::Size(320, 240));
#endif
        sprint_vision.set_all_parameters(all_p);
        sprint_vision.imageProcess(frame, &gabage);

        cv::circle(sprint_vision.src_image_, gabage.center_, 3, cv::Scalar(0, 255, 0), 3);
        cv::rectangle(sprint_vision.src_image_, sprint_vision.result_rect_, cv::Scalar(0, 255, 0), 3);
        if (sprint_vision.init_former_rect_) {
            cv::rectangle(sprint_vision.src_image_, sprint_vision.former_result_rect_, cv::Scalar(255, 0, 0), 1);
        }        

        cv::imshow("living", frame);
        cv::imshow("result_show", sprint_vision. src_image_);
        cv::imshow("threshold", sprint_vision.thresholded_image_);

        char key = cv::waitKey(100);
        if (key == 's') {
            sprint_vision.StoreParameters();
            break;
        }
        else if (key == 'q') {
            break;
        }


        // for SVM test
        // ROI = frame(ROI_Rect).clone();
        // cv::resize(ROI, ROI, cv::Size(32, 32));
        // cv::HOGDescriptor hog_des(Size(32, 32), Size(16,16), Size(8,8), Size(8,8), 9);
        // std::vector<float> hog_vec;
        // hog_des.compute(ROI, hog_vec);

        // cv::Mat t(hog_vec);
        // cv::Mat hog_vec_in_mat = t.t();
        // hog_vec_in_mat.convertTo(hog_vec_in_mat, CV_32FC1);

        // CvSVM classifier;
        // classifier.load("linear_auto.xml");

        // int lable = (int)classifier.predict(hog_vec_in_mat);
        // if (lable == POS_LABLE) {
        //     cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 255, 0), 2);
        // }
        // else {
        //     cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 0, 255), 2);
        // }
        // cv::imshow("frame", frame);
        // if (cv::waitKey(20) == 'q') {
        //     break;
        // }
    }
    return 0;
}
