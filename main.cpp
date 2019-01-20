#include "SprintVision.h"



// #define RUN_ON_DARWIN 

int main(int argc, char const *argv[]) {
    cv::VideoCapture cp(0);
    cv::Mat frame;
    cv::Mat ROI;
    cv::Rect ROI_Rect(0, 0, 400, 400);

    if (!cp.isOpened()) {
        cerr<<"open camera fail"<<endl;
        return -1;
    }

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
        ROI = frame(ROI_Rect).clone();
        cv::resize(ROI, ROI, cv::Size(32, 32));
        cv::HOGDescriptor hog_des(Size(32, 32), Size(16,16), Size(8,8), Size(8,8), 9);
        std::vector<float> hog_vec;
        hog_des.compute(ROI, hog_vec);

        cv::Mat t(hog_vec);
        cv::Mat hog_vec_in_mat = t.t();
        hog_vec_in_mat.convertTo(hog_vec_in_mat, CV_32FC1);

        CvSVM classifier;
        classifier.load("linear_auto.xml");

        int lable = (int)classifier.predict(hog_vec_in_mat);
        if (lable == POS_LABLE) {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 255, 0), 2);
        }
        else {
            cv::rectangle(frame, ROI_Rect, cv::Scalar(0, 0, 255), 2);
        }
        cv::imshow("frame", frame);
        if (cv::waitKey(20) == 'q') {
            break;
        }
    }
    return 0;
}
