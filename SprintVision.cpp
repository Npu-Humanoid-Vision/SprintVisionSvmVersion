#include "SprintVision.h"

SprintVision::SprintVision() {
    final_result_.valid_    = false;
    start_file_num_         = 0;
    max_file_num_           = 500;
    former_result_rect_ = cv::Rect(-1, -1, -1, -1);
    init_former_rect_ = false;
    this->LoadParameters();
}

SprintVision::~SprintVision() {}

void SprintVision::imageProcess(cv::Mat input_image, ImgProcResult* output_result) {
    src_image_ = input_image.clone();
    pretreaded_image_ = Pretreat(src_image_);
    thresholded_image_ = ProcessColor(pretreaded_image_);
    possible_rects_ = GetPossibleRect(thresholded_image_);  // 在这里面得到 与之前距离最近的待选Rect 

    // 喂所有可能的 Rect 给 SVM classifier
    std::vector<cv::Rect> pos_lable_rects; // 存所有 perdict 为 pos lable 的 rect
    cout<<possible_rects_.size()<<endl;
    for (std::vector<cv::Rect>::iterator iter = possible_rects_.begin();
         iter != possible_rects_.end(); iter++) {
        if (iter->area() < 20) {
            continue;
        }
        cout<<*iter<<endl;
        cv::Mat roi_hog_vec = GetHogVec(*iter);
        int lable = (int)svm_classifier_.predict(roi_hog_vec);
        if (lable == POS_LABLE) {
            pos_lable_rects.push_back(*iter);
        }
    }

    if (!init_former_rect_) {// 初始化 former_rect 才能开始利用 former_rect
        if (pos_lable_rects.size() == 1) {// 初始化成功
            init_former_rect_ = true;
            final_result_.center_ = cv::Point(pos_lable_rects[0].x + cvRound(pos_lable_rects[0].width/2.0),
                                            pos_lable_rects[0].y + cvRound(pos_lable_rects[0].height/2.0));
            final_result_.valid_ = true;
            former_result_rect_ = pos_lable_rects[0];
            (*dynamic_cast<SprintResult*>(output_result)) = final_result_;
            return ;
        }
        else {// 不成功
            final_result_.valid_ = false;
            (*dynamic_cast<SprintResult*>(output_result)) = final_result_;
            return ;
        }
    }


    if (pos_lable_rects.size() == 0) { // 未检出pos lable，即未初始化former_rect_
        final_result_.valid_ = false;
        init_former_rect_ = false;
    }
    else if (pos_lable_rects.size() == 1) { // 就是它！
        result_rect_ = pos_lable_rects[0];
        final_result_.valid_ = true;
    }
    else { // 有多个pos lable， 返回 与之前的 Rect 距离最近的待选Rect
        result_rect_ = nearest_rect_;
    }

    if (final_result_.valid_) {
        final_result_.center_ = cv::Point(result_rect_.x + cvRound(result_rect_.width/2.0),
                                          result_rect_.y + cvRound(result_rect_.height/2.0));
        former_result_rect_ = result_rect_;
    }

    (*dynamic_cast<SprintResult*>(output_result)) = final_result_;

#ifndef ADJUST_PARAMETER
    this->WriteImg(src_image_,"src_img",start_file_num_);
    if (final_result_.valid_) {
        cv::rectangle(src_image_, result_rect_, cv::Scalar(0, 255, 0));
    }
    this->WriteImg(src_image_,"center_img",start_file_num_++);
#endif
}

cv::Mat SprintVision::Pretreat(cv::Mat raw_image) {
    // 先获得各个通道先
    cv::Mat t_hsv;
    cv::cvtColor(raw_image, t_hsv, CV_BGR2HSV);
    cv::split(t_hsv, src_hsv_channels_);
    used_channel = src_hsv_channels_[0].clone();

    cv::Mat blured_image;
    cv::GaussianBlur(raw_image, blured_image, cv::Size(3, 3), 0, 0);
    return blured_image;
}

cv::Mat SprintVision::ProcessColor() {
    cv::Mat mask = src_hsv_channels_[1] >= color_s_thre_;
    cv::Mat thre_result;
    if (color_direction_forward_) {
        thre_result = used_channel <= color_max_thre_ & used_channel >= color_min_thre_;
    }
    else {
        thre_result = used_channel >= color_max_thre_ | used_channel <= color_min_thre_;
    }
    thre_result = thre_result & mask;

    for (int i = 0; i < color_erode_times_; i++) {
        cv::erode(thre_result, thre_result, cv::Mat(3, 3, CV_8UC1));
    }
    for (int i = 0; i < color_dilate_times_; i++) {
        cv::dilate(thre_result, thre_result, cv::Mat(3, 3, CV_8UC1));
    }
    return thre_result;
}

std::vector<cv::Rect> SprintVision::GetPossibleRect(cv::Mat binary_image) {
    std::vector<std::vector<cv::Point> > contours;
    std::vector<std::vector<cv::Point> > contours_poly;
    std::vector<cv::Rect> bound_rect;

    cv::Mat image_for_contours = binary_image.clone();
    cv::findContours(image_for_contours, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

    contours_poly.resize(contours.size());
    bound_rect.resize(contours.size());

    // double max_area = 0.0;
    // int max_area_idx = -1;
    int max_inter_area = 0.0;
    int min_dist_idx = -1;
    for (unsigned int i = 0; i < contours.size(); i++) {
        cv::approxPolyDP(contours[i], contours_poly[i], 3, false);
        bound_rect[i] = cv::boundingRect(contours_poly[i]);
        
        // if (cv::contourArea(contours_poly[i], false) > max_area) {
        //     max_area = cv::contourArea(contours_poly[i], false);
        //     max_area_idx = i;
        // }
        if (init_former_rect_) {
            cv::Rect intersection = former_result_rect_ & bound_rect[i];
            if (intersection.area() > max_inter_area) {
                max_inter_area = intersection.area();
                min_dist_idx = i;
            }
        }

    }
    return bound_rect;
}

cv::Mat SprintVision::GetHogVec(cv::Rect roi) {
    cv::Mat roi_in_mat = src_image_(roi).clone();
    // SHOW_IMAGE(roi_in_mat);
    cv::resize(roi_in_mat, roi_in_mat, cv::Size(32, 32)); // 与训练相关参数，之后最好做成文件传入参数
    // SHOW_IMAGE(roi_in_mat);
    cv::HOGDescriptor hog_des(Size(32, 32), Size(16,16), Size(8,8), Size(8,8), 9);
    std::vector<float> hog_vec;
    hog_des.compute(roi_in_mat, hog_vec);

    cv::Mat t(hog_vec);
    // cout<<t<<endl;
    cv::Mat hog_vec_in_mat = t.t();
    // cout<<hog_vec_in_mat<<endl;
    hog_vec_in_mat.convertTo(hog_vec_in_mat, CV_32FC1);
    return hog_vec_in_mat;
}



void SprintVision::LoadParameters() {
#ifdef ADJUST_PARAMETER
    std::ifstream in_file("./7.txt");

#else    
    std::ifstream in_file("../source/data/set_sprint_param/7.txt");
#endif
    if (!in_file) {
        cerr<<"Error:"<<__FILE__
                <<":line"<<__LINE__<<endl
                <<"     Complied on"<<__DATE__
                <<"at"<<__TIME__<<endl;
    }
    int i = 0;
    string line_words;
    cout<<"Loading Parameters"<<endl;
    while (in_file >> line_words) {
        cout<<line_words<<endl;
        std::istringstream ins(line_words);
        switch (i++) {
        case 0:
            ins >> color_min_thre_;
            break;
        case 1:
            ins >> color_max_thre_;
            break;
        case 2:
            ins >> color_direction_forward_;
            break;
        case 3:
            ins >> color_erode_times_;
            break;
        case 4:
            ins >> color_dilate_times_;
            break;
        case 5:
            ins >> color_s_thre_;
            break;
        case 6:
            ins >> svm_model_name_;
            break;
        }
    }
#ifdef ADJUST_PARAMETER
    svm_classifier_.load(svm_model_name_.c_str());
#else
    svm_classifier_.load(("../source/data/set_sprint_param/"+svm_model_name_).c_str());
#endif
}

void SprintVision::StoreParameters() {
    std::ofstream out_file("./7.txt");
    if (!out_file) {
        cerr<<"Error:"<<__FILE__
                <<":line"<<__LINE__<<endl
                <<"     Complied on"<<__DATE__
                <<"at"<<__TIME__<<endl;
    }
    out_file << setw(3) << setfill('0') << color_min_thre_              <<"___color_min_thre"<<endl;
    out_file << setw(3) << setfill('0') << color_max_thre_              <<"___color_max_thre"<<endl;
    out_file << setw(3) << setfill('0') << color_direction_forward_     <<"___color_direction_forward"<<endl;
    out_file << setw(3) << setfill('0') << color_erode_times_           <<"___color_erode_times"<<endl;
    out_file << setw(3) << setfill('0') << color_dilate_times_          <<"___color_dilate_times"<<endl;
    out_file << setw(3) << setfill('0') << color_s_thre_                <<"___color_s_thre"<<endl;
    out_file << svm_model_name_;
    out_file.close();
}

void SprintVision::set_all_parameters(AllParameters ap) {
    color_min_thre_ = ap.c_min_thre;
    color_max_thre_ = ap.c_max_thre;
    color_direction_forward_ = ap.c_direc_forw;
    color_erode_times_ = ap.c_erode_times;
    color_dilate_times_ = ap.c_dilat_times;
    color_s_thre_ = ap.c_s_thre;
}
     
void SprintVision::WriteImg(cv::Mat src, string folder_name, int num) {
    stringstream t_ss;
    string path = "../source/data/con_img/";
    if (start_file_num_ <= max_file_num_) {
        path += folder_name;
        path += "/";

        t_ss << num;
        path += t_ss.str();
        t_ss.str("");
        t_ss.clear();
        // path += std::to_string(num); 

        path += ".jpg";

        cv::imwrite(path,src);
    }
}
