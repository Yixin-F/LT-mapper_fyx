#pragma once

#include "removert/RosParamServer.h"
#include "removert/tictoc.h"

class Segment : public TicToc{

private:

    std::vector<std::string> soure_PointCloudNamestr;
    std::vector<std::string> soure_PointCloudPathstr;

    std::vector<pcl::PointCloud<PointType>::Ptr> soure_PointCloud;
    std::vector<cv::Mat> source_RangeImageIdx;
    std::vector<cv::Mat> source_RangeImage;
    
    std::string savePath = "/home/fyx/lt-mapper/src/segmentation_dataset/";
    std::string savePath_in = savePath + "in/";
    std::string savePath_out = savePath + "out/";

    float kDownsampleVoxelSize = 0.05;
    const float kFlagNoPOINT = 10000.0;

    std::vector<float> rimg_resolution_list_;
    std::pair<float, float> kFOV;
    std::pair<float, float> color_axis;

public:
    Segment();
    ~Segment();

    std::vector<pcl::PointCloud<PointType>::Ptr> gd_PointCloud;
    std::vector<pcl::PointCloud<PointType>::Ptr> far_PointCloud;
    std::vector<pcl::PointCloud<PointType>::Ptr> seg_PointCloud;

    // > remember to clear in every scan segmentation
    std::vector<int> gd_PointCloudIdx;
    std::vector<int> far_PointCloudIdx;
    std::vector<int> seg_PointCloudIdx;

    std::vector<cv::Mat> gd_RangeImage;
    std::vector<cv::Mat> far_RangeImage;
    std::vector<cv::Mat> seg_RangeImage;

    void fsmkdir(std::string _path);
    
    void allocateMemory();

    void readExamplePCD();

    cv::Mat scan2RangeImg(const pcl::PointCloud<PointType>::Ptr& _scan, 
                        const std::pair<float, float> _fov, 
                        const std::pair<int, int> _rimg_size);

    std::vector<int> groundExtract(const cv::Mat& _rimg);
    std::vector<int> farExtract(const cv::Mat& _rimg);
    // TODO: segmentation Aft ground&far Extract

    void prasePointCloudUsingPtIdx(const pcl::PointCloud<PointType>& _sr_PointCloud, const std::vector<int>& _ptIdx);

    void savePtAndRimgResult();

    inline float rad2deg(float radians);
    inline float deg2rad(float degrees);
}; // Segment