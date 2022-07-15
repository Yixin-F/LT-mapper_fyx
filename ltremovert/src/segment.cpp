# include "removert/segment.h"

Segment::Segment(){

    // ? voxelgrid generates warnings frequently, so verbose off + ps. recommend to use octree
    pcl::console::setVerbosityLevel(pcl::console::L_ERROR); 

    allocateMemory();

    readExamplePCD();

    // TODO: rviz visualization

}

/**
 * @brief  分配存储空间
 * 
 */
void Segment::allocateMemory(){

    soure_PointCloudNamestr.clear();
    soure_PointCloudPathstr.clear();

    for(auto& _entry : fs::directory_iterator(savePath_in)) {
        soure_PointCloudNamestr.emplace_back(_entry.path().filename());
        soure_PointCloudPathstr.emplace_back(_entry.path());
    }

    assert(soure_PointCloudNamestr.size() == soure_PointCloudPathstr.size());
    std::cout << "The num of example is:" << " " << soure_PointCloudNamestr.size() << std::endl;

    soure_PointCloud.clear(); // pcl::PointCloud<PointType>::Ptr
    gd_PointCloud.clear();
    far_PointCloud.clear();
    seg_PointCloud.clear();

    source_RangeImageIdx.clear(); // cv::Mat
    gd_PointCloudIdx.clear(); // int
    far_PointCloudIdx.clear();
    seg_PointCloudIdx.clear();

    source_RangeImage.clear(); // cv::Mat
    gd_RangeImage.clear();
    far_RangeImage.clear();
    seg_RangeImage.clear();

    // make save_out file
    if(savePath_out.substr(savePath_out.size()-1, 1) != std::string("/"))
        savePath_out = savePath_out + "/";
    fsmkdir(savePath_out);

    rimg_resolution_list_.emplace_back(1.4);
    rimg_resolution_list_.emplace_back(1.1);
    
    kFOV = std::pair<float, float>(50.0, 360.0);

    color_axis = std::pair<float, float>(0.0, 100.0);
}

void Segment::readExamplePCD(){

    pcl::PointCloud<PointType>::Ptr _scan(new pcl::PointCloud<PointType>());

    for(size_t ii = 0; ii < soure_PointCloudPathstr.size(); ii++){
        // if(isScanFileKITTIFormat_){
        if(false){
            readBin(soure_PointCloudPathstr[ii], _scan);
        }
        else
            pcl::io::loadPCDFile<PointType>(soure_PointCloudPathstr[ii], *_scan);

        // downsample and save
        pcl::VoxelGrid<PointType> downsize_filter;
        pcl::PointCloud<PointType>::Ptr downsampled_scan (new pcl::PointCloud<PointType>);
        downsize_filter.setLeafSize(kDownsampleVoxelSize, kDownsampleVoxelSize, kDownsampleVoxelSize);
        downsize_filter.setInputCloud(_scan);
        downsize_filter.filter(*downsampled_scan);

        soure_PointCloud.emplace_back(downsampled_scan);

        std::cout << soure_PointCloudPathstr[ii] << "\n"
                  << "src size:" << " " << _scan->points.size() << "\n"
                  << "voxel size:" << " " << kDownsampleVoxelSize << "\n"
                  << "Aft downsample size:" << " " << downsampled_scan->points.size() << "\n"
                  << std::endl;
    }

    assert(soure_PointCloud.size() == soure_PointCloudNamestr.size());

}

cv::Mat Segment::scan2RangeImg(const pcl::PointCloud<PointType>::Ptr& _scan, 
                      const std::pair<float, float> _fov, 
                      const std::pair<int, int> _rimg_size)
{
    const float kVFOV = _fov.first;
    const float kHFOV = _fov.second;
    
    const int kNumRimgRow = _rimg_size.first;
    const int kNumRimgCol = _rimg_size.second;

    // @ range image initizliation 
    cv::Mat rimg = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32FC1, cv::Scalar::all(kFlagNoPOINT)); // float matrix

    // @ points to range img 
    int num_points = _scan->points.size();
    #pragma omp parallel for num_threads(8)
    for (int pt_idx = 0; pt_idx < num_points; ++pt_idx)
    {   
        PointType this_point = _scan->points[pt_idx];
        SphericalPoint sph_point = cart2sph(this_point);

        // @ note about vfov: e.g., (+ V_FOV/2) to adjust [-15, 15] to [0, 30]
        // @ min and max is just for the easier (naive) boundary checks. 
        int lower_bound_row_idx {0}; 
        int lower_bound_col_idx {0};
        int upper_bound_row_idx {kNumRimgRow - 1}; 
        int upper_bound_col_idx {kNumRimgCol - 1};
        int pixel_idx_row = int(std::min(std::max(std::round(kNumRimgRow * (1 - (rad2deg(sph_point.el) + (kVFOV/float(2.0))) / (kVFOV - float(0.0)))), float(lower_bound_row_idx)), float(upper_bound_row_idx)));
        int pixel_idx_col = int(std::min(std::max(std::round(kNumRimgCol * ((rad2deg(sph_point.az) + (kHFOV/float(2.0))) / (kHFOV - float(0.0)))), float(lower_bound_col_idx)), float(upper_bound_col_idx)));

        float curr_range = sph_point.r;

        if (curr_range < rimg.at<float>(pixel_idx_row, pixel_idx_col)){
            rimg.at<float>(pixel_idx_row, pixel_idx_col) = curr_range;
        }
    }

    return rimg;
} // scan2RangeImg

void Segment::savePtAndRimgResult(){

    std::cout << "scan2Rimg:" << std::endl;

    for(size_t ii = 0; ii < soure_PointCloud.size(); ii++){
        TicToc rimg_tim(true);
        
        // multi-resolution
        for(float _res: rimg_resolution_list_){
            std::pair<int, int> rimg_shape = resetRimgSize(kFOV, _res);

            cv::Mat _scan_rimg = scan2RangeImg(soure_PointCloud[ii], kFOV, rimg_shape);
            cv::Mat scan_rimg_viz = convertColorMappedImg(_scan_rimg, color_axis); // color
            cv::imwrite(savePath_out + soure_PointCloudNamestr[ii] + "_" + std::to_string(rimg_shape.first) + ".jpg", scan_rimg_viz);
        }
        
        rimg_tim.toc(soure_PointCloudNamestr[ii]);
    }
}


Segment::~Segment(){}

void Segment::fsmkdir(std::string _path){
    if (!fs::is_directory(_path) || !fs::exists(_path)) 
        fs::create_directories(_path);
    else{
        fs::remove(_path);
        fs::create_directories(_path);  // new
    } 
}

inline float Segment::rad2deg(float radians){
    return radians * 180.0 / M_PI; 
}

inline float Segment::deg2rad(float degrees){
    return degrees * M_PI / 180.0; 
}