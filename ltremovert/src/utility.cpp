#include "removert/utility.h"

bool cout_debug = false;

// 读取每一激光帧的bin文件
void readBin(std::string _bin_path, pcl::PointCloud<PointType>::Ptr _pcd_ptr)
{
 	std::fstream input(_bin_path.c_str(), ios::in | ios::binary);
	if(!input.good()){
		cerr << "Could not read file: " << _bin_path << endl;
		exit(EXIT_FAILURE);
	}
	input.seekg(0, ios::beg);
  
	for (int ii=0; input.good() && !input.eof(); ii++) {
		PointType point;

		input.read((char *) &point.x, sizeof(float));
		input.read((char *) &point.y, sizeof(float));
		input.read((char *) &point.z, sizeof(float));
		input.read((char *) &point.intensity, sizeof(float));

		_pcd_ptr->push_back(point);
	}
	input.close();
}

// 按某种分割符delimiter来读取每一行的位姿
std::vector<double> splitPoseLine(std::string _str_line, char _delimiter) {
    std::vector<double> parsed;
    std::stringstream ss(_str_line);
    std::string temp;
    while (getline(ss, temp, _delimiter)) {
        parsed.push_back(std::stod(temp)); // convert string to "double"
    }
    return parsed;
}

// 将扫描的笛卡尔坐标系点云转化进入球形坐标系，也就是rangeimage
SphericalPoint cart2sph(const PointType & _cp)
{ // _cp means cartesian point

    if(cout_debug){
        cout << "Cartesian Point [x, y, z]: [" << _cp.x << ", " << _cp.y << ", " << _cp.z << endl;
    }

    SphericalPoint sph_point {
         std::atan2(_cp.y, _cp.x), 
         std::atan2(_cp.z, std::sqrt(_cp.x*_cp.x + _cp.y*_cp.y)),
         std::sqrt(_cp.x*_cp.x + _cp.y*_cp.y + _cp.z*_cp.z)
    };    
    return sph_point;
}

float rad2deg(float radians)
{
    return radians * 180.0 / M_PI;
}

float deg2rad(float degrees)
{
    return degrees * M_PI / 180.0;
}


void transformGlobalMapToLocal(
    const pcl::PointCloud<PointType>::Ptr &_map_global,
    Eigen::Matrix4d _base_pose_inverse, Eigen::Matrix4d _base2lidar, pcl::PointCloud<PointType>::Ptr &_map_local)
{
    // global to local (global2local)
    _map_local->clear();
    // 将得到的每一帧pcd，由世界坐标系转换到base坐标系
    pcl::transformPointCloud(*_map_global, *_map_local, _base_pose_inverse);
    // 再一个小转换
    pcl::transformPointCloud(*_map_local, *_map_local, _base2lidar); // > _base2lidar == kSE3MatExtrinsicPoseBasetoLiDAR
} // transformGlobalMapToLocal

pcl::PointCloud<PointType>::Ptr parseProjectedPoints(const pcl::PointCloud<PointType>::Ptr &_scan,
                                                                    const std::pair<float, float> _fov, /* e.g., [vfov = 50 (upper 25, lower 25), hfov = 360] */
                                                                    const std::pair<int, int> _rimg_size)
{
    auto [map_rimg, map_rimg_ptidx] = map2RangeImg(_scan, _fov, _rimg_size); // the most time comsuming part 2 -> so openMP applied inside
    pcl::PointCloud<PointType>::Ptr scan_projected(new pcl::PointCloud<PointType>);
    for (int row_idx = 0; row_idx < map_rimg_ptidx.rows; ++row_idx) {
        for (int col_idx = 0; col_idx < map_rimg_ptidx.cols; ++col_idx) {
            if(map_rimg_ptidx.at<int>(row_idx, col_idx) == 0) // 0 means no point (see map2RangeImg)
                continue;

            scan_projected->push_back(_scan->points[map_rimg_ptidx.at<int>(row_idx, col_idx)]);
        }
    }
    return scan_projected;
} // map2RangeImg


std::pair<cv::Mat, cv::Mat> map2RangeImg(const pcl::PointCloud<PointType>::Ptr &_scan,
                                                        const std::pair<float, float> _fov, /* e.g., [vfov = 50 (upper 25, lower 25), hfov = 360] */
                                                        const std::pair<int, int> _rimg_size)
{
    const float kVFOV = _fov.first;
    const float kHFOV = _fov.second;

    const int kNumRimgRow = _rimg_size.first;
    const int kNumRimgCol = _rimg_size.second;

    // @ range image initizliation
    // ragne & index 分开存储
    cv::Mat rimg = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32FC1, cv::Scalar::all(kFlagNoPOINT)); // float matrix, save range value
    cv::Mat rimg_ptidx = cv::Mat(kNumRimgRow, kNumRimgCol, CV_32SC1, cv::Scalar::all(0));      // int matrix, save point (of global map) index

    // @ points to range img
    int num_points = _scan->points.size();

    const int kNumOmpCores = 16; // hard coding for fast dev 
    #pragma omp parallel for num_threads(kNumOmpCores)
    for (int pt_idx = 0; pt_idx < num_points; ++pt_idx)
    {
        PointType this_point = _scan->points[pt_idx];
        SphericalPoint sph_point = cart2sph(this_point); // 进入球形坐标系

        // @ note about vfov: e.g., (+ V_FOV/2) to adjust [-15, 15] to [0, 30]
        // @ min and max is just for the easier (naive) boundary checks.
        int lower_bound_row_idx{0};
        int lower_bound_col_idx{0};
        int upper_bound_row_idx{kNumRimgRow - 1};
        int upper_bound_col_idx{kNumRimgCol - 1};
        int pixel_idx_row = int(std::min(std::max(std::round(kNumRimgRow * (1 - (rad2deg(sph_point.el) + (kVFOV / float(2.0))) / (kVFOV - float(0.0)))), float(lower_bound_row_idx)), float(upper_bound_row_idx)));
        int pixel_idx_col = int(std::min(std::max(std::round(kNumRimgCol * ((rad2deg(sph_point.az) + (kHFOV / float(2.0))) / (kHFOV - float(0.0)))), float(lower_bound_col_idx)), float(upper_bound_col_idx)));

        float curr_range = sph_point.r;

        // @ Theoretically, this if-block would have race condition (i.e., this is a critical section),
        // @ But, the resulting range image is acceptable (watching via Rviz),
        // @      so I just naively applied omp pragma for this whole for-block (2020.10.28)
        // @ Reason: because this for loop is splited by the omp, points in a single splited for range do not race among them,
        // @         also, a point A and B lied in different for-segments do not tend to correspond to the same pixel,
        // #               so we can assume practically there are few race conditions.
        // @ P.S. some explicit mutexing directive makes the code even slower ref: https://stackoverflow.com/questions/2396430/how-to-use-lock-in-openmp
        // ! 意思是会有一些距离不同的点出现在同一pixel中
        // > 这个地方，其实可以最后再平滑一下，不是单纯的判断这个点是否是动态点，一是可以将相邻动态点聚类检查，二是可以实例分割辅助
        if (curr_range < rimg.at<float>(pixel_idx_row, pixel_idx_col))
        {
            rimg.at<float>(pixel_idx_row, pixel_idx_col) = curr_range;
            rimg_ptidx.at<int>(pixel_idx_row, pixel_idx_col) = pt_idx;
        }
    }

    return std::pair<cv::Mat, cv::Mat>(rimg, rimg_ptidx);
} // map2RangeImg

// void octreeDownsampling(const pcl::PointCloud<PointType>::Ptr &_src, pcl::PointCloud<PointType>::Ptr &_to_save)
// {
//     pcl::octree::OctreePointCloudVoxelCentroid<PointType> octree(kDownsampleVoxelSize);
//     octree.setInputCloud(_src);
//     octree.defineBoundingBox();
//     octree.addPointsFromInputCloud();
//     pcl::octree::OctreePointCloudVoxelCentroid<PointType>::AlignedPointTVector centroids;
//     octree.getVoxelCentroids(centroids);

//     // init current map with the downsampled full cloud
//     _to_save->points.assign(centroids.begin(), centroids.end());
//     _to_save->width = 1;
//     _to_save->height = _to_save->points.size(); // make sure again the format of the downsampled point cloud
//     ROS_INFO_STREAM("\033[1;32m Octree Downsampled pointcloud have: " << _to_save->points.size() << " points.\033[0m");
// } // octreeDownsampling

// base到世界坐标系
pcl::PointCloud<PointType>::Ptr local2global(const pcl::PointCloud<PointType>::Ptr &_scan_local, 
                                const Eigen::Matrix4d &_scan_pose, const Eigen::Matrix4d &_lidar2base)
{
    pcl::PointCloud<PointType>::Ptr scan_global(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*_scan_local, *scan_global, _lidar2base);
    pcl::transformPointCloud(*scan_global, *scan_global, _scan_pose);

    return scan_global;
}

pcl::PointCloud<PointType>::Ptr mergeScansWithinGlobalCoordUtil(
    const std::vector<pcl::PointCloud<PointType>::Ptr> &_scans,
    const std::vector<Eigen::Matrix4d> &_scans_poses, Eigen::Matrix4d _lidar2base)
{
    pcl::PointCloud<PointType>::Ptr ptcloud_merged(new pcl::PointCloud<PointType>());

    // NOTE: _scans must be in local coord
    for (std::size_t scan_idx = 0; scan_idx < _scans.size(); scan_idx++)
    {
        auto ii_scan = _scans.at(scan_idx);       // pcl::PointCloud<PointType>::Ptr
        auto ii_pose = _scans_poses.at(scan_idx); // Eigen::Matrix4d

        // local to global (local2global)
        pcl::PointCloud<PointType>::Ptr scan_global_coord(new pcl::PointCloud<PointType>());
        pcl::transformPointCloud(*ii_scan, *scan_global_coord, _lidar2base); // kSE3MatExtrinsicLiDARtoPoseBase
        pcl::transformPointCloud(*scan_global_coord, *scan_global_coord, ii_pose);

        // merge the scan into the global map
        // 将该帧放入世界坐标系中
        *ptcloud_merged += *scan_global_coord;
    }

    return ptcloud_merged;
} // mergeScansWithinGlobalCoord

pcl::PointCloud<PointType>::Ptr global2local(const pcl::PointCloud<PointType>::Ptr &_scan_global, 
                                const Eigen::Matrix4d &_scan_pose_inverse, const Eigen::Matrix4d &_base2lidar)
{
    pcl::PointCloud<PointType>::Ptr scan_local(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*_scan_global, *scan_local, _scan_pose_inverse);
    pcl::transformPointCloud(*scan_local, *scan_local, _base2lidar); // kSE3MatExtrinsicPoseBasetoLiDAR

    return scan_local;
}

// > 八叉树下采样更注重每个voxel的递进关系，更方便进行knn等
void octreeDownsampling(const pcl::PointCloud<PointType>::Ptr &_src, 
                              pcl::PointCloud<PointType>::Ptr &_to_save,
                              const float _kDownsampleVoxelSize)
{
    pcl::octree::OctreePointCloudVoxelCentroid<PointType> octree(_kDownsampleVoxelSize);
    octree.setInputCloud(_src);
    octree.defineBoundingBox();
    octree.addPointsFromInputCloud();
    pcl::octree::OctreePointCloudVoxelCentroid<PointType>::AlignedPointTVector centroids;
    octree.getVoxelCentroids(centroids);

    // init current map with the downsampled full cloud
    _to_save->points.assign(centroids.begin(), centroids.end());
    _to_save->width = 1;
    _to_save->height = _to_save->points.size(); // make sure again the format of the downsampled point cloud
} // octreeDownsampling

// > 重置分辨率
std::pair<int, int> resetRimgSize(const std::pair<float, float> _fov, const float _resize_ratio)
{
    // default is 1 deg x 1 deg 
    float alpha_vfov = _resize_ratio;    
    float alpha_hfov = _resize_ratio;    

    float V_FOV = _fov.first;
    float H_FOV = _fov.second;

    int NUM_RANGE_IMG_ROW = std::round(V_FOV*alpha_vfov);
    int NUM_RANGE_IMG_COL = std::round(H_FOV*alpha_hfov);

    std::pair<int, int> rimg {NUM_RANGE_IMG_ROW, NUM_RANGE_IMG_COL};
    return rimg;
}


std::set<int> convertIntVecToSet(const std::vector<int> & v) 
{ 
    std::set<int> s; 
    for (int x : v) { 
        s.insert(x); 
    } 
    return s; 
} 

void pubRangeImg(cv::Mat& _rimg, 
                sensor_msgs::ImagePtr& _msg,
                image_transport::Publisher& _publiser,
                std::pair<float, float> _caxis)
{
    cv::Mat scan_rimg_viz = convertColorMappedImg(_rimg, _caxis);
    _msg = cvmat2msg(scan_rimg_viz);
    _publiser.publish(_msg);    
} // pubRangeImg


sensor_msgs::ImagePtr cvmat2msg(const cv::Mat &_img)
{
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", _img).toImageMsg();
  return msg;
}


void publishPointcloud2FromPCLptr(const ros::Publisher& _scan_publisher, const pcl::PointCloud<PointType>::Ptr _scan)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*_scan, tempCloud);
    tempCloud.header.stamp = ros::Time::now();
    tempCloud.header.frame_id = "removert";
    _scan_publisher.publish(tempCloud);
} // publishPointcloud2FromPCLptr


sensor_msgs::PointCloud2 publishCloud(ros::Publisher *thisPub, pcl::PointCloud<PointType>::Ptr thisCloud, ros::Time thisStamp, std::string thisFrame)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*thisCloud, tempCloud);
    tempCloud.header.stamp = thisStamp;
    tempCloud.header.frame_id = thisFrame;
    if (thisPub->getNumSubscribers() != 0)
        thisPub->publish(tempCloud);
    return tempCloud;
}

float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

