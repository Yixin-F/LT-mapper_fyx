#include "removert/utility.h"

bool cout_debug = false;

/**
 * @brief  读取激光帧的bin文件
 * 
 * @param[in] _bin_path 
 * @param[in] _pcd_ptr 
 */
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

/**
 * @brief 按照某种分割符来分割某行字符串
 * 
 * @param[in] _str_line    字符串
 * @param[in] _delimiter   分割符类型
 * @return std::vector<double>  double类型的vector
 */
std::vector<double> splitPoseLine(std::string _str_line, char _delimiter) {
    std::vector<double> parsed;
    std::stringstream ss(_str_line);
    std::string temp;
    while (getline(ss, temp, _delimiter)) {
        parsed.push_back(std::stod(temp)); // convert string to "double"
    }
    return parsed;
}

/**
 * @brief  将笛卡尔坐标系中的点转换进球形坐标系
 * 
 * @param[in] _cp   cartesian point
 * @return SphericalPoint  球形点
 */
SphericalPoint cart2sph(const PointType & _cp)
{

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

/**
 * @brief  重置range image分辨率
 * 
 * @param[in] _fov   旧的分辨率
 * @param[in] _resize_ratio    缩放比例
 * @return std::pair<int, int>   新的分辨率
 */
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

/**
 * @brief vector转set，set支持高效的关键字查询
 * 
 * @param[in] v  vecotor容器
 * @return * std::set<int>   set容器
 */
std::set<int> convertIntVecToSet(const std::vector<int> & v) 
{ 
    std::set<int> s; 
    for (int x : v) { 
        s.insert(x); 
    } 
    return s; 
} 

/**
 * @brief  发布range image消息
 * 
 * @param[in] _rimg  range image
 * @param[in] _msg    图片消息指针
 * @param[in] _publiser  发布者句柄
 * @param[in] _caxis  彩色上下界
 */
void pubRangeImg(cv::Mat& _rimg, 
                sensor_msgs::ImagePtr& _msg,
                image_transport::Publisher& _publiser,
                std::pair<float, float> _caxis)
{
    cv::Mat scan_rimg_viz = convertColorMappedImg(_rimg, _caxis);
    _msg = cvmat2msg(scan_rimg_viz);
    _publiser.publish(_msg);    
} // pubRangeImg

/**
 * @brief  
 * 
 * @param[in] _img 
 * @return sensor_msgs::ImagePtr 
 */
sensor_msgs::ImagePtr cvmat2msg(const cv::Mat &_img)
{
  sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", _img).toImageMsg();
  return msg;
}

/**
 * @brief 发布激光帧
 * 
 * @param[in] _scan_publisher  发布者句柄
 * @param[in] _scan   发布帧
 */
void publishPointcloud2FromPCLptr(const ros::Publisher& _scan_publisher, const pcl::PointCloud<PointType>::Ptr _scan)
{
    sensor_msgs::PointCloud2 tempCloud;
    pcl::toROSMsg(*_scan, tempCloud);
    tempCloud.header.stamp = ros::Time::now();
    tempCloud.header.frame_id = "removert";
    _scan_publisher.publish(tempCloud);
} // publishPointcloud2FromPCLptr

/**
 * @brief  发布点云
 * 
 * @param[in] thisPub   发布者句柄
 * @param[in] thisCloud   待发布点云
 * @param[in] thisStamp   时间戳
 * @param[in] thisFrame   frame_id
 * @return sensor_msgs::PointCloud2   待发布点云
 */
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

/**
 * @brief   计算点到base距离
 * 
 * @param[in] p   点
 * @return float   举例
 */
float pointDistance(PointType p)
{
    return sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
}

/**
 * @brief   计算两点之间距离
 * 
 * @param[in] p1   点1
 * @param[in] p2   点2
 * @return float   距离
 */
float pointDistance(PointType p1, PointType p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y) + (p1.z-p2.z)*(p1.z-p2.z));
}

