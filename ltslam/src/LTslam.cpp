#include "ltslam/LTslam.h"


gtsam::Pose3 LTslam::getPoseOfIsamUsingKey (const Key _key) {
    const Value& pose_value = isam->calculateEstimate(_key);
    auto p_pose_value = dynamic_cast<const gtsam::GenericValue<gtsam::Pose3>*>(&pose_value);
    gtsam::Pose3 pose = gtsam::Pose3{p_pose_value->value()};
    return pose;
}

void LTslam::writeAllSessionsTrajectories(std::string _postfix = "")
{
    // parse
    std::map<int, gtsam::Pose3> parsed_anchor_transforms;
    std::map<int, std::vector<gtsam::Pose3>> parsed_poses;

    isamCurrentEstimate = isam->calculateEstimate();
    for(const auto& key_value: isamCurrentEstimate) {

        int curr_node_idx = int(key_value.key); // typedef std::uint64_t Key

        std::vector<int> parsed_digits;
        collect_digits(parsed_digits, curr_node_idx);
        int session_idx = parsed_digits.at(0);
        int anchor_node_idx = genAnchorNodeIdx(session_idx);

        auto p = dynamic_cast<const gtsam::GenericValue<gtsam::Pose3>*>(&key_value.value);
        if (!p) continue;
        gtsam::Pose3 curr_node_pose = gtsam::Pose3{p->value()};

        if( curr_node_idx == anchor_node_idx ) { 
            // anchor node 
            parsed_anchor_transforms[session_idx] = curr_node_pose;
        } else { 
            // general nodes
            parsed_poses[session_idx].push_back(curr_node_pose);
        }
    }

    std::map<int, std::string> session_names;
    for(auto& _sess_pair: sessions_)
    {
        auto& _sess = _sess_pair.second;
        session_names[_sess.index_] = _sess.name_;
    }

    // write
    for(auto& _session_info: parsed_poses) {
        int session_idx = _session_info.first;

    	std::string filename_local = save_directory_ + session_names[session_idx] + "_local_" + _postfix + ".txt";
    	std::string filename_central = save_directory_ + session_names[session_idx] + "_central_" + _postfix + ".txt";
        cout << filename_central << endl;

        std::fstream stream_local(filename_local.c_str(), std::fstream::out);
        std::fstream stream_central(filename_central.c_str(), std::fstream::out);

        gtsam::Pose3 anchor_transform = parsed_anchor_transforms[session_idx];
        for(auto& _pose: _session_info.second) {
            writePose3ToStream(stream_local, _pose);

            gtsam::Pose3 pose_central = anchor_transform * _pose; // se3 compose (oplus) 
            writePose3ToStream(stream_central, pose_central);
        }
    }

} // writeAllSessionsTrajectories


LTslam::LTslam()
: poseOrigin(gtsam::Pose3(gtsam::Rot3::RzRyRx(0.0, 0.0, 0.0), gtsam::Point3(0.0, 0.0, 0.0))) 
{
} // ctor


LTslam::~LTslam() { }
// dtor

void LTslam::run( void )
{
    initOptimizer();
    initNoiseConstants();

    loadAllSessions();
    addAllSessionsToGraph();

    optimizeMultisesseionGraph(true); // optimize the graph with existing edges 
    writeAllSessionsTrajectories(std::string("bfr_intersession_loops"));

    detectInterSessionSCloops(); // detectInterSessionRSloops was internally done while sc detection 
    addSCloops();
    optimizeMultisesseionGraph(true); // optimize the graph with existing edges + SC loop edges

    bool toOpt = addRSloops(); // using the optimized estimates (rough alignment using SC)
    optimizeMultisesseionGraph(toOpt); // optimize the graph with existing edges + SC loop edges + RS loop edges

    writeAllSessionsTrajectories(std::string("aft_intersession_loops"));
}

void LTslam::initNoiseConstants()
{
    // Variances Vector6 order means
    // : rad*rad, rad*rad, rad*rad, meter*meter, meter*meter, meter*meter
    {
        gtsam::Vector Vector6(6);
        Vector6 << 1e-12, 1e-12, 1e-12, 1e-12, 1e-12, 1e-12;
        priorNoise = noiseModel::Diagonal::Variances(Vector6);
    }
    {
        gtsam::Vector Vector6(6);
        Vector6 << 1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4;
        odomNoise = noiseModel::Diagonal::Variances(Vector6);
    }
    {
        gtsam::Vector Vector6(6);
        Vector6 << 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3;
        loopNoise = noiseModel::Diagonal::Variances(Vector6);
    }
    {
        gtsam::Vector Vector6(6);
        Vector6 << M_PI*M_PI, M_PI*M_PI, M_PI*M_PI, 1e8, 1e8, 1e8;
        // Vector6 << 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3;
        largeNoise = noiseModel::Diagonal::Variances(Vector6);
    }

    float robustNoiseScore = 0.5; // constant is ok...
    gtsam::Vector robustNoiseVector6(6); 
    robustNoiseVector6 << robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore, robustNoiseScore;
    robustNoise = gtsam::noiseModel::Robust::Create(
        gtsam::noiseModel::mEstimator::Cauchy::Create(1), // optional: replacing Cauchy by DCS or GemanMcClure, but with a good front-end loop detector, Cauchy is empirically enough.
        gtsam::noiseModel::Diagonal::Variances(robustNoiseVector6)
    ); // - checked it works. but with robust kernel, map modification may be delayed (i.e,. requires more true-positive loop factors)
}


void LTslam::initOptimizer()
{
    ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.1;
    parameters.relinearizeSkip = 1; // TODO: study later
    isam = new ISAM2(parameters);
}


void LTslam::updateSessionsPoses()
{
    for(auto& _sess_pair: sessions_)
    {
        auto& _sess = _sess_pair.second;
        gtsam::Pose3 anchor_transform = isamCurrentEstimate.at<gtsam::Pose3>(genAnchorNodeIdx(_sess.index_));
        // cout << anchor_transform << endl;
        _sess.updateKeyPoses(isam, anchor_transform);
    }
} // updateSessionsPoses


void LTslam::optimizeMultisesseionGraph(bool _toOpt)
{
    if(!_toOpt)
        return;

    isam->update(gtSAMgraph, initialEstimate);
    isam->update();
    isam->update();
    isam->update();
    isam->update();
    isam->update();
    isamCurrentEstimate = isam->calculateEstimate(); // must be followed by update 

    gtSAMgraph.resize(0);
    initialEstimate.clear();

    updateSessionsPoses(); 

    if(is_display_debug_msgs_) {
        std::cout << "**********************************************" << std::endl;
        std::cout << "***** variable values after optimization *****" << std::endl;
        std::cout << std::endl;
        isamCurrentEstimate.print("Current estimate: ");
        std::cout << std::endl;
        // std::ofstream os("/home/user/Documents/catkin2021/catkin_ltmapper/catkin_ltmapper_dev/src/ltmapper/data/3d/kaist/PoseGraphExample.dot");
        // gtSAMgraph.saveGraph(os, isamCurrentEstimate);
    }
} // optimizeMultisesseionGraph


std::optional<gtsam::Pose3> LTslam::doICPVirtualRelative( // for SC loop
    Session& target_sess, Session& source_sess, 
    const int& loop_idx_target_session, const int& loop_idx_source_session)
{
    // 20201228: get relative virtual measurements using ICP (refer LIO-SAM's code)

    // parse pointclouds
    mtx.lock();
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());

    int base_key = 0; // its okay. (using the origin for sc loops' co-base) 
    int historyKeyframeSearchNum = 25; // TODO move to yaml 

    source_sess.loopFindNearKeyframesLocalCoord(cureKeyframeCloud, loop_idx_source_session, 0);
    target_sess.loopFindNearKeyframesLocalCoord(targetKeyframeCloud, loop_idx_target_session, historyKeyframeSearchNum); 
    mtx.unlock(); // unlock after loopFindNearKeyframesWithRespectTo because many new in the loopFindNearKeyframesWithRespectTo

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align pointclouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);
 
    // giseop 
    // TODO icp align with initial 

    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
        mtx.lock();
        std::cout << "  [SC loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this SC loop." << std::endl;
        mtx.unlock();
        return std::nullopt;
    } else {
        mtx.lock();
        std::cout << "  [SC loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this SC loop." << std::endl;
        mtx.unlock();
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();
    pcl::getTranslationAndEulerAngles (correctionLidarFrame, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = Pose3(Rot3::RzRyRx(0.0, 0.0, 0.0), Point3(0.0, 0.0, 0.0));

    return poseFrom.between(poseTo);
} // doICPVirtualRelative


std::optional<gtsam::Pose3> LTslam::doICPGlobalRelative( // For RS loop
    Session& target_sess, Session& source_sess, 
    const int& loop_idx_target_session, const int& loop_idx_source_session)
{
    // parse pointclouds
    mtx.lock();
    pcl::PointCloud<PointType>::Ptr cureKeyframeCloud(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr targetKeyframeCloud(new pcl::PointCloud<PointType>());

    int base_key = 0; // its okay. (using the origin for sc loops' co-base) 
    int historyKeyframeSearchNum = 25; // TODO move to yaml 

    source_sess.loopFindNearKeyframesCentralCoord(cureKeyframeCloud, loop_idx_source_session, 0);
    target_sess.loopFindNearKeyframesCentralCoord(targetKeyframeCloud, loop_idx_target_session, historyKeyframeSearchNum); 
    mtx.unlock(); // unlock after loopFindNearKeyframesWithRespectTo because many new in the loopFindNearKeyframesWithRespectTo

    // ICP Settings
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(150); // giseop , use a value can cover 2*historyKeyframeSearchNum range in meter 
    icp.setMaximumIterations(100);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    icp.setRANSACIterations(0);

    // Align pointclouds
    icp.setInputSource(cureKeyframeCloud);
    icp.setInputTarget(targetKeyframeCloud);
    pcl::PointCloud<PointType>::Ptr unused_result(new pcl::PointCloud<PointType>());
    icp.align(*unused_result);
 
    // giseop 
    // TODO icp align with initial 

    if (icp.hasConverged() == false || icp.getFitnessScore() > loopFitnessScoreThreshold) {
        mtx.lock();
        std::cout << "  [RS loop] ICP fitness test failed (" << icp.getFitnessScore() << " > " << loopFitnessScoreThreshold << "). Reject this RS loop." << std::endl;
        mtx.unlock();
        return std::nullopt;
    } else {
        mtx.lock();
        std::cout << "  [RS loop] ICP fitness test passed (" << icp.getFitnessScore() << " < " << loopFitnessScoreThreshold << "). Add this RS loop." << std::endl;
        mtx.unlock();
    }

    // Get pose transformation
    float x, y, z, roll, pitch, yaw;
    Eigen::Affine3f correctionLidarFrame;
    correctionLidarFrame = icp.getFinalTransformation();

    Eigen::Affine3f tWrong = pclPointToAffine3f(source_sess.cloudKeyPoses6D->points[loop_idx_source_session]);
    Eigen::Affine3f tCorrect = correctionLidarFrame * tWrong;// pre-multiplying -> successive rotation about a fixed frame
    pcl::getTranslationAndEulerAngles (tCorrect, x, y, z, roll, pitch, yaw);
    gtsam::Pose3 poseFrom = Pose3(Rot3::RzRyRx(roll, pitch, yaw), Point3(x, y, z));
    gtsam::Pose3 poseTo = pclPointTogtsamPose3(target_sess.cloudKeyPoses6D->points[loop_idx_target_session]);

    return poseFrom.between(poseTo);
} // doICPGlobalRelative


void LTslam::detectInterSessionSCloops() // using ScanContext
{
    auto& target_sess = sessions_.at(target_sess_idx); 
    auto& source_sess = sessions_.at(source_sess_idx);

    // Detect loop closures: Find loop edge index pairs 
    SCLoopIdxPairs_.clear();
    RSLoopIdxPairs_.clear();
    auto& target_scManager = target_sess.scManager;
    auto& source_scManager = source_sess.scManager;
    for (int source_node_idx=0; source_node_idx < int(source_scManager.polarcontexts_.size()); source_node_idx++)
    {
        std::vector<float> source_node_key = source_scManager.polarcontext_invkeys_mat_.at(source_node_idx);
        Eigen::MatrixXd source_node_scd = source_scManager.polarcontexts_.at(source_node_idx);

        auto detectResult = target_scManager.detectLoopClosureIDBetweenSession(source_node_key, source_node_scd); // first: nn index, second: yaw diff 

        int loop_idx_source_session = source_node_idx;
        int loop_idx_target_session = detectResult.first;

        if(loop_idx_target_session == -1) { // TODO using NO_LOOP_FOUND rather using -1 
            RSLoopIdxPairs_.emplace_back(std::pair(-1, loop_idx_source_session)); // -1 will be later be found (nn pose). 
            continue;
        }

        SCLoopIdxPairs_.emplace_back(std::pair(loop_idx_target_session, loop_idx_source_session));
    }

    ROS_INFO_STREAM("\033[1;32m Total " << SCLoopIdxPairs_.size() << " inter-session loops are found. \033[0m");
} // detectInterSessionSCloops


void LTslam::detectInterSessionRSloops() // using ScanContext
{
    
} // detectInterSessionRSloops


void LTslam::addAllSessionsToGraph()
{
    for(auto& _sess_pair: sessions_)    
    {
        auto& _sess = _sess_pair.second;
        initTrajectoryByAnchoring(_sess);
        addSessionToCentralGraph(_sess);   
    }
} // addAllSessionsToGraph


std::vector<std::pair<int, int>> LTslam::equisampleElements(
    const std::vector<std::pair<int, int>>& _input_pair, float _gap, int _num_sampled)
{
    std::vector<std::pair<int, int>> sc_loop_idx_pairs_sampled;

    int equisampling_counter { 0 }; 

    std::vector<int> equisampled_idx;
    for (int i=0; i<_num_sampled; i++)
        equisampled_idx.emplace_back(std::round(float(i) * _gap));

    for (auto& _idx: equisampled_idx)
        sc_loop_idx_pairs_sampled.emplace_back(_input_pair.at(_idx));

    return sc_loop_idx_pairs_sampled;
}

void LTslam::addSCloops()
{
    if(SCLoopIdxPairs_.empty()) 
        return;

    // equi-sampling sc loops 
    int num_scloops_all_found = int(SCLoopIdxPairs_.size());
    int num_scloops_to_be_added = std::min( num_scloops_all_found, kNumSCLoopsUpperBound );
    int equisampling_gap = num_scloops_all_found / num_scloops_to_be_added;

    auto sc_loop_idx_pairs_sampled = equisampleElements(SCLoopIdxPairs_, equisampling_gap, num_scloops_to_be_added);
    auto num_scloops_sampled = sc_loop_idx_pairs_sampled.size();

    // add selected sc loops 
    auto& target_sess = sessions_.at(target_sess_idx); 
    auto& source_sess = sessions_.at(source_sess_idx);

    std::vector<int> idx_added_loops; 
    idx_added_loops.reserve(num_scloops_sampled);
    #pragma omp parallel for num_threads(numberOfCores)
    for (int ith = 0; ith < num_scloops_sampled; ith++) 
    {
        auto& _loop_idx_pair = sc_loop_idx_pairs_sampled.at(ith);
        int loop_idx_target_session = _loop_idx_pair.first;
        int loop_idx_source_session = _loop_idx_pair.second;

        auto relative_pose_optional = doICPVirtualRelative(target_sess, source_sess, loop_idx_target_session, loop_idx_source_session); 

        if(relative_pose_optional) {
            mtx.lock();
            gtsam::Pose3 relative_pose = relative_pose_optional.value();
            gtSAMgraph.add( BetweenFactorWithAnchoring<gtsam::Pose3>(
                genGlobalNodeIdx(target_sess_idx, loop_idx_target_session), genGlobalNodeIdx(source_sess_idx, loop_idx_source_session),
                genAnchorNodeIdx(target_sess_idx), genAnchorNodeIdx(source_sess_idx), 
                relative_pose, robustNoise) );
            mtx.unlock();

            // debug msg (would be removed later)
            mtx.lock();
            idx_added_loops.emplace_back(loop_idx_target_session); 
            cout << "SCdetector found an inter-session edge between " 
                << genGlobalNodeIdx(target_sess_idx, loop_idx_target_session) << " and " << genGlobalNodeIdx(source_sess_idx, loop_idx_source_session) 
                << " (anchor nodes are " << genAnchorNodeIdx(target_sess_idx) << " and " << genAnchorNodeIdx(source_sess_idx) << ")" << endl;
            mtx.unlock();
        }
    }
} // addSCloops


double LTslam::calcInformationGainBtnTwoNodes(const int loop_idx_target_session, const int loop_idx_source_session)
{
    auto pose_s1 = isamCurrentEstimate.at<gtsam::Pose3>( genGlobalNodeIdx(target_sess_idx, loop_idx_target_session) ); // node: s1 is the central 
    auto pose_s2 = isamCurrentEstimate.at<gtsam::Pose3>( genGlobalNodeIdx(source_sess_idx, loop_idx_source_session) );
    auto pose_s1_anchor = isamCurrentEstimate.at<gtsam::Pose3>( genAnchorNodeIdx(target_sess_idx) );
    auto pose_s2_anchor = isamCurrentEstimate.at<gtsam::Pose3>( genAnchorNodeIdx(source_sess_idx) );

    gtsam::Pose3 hx1 = traits<gtsam::Pose3>::Compose(pose_s1_anchor, pose_s1); // for the updated jacobian, see line 60, 219, https://gtsam.org/doxygen/a00053_source.html
    gtsam::Pose3 hx2 = traits<gtsam::Pose3>::Compose(pose_s2_anchor, pose_s2); 
    gtsam::Pose3 estimated_relative_pose = traits<gtsam::Pose3>::Between(hx1, hx2); 

    gtsam::Matrix H_s1, H_s2, H_s1_anchor, H_s2_anchor;
    auto loop_factor = BetweenFactorWithAnchoring<gtsam::Pose3>(
        genGlobalNodeIdx(target_sess_idx, loop_idx_target_session), genGlobalNodeIdx(source_sess_idx, loop_idx_source_session),
        genAnchorNodeIdx(target_sess_idx), genAnchorNodeIdx(source_sess_idx), 
        estimated_relative_pose, robustNoise);
    loop_factor.evaluateError(pose_s1, pose_s2, pose_s1_anchor, pose_s2_anchor, 
                                 H_s1,    H_s2,    H_s1_anchor,    H_s2_anchor);

    gtsam::Matrix pose_s1_cov = isam->marginalCovariance(genGlobalNodeIdx(target_sess_idx, loop_idx_target_session)); // note: typedef Eigen::MatrixXd  gtsam::Matrix
    gtsam::Matrix pose_s2_cov = isam->marginalCovariance(genGlobalNodeIdx(source_sess_idx, loop_idx_source_session));

    // calc S and information gain 
    gtsam::Matrix Sy = Eigen::MatrixXd::Identity(6, 6); // measurement noise, assume fixed
    gtsam::Matrix S = Sy + (H_s1*pose_s1_cov*H_s1.transpose() + H_s2*pose_s2_cov*H_s2.transpose());
    double Sdet = S.determinant(); 
    double information_gain = 0.5 * log( Sdet / Sy.determinant());

    return information_gain;
}

void LTslam::findNearestRSLoopsTargetNodeIdx() // based-on information gain 
{
    std::vector<std::pair<int, int>> validRSLoopIdxPairs;

    for(std::size_t i=0; i<RSLoopIdxPairs_.size(); i++)
    {
        // curr query pose 
        auto rsloop_idx_pair = RSLoopIdxPairs_.at(i);
        auto rsloop_idx_source_session = rsloop_idx_pair.second;
        auto rsloop_global_idx_source_session = genGlobalNodeIdx(source_sess_idx, rsloop_idx_source_session);

        auto source_node_idx = rsloop_idx_source_session;
        auto query_pose = isamCurrentEstimate.at<gtsam::Pose3>(rsloop_global_idx_source_session);
        gtsam::Pose3 query_sess_anchor_transform = isamCurrentEstimate.at<gtsam::Pose3>(genAnchorNodeIdx(source_sess_idx));
        auto query_pose_central_coord = query_sess_anchor_transform * query_pose;

        // find nn pose idx in the target sess 
        auto& target_sess = sessions_.at(target_sess_idx); 
        std::vector<int> target_node_idxes_within_ball;
        for (int target_node_idx=0; target_node_idx < int(target_sess.nodes_.size()); target_node_idx++) {
            auto target_pose = isamCurrentEstimate.at<gtsam::Pose3>(genGlobalNodeIdx(target_sess_idx, target_node_idx));
            if( poseDistance(query_pose_central_coord, target_pose) < 10.0 ) // 10 is a hard-coding for fast test
            {
                target_node_idxes_within_ball.push_back(target_node_idx);
                // cout << "(all) RS pair detected: " << target_node_idx << " <-> " << source_node_idx << endl;    
            }
        }

        // if no nearest one, skip 
        if(target_node_idxes_within_ball.empty())
            continue;

        // selected a single one having maximum information gain  
        int selected_near_target_node_idx; 
        double max_information_gain {0.0};     
        for (int i=0; i<target_node_idxes_within_ball.size(); i++) 
        {
            auto nn_target_node_idx = target_node_idxes_within_ball.at(i);
            double this_information_gain = calcInformationGainBtnTwoNodes(nn_target_node_idx, source_node_idx);
            if(this_information_gain > max_information_gain) {
                selected_near_target_node_idx = nn_target_node_idx;
                max_information_gain = this_information_gain;
            }
        }

        // cout << "RS pair detected: " << selected_near_target_node_idx << " <-> " << source_node_idx << endl;    
        // cout << "info gain: " << max_information_gain << endl;    
             
        validRSLoopIdxPairs.emplace_back(std::pair<int, int>{selected_near_target_node_idx, source_node_idx});
    }

    // update 
    RSLoopIdxPairs_.clear();
    RSLoopIdxPairs_.resize((int)(validRSLoopIdxPairs.size()));
    std::copy( validRSLoopIdxPairs.begin(), validRSLoopIdxPairs.end(), RSLoopIdxPairs_.begin() );
}


bool LTslam::addRSloops()
{
    if( kNumRSLoopsUpperBound == 0 )
        return false;

    // find nearest target node idx 
    findNearestRSLoopsTargetNodeIdx();

    // parse RS loop src idx
    int num_rsloops_all_found = int(RSLoopIdxPairs_.size());
    if( num_rsloops_all_found == 0 )
        return false;

    int num_rsloops_to_be_added = std::min( num_rsloops_all_found, kNumRSLoopsUpperBound );
    int equisampling_gap = num_rsloops_all_found / num_rsloops_to_be_added;

    auto rs_loop_idx_pairs_sampled = equisampleElements(RSLoopIdxPairs_, equisampling_gap, num_rsloops_to_be_added);
    auto num_rsloops_sampled = rs_loop_idx_pairs_sampled.size();

    cout << "num of RS pair: " << num_rsloops_all_found << endl;         
    cout << "num of sampled RS pair: " << num_rsloops_sampled << endl;         

    // add selected rs loops 
    auto& target_sess = sessions_.at(target_sess_idx); 
    auto& source_sess = sessions_.at(source_sess_idx);

    #pragma omp parallel for num_threads(numberOfCores)
    for (int ith = 0; ith < num_rsloops_sampled; ith++) 
    {
        auto& _loop_idx_pair = rs_loop_idx_pairs_sampled.at(ith);
        int loop_idx_target_session = _loop_idx_pair.first;
        int loop_idx_source_session = _loop_idx_pair.second;

        auto relative_pose_optional = doICPGlobalRelative(target_sess, source_sess, loop_idx_target_session, loop_idx_source_session); 

        if(relative_pose_optional) {
            mtx.lock();
            gtsam::Pose3 relative_pose = relative_pose_optional.value();
            gtSAMgraph.add( BetweenFactorWithAnchoring<gtsam::Pose3>(
                genGlobalNodeIdx(target_sess_idx, loop_idx_target_session), genGlobalNodeIdx(source_sess_idx, loop_idx_source_session),
                genAnchorNodeIdx(target_sess_idx), genAnchorNodeIdx(source_sess_idx), 
                relative_pose, robustNoise) );
            mtx.unlock();

            // debug msg (would be removed later)
            mtx.lock();
            cout << "RS loop detector found an inter-session edge between " 
                << genGlobalNodeIdx(target_sess_idx, loop_idx_target_session) << " and " << genGlobalNodeIdx(source_sess_idx, loop_idx_source_session) 
                << " (anchor nodes are " << genAnchorNodeIdx(target_sess_idx) << " and " << genAnchorNodeIdx(source_sess_idx) << ")" << endl;
            mtx.unlock();
        }
    }

    return true;
} // addRSloops


void LTslam::initTrajectoryByAnchoring(const Session& _sess)
{
    int this_session_anchor_node_idx = genAnchorNodeIdx(_sess.index_);

    if(_sess.is_base_session_) {
        gtSAMgraph.add(PriorFactor<gtsam::Pose3>(this_session_anchor_node_idx, poseOrigin, priorNoise));
    } else {
        gtSAMgraph.add(PriorFactor<gtsam::Pose3>(this_session_anchor_node_idx, poseOrigin, largeNoise));
    }

    initialEstimate.insert(this_session_anchor_node_idx, poseOrigin);
} // initTrajectoryByAnchoring


void LTslam::addSessionToCentralGraph(const Session& _sess)
{
    // add nodes 
    for( auto& _node: _sess.nodes_)
    {        
        int node_idx = _node.second.idx;
        auto& curr_pose = _node.second.initial;

        int prev_node_global_idx = genGlobalNodeIdx(_sess.index_, node_idx - 1);
        int curr_node_global_idx = genGlobalNodeIdx(_sess.index_, node_idx);

        gtsam::Vector Vector6(6);
        if(node_idx == 0) { // TODO consider later if the initial node idx is not zero (but if using SC-LIO-SAM, don't care)
            // prior node 
            gtSAMgraph.add(PriorFactor<gtsam::Pose3>(curr_node_global_idx, curr_pose, priorNoise));
            initialEstimate.insert(curr_node_global_idx, curr_pose);
        } else { 
            // odom nodes 
            initialEstimate.insert(curr_node_global_idx, curr_pose);
        }
    }
    
    // add edges 
    for( auto& _edge: _sess.edges_)
    {
        int from_node_idx = _edge.second.from_idx;
        int to_node_idx = _edge.second.to_idx;
        
        int from_node_global_idx = genGlobalNodeIdx(_sess.index_, from_node_idx);
        int to_node_global_idx = genGlobalNodeIdx(_sess.index_, to_node_idx);

        gtsam::Pose3 relative_pose = _edge.second.relative;
        if( std::abs(to_node_idx - from_node_idx) == 1) {
            // odom edge (temporally consecutive)
            gtSAMgraph.add(BetweenFactor<gtsam::Pose3>(from_node_global_idx, to_node_global_idx, relative_pose, odomNoise));
            if(is_display_debug_msgs_) cout << "add an odom edge between " << from_node_global_idx << " and " << to_node_global_idx << endl;
        } else {
            // loop edge
            gtSAMgraph.add(BetweenFactor<gtsam::Pose3>(from_node_global_idx, to_node_global_idx, relative_pose, robustNoise));
            if(is_display_debug_msgs_) cout << "add a loop edge between " << from_node_global_idx << " and " << to_node_global_idx << endl;
        }
    }

}


void LTslam::loadAllSessions() 
{
    // pose 
    ROS_INFO_STREAM("\033[1;32m Load sessions' pose dasa from: " << sessions_dir_ << "\033[0m");
    for(auto& _session_entry : fs::directory_iterator(sessions_dir_)) 
    {
        std::string session_name = _session_entry.path().filename();        
        if( !isTwoStringSame(session_name, central_sess_name_) & !isTwoStringSame(session_name, query_sess_name_) ) {
            continue; // jan. 2021. currently designed for two-session ver. (TODO: be generalized for N-session co-optimization)
        }

        // save a session (read graph txt flie and load nodes and edges internally)
        int session_idx;
        if(isTwoStringSame(session_name, central_sess_name_))
            session_idx = target_sess_idx;
        else
            session_idx = source_sess_idx;

        std::string session_dir_path = _session_entry.path();

        // sessions_.emplace_back(Session(session_idx, session_name, session_dir_path, isTwoStringSame(session_name, central_sess_name_)));
        sessions_.insert( std::make_pair(session_idx, 
                                         Session(session_idx, session_name, session_dir_path, isTwoStringSame(session_name, central_sess_name_))) );

        // LTslam::num_sessions++; // incr the global index // TODO: make this private and provide incrSessionIdx
    }

    std::cout << std::boolalpha;   
    ROS_INFO_STREAM("\033[1;32m Total : " << sessions_.size() << " sessions are loaded.\033[0m");
    std::for_each( sessions_.begin(), sessions_.end(), [](auto& _sess_pair) { 
                cout << " — " << _sess_pair.second.name_ << " (is central: " << _sess_pair.second.is_base_session_ << ")" << endl; 
                } );

} // loadSession

