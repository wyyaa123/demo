#pragma once
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <string>

#include <tbb/tbb.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

#include "patch.h"
#include "edgeCluster.h"

class Frame
{
public:
    Frame(int id) : frame_id(id) {}
    Frame() : frame_id(-1) {}

    void semToPatches(const cv::Mat &semantic, const cv::Mat &depth = cv::Mat(), float fx = 0, float fy = 0, float cx = 0, float cy = 0);
    // void genPatchSemEdge(const cv::Mat& canny);
    void genPatchSemEdge(const cv::Mat &semantic);

    int frame_id;

    cv::Mat gray;
    cv::Mat color;
    cv::Mat canny;

    cv::Mat magnitude;
    cv::Mat angle;
    
    std::vector<Patch> sem_patches; // Map from label to patches <sem, patches>
private:
    void preprocess(cv::Mat& canny);
    Edge postprocess(const Edge& edge, int step, double curvatureDiffThresh, int removeRadius = 0);
    std::vector<EdgeCluster> regionGrowthClusteringOCanny(const cv::Mat &pre_canny, float angle_Thres, const cv::Point &offset = cv::Point());
};

inline float calcAngleBias(float angle_1, float angle_2);
inline double computeCurvature(const orderedEdgePoint &p0, const orderedEdgePoint &p1, const orderedEdgePoint &p2);
