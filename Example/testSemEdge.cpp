#include "frame.h"
#include <algorithm>
#include <iostream>
#include <limits>

struct Bezier2D {
    cv::Point2f p0;
    cv::Point2f p1;
    cv::Point2f p2;
    cv::Point2f p3;
    bool valid = false;
};

struct EdgeWithClass {
    const Edge* edge = nullptr;
    int cls = -1;
};

static cv::Point2f bezierPoint(const Bezier2D& c, float t)
{
    const float u = 1.0f - t;
    const float b0 = u * u * u;
    const float b1 = 3.0f * u * u * t;
    const float b2 = 3.0f * u * t * t;
    const float b3 = t * t * t;
    return c.p0 * b0 + c.p1 * b1 + c.p2 * b2 + c.p3 * b3;
}

static std::vector<cv::Point2f> edgeToPoints(const Edge& edge)
{
    std::vector<cv::Point2f> pts;
    pts.reserve(edge.mvPoints.size());
    for (const auto& p : edge.mvPoints) {
        pts.emplace_back(static_cast<float>(p.x), static_cast<float>(p.y));
    }
    return pts;
}

static std::vector<float> chordLengthParameters(const std::vector<cv::Point2f>& pts)
{
    std::vector<float> t(pts.size(), 0.0f);
    if (pts.size() < 2) {
        return t;
    }

    float total = 0.0f;
    for (size_t i = 1; i < pts.size(); ++i) {
        total += cv::norm(pts[i] - pts[i - 1]);
        t[i] = total;
    }

    if (total < 1e-6f) {
        return t;
    }
    for (size_t i = 1; i < pts.size(); ++i) {
        t[i] /= total;
    }
    return t;
}

static Bezier2D fitCubicBezier2D(const std::vector<cv::Point2f>& pts)
{
    Bezier2D curve;
    if (pts.size() < 4) {
        return curve;
    }

    curve.p0 = pts.front();
    curve.p3 = pts.back();

    const std::vector<float> t = chordLengthParameters(pts);

    float sum_a2 = 0.0f;
    float sum_ab = 0.0f;
    float sum_b2 = 0.0f;
    cv::Point2f sum_abx(0.0f, 0.0f);
    cv::Point2f sum_bbx(0.0f, 0.0f);

    for (size_t i = 0; i < pts.size(); ++i) {
        const float ti = t[i];
        const float u = 1.0f - ti;
        const float a = 3.0f * u * u * ti;
        const float b = 3.0f * u * ti * ti;
        const cv::Point2f b_i = pts[i] - curve.p0 * (u * u * u) - curve.p3 * (ti * ti * ti);

        sum_a2 += a * a;
        sum_ab += a * b;
        sum_b2 += b * b;
        sum_abx += b_i * a;
        sum_bbx += b_i * b;
    }

    const float det = sum_a2 * sum_b2 - sum_ab * sum_ab;
    if (std::fabs(det) < 1e-6f) {
        return curve;
    }

    const float inv_det = 1.0f / det;
    curve.p1 = (sum_abx * sum_b2 - sum_bbx * sum_ab) * inv_det;
    curve.p2 = (sum_bbx * sum_a2 - sum_abx * sum_ab) * inv_det;
    curve.valid = true;
    return curve;
}

static float pointToBezierDistance(const Bezier2D& curve, const cv::Point2f& p, int samples)
{
    float min_dist = std::numeric_limits<float>::max();
    for (int i = 0; i <= samples; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(samples);
        const cv::Point2f q = bezierPoint(curve, t);
        min_dist = std::min((double)min_dist, cv::norm(p - q));
    }
    return min_dist;
}

static float edgeToBezierResidual(const Edge& edge, const Bezier2D& curve)
{
    if (!curve.valid || edge.mvPoints.empty()) {
        return std::numeric_limits<float>::max();
    }

    const int samples = std::max(20, std::min(120, static_cast<int>(edge.mvPoints.size())));
    float sum = 0.0f;
    for (const auto& p : edge.mvPoints) {
        const cv::Point2f pt(static_cast<float>(p.x), static_cast<float>(p.y));
        sum += pointToBezierDistance(curve, pt, samples);
    }
    return sum / static_cast<float>(edge.mvPoints.size());
}

static cv::Point2f edgeCentroid(const Edge& edge)
{
    cv::Point2f centroid(0.0f, 0.0f);
    if (edge.mvPoints.empty()) {
        return centroid;
    }
    for (const auto& p : edge.mvPoints) {
        centroid.x += static_cast<float>(p.x);
        centroid.y += static_cast<float>(p.y);
    }
    centroid.x /= static_cast<float>(edge.mvPoints.size());
    centroid.y /= static_cast<float>(edge.mvPoints.size());
    return centroid;
}

static cv::Matx33f skewSymmetric(const cv::Vec3f& t)
{
    return cv::Matx33f(0.0f, -t[2], t[1], t[2], 0.0f, -t[0], -t[1], t[0], 0.0f);
}

static float pointLineDistance(const cv::Vec3f& line, const cv::Point2f& p)
{
    const float a = line[0];
    const float b = line[1];
    const float c = line[2];
    const float denom = std::sqrt(a * a + b * b);
    if (denom < 1e-6f) {
        return std::numeric_limits<float>::max();
    }
    return std::fabs(a * p.x + b * p.y + c) / denom;
}

static std::vector<EdgeWithClass> collectEdgesWithClass(const Frame& frame)
{
    std::vector<EdgeWithClass> edges;
    for (const auto& patch : frame.sem_patches) {
        for (const auto& edge : patch.edges) {
            edges.push_back(EdgeWithClass{&edge, patch.cls});
        }
    }
    return edges;
}

static void runBezier2DMatchDemo(const Frame& frame0,
                                const Frame& frame1,
                                const cv::Matx33f& K,
                                const cv::Matx33f& R_10,
                                const cv::Vec3f& t_10)
{
    const auto edges0 = collectEdgesWithClass(frame0);
    const auto edges1 = collectEdgesWithClass(frame1);
    if (edges0.empty() || edges1.empty()) {
        std::cout << "No edges for matching." << std::endl;
        return;
    }

    const cv::Matx33f Kinv = K.inv();
    const cv::Matx33f F = Kinv.t() * skewSymmetric(t_10) * R_10 * Kinv;

    const float epipolar_thd = 2.0f;
    const int min_points = 8;

    for (const auto& e0 : edges0) {
        if (static_cast<int>(e0.edge->mvPoints.size()) < min_points) {
            continue;
        }

        const std::vector<cv::Point2f> pts0 = edgeToPoints(*e0.edge);
        const Bezier2D curve0 = fitCubicBezier2D(pts0);
        if (!curve0.valid) {
            continue;
        }

        const cv::Point2f c0 = edgeCentroid(*e0.edge);
        const cv::Vec3f x0(c0.x, c0.y, 1.0f);
        const cv::Vec3f line1 = F * x0;

        float best_res = std::numeric_limits<float>::max();
        int best_idx = -1;

        for (size_t j = 0; j < edges1.size(); ++j) {
            const auto& e1 = edges1[j];
            if (e1.cls != e0.cls) {
                continue;
            }
            if (static_cast<int>(e1.edge->mvPoints.size()) < min_points) {
                continue;
            }

            const cv::Point2f c1 = edgeCentroid(*e1.edge);
            const float epi_dist = pointLineDistance(line1, c1);
            if (epi_dist > epipolar_thd) {
                continue;
            }

            const float residual = edgeToBezierResidual(*e1.edge, curve0);
            if (residual < best_res) {
                best_res = residual;
                best_idx = static_cast<int>(j);
            }
        }

        if (best_idx >= 0) {
            std::cout << "Edge cls " << e0.cls
                      << ": best residual = " << best_res
                      << " (match index " << best_idx << ")" << std::endl;
        }
    }
}


cv::Mat visualizeEdgesRaw(cv::Mat imgBackGround, std::vector<Edge>& edges)
{
    cv::Mat image_viz = imgBackGround.clone();

    if(image_viz.channels() == 3){
        cv::cvtColor(image_viz, image_viz, cv::COLOR_BGR2GRAY);
        for (int y = 0; y < image_viz.rows; ++y) {
            for (int x = 0; x < image_viz.cols; ++x) {
                // 获取当前像素值
                uchar& pixel = image_viz.at<uchar>(y, x);
                
                // 将像素值减半，确保不会小于0
                pixel = static_cast<uchar>(pixel / 1.3);
            }
        }
    }
    cv::cvtColor(image_viz, image_viz, cv::COLOR_GRAY2BGR);
    int maxLabel = 0;
    for(int i = 0; i < edges.size(); ++i){
        for(int j = 0; j < edges[i].mvPoints.size(); ++j){
            orderedEdgePoint curr = edges[i].mvPoints[j];
            cv::circle(image_viz, cv::Point(curr.x, curr.y), 1, cv::Scalar(0,200,0), 1, cv::LINE_AA);
        }
    }
    return image_viz;
}

cv::Mat visualizeEdgesOrient(cv::Mat imgBackGround, std::vector<Edge>& edges)
{
    cv::Mat hsvTabel(1,180,CV_8UC3,cv::Scalar::all(0));
    for(int i = 0; i < 180; i++){
        hsvTabel.at<cv::Vec3b>(0,i) = cv::Vec3b(i,255,255);
    }
    cv::cvtColor(hsvTabel, hsvTabel, cv::COLOR_HSV2BGR);

    cv::Mat image_viz = imgBackGround.clone();

    if(image_viz.channels() == 3){
        cv::cvtColor(image_viz, image_viz, cv::COLOR_BGR2GRAY);
        for (int y = 0; y < image_viz.rows; ++y) {
            for (int x = 0; x < image_viz.cols; ++x) {
                // 获取当前像素值
                uchar& pixel = image_viz.at<uchar>(y, x);
                
                // 将像素值减半，确保不会小于0
                pixel = static_cast<uchar>(pixel / 1.3);
            }
        }
    }
    cv::cvtColor(image_viz, image_viz, cv::COLOR_GRAY2BGR);
    int maxLabel = 0;
    for(int i = 0; i < edges.size(); ++i){
        for(int j = 0; j < edges[i].mvPoints.size(); ++j){
            orderedEdgePoint curr = edges[i].mvPoints[j];
            float angle = curr.imgGradAngle;
            int ratio = int(angle / 2.0);
            cv::Vec3b color = hsvTabel.at<cv::Vec3b>(0,ratio);
            cv::circle(image_viz, cv::Point(curr.x, curr.y), 1, color, 1, cv::LINE_AA);
        }
    }
    return image_viz;
}

cv::Mat visualizeEdges(cv::Mat imgBackGround, std::vector<Edge>& edges)
{
    cv::RNG rng(66);
    cv::Mat image_viz = imgBackGround.clone();

    if(image_viz.channels() == 3){
        cv::cvtColor(image_viz, image_viz, cv::COLOR_BGR2GRAY);
        for (int y = 0; y < image_viz.rows; ++y) {
            for (int x = 0; x < image_viz.cols; ++x) {
                // 获取当前像素值
                uchar& pixel = image_viz.at<uchar>(y, x);
                
                // 将像素值减半，确保不会小于0
                pixel = static_cast<uchar>(pixel / 1.3);
            }
        }
    }
    cv::cvtColor(image_viz, image_viz, cv::COLOR_GRAY2BGR);
    int maxLabel = 0;
    for(int i = 0; i < edges.size(); ++i){
        int b = rng.uniform(0, 255);
        int g = rng.uniform(0, 255);
        int r = rng.uniform(0, 255);
        cv::Vec3b color = cv::Vec3b(b,g,r);
        //if(mvEdgeClusters[i].mvPoints.size()<15) continue;
        for(int j = 0; j < edges[i].mvPoints.size(); ++j){
            orderedEdgePoint curr = edges[i].mvPoints[j];
            image_viz.at<cv::Vec3b>(curr.y,curr.x) = color;
            cv::circle(image_viz, cv::Point(curr.x, curr.y), 1, color, 1, cv::LINE_AA);
        }
    }
    return image_viz;
}

cv::Mat visualizeEdgesOrganized(cv::Mat imgBackGround, std::vector<Edge>& edges)
{
    cv::Mat valueTabel(256,1,CV_8UC1);
    cv::Mat ColorTabel;
    for(int i = 0; i<256; i++){
        valueTabel.at<uint8_t>(i,0)=i;
    }
    cv::applyColorMap(valueTabel,ColorTabel,cv::COLORMAP_PARULA);
    cv::Mat image_viz = imgBackGround.clone();
    if(image_viz.channels() == 3){
        cv::cvtColor(image_viz, image_viz, cv::COLOR_BGR2GRAY);
        for (int y = 0; y < image_viz.rows; ++y) {
            for (int x = 0; x < image_viz.cols; ++x) {
                // 获取当前像素值
                uchar& pixel = image_viz.at<uchar>(y, x);
                
                // 将像素值减半，确保不会小于0
                pixel = static_cast<uchar>(pixel / 1.3);
            }
        }
    }
    cv::cvtColor(image_viz, image_viz, cv::COLOR_GRAY2BGR);
    int maxLabel = 0;
    for(int i = 0; i < edges.size(); ++i){
        //if(mvEdgeClusters[i].mvPoints.size()<15) continue;
        for(int j = 0; j < edges[i].mvPoints.size(); ++j){
            float proportion = float(j)/float(edges[i].mvPoints.size());
            int idx = cvRound(proportion * 255);
            orderedEdgePoint curr = edges[i].mvPoints[j];
            cv::Vec3b color = ColorTabel.at<cv::Vec3b>(idx,0);
            image_viz.at<cv::Vec3b>(curr.y,curr.x) = color;
            cv::circle(image_viz, cv::Point(curr.x, curr.y), 1, color, 1, cv::LINE_AA);
        }
    }
    return image_viz;
}

int main(int argc, char** argv) {
    cv::Mat color = cv::imread("/home/wyyaa123/demo/datasets/Real/D435I/2026-04-24-08-20-35/camera_infra1_image_rect_raw/1777018894246603012.png", cv::IMREAD_COLOR);
    cv::Mat depth = cv::imread("/home/wyyaa123/demo/datasets/Real/D435I/2026-04-24-08-20-35/depth/1777018894246603012.png", cv::IMREAD_UNCHANGED);
    cv::Mat semantic = cv::imread("/home/wyyaa123/demo/datasets/Real/D435I/2026-04-24-08-20-35/camera_infra1_semantic/merged/1777018894246603012.png", cv::IMREAD_GRAYSCALE);

    cv::Mat color2 = color.clone();
    cv::Mat depth2 = depth.clone();
    cv::Mat semantic2 = semantic.clone();

    Frame frame;
    frame.color = color;
    cv::cvtColor(color, frame.gray, cv::COLOR_BGR2GRAY);

    Frame frame2;
    frame2.color = color2;
    cv::cvtColor(color2, frame2.gray, cv::COLOR_BGR2GRAY);

    float fx = 394.7194240503013;
    float fy = 394.4889497892937;
    float cx = 323.98920121606034;
    float cy = 237.27629836778792;

    float depth_scale = 1000.0; // 如果深度是毫米，需要除1000

    cv::Mat canny;
    cv::Mat sem = semantic.clone();
    frame.semToPatches(sem, depth, fx, fy, cx, cy);
    cv::Canny(frame.color, canny, 50, 150, 3, true);

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    frame.genPatchSemEdge(canny);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time taken: " << elapsed_seconds.count() * 1000 << " milliseconds" << std::endl;

    cv::Mat canny2;
    cv::Mat sem2 = semantic2.clone();
    frame2.semToPatches(sem2, depth2, fx, fy, cx, cy);
    cv::Canny(frame2.color, canny2, 50, 150, 3, true);
    frame2.genPatchSemEdge(canny2);

    cv::Matx33f K(fx, 0.0f, cx,
                 0.0f, fy, cy,
                 0.0f, 0.0f, 1.0f);
    cv::Matx33f R_10 = cv::Matx33f::eye();
    cv::Vec3f t_10(0.0f, 0.0f, 0.0f);
    runBezier2DMatchDemo(frame, frame2, K, R_10, t_10);

    std::vector<Edge> edges;
    for (int i = 0; i < frame.sem_patches.size(); ++i) {
        edges.insert(edges.end(), frame.sem_patches[i].edges.begin(), frame.sem_patches[i].edges.end());
    }

    cv::Mat imgSeg = visualizeEdges(frame.color, edges);
    cv::Mat imgSeg_2 = visualizeEdgesOrganized(frame.color, edges);
    cv::Mat imgSeg_3 = visualizeEdgesRaw(frame.color, edges);
    cv::Mat imgSeg_4 = visualizeEdgesOrient(frame.color, edges);

    // 创建一个大图像来存放四宫格
    cv::Mat collage(frame.gray.rows * 2, frame.gray.cols * 2, frame.color.type());
    
    // 将四个图像放置到四宫格的相应位置
    // 左上角
    imgSeg_3.copyTo(collage(cv::Rect(0, 0, frame.gray.cols, frame.gray.rows)));
    // 右上角
    imgSeg_4.copyTo(collage(cv::Rect(frame.gray.cols, 0, frame.gray.cols, frame.gray.rows)));
    // 左下角
    imgSeg.copyTo(collage(cv::Rect(0, frame.gray.rows, frame.gray.cols, frame.gray.rows)));
    // 右下角
    imgSeg_2.copyTo(collage(cv::Rect(frame.gray.cols, frame.gray.rows, frame.gray.cols, frame.gray.rows)));
    
    cv::imwrite("edges_visualization.png", collage);

    // 显示四宫格
    cv::imshow("O-EDGE", collage);
    cv::waitKey(0); // 等待按键



    cv::destroyAllWindows();

    return 0;
}
