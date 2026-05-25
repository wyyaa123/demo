#include "frame.h"
#include <iostream>


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
    cv::Mat color = cv::imread("/home/wyyaa123/demo/datasets/Real/D435I/2026-04-24-08-20-35/camera_infra1_image_rect_raw/1777018836360285044.png", cv::IMREAD_COLOR);
    cv::Mat depth = cv::imread("/home/wyyaa123/demo/datasets/Real/D435I/2026-04-24-08-20-35/depth/1777018836360285044.png", cv::IMREAD_UNCHANGED);
    cv::Mat semantic = cv::imread("/home/wyyaa123/demo/datasets/Real/D435I/2026-04-24-08-20-35/camera_infra1_semantic/merged/1777018836360285044.png", cv::IMREAD_GRAYSCALE);

    Frame frame;
    frame.color = color;
    cv::cvtColor(color, frame.gray, cv::COLOR_BGR2GRAY);

    float fx = 394.7194240503013;
    float fy = 394.4889497892937;
    float cx = 323.98920121606034;
    float cy = 237.27629836778792;

    float depth_scale = 1000.0; // 如果深度是毫米，需要除1000

    // cv::Mat canny;
    // cv::Mat sem = semantic.clone();
    // frame.semToPatches(sem, depth, fx, fy, cx, cy);
    // cv::Canny(frame.color, canny, 50, 150, 3, true);

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    frame.genPatchSemEdge(semantic);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::cout << "Time taken: " << elapsed_seconds.count() * 1000 << " milliseconds" << std::endl;

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
