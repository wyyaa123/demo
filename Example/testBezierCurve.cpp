#include "bezierCurve.h"
#include "frame.h"


int main(int argc, char** argv) {
    cv::Mat color = cv::imread("/home/wyyaa123/demo/datasets/Real/D435I/2026-04-24-08-20-35/camera_infra1_image_rect_raw/1777018836360285044.png", cv::IMREAD_COLOR);
    cv::Mat semantic = cv::imread("/home/wyyaa123/demo/datasets/Real/D435I/2026-04-24-08-20-35/camera_infra1_semantic/merged/1777018836360285044.png", cv::IMREAD_GRAYSCALE);
    
    Frame frame;
    frame.color = color;
    cv::cvtColor(color, frame.gray, cv::COLOR_BGR2GRAY);

    float fx = 394.7194240503013;
    float fy = 394.4889497892937;
    float cx = 323.98920121606034;
    float cy = 237.27629836778792;
    
    frame.genPatchSemEdge(semantic);
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    // cv::namedWindow("Viz", cv::WINDOW_NORMAL);
    for (const auto& patch : frame.sem_patches) {
        for (const auto& edge: patch.edges) {
            // cv::Mat viz = cv::Mat::zeros(frame.gray.size(), CV_8UC3);
            // for (const auto& pt : edge.mvPoints) {
            //     cv::circle(viz, cv::Point(pt.x, pt.y), 2, cv::Scalar(0, 255, 0), -1);
            // }
            std::vector<std::vector<orderedEdgePoint>> beziers = fitBezierAdaptive(edge.mvPoints, 3);
            // printf("Fitted %zu Bezier curves for edge with %zu points\n", beziers.size(), edge.mvPoints.size());
            // std::cout << "Control Points for Edge:" << std::endl;
            // for (const auto& bezier : beziers) {
                // printf("Bezier curve with %zu control points:\n", bezier.size());
                
            //     for (const auto& pt : bezier) {
            //         std::cout << "(" << pt.x << ", " << pt.y << ")" << std::endl;
            //     }
            // }
            // cv::imshow("Viz", viz);
            // cv::waitKey(0);
            // printf("------------------------------------------------\n");
        }
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> elapsed_seconds = end - start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " milliseconds" << std::endl;

    return 0;
}