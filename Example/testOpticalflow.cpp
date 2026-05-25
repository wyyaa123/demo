#include "frame.h"
#include <iostream>
#include <vector>

int main(int argc, char** argv) 
{

    if (argc < 5)
    {
        std::cerr << "Usage: " << argv[0] << " <image1> <image2> <semantic1> <semantic2>" << std::endl;
        return 1;
    }

    cv::Mat image1 = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    cv::Mat image2 = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
    cv::Mat semantic1 = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
    cv::Mat semantic2 = cv::imread(argv[4], cv::IMREAD_GRAYSCALE);

    if (image1.empty() || image2.empty() || semantic1.empty() || semantic2.empty())
    {
        std::cerr << "Failed to load input images." << std::endl;
        return 1;
    }

    Frame frame1, frame2;
    frame1.gray = image1;
    frame2.gray = image2;
    
    frame1.genPatchSemEdge(semantic1);
    frame2.genPatchSemEdge(semantic2);

    struct EdgePointRef
    {
        size_t patch_idx;
        size_t edge_idx;
        size_t point_idx;
    };

    std::vector<cv::Point2f> pts1;
    std::vector<EdgePointRef> refs;
    pts1.reserve(10000);
    refs.reserve(10000);

    for (size_t p = 0; p < frame1.sem_patches.size(); ++p)
    {
        const auto &patch = frame1.sem_patches[p];
        for (size_t e = 0; e < patch.edges.size(); ++e)
        {
            const auto &edge = patch.edges[e];
            for (size_t i = 0; i < edge.mvPoints.size(); ++i)
            {
                const auto &pt = edge.mvPoints[i];
                pts1.emplace_back(static_cast<float>(pt.x), static_cast<float>(pt.y));
                refs.push_back({p, e, i});
            }
        }
    }

    if (pts1.empty())
    {
        std::cout << "No edge points in frame1.sem_patches." << std::endl;
        return 0;
    }

    std::vector<cv::Point2f> pts2;
    std::vector<uchar> status;
    cv::Mat flow;
    cv::calcOpticalFlowFarneback(frame1.gray, frame2.gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

    pts2.resize(pts1.size());
    status.assign(pts1.size(), 0);

    const int width = frame1.gray.cols;
    const int height = frame1.gray.rows;
    for (size_t i = 0; i < pts1.size(); ++i)
    {
        const cv::Point2f &pt = pts1[i];
        const int x = cvRound(pt.x);
        const int y = cvRound(pt.y);
        if (x < 0 || x >= width || y < 0 || y >= height)
            continue;

        const cv::Point2f flow_xy = flow.at<cv::Point2f>(y, x);
        const cv::Point2f next_pt = pt + flow_xy;
        if (next_pt.x < 0.0f || next_pt.x >= static_cast<float>(width) ||
            next_pt.y < 0.0f || next_pt.y >= static_cast<float>(height))
            continue;

        pts2[i] = next_pt;
        status[i] = 1;
    }

    std::vector<std::vector<std::vector<cv::Point2f>>> tracked_points;
    tracked_points.resize(frame1.sem_patches.size());
    for (size_t p = 0; p < frame1.sem_patches.size(); ++p)
    {
        tracked_points[p].resize(frame1.sem_patches[p].edges.size());
        for (size_t e = 0; e < frame1.sem_patches[p].edges.size(); ++e)
        {
            tracked_points[p][e].resize(frame1.sem_patches[p].edges[e].mvPoints.size(), cv::Point2f(-1.0f, -1.0f));
        }
    }

    int valid_count = 0;
    for (size_t i = 0; i < status.size(); ++i)
    {
        if (!status[i])
            continue;

        const auto &ref = refs[i];
        tracked_points[ref.patch_idx][ref.edge_idx][ref.point_idx] = pts2[i];
        ++valid_count;
    }

    std::cout << "Tracked points: " << valid_count << " / " << pts1.size() << std::endl;

    const cv::Scalar kColors[] = {
        cv::Scalar(0, 0, 255),
        cv::Scalar(0, 255, 0),
        cv::Scalar(255, 0, 0),
        cv::Scalar(0, 255, 255),
        cv::Scalar(255, 0, 255),
        cv::Scalar(255, 255, 0),
        cv::Scalar(255, 128, 0),
        cv::Scalar(128, 255, 0)
    };
    const size_t color_count = sizeof(kColors) / sizeof(kColors[0]);

    auto draw_sem_edges = [&](const Frame &frame, const std::string &out_name)
    {
        cv::Mat vis;
        cv::cvtColor(frame.gray, vis, cv::COLOR_GRAY2BGR);

        for (size_t p = 0; p < frame.sem_patches.size(); ++p)
        {
            const cv::Scalar color = kColors[p % color_count];
            const auto &patch = frame.sem_patches[p];
            for (size_t e = 0; e < patch.edges.size(); ++e)
            {
                const auto &edge = patch.edges[e];
                for (size_t i = 0; i < edge.mvPoints.size(); ++i)
                {
                    const auto &pt = edge.mvPoints[i];
                    cv::circle(vis,
                               cv::Point2f(static_cast<float>(pt.x), static_cast<float>(pt.y)),
                               1, color, -1, cv::LINE_AA);
                }
            }
        }

        if (!cv::imwrite(out_name, vis))
        {
            std::cerr << "Failed to write " << out_name << std::endl;
        }
        else
        {
            std::cout << "Saved edge visualization to " << out_name << std::endl;
        }
    };

    draw_sem_edges(frame1, "edges_frame1.png");
    draw_sem_edges(frame2, "edges_frame2.png");

    cv::Mat vis;
    cv::cvtColor(frame2.gray, vis, cv::COLOR_GRAY2BGR);

    for (size_t p = 0; p < tracked_points.size(); ++p)
    {
        const cv::Scalar color = kColors[p % color_count];
        for (size_t e = 0; e < tracked_points[p].size(); ++e)
        {
            for (size_t i = 0; i < tracked_points[p][e].size(); ++i)
            {
                const cv::Point2f &pt = tracked_points[p][e][i];
                if (pt.x < 0.0f || pt.y < 0.0f)
                    continue;
                cv::circle(vis, pt, 1, color, -1, cv::LINE_AA);
            }
        }
    }

    if (!cv::imwrite("tracked_edges.png", vis))
    {
        std::cerr << "Failed to write tracked_edges.png" << std::endl;
    }
    else
    {
        std::cout << "Saved tracked visualization to tracked_edges.png" << std::endl;
    }

    


    return 0;
}
