#include "frame.h"

void Frame::semToPatches(const cv::Mat &semantic, const cv::Mat &depth, float fx, float fy, float cx, float cy)
{
    for (int i = 1; i < 4; ++i)
    {
        cv::Mat filtered_semantic = cv::Mat::zeros(semantic.size(), CV_8UC1);
        filtered_semantic.setTo(255, semantic == i);

        if (cv::countNonZero(filtered_semantic) == 0)
            continue;

        // cv::Mat close_kernel = cv::Mat::ones(3, 3, CV_8UC1);
        // cv::morphologyEx(filtered_semantic, filtered_semantic, cv::MORPH_CLOSE, close_kernel);

        // cv::Mat open_kernel = cv::Mat::ones(3, 3, CV_8UC1);
        // cv::morphologyEx(filtered_semantic, filtered_semantic, cv::MORPH_OPEN, open_kernel);

        cv::Mat label, stats, centroids;
        cv::connectedComponentsWithStats(filtered_semantic, label, stats, centroids, 4, CV_16U);

        for (int j = 1; j < centroids.rows; j++)
        {
            // Exclude background
            Patch patch;
            patch.area = stats.at<int>(j, cv::CC_STAT_AREA);
            patch.top = stats.at<int>(j, cv::CC_STAT_TOP);   // y
            patch.left = stats.at<int>(j, cv::CC_STAT_LEFT); // x
            patch.width = stats.at<int>(j, cv::CC_STAT_WIDTH);
            patch.height = stats.at<int>(j, cv::CC_STAT_HEIGHT);
            patch.cls = i;

            patch.color = cv::Vec3b(0, 165, 255); // Orange for class 1, White for class 2

            // Clean redundant patches.
            // 1. Remove patches with too few points.
            if (patch.area < areaThd)
                continue;

            // 2. Remove patches with too big W-H ratio.
            double WHratio = std::max(patch.width, patch.height) / std::min(patch.width, patch.height);
            if (WHratio > 10)
                continue;

            const uint16_t *label_ptr = label.ptr<uint16_t>();
            const int label_step = label.step / sizeof(uint16_t);

            // Count every patch.
            patch.centroid = Eigen::Vector3f::Zero();
            for (int x = patch.left; x < label.cols; ++x)
            {
                for (int y = patch.top; y < label.rows; ++y)
                {
                    const uint16_t cur_label = label_ptr[y * label_step + x];
                    if (cur_label == j)
                    {
                        if (!depth.empty())
                        {
                            ushort d = depth.at<ushort>(y, x);
                            if (d == 0)
                                continue; // 无效深度

                            float z = d / 1000.0f; // Convert from mm to m
                            float x3d = (x - cx) * z / fx;
                            float y3d = (y - cy) * z / fy;
                            patch.points.emplace_back(x3d, y3d, z);
                            patch.centroid += Eigen::Vector3f(x3d, y3d, z);
                        }
                        else
                        {
                            throw std::runtime_error("Depth map is required for 3D point generation.");
                            // std::cerr << "Depth map is required for 3D point generation." << std::endl;
                            return;
                        }
                    }
                }
            }
            patch.centroid /= static_cast<float>(patch.points.size());

            patch.cov = Eigen::Matrix3f::Zero();
            for (const auto &p : patch.points)
            {
                Eigen::Vector3f centered = p - patch.centroid;
                patch.cov += centered * centered.transpose();
            }
            patch.cov /= static_cast<float>(patch.points.size());
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(patch.cov);
            patch.ev = es.eigenvectors();

            // 保持右手坐标系（行列式为正）
            if (patch.ev.determinant() < 0)
                patch.ev.col(0) = -patch.ev.col(0);

            {
                cv::Mat mask = cv::Mat::zeros(semantic.size(), CV_8UC1);
                mask = mask.setTo(255, label == j);

                int dilationSize = 2;
                cv::Mat dilationElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1), cv::Point(dilationSize, dilationSize));
                cv::dilate(mask, mask, dilationElement);

                patch.mask = mask;
                std::vector<cv::Point> nz_points;
                cv::findNonZero(mask, nz_points);
                patch.roi = nz_points.empty() ? cv::Rect() : cv::boundingRect(nz_points);
            }

            sem_patches.emplace_back(patch);
        }
    }
}

void Frame::preprocess(cv::Mat &canny)
{
    for (int i = 0; i < canny.rows; ++i)
    {
        uint8_t *current = canny.ptr<uint8_t>(i);
        uint8_t *above = i > 0 ? canny.ptr<uint8_t>(i - 1) : nullptr;
        uint8_t *below = i < canny.rows - 1 ? canny.ptr<uint8_t>(i + 1) : nullptr;

        for (int j = 0; j < canny.cols; ++j)
        {
            if (current[j] == 0)
                continue; // 跳过非边缘点

            int left = j > 0 ? current[j - 1] : 0;
            int right = j < canny.cols - 1 ? current[j + 1] : 0;
            int up = above ? above[j] : 0;
            int down = below ? below[j] : 0;

            bool connected = (left > 0 && up > 0) || (right > 0 && up > 0) ||
                             (left > 0 && down > 0) || (right > 0 && down > 0);

            if (connected)
            {
                current[j] = 0;
            }
        }
    }
}

float calcAngleBias(float angle_1, float angle_2)
{
    float res = fabs(angle_1 - angle_2);
    if (res > 180)
    {
        res = 360 - res;
    }
    return res;
}

std::vector<EdgeCluster> Frame::regionGrowthClusteringOCanny(const cv::Mat &pre_canny, float angle_Thres, const cv::Point &offset)
{
    std::vector<EdgeCluster> edge_clusters;
    //-- 判断有没有遍历到当前点的矩阵 0 表示没有遍历到，1 表示遍历到
    cv::Mat visitedMat(pre_canny.size(), CV_8UC1, cv::Scalar::all(0));

    //-- 预定义图像的指针
    uint8_t *canny_ptr = pre_canny.data;
    uint8_t *visited_ptr = visitedMat.data;
    const float *angle_ptr = angle.ptr<float>();
    const int canny_step = static_cast<int>(pre_canny.step);
    const int visited_step = static_cast<int>(visitedMat.step);
    const int angle_step = static_cast<int>(angle.step / sizeof(float));
    const int width = pre_canny.cols;
    const int height = pre_canny.rows;

    static const int kDx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    static const int kDy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            //-- 如果该点是canny得到的边缘且当前没遍历过这个点，就以该点为初始，区域生长地找到一个区域
            if (visited_ptr[y * visited_step + x] != 0 || canny_ptr[y * canny_step + x] != 255)
                continue;

            //-- 创建一个新的聚类
            std::vector<edgePoint> current_cluster;
            current_cluster.reserve(128);
            //-- 更新当前位置的访问状态
            visited_ptr[y * visited_step + x] = 1;
            //-- 每个 cluster 中的点均会获得从0开始的编号
            int point_id = 0;

            //-- 创建一个边缘点，由于是该边缘的第一个点，设置ID为0，没有父节点，设为-1
            edgePoint curr_edge_point(cv::Point(x, y), point_id, -1);
            curr_edge_point.isRoot = true;
            point_id++;

            //-- 广度优先的区域生长方法的遍历队列（vector + head 代替 queue 提升性能）
            std::vector<edgePoint> open_list;
            open_list.reserve(256);
            open_list.push_back(curr_edge_point);
            size_t head = 0;

            //-- 基于图搜索（深度优先或广度优先）的区域生长遍历
            while (head < open_list.size())
            {
                //-- vector 使用 head 索引获取当前元素
                const edgePoint current_point = open_list[head++];

                const int cx = current_point.pixel.x;
                const int cy = current_point.pixel.y;

                // if(canny.at<uint8_t>(current_point.pixel.y, current_point.pixel.x)==0) continue;

                current_cluster.push_back(current_point);

                //-- 扩展当前这个像素的相邻区域，得到新的满足条件的像素加入open_list
                const float curr_angle = angle_ptr[(cy + offset.y) * angle_step + (cx + offset.x)];

                //-- 向 8 邻域进行扩展
                for (int k = 0; k < 8; ++k)
                {
                    const int nx = cx + kDx[k];
                    const int ny = cy + kDy[k];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                    {
                        continue;
                    }

                    const int visited_idx = ny * visited_step + nx;
                    if (visited_ptr[visited_idx] != 0)
                    {
                        continue;
                    }

                    if (canny_ptr[ny * canny_step + nx] != 255)
                    {
                        continue;
                    }

                    const float neigh_angle = angle_ptr[(ny + offset.y) * angle_step + (nx + offset.x)];

                    float angle_bias = calcAngleBias(neigh_angle, curr_angle);
                    if (angle_bias >= angle_Thres)
                    {
                        continue;
                    }

                    //-- 角度满足要求的点已经是可以聚类的点，因此可以赋ID，放到 queue 中继续扩展
                    visited_ptr[visited_idx] = 1;
                    //-- 该点由 current_point 扩展得到，因此其父 ID 为 current_point.point_id
                    open_list.emplace_back(cv::Point(nx, ny), point_id, current_point.point_id);
                    point_id++;
                }
            }

            //-- region growth 结束，一组聚类生成，判断聚类大小，只取大序列
            if (current_cluster.size() > 10)
            {
                //-- 结束一组聚类，此时得到一个完整的current_cluster
                edge_clusters.emplace_back(std::move(current_cluster));
            }
        }
    }

    return edge_clusters;
}

// void Frame::genPatchSemEdge(const cv::Mat &canny)
// {
//     cv::Mat grad_x, grad_y;
//     cv::Scharr(gray, grad_x, CV_32F, 1, 0);
//     cv::Scharr(gray, grad_y, CV_32F, 0, 1);

//     magnitude.create(gray.size(), CV_32F);
//     angle.create(gray.size(), CV_32F);
//     cv::cartToPolar(grad_x, grad_y, magnitude, angle, true);

//     const float *angle_ptr = angle.ptr<float>();
//     const int angle_step = angle.step / sizeof(float);
//     for (auto &patch : sem_patches)
//     {
//         if (patch.mask.empty())
//             continue;

//         const cv::Rect roi = patch.roi;
//         cv::Mat canny_in_mask;
//         canny(roi).copyTo(canny_in_mask, patch.mask(roi));

//         preprocess(canny_in_mask);
//         std::vector<EdgeCluster> edge_clusters = regionGrowthClusteringOCanny(canny_in_mask, 20.0f, roi.tl());

//         patch.edges.reserve(edge_clusters.size());
//         for (int i = 0; i < edge_clusters.size(); ++i)
//         {
//             Edge temp_edge(i);
//             const auto &cluster_points = edge_clusters[i].organize();

//             temp_edge.mvPoints.reserve(cluster_points.size());

//             for (const auto &point : cluster_points)
//             {
//                 const int x = static_cast<int>(point.pixel.x) + roi.x;
//                 const int y = static_cast<int>(point.pixel.y) + roi.y;

//                 // angle(0~360) of the image gradient
//                 const float angle = angle_ptr[y * angle_step + x];

//                 // an orderedEdgePoint is initially constructed by coordinate (x,y) and gradient angle
//                 temp_edge.mvPoints.emplace_back(x, y, angle);
//             }
//             Edge curr_edge = postprocess(temp_edge, 5, 0.08, 1);
//             if (curr_edge.mvPoints.size() < 10)
//                 continue;                                // 过滤掉过短的边缘
//             patch.edges.push_back(std::move(curr_edge)); // 移动语义减少拷贝
//         }
//     }
// }

void Frame::genPatchSemEdge(const cv::Mat &semantic)
{
    cv::Mat canny;
    cv::Canny(gray, canny, 50, 150, 3, true);

    cv::Mat grad_x, grad_y;
    cv::Scharr(gray, grad_x, CV_32F, 1, 0);
    cv::Scharr(gray, grad_y, CV_32F, 0, 1);

    magnitude.create(gray.size(), CV_32F);
    angle.create(gray.size(), CV_32F);
    cv::cartToPolar(grad_x, grad_y, magnitude, angle, true);

    const float *angle_ptr = angle.ptr<float>();
    const int angle_step = angle.step / sizeof(float);

    for (int i = 1; i < 4; ++i)
    {
        cv::Mat filtered_semantic = cv::Mat::zeros(semantic.size(), CV_8UC1);
        filtered_semantic.setTo(255, semantic == i);

        if (cv::countNonZero(filtered_semantic) == 0)
            continue;

        cv::Mat label, stats, centroids;
        cv::connectedComponentsWithStats(filtered_semantic, label, stats, centroids, 4, CV_16U);

        for (int j = 1; j < centroids.rows; j++)
        {
            // Exclude background
            Patch patch;
            patch.area = stats.at<int>(j, cv::CC_STAT_AREA);
            patch.top = stats.at<int>(j, cv::CC_STAT_TOP);   // y
            patch.left = stats.at<int>(j, cv::CC_STAT_LEFT); // x
            patch.width = stats.at<int>(j, cv::CC_STAT_WIDTH);
            patch.height = stats.at<int>(j, cv::CC_STAT_HEIGHT);
            patch.cls = i;

            // Clean redundant patches.
            // 1. Remove patches with too few points.
            if (patch.area < areaThd)
                continue;

            // 2. Remove patches with too big W-H ratio.
            double WHratio = std::max(patch.width, patch.height) / std::min(patch.width, patch.height);
            if (WHratio > 10)
                continue;

            cv::Mat mask = cv::Mat::zeros(semantic.size(), CV_8UC1);
            mask = mask.setTo(255, label == j);

            int dilationSize = 2;
            cv::Mat dilationElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1), cv::Point(dilationSize, dilationSize));
            cv::dilate(mask, mask, dilationElement);

            patch.mask = mask;
            std::vector<cv::Point> nz_points;
            cv::findNonZero(mask, nz_points);
            patch.roi = nz_points.empty() ? cv::Rect() : cv::boundingRect(nz_points);

            cv::Mat canny_in_mask;
            canny(patch.roi).copyTo(canny_in_mask, patch.mask(patch.roi));

            preprocess(canny_in_mask);
            std::vector<EdgeCluster> edge_clusters = regionGrowthClusteringOCanny(canny_in_mask, 20.0f, patch.roi.tl());

            patch.edges.reserve(edge_clusters.size());
            for (size_t i = 0; i < edge_clusters.size(); ++i)
            {
                Edge temp_edge(i);
                const auto &cluster_points = edge_clusters[i].organize();
                
                temp_edge.mvPoints.reserve(cluster_points.size());
                for (const auto &point : cluster_points)
                {
                    const int x = static_cast<int>(point.pixel.x) + patch.roi.x;
                    const int y = static_cast<int>(point.pixel.y) + patch.roi.y;

                    // angle(0~360) of the image gradient
                    const float angle = angle_ptr[y * angle_step + x];

                    // an orderedEdgePoint is initially constructed by coordinate (x,y) and gradient angle
                    temp_edge.mvPoints.emplace_back(x, y, angle);
                }
                // Edge curr_edge = postprocess(temp_edge, 5, 0.08, 1);
                // if (curr_edge.mvPoints.size() < 10)
                //     continue;                                // 过滤掉过短的边缘
                // patch.edges.push_back(std::move(curr_edge)); // 移动语义减少拷贝
                patch.edges.push_back(std::move(temp_edge)); // 移动语义减少拷贝
            }
            sem_patches.emplace_back(patch);
        }
    }
}

double computeCurvature(const orderedEdgePoint &p0, const orderedEdgePoint &p1, const orderedEdgePoint &p2)
{
    orderedEdgePoint a = p1 - p0;
    orderedEdgePoint b = p2 - p1;
    orderedEdgePoint c = p2 - p0;

    double la = a.norm();
    double lb = b.norm();
    double lc = c.norm();

    if (la < 1e-6 || lb < 1e-6 || lc < 1e-6)
        return 0.0;

    double cross = a.x * b.y - a.y * b.x;

    return 2.0 * cross / (la * lb * lc);
}

Edge Frame::postprocess(const Edge &edge, int step, double curvatureDiffThresh, int removeRadius)
{
    Edge postEdge(edge.edge_ID);

    const int n = edge.mvPoints.size();
    if (n <= 2 * step + 1)
        return edge;

    std::vector<double> curvature(n, 0.0);
    std::vector<uint8_t> valid(n, 0);

    for (int i = 1; i < n - 1; ++i)
    {
        const orderedEdgePoint &p0 = edge.mvPoints[std::max(i - step, 0)];
        const orderedEdgePoint &p1 = edge.mvPoints[i];
        const orderedEdgePoint &p2 = edge.mvPoints[std::min(i + step, n - 1)];

        curvature[i] = computeCurvature(p0, p1, p2);
        valid[i] = 1;
    }

    std::vector<uint8_t> removeMask(n, 0);

    for (int i = step + 1; i < n - step; ++i)
    {
        if (!valid[i] || !valid[i - 1])
            continue;

        double diff = std::abs(curvature[i] - curvature[i - 1]);

        if (diff > curvatureDiffThresh)
        {
            removeMask[i] = 1;
            break;
        }
    }

    for (int i = 0; i < n; ++i)
    {
        if (removeMask[i])
            break;
        postEdge.mvPoints.push_back(edge.mvPoints[i]);
    }

    return postEdge;
}
