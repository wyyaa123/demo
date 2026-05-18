#include "frame.h"

#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <vector>
#include <string>

int main()
{
    cv::Mat depth = cv::imread("datasets/Real/D435I/2026-04-24-08-20-35/depth/1777018884637473345.png", cv::IMREAD_UNCHANGED);
    cv::Mat semantic = cv::imread("datasets/Real/D435I/2026-04-24-08-20-35/semantic/1777018884637473345.png", cv::IMREAD_GRAYSCALE);
    Frame frame;

    float fx = 394.7194240503013;
    float fy = 394.4889497892937;
    float cx = 323.98920121606034;
    float cy = 237.27629836778792;

    float depth_scale = 1000.0; // 如果深度是毫米，需要除1000

    // 创建点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);

    cv::Mat sem = semantic.clone();
    std::vector<Patch> filtered_patches;
    frame.semToPatches(sem, filtered_patches, depth, fx, fy, cx, cy);

    // 保存每个 patch 的局部坐标变换，用于可视化坐标轴
    std::vector<Eigen::Affine3f> patch_axes;
    std::vector<std::string> patch_axis_ids;
    int patch_idx = 0;

    for (auto& patch: filtered_patches)
    {
        // if (patch.cls != 1) { patch_idx++; continue; }

        std::vector<Eigen::Vector3f> pts3d;

        for (auto& point: patch.points)
        {
            pcl::PointXYZRGB pcl_point;
            pcl_point.x = point.x();
            pcl_point.y = point.y();
            pcl_point.z = point.z();

            // 获取颜色
            uint32_t rgb = (static_cast<uint32_t>(patch.color[2]) << 16 |  // R
                            static_cast<uint32_t>(patch.color[1]) <<  8 |  // G
                            static_cast<uint32_t>(patch.color[0]));        // B
            pcl_point.rgb = *reinterpret_cast<float *>(&rgb);

            cloud->points.push_back(pcl_point);
        }

        Eigen::Affine3f transform = Eigen::Affine3f::Identity();
        transform.linear() = patch.ev;
        transform.translation() = patch.centroid;

        patch_axes.push_back(transform);
        patch_axis_ids.push_back(std::string("patch_axis_") + std::to_string(patch_idx));

        patch_idx++;
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;

    std::cout << "Point cloud size: " << cloud->points.size() << std::endl;

    // 1. 创建 PCLVisualizer 对象
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

    // 2. 设置背景和点云
    viewer->setBackgroundColor(255, 255, 255); // 深灰色背景
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

    viewer->setCameraPosition(0.5, -0.5, -2.0,
                              0.0,  0.0,  1.0,
                              0.0, -1.0,  0.0);

    // 4. 添加坐标轴参考（红色X, 绿色Y, 蓝色Z）
    viewer->addCoordinateSystem(0.2);

    // 为每个 patch 添加局部坐标轴与质心标记
    for (size_t i = 0; i < patch_axes.size(); ++i)
    {
        viewer->addCoordinateSystem(1.0, patch_axes[i], patch_axis_ids[i]);

        // 在质心处添加小球（黄色）便于观察
        Eigen::Vector3f t = patch_axes[i].translation();
        pcl::PointXYZ center_point;
        center_point.x = t.x();
        center_point.y = t.y();
        center_point.z = t.z();
        std::string sphere_id = std::string("centroid_") + std::to_string(i);
        viewer->addSphere(center_point, 0.02, 1.0, 1.0, 0.0, sphere_id);
    }

    // 5. 渲染循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }

    return 0;
}
