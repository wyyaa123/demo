#include <iostream>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

int main()
{
    // 读取图像
    cv::Mat color = cv::imread("/home/wyyaa123/demo/datasets/Real/camera_infra1_image_rect_raw/1777018836259954214.png", cv::IMREAD_COLOR);
    cv::Mat depth = cv::imread("/home/wyyaa123/demo/datasets/Real/depth/1777018836259954214.png", cv::IMREAD_UNCHANGED); // 保留深度原始值

    // 相机内参（你需要替换成自己的）
    float fx = 394.7194240503013;
    float fy = 394.4889497892937;
    float cx = 323.98920121606034;
    float cy = 237.27629836778792;

    float depth_scale = 1000.0; // 如果深度是毫米，需要除1000

    // 创建点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZRGB>);

    // 遍历每个像素
    for (int v = 0; v < depth.rows; v++)
    {
        for (int u = 0; u < depth.cols; u++)
        {
            ushort d = depth.ptr<ushort>(v)[u];
            if (d == 0)
                continue; // 无效深度

            float z = d / depth_scale;
            float x = (u - cx) * z / fx;
            float y = (v - cy) * z / fy;

            pcl::PointXYZRGB point;
            point.x = x;
            point.y = y;
            point.z = z;

            // 获取颜色
            cv::Vec3b color_pixel = color.at<cv::Vec3b>(v, u);
            uint32_t rgb = (static_cast<uint32_t>(color_pixel[2]) << 16 | // R
                            static_cast<uint32_t>(color_pixel[1]) << 8 |  // G
                            static_cast<uint32_t>(color_pixel[0]));       // B
            point.rgb = *reinterpret_cast<float *>(&rgb);

            cloud->points.push_back(point);
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = false;

    std::cout << "Point cloud size: " << cloud->points.size() << std::endl;

    // 1. 创建 PCLVisualizer 对象
    pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));

    // 2. 设置背景和点云
    viewer->setBackgroundColor(0.05, 0.05, 0.05); // 深灰色背景
    viewer->addPointCloud<pcl::PointXYZRGB>(cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");

    viewer->setCameraPosition(0.5, -0.5, -2.0,
                              0.0,  0.0,  1.0,
                              0.0, -1.0,  0.0);

    // 4. 添加坐标轴参考（红色X, 绿色Y, 蓝色Z）
    viewer->addCoordinateSystem(0.2);

    // 5. 渲染循环
    while (!viewer->wasStopped())
    {
        viewer->spinOnce(100);
    }

    return 0;
}