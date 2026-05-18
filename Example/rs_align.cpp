#include <iostream>
#include <string>

#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>

int main()
{
    try
    {
        rs2::pipeline pipe;
        rs2::config cfg;

        // D435i supports these streams; 640x480@30 is a common stable setup.
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

        rs2::pipeline_profile profile = pipe.start(cfg);

        rs2::device dev = profile.get_device();
        std::string device_name = dev.get_info(RS2_CAMERA_INFO_NAME);
        std::cout << "Device: " << device_name << std::endl;

        rs2::depth_sensor depth_sensor = dev.first<rs2::depth_sensor>();
        float depth_scale = depth_sensor.get_depth_scale();
        std::cout << "Depth scale: " << depth_scale << " m/unit" << std::endl;

        // Align depth to color so both images share the same pixel coordinates.
        rs2::align align_to_color(RS2_STREAM_COLOR);

        cv::namedWindow("color", cv::WINDOW_AUTOSIZE);
        cv::namedWindow("depth_aligned", cv::WINDOW_AUTOSIZE);

        while (true)
        {
            rs2::frameset frames = pipe.wait_for_frames();
            rs2::frameset aligned_frames = align_to_color.process(frames);

            rs2::video_frame color_frame = aligned_frames.get_color_frame();
            rs2::depth_frame depth_frame = aligned_frames.get_depth_frame();

            if (!color_frame || !depth_frame)
            {
                continue;
            }

            const int w = color_frame.get_width();
            const int h = color_frame.get_height();

            cv::Mat color(cv::Size(w, h), CV_8UC3,
                          const_cast<void *>(color_frame.get_data()),
                          cv::Mat::AUTO_STEP);

            cv::Mat depth_raw(cv::Size(w, h), CV_16UC1,
                              const_cast<void *>(depth_frame.get_data()),
                              cv::Mat::AUTO_STEP);

            // For display only: map depth to 8-bit and apply color map.
            cv::Mat depth_8u;
            cv::convertScaleAbs(depth_raw, depth_8u, 0.03);
            cv::Mat depth_color;
            cv::applyColorMap(depth_8u, depth_color, cv::COLORMAP_JET);

            // Example: query metric depth at image center from aligned depth.
            const int cx = w / 2;
            const int cy = h / 2;
            float distance_m = depth_frame.get_distance(cx, cy);

            cv::Mat color_show = color.clone();
            cv::circle(color_show, cv::Point(cx, cy), 4, cv::Scalar(0, 255, 0), -1);
            cv::putText(color_show,
                        "center depth: " + std::to_string(distance_m) + " m",
                        cv::Point(20, 30),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.8,
                        cv::Scalar(0, 255, 0),
                        2);

            cv::imshow("color", color_show);
            cv::imshow("depth_aligned", depth_color);

            char key = static_cast<char>(cv::waitKey(1));
            if (key == 27 || key == 'q')
            {
                cv::imwrite("color_aligned.png", color);
                cv::imwrite("depth_aligned.png", depth_raw);
                break;
            }
        }

        pipe.stop();
        return 0;
    }
    catch (const rs2::error &e)
    {
        std::cerr << "RealSense error: " << e.get_failed_function() << "(" << e.get_failed_args() << ")\n"
                  << e.what() << std::endl;
        return 1;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
