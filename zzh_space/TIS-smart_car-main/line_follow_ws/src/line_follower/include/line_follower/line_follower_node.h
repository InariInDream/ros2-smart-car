#pragma once

#include <chrono>
#include <optional>
#include <string>
#include <vector>

#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/string.hpp>

namespace line_follower
{

enum class TrackState
{
  LOST = 0,
  TRACKING,
  YELLOW_APPROACH,
  YELLOW_LINE_DETECTED,
  RED_AVOIDING,
  FINISH_STOP
};

struct ProcessingResult
{
  bool line_detected{false};
  bool yellow_approach_detected{false};
  bool yellow_line_detected{false};
  double yellow_area_ratio{0.0};
  double error{0.0};
  cv::Point2f centroid{0.0F, 0.0F};
  cv::Rect bounding_box{0, 0, 0, 0};
  cv::Mat binary_mask;
  cv::Mat roi_bgr;
  TrackState state{TrackState::LOST};
  int roi_top{0};
  int roi_bottom{0};
};

class LineFollowerNode : public rclcpp::Node
{
public:
  LineFollowerNode();

private:
  void declare_parameters();
  void setup_interfaces();

  void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg);
  void start_callback(const std_msgs::msg::Bool::SharedPtr msg);
  std::optional<ProcessingResult> process_image(const cv::Mat & frame_bgr) const;

  std::string state_to_string(TrackState state) const;

  void publish_cmd_vel(double linear_speed, double yaw_rate, bool publish_stop);
  void publish_state(TrackState state);
  void publish_debug_image(
    const sensor_msgs::msg::Image::ConstSharedPtr & source,
    const cv::Mat & frame_bgr,
    const ProcessingResult & result) const;
  void publish_centroid(double norm_x, double norm_y);
  bool detect_red(
    const cv::Mat & frame_bgr, int roi_top, int roi_bottom,
    const cv::Point2d & line_centroid_roi, double & side_sign,
    cv::Point2d & red_centroid_img, cv::Point2d & div_base_img,
    cv::Point2d & div_target_img) const;

  static double clamp(double value, double min_value, double max_value);

  // Parameters
  std::string image_topic_;
  std::string cmd_vel_topic_;
  std::string debug_image_topic_;
  std::string line_state_topic_;
  std::string start_topic_;
  std::string stop_cmd_topic_;
  std::string line_centroid_topic_;

  double target_speed_{0.6};
  double max_speed_{1.0};
  double min_speed_{0.0};
  double max_yaw_rate_{1.2};
  double kp_{1.5};
  double left_turn_ratio_{1.15};

  double roi_top_ratio_{0.5};
  double roi_bottom_ratio_{0.05};
  int gaussian_kernel_{5};
  double binary_threshold_{160.0};
  int morph_kernel_{5};
  int morph_iterations_{2};
  double min_contour_area_{2000.0};
  bool invert_binary_{false};
  bool use_white_filter_{false};
  int white_h_low_{0};
  int white_h_high_{180};
  int white_s_max_{60};
  int white_v_min_{180};

  bool use_yellow_filter_{false};
  int yellow_h_low_{20};
  int yellow_h_high_{40};
  int yellow_s_min_{80};
  int yellow_v_min_{120};
  double yellow_area_ratio_threshold_{0.1};
  bool yellow_approach_enabled_{true};
  double yellow_approach_area_ratio_threshold_{0.01};
  double yellow_approach_speed_{0.12};
  int yellow_stop_confirm_frames_{3};
  double yellow_clear_area_ratio_threshold_{0.008};
  double yellow_stop_duration_{1.0};
  int yellow_slow_count_{3};
  double yellow_slow_speed_{0.3};
  double yellow_detection_cooldown_{2.0};
  double yellow_resume_yaw_rate_{0.0};
  double yellow_resume_yaw_duration_{0.0};
  int yellow_resume_yaw_count_{1};
  std::vector<double> yellow_resume_yaw_sequence_;
  double yellow_resume_yaw_active_rate_{0.0};
  int yellow_permanent_count_{5};
  bool enable_intersection_probe_{false};
  int intersection_band_height_{50};
  double intersection_dark_ratio_threshold_{0.9};
  double intersection_turn_speed_{0.2};
  double intersection_turn_yaw_rate_{0.8};
  double intersection_turn_duration_{1.0};
  double intersection_cooldown_{2.0};
  bool enable_red_filter_{false};
  int red_h_low_1_{0};
  int red_h_high_1_{10};
  int red_h_low_2_{160};
  int red_h_high_2_{180};
  int red_s_min_{100};
  int red_v_min_{80};
  double red_area_ratio_threshold_{0.005};
  double red_turn_yaw_{0.6};
  double red_turn_linear_{0.0};
  double red_turn_duration_{1.0};
  int red_vote_frames_{8};
  double red_divider_angle_deg_{0.0};
  double red_divider_angle_rad_{0.0};
  double red_min_contour_area_{80.0};
  double red_cooldown_{0.6};
  bool red_debug_overlay_{true};

  bool enable_finish_stop_{false};
  std::string finish_stop_mode_{"yellow_count"};
  int finish_stop_yellow_count_{0};
  double finish_stop_area_ratio_threshold_{0.05};
  int finish_stop_confirm_frames_{3};
  bool finish_stop_hold_forever_{true};

  double line_lost_stop_time_{0.8};
  double speed_gain_damping_{0.5};
  bool publish_debug_image_{true};

  // Interfaces
  image_transport::Subscriber image_sub_;
  image_transport::Publisher debug_pub_;
  bool debug_enabled_{false};

  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_vel_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr stop_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Point>::SharedPtr centroid_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr start_sub_;

  mutable cv::Mat morph_element_;
  rclcpp::Time last_detection_time_;
  rclcpp::Time yellow_line_detection_time_;
  rclcpp::Time yellow_stop_start_time_;
  TrackState last_state_{TrackState::LOST};
  bool yellow_line_stop_active_{false};
  bool yellow_line_latched_{false};
  int yellow_stop_confirm_count_{0};
  int yellow_line_detection_count_{0};
  bool yellow_slowdown_active_{false};
  rclcpp::Time yellow_resume_yaw_until_{0, 0, RCL_ROS_TIME};
  int yellow_resume_yaw_applied_{0};
  bool permanent_stop_{false};
  bool start_enabled_{false};
  bool intersection_active_{false};
  rclcpp::Time intersection_end_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time intersection_cooldown_until_{0, 0, RCL_ROS_TIME};
  int red_left_votes_{0};
  int red_right_votes_{0};
  bool red_turning_{false};
  rclcpp::Time red_turn_end_time_{0, 0, RCL_ROS_TIME};
  rclcpp::Time red_cooldown_until_{0, 0, RCL_ROS_TIME};
  double red_turn_yaw_cmd_{0.0};
  bool finish_stop_active_{false};
  rclcpp::Time finish_stop_start_time_{0, 0, RCL_ROS_TIME};
  int finish_stop_confirm_count_{0};

  // Debug overlay for red avoidance
  mutable bool last_red_overlay_valid_{false};
  mutable cv::Point2d last_red_center_img_{0.0, 0.0};
  mutable cv::Point2d last_div_base_img_{0.0, 0.0};
  mutable cv::Point2d last_div_target_img_{0.0, 0.0};
  mutable double last_red_side_sign_{0.0};
};

}  // namespace line_follower
