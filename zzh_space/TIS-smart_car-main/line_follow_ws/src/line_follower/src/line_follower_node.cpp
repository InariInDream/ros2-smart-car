#include "line_follower/line_follower_node.h"

#include <algorithm>
#include <cmath>
#include <functional>

#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/point.hpp>
#include <opencv2/imgproc.hpp>
#include <std_msgs/msg/bool.hpp>

namespace line_follower
{

LineFollowerNode::LineFollowerNode()
: Node("line_follower_node")
{
  declare_parameters();
  setup_interfaces();

  const int kernel_size = std::max<int>(1, morph_kernel_);
  morph_element_ = cv::getStructuringElement(
    cv::MORPH_RECT, cv::Size(kernel_size, kernel_size));
  const auto current_time = this->now();
  last_detection_time_ = current_time;
  yellow_line_detection_time_ =
    current_time - rclcpp::Duration::from_seconds(std::max(0.0, yellow_detection_cooldown_));
  yellow_stop_start_time_ = current_time;
  last_state_ = TrackState::LOST;
  yellow_line_stop_active_ = false;
  yellow_line_latched_ = false;
  yellow_stop_confirm_count_ = 0;
  yellow_line_detection_count_ = 0;
  permanent_stop_ = false;
  finish_stop_active_ = false;
  finish_stop_start_time_ = current_time;
  finish_stop_confirm_count_ = 0;
  start_enabled_ = false;
  red_left_votes_ = 0;
  red_right_votes_ = 0;
  red_turning_ = false;
  red_turn_end_time_ = current_time;
  red_cooldown_until_ = current_time;
  last_red_overlay_valid_ = false;

  RCLCPP_INFO(
    this->get_logger(),
    "Line follower subscribing to '%s', publishing cmd_vel on '%s', start_topic: '%s'",
    image_topic_.c_str(), cmd_vel_topic_.c_str(), start_topic_.c_str());
}

void LineFollowerNode::declare_parameters()
{
  this->declare_parameter("image_topic", "/ascamera/camera_publisher/rgb/image0");
  this->declare_parameter("cmd_vel_topic", "/cmd_vel");
  this->declare_parameter("debug_image_topic", "line_follower/debug_image");
  this->declare_parameter("line_state_topic", "line_follower/track_state");
  this->declare_parameter("start_topic", "line_follower/start_cmd");
  this->declare_parameter("stop_cmd_topic", "/cmd_vel_stop");
  this->declare_parameter("line_centroid_topic", "line_follower/centroid");

  this->declare_parameter("target_speed", 0.6);
  this->declare_parameter("max_speed", 1.0);
  this->declare_parameter("min_speed", 0.0);
  this->declare_parameter("max_yaw_rate", 1.2);
  this->declare_parameter("kp", 1.8);
  this->declare_parameter("left_turn_ratio", 1.15);

  this->declare_parameter("roi_top_ratio", 0.5);
  this->declare_parameter("roi_bottom_ratio", 0.05);
  this->declare_parameter("gaussian_kernel", 5);
  this->declare_parameter("binary_threshold", 160.0);
  this->declare_parameter("morph_kernel", 5);
  this->declare_parameter("morph_iterations", 2);
  this->declare_parameter("min_contour_area", 2000.0);
  this->declare_parameter("invert_binary", false);
  this->declare_parameter("use_white_filter", false);
  this->declare_parameter("white_h_low", 0);
  this->declare_parameter("white_h_high", 180);
  this->declare_parameter("white_s_max", 60);
  this->declare_parameter("white_v_min", 180);

  this->declare_parameter("use_yellow_filter", false);
  this->declare_parameter("yellow_h_low", 20);
  this->declare_parameter("yellow_h_high", 40);
  this->declare_parameter("yellow_s_min", 80);
  this->declare_parameter("yellow_v_min", 120);
  this->declare_parameter("yellow_area_ratio_threshold", 0.1);
  this->declare_parameter("yellow_approach_enabled", true);
  this->declare_parameter("yellow_approach_area_ratio_threshold", 0.01);
  this->declare_parameter("yellow_approach_speed", 0.12);
  this->declare_parameter("yellow_stop_confirm_frames", 3);
  this->declare_parameter("yellow_clear_area_ratio_threshold", 0.008);
  this->declare_parameter("yellow_stop_duration", 1.0);
  this->declare_parameter("yellow_slow_count", 3);
  this->declare_parameter("yellow_slow_speed", 0.3);
  this->declare_parameter("yellow_detection_cooldown", 2.0);
  this->declare_parameter("yellow_resume_yaw_rate", 1.0);
  this->declare_parameter("yellow_resume_yaw_duration", 0.5);
  this->declare_parameter("yellow_resume_yaw_count", 1);
  this->declare_parameter("yellow_resume_yaw_sequence", std::vector<double>{});
  this->declare_parameter("yellow_permanent_count", 5);
  this->declare_parameter("enable_intersection_probe", false);
  this->declare_parameter("intersection_band_height", 50);
  this->declare_parameter("intersection_dark_ratio_threshold", 0.9);
  this->declare_parameter("intersection_turn_speed", 0.2);
  this->declare_parameter("intersection_turn_yaw_rate", 0.8);
  this->declare_parameter("intersection_turn_duration", 1.0);
  this->declare_parameter("intersection_cooldown", 2.0);
  // 红色避障
  this->declare_parameter("enable_red_filter", false);
  this->declare_parameter("red_h_low_1", 0);
  this->declare_parameter("red_h_high_1", 10);
  this->declare_parameter("red_h_low_2", 160);
  this->declare_parameter("red_h_high_2", 180);
  this->declare_parameter("red_s_min", 100);
  this->declare_parameter("red_v_min", 80);
  this->declare_parameter("red_area_ratio_threshold", 0.005);
  this->declare_parameter("red_turn_yaw", 0.6);
  this->declare_parameter("red_turn_linear", 0.0);
  this->declare_parameter("red_turn_duration", 1.0);
  this->declare_parameter("red_vote_frames", 8);
  this->declare_parameter("red_divider_angle_deg", 0.0);
  this->declare_parameter("red_min_contour_area", 80.0);
  this->declare_parameter("red_cooldown", 0.6);
  this->declare_parameter("red_debug_overlay", true);

  this->declare_parameter("enable_finish_stop", false);
  this->declare_parameter("finish_stop_mode", "yellow_count");
  this->declare_parameter("finish_stop_yellow_count", 0);
  this->declare_parameter("finish_stop_area_ratio_threshold", 0.05);
  this->declare_parameter("finish_stop_confirm_frames", 3);
  this->declare_parameter("finish_stop_hold_forever", true);

  this->declare_parameter("line_lost_stop_time", 0.8);
  this->declare_parameter("speed_gain_damping", 0.5);
  this->declare_parameter("publish_debug_image", true);

  image_topic_ = this->get_parameter("image_topic").as_string();
  cmd_vel_topic_ = this->get_parameter("cmd_vel_topic").as_string();
  debug_image_topic_ = this->get_parameter("debug_image_topic").as_string();
  line_state_topic_ = this->get_parameter("line_state_topic").as_string();
  start_topic_ = this->get_parameter("start_topic").as_string();
  stop_cmd_topic_ = this->get_parameter("stop_cmd_topic").as_string();
  line_centroid_topic_ = this->get_parameter("line_centroid_topic").as_string();

  target_speed_ = this->get_parameter("target_speed").as_double();
  max_speed_ = this->get_parameter("max_speed").as_double();
  min_speed_ = this->get_parameter("min_speed").as_double();
  max_yaw_rate_ = this->get_parameter("max_yaw_rate").as_double();
  kp_ = this->get_parameter("kp").as_double();
  left_turn_ratio_ = this->get_parameter("left_turn_ratio").as_double();

  roi_top_ratio_ = this->get_parameter("roi_top_ratio").as_double();
  roi_bottom_ratio_ = this->get_parameter("roi_bottom_ratio").as_double();

  gaussian_kernel_ = this->get_parameter("gaussian_kernel").as_int();
  if (gaussian_kernel_ <= 0) {
    gaussian_kernel_ = 1;
  }
  if (gaussian_kernel_ % 2 == 0) {
    ++gaussian_kernel_;
  }

  binary_threshold_ = this->get_parameter("binary_threshold").as_double();
  morph_kernel_ =
    std::max<int>(1, static_cast<int>(this->get_parameter("morph_kernel").as_int()));
  morph_iterations_ =
    std::max<int>(1, static_cast<int>(this->get_parameter("morph_iterations").as_int()));
  min_contour_area_ = this->get_parameter("min_contour_area").as_double();

  invert_binary_ = this->get_parameter("invert_binary").as_bool();
  use_white_filter_ = this->get_parameter("use_white_filter").as_bool();
  white_h_low_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("white_h_low").as_int()), 0, 180);
  white_h_high_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("white_h_high").as_int()), 0, 180);
  if (white_h_low_ > white_h_high_) {
    std::swap(white_h_low_, white_h_high_);
  }
  white_s_max_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("white_s_max").as_int()), 0, 255);
  white_v_min_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("white_v_min").as_int()), 0, 255);

  use_yellow_filter_ = this->get_parameter("use_yellow_filter").as_bool();
  yellow_h_low_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("yellow_h_low").as_int()), 0, 180);
  yellow_h_high_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("yellow_h_high").as_int()), 0, 180);
  if (yellow_h_low_ > yellow_h_high_) {
    std::swap(yellow_h_low_, yellow_h_high_);
  }
  yellow_s_min_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("yellow_s_min").as_int()), 0, 255);
  yellow_v_min_ = std::clamp<int>(
    static_cast<int>(this->get_parameter("yellow_v_min").as_int()), 0, 255);
  yellow_area_ratio_threshold_ = this->get_parameter("yellow_area_ratio_threshold").as_double();
  yellow_approach_enabled_ = this->get_parameter("yellow_approach_enabled").as_bool();
  yellow_approach_area_ratio_threshold_ =
    this->get_parameter("yellow_approach_area_ratio_threshold").as_double();
  yellow_approach_speed_ = std::clamp(
    this->get_parameter("yellow_approach_speed").as_double(), min_speed_, max_speed_);
  yellow_stop_confirm_frames_ = std::max<int>(
    1, static_cast<int>(this->get_parameter("yellow_stop_confirm_frames").as_int()));
  yellow_clear_area_ratio_threshold_ =
    this->get_parameter("yellow_clear_area_ratio_threshold").as_double();
  yellow_stop_duration_ = this->get_parameter("yellow_stop_duration").as_double();
  yellow_slow_count_ = std::max<int>(0, this->get_parameter("yellow_slow_count").as_int());
  yellow_permanent_count_ = std::max<int>(
    0, static_cast<int>(this->get_parameter("yellow_permanent_count").as_int()));
  if (yellow_permanent_count_ > 0) {
    yellow_slow_count_ = std::min(yellow_slow_count_, yellow_permanent_count_);
  }
  yellow_slow_speed_ = std::clamp(
    this->get_parameter("yellow_slow_speed").as_double(), min_speed_, max_speed_);
  yellow_detection_cooldown_ =
    std::max(0.0, this->get_parameter("yellow_detection_cooldown").as_double());
  yellow_resume_yaw_rate_ = this->get_parameter("yellow_resume_yaw_rate").as_double();
  yellow_resume_yaw_duration_ =
    std::max(0.0, this->get_parameter("yellow_resume_yaw_duration").as_double());
  yellow_resume_yaw_count_ =
    std::max<int>(0, this->get_parameter("yellow_resume_yaw_count").as_int());
  yellow_resume_yaw_sequence_ =
    this->get_parameter("yellow_resume_yaw_sequence").as_double_array();
  enable_intersection_probe_ = this->get_parameter("enable_intersection_probe").as_bool();
  intersection_band_height_ = std::max<int>(
    1, static_cast<int>(this->get_parameter("intersection_band_height").as_int()));
  intersection_dark_ratio_threshold_ =
    this->get_parameter("intersection_dark_ratio_threshold").as_double();
  intersection_turn_speed_ = this->get_parameter("intersection_turn_speed").as_double();
  intersection_turn_yaw_rate_ = this->get_parameter("intersection_turn_yaw_rate").as_double();
  intersection_turn_duration_ = this->get_parameter("intersection_turn_duration").as_double();
  intersection_cooldown_ = this->get_parameter("intersection_cooldown").as_double();
  enable_red_filter_ = this->get_parameter("enable_red_filter").as_bool();
  red_h_low_1_ = this->get_parameter("red_h_low_1").as_int();
  red_h_high_1_ = this->get_parameter("red_h_high_1").as_int();
  red_h_low_2_ = this->get_parameter("red_h_low_2").as_int();
  red_h_high_2_ = this->get_parameter("red_h_high_2").as_int();
  red_s_min_ = this->get_parameter("red_s_min").as_int();
  red_v_min_ = this->get_parameter("red_v_min").as_int();
  red_area_ratio_threshold_ = this->get_parameter("red_area_ratio_threshold").as_double();
  red_turn_yaw_ = this->get_parameter("red_turn_yaw").as_double();
  red_turn_linear_ = this->get_parameter("red_turn_linear").as_double();
  red_turn_duration_ = this->get_parameter("red_turn_duration").as_double();
  red_vote_frames_ = std::max<int>(1, this->get_parameter("red_vote_frames").as_int());
  red_divider_angle_deg_ = this->get_parameter("red_divider_angle_deg").as_double();
  red_divider_angle_rad_ = red_divider_angle_deg_ * (M_PI / 180.0);
  red_min_contour_area_ =
    std::max(0.0, this->get_parameter("red_min_contour_area").as_double());
  red_cooldown_ = std::max(0.0, this->get_parameter("red_cooldown").as_double());
  red_debug_overlay_ = this->get_parameter("red_debug_overlay").as_bool();

  enable_finish_stop_ = this->get_parameter("enable_finish_stop").as_bool();
  finish_stop_mode_ = this->get_parameter("finish_stop_mode").as_string();
  finish_stop_yellow_count_ = std::max<int>(
    0, static_cast<int>(this->get_parameter("finish_stop_yellow_count").as_int()));
  finish_stop_area_ratio_threshold_ =
    std::max(0.0, this->get_parameter("finish_stop_area_ratio_threshold").as_double());
  finish_stop_confirm_frames_ = std::max<int>(
    1, static_cast<int>(this->get_parameter("finish_stop_confirm_frames").as_int()));
  finish_stop_hold_forever_ = this->get_parameter("finish_stop_hold_forever").as_bool();

  line_lost_stop_time_ = this->get_parameter("line_lost_stop_time").as_double();
  speed_gain_damping_ = this->get_parameter("speed_gain_damping").as_double();
  publish_debug_image_ = this->get_parameter("publish_debug_image").as_bool();
}

void LineFollowerNode::setup_interfaces()
{
  cmd_vel_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);
  stop_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(stop_cmd_topic_, 10);
  state_pub_ = this->create_publisher<std_msgs::msg::String>(line_state_topic_, 10);
  centroid_pub_ = this->create_publisher<geometry_msgs::msg::Point>(line_centroid_topic_, 10);

  image_sub_ = image_transport::create_subscription(
    this,
    image_topic_,
    std::bind(&LineFollowerNode::image_callback, this, std::placeholders::_1),
    "raw");

  start_sub_ = this->create_subscription<std_msgs::msg::Bool>(
    start_topic_, 10, std::bind(&LineFollowerNode::start_callback, this, std::placeholders::_1));

  if (publish_debug_image_) {
    debug_pub_ = image_transport::create_publisher(this, debug_image_topic_);
    debug_enabled_ = true;
  }
}

void LineFollowerNode::start_callback(const std_msgs::msg::Bool::SharedPtr msg)
{
  bool new_state = msg->data;
  if (new_state == start_enabled_) {
    return;
  }
  start_enabled_ = new_state;
  RCLCPP_INFO(this->get_logger(), "Start signal changed to: %s", start_enabled_ ? "true" : "false");
}

void LineFollowerNode::image_callback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
{
  cv::Mat frame_bgr;
  try {
    frame_bgr = cv_bridge::toCvShare(msg, "bgr8")->image.clone();
  } catch (const cv_bridge::Exception & e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }

  auto result_opt = process_image(frame_bgr);
  if (result_opt && result_opt->line_detected) {
    if (debug_enabled_) {
      publish_debug_image(msg, frame_bgr, *result_opt);
    }
    // 发布归一化质心（0~1，基于 ROI 宽度）
    double norm_x = 0.5;
    double norm_y = 0.5;
    if (frame_bgr.cols > 0 && frame_bgr.rows > 0) {
      norm_x = result_opt->centroid.x / static_cast<double>(frame_bgr.cols);
      norm_y = (static_cast<double>(result_opt->roi_top) + result_opt->centroid.y) /
        static_cast<double>(frame_bgr.rows);
      norm_x = clamp(norm_x, 0.0, 1.0);
      norm_y = clamp(norm_y, 0.0, 1.0);
    }
    publish_centroid(norm_x, norm_y);
  }

  if (!start_enabled_) {
    static rclcpp::Time last_log_time = this->now();
    rclcpp::Time current_time = this->now();
    if ((current_time - last_log_time).seconds() > 2.0) {
      RCLCPP_INFO(this->get_logger(), "Line follower is not started, waiting for start signal...");
      last_log_time = current_time;
    }
    publish_cmd_vel(0.0, 0.0, true);
    return;
  }

  auto now = this->now();

  double linear_speed = 0.0;
  double yaw_rate = 0.0;
  TrackState current_state = TrackState::LOST;
  bool publish_stop_cmd = false;

  // 红色避障检测与投票（以“底部中心→巡线质心”的分界线判断左右）。
  if (!red_debug_overlay_) {
    last_red_overlay_valid_ = false;
  }
  if (enable_red_filter_ && result_opt && result_opt->line_detected &&
    !red_turning_ && now >= red_cooldown_until_)
  {
    int roi_top = result_opt->roi_top;
    int roi_bottom = result_opt->roi_bottom;
    cv::Point2d line_centroid_roi(result_opt->centroid.x, result_opt->centroid.y);

    double side_sign = 0.0;  // >0 红在分界线右侧（向左转），<0 红在左侧（向右转）
    cv::Point2d red_center_img, div_base_img, div_target_img;
    bool red_seen = detect_red(
      frame_bgr, roi_top, roi_bottom, line_centroid_roi, side_sign,
      red_center_img, div_base_img, div_target_img);
    last_red_overlay_valid_ = red_debug_overlay_ && red_seen;
    if (red_seen && red_debug_overlay_) {
      last_red_center_img_ = red_center_img;
      last_div_base_img_ = div_base_img;
      last_div_target_img_ = div_target_img;
      last_red_side_sign_ = side_sign;
    }
    if (red_seen) {
      if (side_sign < 0.0) {            // 红在左 -> 向右转
        red_left_votes_++;
        red_right_votes_ = 0;
      } else {                          // 红在右 -> 向左转
        red_right_votes_++;
        red_left_votes_ = 0;
      }
      if (red_left_votes_ >= red_vote_frames_) {
        int dir = -1;  // 右转
        red_turn_yaw_cmd_ = clamp(dir * red_turn_yaw_, -max_yaw_rate_, max_yaw_rate_);
        red_turning_ = true;
        red_turn_end_time_ = now + rclcpp::Duration::from_seconds(red_turn_duration_);
        red_cooldown_until_ = red_turn_end_time_ + rclcpp::Duration::from_seconds(red_cooldown_);
        red_left_votes_ = 0;
        red_right_votes_ = 0;
        RCLCPP_INFO(this->get_logger(), "Red area on the left, avoiding to the right.");
      } else if (red_right_votes_ >= red_vote_frames_) {
        int dir = 1;  // 左转
        red_turn_yaw_cmd_ = clamp(dir * red_turn_yaw_, -max_yaw_rate_, max_yaw_rate_);
        red_turning_ = true;
        red_turn_end_time_ = now + rclcpp::Duration::from_seconds(red_turn_duration_);
        red_cooldown_until_ = red_turn_end_time_ + rclcpp::Duration::from_seconds(red_cooldown_);
        red_left_votes_ = 0;
        red_right_votes_ = 0;
        RCLCPP_INFO(this->get_logger(), "Red area on the right, avoiding to the left.");
      }
    } else {
      red_left_votes_ = 0;
      red_right_votes_ = 0;
    }
  }

  const double yellow_area_ratio = result_opt ? result_opt->yellow_area_ratio : 0.0;
  const bool yellow_stop_candidate = result_opt && result_opt->yellow_line_detected;
  if (yellow_stop_candidate) {
    yellow_stop_confirm_count_++;
  } else if (!result_opt || yellow_area_ratio <= yellow_clear_area_ratio_threshold_) {
    yellow_stop_confirm_count_ = 0;
    yellow_line_latched_ = false;
    finish_stop_confirm_count_ = 0;
  }

  const bool yellow_stop_confirmed =
    yellow_stop_confirm_count_ >= yellow_stop_confirm_frames_;
  if (yellow_stop_confirmed && !yellow_line_latched_) {
    yellow_line_latched_ = true;
    double since_last_detection = (now - yellow_line_detection_time_).seconds();
    bool can_count_detection =
      yellow_detection_cooldown_ <= 0.0 || since_last_detection >= yellow_detection_cooldown_;
    if (can_count_detection) {
      yellow_line_detection_time_ = now;
      yellow_line_detection_count_++;
      RCLCPP_INFO(
        this->get_logger(),
        "Yellow line confirmed. Count: %d, area_ratio: %.4f",
        yellow_line_detection_count_, yellow_area_ratio);
      if (!yellow_slowdown_active_ && yellow_slow_count_ > 0 &&
        yellow_line_detection_count_ >= yellow_slow_count_)
      {
        yellow_slowdown_active_ = true;
        RCLCPP_INFO(
          this->get_logger(),
          "Yellow slowdown activated after %d detections. Speed capped at %.2f m/s.",
          yellow_line_detection_count_, yellow_slow_speed_);
      }
      if (yellow_permanent_count_ > 0 &&
        yellow_line_detection_count_ >= yellow_permanent_count_ && !permanent_stop_)
      {
        permanent_stop_ = true;
        RCLCPP_INFO(this->get_logger(), "Yellow line detected %d times! Permanent stop activated.",
          yellow_line_detection_count_);
      }
    }
    if (!yellow_line_stop_active_) {
      yellow_line_stop_active_ = true;
      yellow_stop_start_time_ = now;
      yellow_resume_yaw_until_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
    }
  }

  if (enable_finish_stop_ && finish_stop_mode_ == "yellow_count" &&
    finish_stop_yellow_count_ > 0 &&
    yellow_line_detection_count_ >= finish_stop_yellow_count_ &&
    yellow_stop_confirmed &&
    yellow_area_ratio >= finish_stop_area_ratio_threshold_)
  {
    finish_stop_confirm_count_++;
    if (finish_stop_confirm_count_ >= finish_stop_confirm_frames_ && !finish_stop_active_) {
      finish_stop_active_ = true;
      finish_stop_start_time_ = now;
      RCLCPP_INFO(
        this->get_logger(),
        "Finish stop activated by yellow_count=%d, area_ratio=%.4f.",
        yellow_line_detection_count_, yellow_area_ratio);
    }
  } else if (!yellow_stop_confirmed || yellow_area_ratio <= yellow_clear_area_ratio_threshold_) {
    finish_stop_confirm_count_ = 0;
  }

  double since_yellow_stop = (now - yellow_stop_start_time_).seconds();

  // 交叉口探针触发（固定左转）
  if (enable_intersection_probe_ && result_opt && result_opt->line_detected &&
    !intersection_active_ && now >= intersection_cooldown_until_ &&
    !yellow_line_stop_active_ && !permanent_stop_ && !finish_stop_active_)
  {
    const cv::Mat & mask = result_opt->binary_mask;
    if (!mask.empty()) {
      int band_h = std::min(intersection_band_height_, mask.rows);
      if (band_h > 0) {
        cv::Rect band_rect(0, 0, mask.cols, band_h);
        cv::Mat band = mask(band_rect);
        int nonzero = cv::countNonZero(band);
        double total = static_cast<double>(band.total());
        double black_ratio = total > 0.0 ? 1.0 - static_cast<double>(nonzero) / total : 0.0;
        if (black_ratio >= intersection_dark_ratio_threshold_) {
          intersection_active_ = true;
          intersection_end_time_ =
            now + rclcpp::Duration::from_seconds(intersection_turn_duration_);
          intersection_cooldown_until_ =
            now + rclcpp::Duration::from_seconds(intersection_cooldown_);
          RCLCPP_INFO(
            this->get_logger(),
            "Intersection probe triggered: black_ratio=%.2f band_h=%d",
            black_ratio, band_h);
        }
      }
    }
  }

  if (finish_stop_active_ &&
    (finish_stop_hold_forever_ ||
    (now - finish_stop_start_time_).seconds() < yellow_stop_duration_))
  {
    current_state = TrackState::FINISH_STOP;
    linear_speed = 0.0;
    yaw_rate = 0.0;
    publish_stop_cmd = true;
  } else if (finish_stop_active_) {
    finish_stop_active_ = false;
  }

  if (!publish_stop_cmd && permanent_stop_) {
    current_state = TrackState::YELLOW_LINE_DETECTED;
    linear_speed = 0.0;
    yaw_rate = 0.0;
    publish_stop_cmd = true;
  } else if (!publish_stop_cmd && yellow_line_stop_active_ && since_yellow_stop < yellow_stop_duration_) {
    current_state = TrackState::YELLOW_LINE_DETECTED;
    linear_speed = 0.0;
    yaw_rate = 0.0;
    publish_stop_cmd = true;
  } else if (!publish_stop_cmd && yellow_line_stop_active_ && since_yellow_stop >= yellow_stop_duration_) {
    yellow_line_stop_active_ = false;
    bool can_resume_yaw =
      yellow_resume_yaw_applied_ < yellow_resume_yaw_count_ &&
      std::abs(yellow_resume_yaw_rate_) > 0.0 &&
      yellow_resume_yaw_duration_ > 0.0;
    if (can_resume_yaw) {
      size_t idx = static_cast<size_t>(yellow_resume_yaw_applied_);
      double chosen_yaw = yellow_resume_yaw_rate_;
      if (idx < yellow_resume_yaw_sequence_.size()) {
        chosen_yaw = yellow_resume_yaw_sequence_[idx];
      }
      yellow_resume_yaw_active_rate_ = chosen_yaw;
      yellow_resume_yaw_until_ = now + rclcpp::Duration::from_seconds(yellow_resume_yaw_duration_);
      yellow_resume_yaw_applied_++;
    } else {
      yellow_resume_yaw_until_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
      yellow_resume_yaw_active_rate_ = 0.0;
    }
    // RCLCPP_INFO(this->get_logger(), "Yellow line stop duration ended");
    if (result_opt && result_opt->line_detected) {
      const auto & result = *result_opt;
      last_detection_time_ = now;
      current_state = TrackState::TRACKING;

      double kp_final = (result.error < 0) ? kp_ * left_turn_ratio_ : kp_; double yaw_cmd = clamp(-(kp_final * result.error), -max_yaw_rate_, max_yaw_rate_);
      double composite_err = std::abs(result.error);
      double speed_scale = 1.0 - speed_gain_damping_ * std::min(1.0, composite_err);
      double speed_cmd = clamp(target_speed_ * speed_scale, min_speed_, max_speed_);

      if (now <= yellow_resume_yaw_until_) {
        yaw_cmd = clamp(yellow_resume_yaw_active_rate_, -max_yaw_rate_, max_yaw_rate_);
      }

      linear_speed = speed_cmd;
      yaw_rate = yaw_cmd;
    }
  } else if (!publish_stop_cmd && result_opt && result_opt->line_detected) {
    const auto & result = *result_opt;
    last_detection_time_ = now;
    current_state = TrackState::TRACKING;

    double kp_final = (result.error < 0) ? kp_ * left_turn_ratio_ : kp_; double yaw_cmd = clamp(-(kp_final * result.error), -max_yaw_rate_, max_yaw_rate_);
    double composite_err = std::abs(result.error);
    double speed_scale = 1.0 - speed_gain_damping_ * std::min(1.0, composite_err);
    double speed_cmd = clamp(target_speed_ * speed_scale, min_speed_, max_speed_);

    if (now <= yellow_resume_yaw_until_) {
      yaw_cmd = clamp(yellow_resume_yaw_active_rate_, -max_yaw_rate_, max_yaw_rate_);
    }

    linear_speed = speed_cmd;
    yaw_rate = yaw_cmd;
  } else if (!publish_stop_cmd) {
    double since_detection = (now - last_detection_time_).seconds();
    if (since_detection >= line_lost_stop_time_) {
      current_state = TrackState::LOST;
      linear_speed = 0.0;
      yaw_rate = 0.0;
      publish_stop_cmd = true;
    } else {
      current_state = last_state_;
      linear_speed = 0.0;
      yaw_rate = 0.0;
    }
  }

  if (!publish_stop_cmd && intersection_active_) {
    if (now <= intersection_end_time_) {
      linear_speed = intersection_turn_speed_;
      yaw_rate = clamp(intersection_turn_yaw_rate_, -max_yaw_rate_, max_yaw_rate_);
      current_state = TrackState::TRACKING;
    } else {
      intersection_active_ = false;
    }
  }

  if (!publish_stop_cmd && yellow_approach_enabled_ && result_opt &&
    result_opt->yellow_approach_detected && !yellow_line_stop_active_ &&
    !permanent_stop_ && !finish_stop_active_ && linear_speed > yellow_approach_speed_)
  {
    linear_speed = yellow_approach_speed_;
    current_state = TrackState::YELLOW_APPROACH;
  }

  // 红色避障优先覆盖普通巡线速度/转向，但不覆盖停车命令。
  if (!publish_stop_cmd && red_turning_ && this->now() < red_turn_end_time_) {
    linear_speed = red_turn_linear_;
    yaw_rate = red_turn_yaw_cmd_;
    current_state = TrackState::RED_AVOIDING;
  } else if (red_turning_) {
    red_turning_ = false;
  }

  if (!publish_stop_cmd && yellow_slowdown_active_ && !permanent_stop_ &&
    !finish_stop_active_ && linear_speed > yellow_slow_speed_)
  {
    linear_speed = yellow_slow_speed_;
  }

  if (!start_enabled_) {
    publish_stop_cmd = true;
  }

  publish_cmd_vel(linear_speed, yaw_rate, publish_stop_cmd);
  if (current_state != last_state_ || current_state != TrackState::TRACKING) {
    publish_state(current_state);
  }
  last_state_ = current_state;
}

std::optional<ProcessingResult> LineFollowerNode::process_image(const cv::Mat & frame_bgr) const
{
  if (frame_bgr.empty()) {
    return std::nullopt;
  }

  const int height = frame_bgr.rows;
  const int width = frame_bgr.cols;

  int roi_top = std::clamp(
    static_cast<int>(height * (1.0 - roi_top_ratio_)), 0, height - 1);
  int roi_bottom = std::clamp(
    height - static_cast<int>(height * roi_bottom_ratio_), roi_top + 1, height);
  const int roi_height = roi_bottom - roi_top;

  cv::Rect roi_rect(0, roi_top, width, roi_height);
  cv::Mat roi = frame_bgr(roi_rect);

  cv::Mat blurred;
  cv::GaussianBlur(roi, blurred, cv::Size(gaussian_kernel_, gaussian_kernel_), 0);

  cv::Mat gray;
  cv::cvtColor(blurred, gray, cv::COLOR_BGR2GRAY);

  cv::Mat hsv;
  cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

  cv::Mat white_mask;
  if (use_white_filter_) {
    cv::Scalar lower(
      static_cast<double>(white_h_low_), 0.0, static_cast<double>(white_v_min_));
    cv::Scalar upper(
      static_cast<double>(white_h_high_), static_cast<double>(white_s_max_), 255.0);
    cv::inRange(hsv, lower, upper, white_mask);
  }

  cv::Mat yellow_mask;
  bool yellow_line_detected = false;
  bool yellow_approach_detected = false;
  double yellow_area_ratio = 0.0;
  if (use_yellow_filter_) {
    cv::Scalar lower(
      static_cast<double>(yellow_h_low_), static_cast<double>(yellow_s_min_), static_cast<double>(yellow_v_min_));
    cv::Scalar upper(
      static_cast<double>(yellow_h_high_), 255.0, 255.0);
    cv::inRange(hsv, lower, upper, yellow_mask);

    int yellow_pixels = cv::countNonZero(yellow_mask);
    yellow_area_ratio = static_cast<double>(yellow_pixels) / static_cast<double>(yellow_mask.total());

    if (yellow_area_ratio >= yellow_area_ratio_threshold_) {
      yellow_line_detected = true;
    }
    if (yellow_approach_enabled_ &&
      yellow_area_ratio >= yellow_approach_area_ratio_threshold_)
    {
      yellow_approach_detected = true;
    }
  }

  int threshold_type = invert_binary_ ? cv::THRESH_BINARY_INV : cv::THRESH_BINARY;
  cv::Mat binary;
  cv::threshold(gray, binary, binary_threshold_, 255, threshold_type);

  if (use_white_filter_ && !white_mask.empty() && !invert_binary_) {
    cv::bitwise_or(binary, white_mask, binary);
  }

  cv::morphologyEx(
    binary, binary, cv::MORPH_CLOSE, morph_element_, cv::Point(-1, -1), morph_iterations_);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

  if (contours.empty()) {
    ProcessingResult result;
    result.yellow_approach_detected = yellow_approach_detected;
    result.yellow_line_detected = yellow_line_detected;
    result.yellow_area_ratio = yellow_area_ratio;
    result.roi_bgr = roi.clone();
    result.binary_mask = binary.clone();
    result.roi_top = roi_rect.y;
    result.roi_bottom = roi_rect.y + roi_rect.height;
    return result;
  }

  double max_area = 0.0;
  size_t best_index = 0;
  for (size_t i = 0; i < contours.size(); ++i) {
    double area = cv::contourArea(contours[i]);
    if (area > max_area) {
      max_area = area;
      best_index = i;
    }
  }

  if (max_area < min_contour_area_) {
    ProcessingResult result;
    result.yellow_approach_detected = yellow_approach_detected;
    result.yellow_line_detected = yellow_line_detected;
    result.yellow_area_ratio = yellow_area_ratio;
    result.roi_bgr = roi.clone();
    result.binary_mask = binary.clone();
    result.roi_top = roi_rect.y;
    result.roi_bottom = roi_rect.y + roi_rect.height;
    return result;
  }

  ProcessingResult result;
  result.line_detected = true;
  result.yellow_approach_detected = yellow_approach_detected;
  result.yellow_line_detected = yellow_line_detected;
  result.yellow_area_ratio = yellow_area_ratio;
  result.bounding_box = cv::boundingRect(contours[best_index]);
  result.roi_bgr = roi.clone();
  result.binary_mask = binary.clone();
  result.roi_top = roi_rect.y;
  result.roi_bottom = roi_rect.y + roi_rect.height;

  const auto moments = cv::moments(contours[best_index]);
  if (moments.m00 != 0.0) {
    result.centroid.x = static_cast<float>(moments.m10 / moments.m00);
    result.centroid.y = static_cast<float>(moments.m01 / moments.m00);
  } else {
    result.centroid = cv::Point2f(
      static_cast<float>(result.bounding_box.x + result.bounding_box.width / 2.0),
      static_cast<float>(result.bounding_box.y + result.bounding_box.height / 2.0));
  }

  result.error = (static_cast<double>(result.centroid.x) - width / 2.0) / (width / 2.0);
  result.error = clamp(result.error, -1.0, 1.0);

  result.state = TrackState::TRACKING;

  return result;
}

bool LineFollowerNode::detect_red(
  const cv::Mat & frame_bgr, int roi_top, int roi_bottom,
  const cv::Point2d & line_centroid_roi, double & side_sign,
  cv::Point2d & red_centroid_img, cv::Point2d & div_base_img,
  cv::Point2d & div_target_img) const
{
  side_sign = 0.0;
  if (!enable_red_filter_ || frame_bgr.empty()) {
    return false;
  }
  roi_top = std::clamp(roi_top, 0, frame_bgr.rows - 1);
  roi_bottom = std::clamp(roi_bottom, roi_top + 1, frame_bgr.rows);
  cv::Rect roi_rect(0, roi_top, frame_bgr.cols, roi_bottom - roi_top);
  cv::Mat roi = frame_bgr(roi_rect);

  cv::Mat hsv;
  cv::cvtColor(roi, hsv, cv::COLOR_BGR2HSV);

  cv::Mat mask1, mask2, mask;
  cv::inRange(
    hsv,
    cv::Scalar(static_cast<double>(red_h_low_1_), static_cast<double>(red_s_min_), static_cast<double>(red_v_min_)),
    cv::Scalar(static_cast<double>(red_h_high_1_), 255.0, 255.0),
    mask1);
  cv::inRange(
    hsv,
    cv::Scalar(static_cast<double>(red_h_low_2_), static_cast<double>(red_s_min_), static_cast<double>(red_v_min_)),
    cv::Scalar(static_cast<double>(red_h_high_2_), 255.0, 255.0),
    mask2);
  cv::bitwise_or(mask1, mask2, mask);

  int red_pixels = cv::countNonZero(mask);
  double area_ratio = static_cast<double>(red_pixels) / static_cast<double>(mask.total());
  if (area_ratio < red_area_ratio_threshold_) {
    return false;
  }

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  if (contours.empty()) {
    return false;
  }
  double max_area = 0.0;
  size_t best = 0;
  for (size_t i = 0; i < contours.size(); ++i) {
    double a = cv::contourArea(contours[i]);
    if (a > max_area) {
      max_area = a;
      best = i;
    }
  }
  if (max_area < red_min_contour_area_) {
    return false;
  }

  cv::Point2d red_centroid;
  const auto moments = cv::moments(contours[best]);
  if (moments.m00 != 0.0) {
    red_centroid = cv::Point2d(moments.m10 / moments.m00, moments.m01 / moments.m00);
  } else {
    cv::Rect box = cv::boundingRect(contours[best]);
    red_centroid = cv::Point2d(
      static_cast<double>(box.x) + static_cast<double>(box.width) * 0.5,
      static_cast<double>(box.y) + static_cast<double>(box.height) * 0.5);
  }

  // 以底部中心为根点，连向巡线质心的向量，判断红色质心在该向量的左/右
  cv::Point2d base(static_cast<double>(mask.cols) * 0.5, static_cast<double>(mask.rows));
  cv::Point2d v_line = line_centroid_roi - base;
  if (red_divider_angle_rad_ != 0.0) {
    double angle = v_line.x >= 0.0 ? red_divider_angle_rad_ : -red_divider_angle_rad_;
    double c = std::cos(angle);
    double s = std::sin(angle);
    v_line = cv::Point2d(v_line.x * c - v_line.y * s, v_line.x * s + v_line.y * c);
  }
  cv::Point2d v_red = red_centroid - base;
  double cross = v_line.x * v_red.y - v_line.y * v_red.x;  // 图像坐标系中 >0 约等于红在右侧
  side_sign = cross >= 0.0 ? 1.0 : -1.0;
  // 输出到整图坐标用于调试叠加
  red_centroid_img = cv::Point2d(red_centroid.x, red_centroid.y + roi_top);
  div_base_img = cv::Point2d(base.x, base.y + roi_top);
  cv::Point2d rotated_target = base + v_line;
  div_target_img = cv::Point2d(rotated_target.x, rotated_target.y + roi_top);
  return true;
}

std::string LineFollowerNode::state_to_string(TrackState state) const
{
  switch (state) {
    case TrackState::TRACKING:
      return "tracking";
    case TrackState::YELLOW_APPROACH:
      return "yellow_approach";
    case TrackState::YELLOW_LINE_DETECTED:
      return "yellow_line_detected";
    case TrackState::RED_AVOIDING:
      return "red_avoiding";
    case TrackState::FINISH_STOP:
      return "finish_stop";
    case TrackState::LOST:
    default:
      return "lost";
  }
}

void LineFollowerNode::publish_cmd_vel(double linear_speed, double yaw_rate, bool publish_stop){
  geometry_msgs::msg::Twist cmd;
  cmd.linear.x = linear_speed;
  cmd.angular.z = yaw_rate;
  cmd_vel_pub_->publish(cmd);

  // If we are stopping due to yellow/permanent stop, also publish stop command to stop topic
  if (stop_pub_ && publish_stop && (linear_speed == 0.0 && yaw_rate == 0.0))
  {
    stop_pub_->publish(cmd);
  }
}

void LineFollowerNode::publish_state(TrackState state)
{
  std_msgs::msg::String msg;
  msg.data = state_to_string(state);
  state_pub_->publish(msg);
}

void LineFollowerNode::publish_centroid(double norm_x, double norm_y)
{
  if (!centroid_pub_) {
    return;
  }
  geometry_msgs::msg::Point msg;
  msg.x = norm_x;
  msg.y = norm_y;
  centroid_pub_->publish(msg);
}

void LineFollowerNode::publish_debug_image(
  const sensor_msgs::msg::Image::ConstSharedPtr & source,
  const cv::Mat & frame_bgr,
  const ProcessingResult & result) const
{
  if (!debug_enabled_ || result.binary_mask.empty()) {
    return;
  }

  cv::Mat overlay = frame_bgr.clone();
  cv::Mat mask_color;
  cv::cvtColor(result.binary_mask, mask_color, cv::COLOR_GRAY2BGR);
  cv::Mat blended;
  cv::addWeighted(result.roi_bgr, 0.6, mask_color, 0.4, 0.0, blended);

  blended.copyTo(
    overlay(cv::Rect(0, result.roi_top, result.roi_bgr.cols, result.roi_bgr.rows)));

  cv::rectangle(
    overlay,
    cv::Rect(
      result.bounding_box.x,
      result.bounding_box.y + result.roi_top,
      result.bounding_box.width,
      result.bounding_box.height),
    cv::Scalar(0, 255, 0),
    2);

  cv::Point centroid(
    static_cast<int>(result.centroid.x),
    static_cast<int>(result.centroid.y) + result.roi_top);
  cv::circle(overlay, centroid, 8, cv::Scalar(0, 0, 255), -1);

  cv::line(
    overlay,
    cv::Point(overlay.cols / 2, result.roi_top),
    cv::Point(overlay.cols / 2, result.roi_bottom),
    cv::Scalar(0, 255, 255),
    2);

  if (result.yellow_line_detected) {
    cv::putText(overlay, "YELLOW STOP DETECTED!", cv::Point(10, 30),
      cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
  } else if (result.yellow_approach_detected) {
    cv::putText(overlay, "YELLOW APPROACH", cv::Point(10, 30),
      cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 255), 2);
  }
  cv::putText(
    overlay,
    "yellow_area=" + std::to_string(result.yellow_area_ratio).substr(0, 6),
    cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);

  if (red_debug_overlay_) {
    // 分界线用于判断红色目标相对巡线方向的左右位置。
    cv::Point2d base(static_cast<double>(frame_bgr.cols) * 0.5, static_cast<double>(result.roi_bottom));
    cv::Point2d v_line(
      static_cast<double>(result.centroid.x) - base.x,
      static_cast<double>(result.centroid.y + result.roi_top) - base.y);
    if (red_divider_angle_rad_ != 0.0) {
      double angle = v_line.x >= 0.0 ? red_divider_angle_rad_ : -red_divider_angle_rad_;
      double c = std::cos(angle);
      double s = std::sin(angle);
      v_line = cv::Point2d(v_line.x * c - v_line.y * s, v_line.x * s + v_line.y * c);
    }
    cv::Point2d target = base + v_line;
    cv::line(overlay, base, target, cv::Scalar(0, 255, 255), 2);
    if (last_red_overlay_valid_) {
      cv::circle(overlay, last_red_center_img_, 8, cv::Scalar(0, 0, 255), -1);
      std::string avoid_text =
        last_red_side_sign_ < 0.0 ? "RED LEFT -> TURN RIGHT" : "RED RIGHT -> TURN LEFT";
      cv::putText(overlay, avoid_text, cv::Point(10, 90),
        cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
    }
  }

  cv_bridge::CvImage debug_image;
  debug_image.header = source->header;
  debug_image.encoding = "bgr8";
  debug_image.image = overlay;

  debug_pub_.publish(debug_image.toImageMsg());
}

double LineFollowerNode::clamp(double value, double min_value, double max_value)
{
  return std::max(min_value, std::min(value, max_value));
}

}  // namespace line_follower




