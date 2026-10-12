#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <vision_msgs/msg/detection2_d_array.hpp>
#include <geometry_msgs/msg/point.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

namespace
{
constexpr double kDefaultConfThreshold = 0.6;
constexpr int kDefaultDetFrameWidth = 640;
constexpr int kDefaultDetFrameHeight = 480;
constexpr int kDefaultFrameWindow = 5;
constexpr double kDefaultTurnCooldown = 0.3;
constexpr double kDefaultTurnDuration = 0.6;
constexpr double kDefaultTurnLinear = 0.3;
constexpr double kDefaultYawGain = 1.0;
constexpr double kDefaultMaxYaw = 1.2;
constexpr double kDefaultYAlignTolerance = 0.05;
constexpr double kDefaultTurnDirectionSign = -1.0;
constexpr double kDefaultCentroidTimeout = 0.5;
constexpr int kDefaultClassVoteWindow = 4;
constexpr int kDefaultSameClassRequired = 2;
constexpr const char * kDefaultLineCentroidTopic = "line_follower/centroid";
constexpr const char * kDefaultStateTopic = "line_follower/track_state";
}  // namespace

class DetectionTurnController : public rclcpp::Node
{
public:
  DetectionTurnController()
  : Node("detection_turn_controller")
  {
    detection_topic_ = this->declare_parameter<std::string>("detection_topic", "/yolo_detections");
    state_topic_ = this->declare_parameter<std::string>("state_topic", kDefaultStateTopic);
    cmd_vel_topic_ = this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel_detection");

    conf_threshold_ = this->declare_parameter<double>("confidence_threshold", kDefaultConfThreshold);
    det_frame_width_ = this->declare_parameter<int>("det_frame_width", kDefaultDetFrameWidth);
    det_frame_height_ = this->declare_parameter<int>("det_frame_height", kDefaultDetFrameHeight);
    frame_window_ = this->declare_parameter<int>("frame_window", kDefaultFrameWindow);
    turn_cooldown_ = this->declare_parameter<double>("turn_cooldown", kDefaultTurnCooldown);
    turn_duration_ = this->declare_parameter<double>("turn_duration", kDefaultTurnDuration);
    turn_linear_ = this->declare_parameter<double>("turn_linear", kDefaultTurnLinear);
    yaw_gain_ = this->declare_parameter<double>("yaw_gain", kDefaultYawGain);
    max_yaw_rate_ = this->declare_parameter<double>("max_yaw_rate", kDefaultMaxYaw);
    y_align_tolerance_ = this->declare_parameter<double>("y_align_tolerance", kDefaultYAlignTolerance);
    turn_direction_sign_ = this->declare_parameter<double>("turn_direction_sign", kDefaultTurnDirectionSign);
    class_vote_window_ = std::max<int>(
      1, this->declare_parameter<int>("class_vote_window", kDefaultClassVoteWindow));
    same_class_required_ = std::max<int>(
      1, this->declare_parameter<int>("same_class_required", kDefaultSameClassRequired));

    line_centroid_topic_ = this->declare_parameter<std::string>(
      "line_centroid_topic", kDefaultLineCentroidTopic);
    centroid_timeout_ = this->declare_parameter<double>("centroid_timeout", kDefaultCentroidTimeout);

    detection_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
      detection_topic_, rclcpp::QoS(10).best_effort(),
      std::bind(&DetectionTurnController::detection_callback, this, std::placeholders::_1));

    centroid_sub_ = this->create_subscription<geometry_msgs::msg::Point>(
      line_centroid_topic_, rclcpp::QoS(10),
      std::bind(&DetectionTurnController::centroid_callback, this, std::placeholders::_1));

    state_sub_ = this->create_subscription<std_msgs::msg::String>(
      state_topic_, rclcpp::QoS(10),
      std::bind(&DetectionTurnController::state_callback, this, std::placeholders::_1));

    cmd_pub_ = this->create_publisher<geometry_msgs::msg::Twist>(cmd_vel_topic_, 10);

    publish_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(50), std::bind(&DetectionTurnController::on_timer, this));

    RCLCPP_INFO(
      get_logger(),
      "Detection turn controller listening detections '%s', centroid '%s', publishing '%s'",
      detection_topic_.c_str(), line_centroid_topic_.c_str(), cmd_vel_topic_.c_str());
  }

private:
  struct Detection
  {
    std::string class_id;
    double center_x;
    double center_y;
    double width;
    double height;
    double confidence;
  };

  void detection_callback(const vision_msgs::msg::Detection2DArray::SharedPtr msg)
  {
    if (!msg) {
      return;
    }
    
    // 使用clock获取时间，避免时间源不匹配
    auto now = this->get_clock()->now();

    // 正在转弯或冷却中，忽略。0 时间戳表示还没有进入过冷却。
    if (turning_ ||
      (turn_cooldown_until_.nanoseconds() != 0 && now < turn_cooldown_until_))
    {
      return;
    }

    // 检查质心是否有效 - 使用seconds()相减而不是直接相减rclcpp::Time
    double time_since_last_centroid = (now.seconds() - last_centroid_time_seconds_);
    if (time_since_last_centroid > centroid_timeout_ || last_centroid_time_seconds_ == 0.0) {
      return;
    }

    // 启动新一轮
    if (!session_active_) {
      session_active_ = true;
      session_frames_ = 0;
      best_detection_ = Detection{};
      best_x_diff_ = 0.0;
      best_y_diff_ = 0.0;
      session_class_ids_.clear();
    }

    // 收集本帧检测（从YOLO的Detection2DArray格式）
    std::vector<Detection> detections;
    for (const auto & detection : msg->detections) {
      // 检查是否是占位检测（99999.0表示没有检测）
      if (detection.bbox.center.position.x > 99990.0) {
        continue;
      }

      double cx = detection.bbox.center.position.x;
      double cy = detection.bbox.center.position.y;
      double width = detection.bbox.size_x;
      double height = detection.bbox.size_y;
      
      // 获取置信度（从results中）
      double conf = 1.0;
      std::string class_id;
      if (!detection.results.empty()) {
        conf = detection.results[0].hypothesis.score;
        class_id = detection.results[0].hypothesis.class_id;
      }

      if (conf < conf_threshold_) {
        continue;
      }

      detections.push_back(Detection{class_id, cx, cy, width, height, conf});
    }

    if (!detections.empty()) {
      // 选择置信度最高的检测
      const auto chosen = *std::max_element(
        detections.begin(), detections.end(),
        [](const Detection & a, const Detection & b) {
          return a.confidence < b.confidence;
        });

      const double half_w = det_frame_width_ > 0 ? static_cast<double>(det_frame_width_) * 0.5 : 320.0;
      const double half_h = det_frame_height_ > 0 ? static_cast<double>(det_frame_height_) * 0.5 : 240.0;

      double det_x_norm = (half_w > 0.0) ? (chosen.center_x / (half_w * 2.0)) : 0.5;
      double det_y_norm = (half_h > 0.0) ? (chosen.center_y / (half_h * 2.0)) : 0.5;
      det_x_norm = std::clamp(det_x_norm, 0.0, 1.0);
      det_y_norm = std::clamp(det_y_norm, 0.0, 1.0);

      double y_diff = det_y_norm - last_centroid_norm_y_;
      if (std::abs(y_diff) <= y_align_tolerance_) {
        double x_diff = det_x_norm - last_centroid_norm_x_;
        if (!chosen.class_id.empty()) {
          session_class_ids_.push_back(chosen.class_id);
          if (static_cast<int>(session_class_ids_.size()) > class_vote_window_) {
            session_class_ids_.erase(session_class_ids_.begin());
          }
        }
        if (chosen.confidence > best_detection_.confidence) {
          best_detection_ = chosen;
          best_x_diff_ = x_diff;
          best_y_diff_ = y_diff;
        }
      }
    }

    session_frames_++;

    if (session_frames_ >= frame_window_) {
      trigger_turn(now);
      session_active_ = false;
    }
  }

  void trigger_turn(const rclcpp::Time & now)
  {
    if (best_detection_.confidence <= 0.0) {
      return;
    }
    if (!best_detection_.class_id.empty() && same_class_required_ > 1) {
      std::unordered_map<std::string, int> counts;
      for (const auto & class_id : session_class_ids_) {
        counts[class_id]++;
      }
      if (counts[best_detection_.class_id] < same_class_required_) {
        RCLCPP_DEBUG(
          get_logger(),
          "Skip detection turn: class '%s' voted %d/%d.",
          best_detection_.class_id.c_str(), counts[best_detection_.class_id],
          same_class_required_);
        return;
      }
    }

    double yaw = std::clamp(turn_direction_sign_ * best_x_diff_ * yaw_gain_, -max_yaw_rate_, max_yaw_rate_);
    if (yaw == 0.0) {
      return;
    }
    double lin = turn_linear_;
    double dur = turn_duration_;

    geometry_msgs::msg::Twist cmd;
    cmd.linear.x = lin;
    cmd.angular.z = yaw;

    turning_cmd_ = cmd;
    turning_end_time_ = now + rclcpp::Duration::from_seconds(dur);
    turn_cooldown_until_ = turning_end_time_ + rclcpp::Duration::from_seconds(turn_cooldown_);
    turning_ = true;

    RCLCPP_INFO(
      get_logger(), "Triggered turn: yaw=%.3f, linear=%.2f, duration=%.2f",
      yaw, lin, dur);
  }

  void centroid_callback(const geometry_msgs::msg::Point::SharedPtr msg)
  {
    auto now = this->get_clock()->now();
    last_centroid_time_seconds_ = now.seconds();

    if (msg->x >= 0.0 && msg->x <= 1.0 && msg->y >= 0.0 && msg->y <= 1.0) {
      last_centroid_norm_x_ = std::clamp(msg->x, 0.0, 1.0);
      last_centroid_norm_y_ = std::clamp(msg->y, 0.0, 1.0);
      return;
    }
    
    const double half_w = det_frame_width_ > 0 ? static_cast<double>(det_frame_width_) * 0.5 : 320.0;
    const double half_h = det_frame_height_ > 0 ? static_cast<double>(det_frame_height_) * 0.5 : 240.0;

    last_centroid_norm_x_ = (half_w > 0.0) ? (msg->x / (half_w * 2.0)) : 0.5;
    last_centroid_norm_y_ = (half_h > 0.0) ? (msg->y / (half_h * 2.0)) : 0.5;
    last_centroid_norm_x_ = std::clamp(last_centroid_norm_x_, 0.0, 1.0);
    last_centroid_norm_y_ = std::clamp(last_centroid_norm_y_, 0.0, 1.0);
  }

  void state_callback(const std_msgs::msg::String::SharedPtr msg)
  {
    (void)msg;
  }

  void on_timer()
  {
    auto now = this->get_clock()->now();
    
    if (turning_) {
      cmd_pub_->publish(turning_cmd_);
      if (now >= turning_end_time_) {
        turning_ = false;
        geometry_msgs::msg::Twist stop_cmd;
        stop_cmd.linear.x = 0.0;
        stop_cmd.angular.z = 0.0;
        cmd_pub_->publish(stop_cmd);
        RCLCPP_INFO(get_logger(), "Turn completed");
      }
    }
  }

  std::string detection_topic_;
  std::string state_topic_;
  std::string cmd_vel_topic_;
  std::string line_centroid_topic_;

  double conf_threshold_;
  int det_frame_width_;
  int det_frame_height_;
  int frame_window_;
  double turn_cooldown_;
  double turn_duration_;
  double turn_linear_;
  double yaw_gain_;
  double max_yaw_rate_;
  double y_align_tolerance_;
  double turn_direction_sign_;
  int class_vote_window_;
  int same_class_required_;
  double centroid_timeout_;

  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Point>::SharedPtr centroid_sub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr state_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::TimerBase::SharedPtr publish_timer_;

  Detection best_detection_;
  std::vector<std::string> session_class_ids_;
  double best_x_diff_{0.0};
  double best_y_diff_{0.0};
  double last_centroid_norm_x_{0.5};
  double last_centroid_norm_y_{0.5};
  double last_centroid_time_seconds_{0.0};  // 改用double存储seconds
  bool session_active_{false};
  int session_frames_{0};
  bool turning_{false};
  geometry_msgs::msg::Twist turning_cmd_;
  rclcpp::Time turning_end_time_;
  rclcpp::Time turn_cooldown_until_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DetectionTurnController>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
