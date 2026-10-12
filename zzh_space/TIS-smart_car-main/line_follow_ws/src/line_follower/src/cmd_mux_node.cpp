#include <memory>
#include <string>

#include <geometry_msgs/msg/twist.hpp>
#include <rclcpp/rclcpp.hpp>

class CmdMuxNode : public rclcpp::Node
{
public:
  CmdMuxNode()
  : Node("cmd_mux_node")
  {
    high_topic_ = this->declare_parameter<std::string>("high_topic", "/cmd_vel_detection");
    low_topic_ = this->declare_parameter<std::string>("low_topic", "/cmd_vel_follower");
    output_topic_ = this->declare_parameter<std::string>("output_topic", "/cmd_vel");
    high_timeout_ = this->declare_parameter<double>("high_timeout", 0.5);
    stop_topic_ = this->declare_parameter<std::string>("stop_topic", "/cmd_vel_stop");
    stop_timeout_ = this->declare_parameter<double>("stop_timeout", 0.5);

    high_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      high_topic_, rclcpp::QoS(10),
      std::bind(&CmdMuxNode::high_callback, this, std::placeholders::_1));
    low_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      low_topic_, rclcpp::QoS(10),
      std::bind(&CmdMuxNode::low_callback, this, std::placeholders::_1));
    stop_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      stop_topic_, rclcpp::QoS(10),
      std::bind(&CmdMuxNode::stop_callback, this, std::placeholders::_1));

    pub_ = this->create_publisher<geometry_msgs::msg::Twist>(output_topic_, 10);

    RCLCPP_INFO(
      get_logger(),
      "Cmd mux: stop=%s timeout=%.2f, high=%s timeout=%.2f, low=%s, output=%s",
      stop_topic_.c_str(), stop_timeout_, high_topic_.c_str(), high_timeout_,
      low_topic_.c_str(), output_topic_.c_str());
  }

private:
  void high_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    auto now = this->get_clock()->now();
    double now_seconds = now.seconds();
    
    // If recent stop exists, keep stop priority
    if (now_seconds - last_stop_time_seconds_ <= stop_timeout_) {
      publish(last_stop_msg_);
      return;
    }
    last_high_msg_ = *msg;
    last_high_time_seconds_ = now_seconds;
    publish(*msg);
  }

  void stop_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    auto now = this->get_clock()->now();
    last_stop_msg_ = *msg;
    last_stop_time_seconds_ = now.seconds();
    publish(*msg);
  }

  void low_callback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    auto now = this->get_clock()->now();
    double now_seconds = now.seconds();
    
    // Highest priority: recent stop cmd
    if (now_seconds - last_stop_time_seconds_ <= stop_timeout_) {
      publish(last_stop_msg_);
      return;
    }
    // Next: high priority if not stale
    if (now_seconds - last_high_time_seconds_ <= high_timeout_) {
      publish(last_high_msg_);
      return;
    }
    // Else forward low priority
    if (pub_) {
      publish(*msg);
    }
  }

  void publish(const geometry_msgs::msg::Twist & msg)
  {
    pub_->publish(msg);
  }

  std::string high_topic_;
  std::string low_topic_;
  std::string output_topic_;
  std::string stop_topic_;
  double high_timeout_{0.5};
  double stop_timeout_{0.5};

  double last_high_time_seconds_{0.0};
  double last_stop_time_seconds_{0.0};
  geometry_msgs::msg::Twist last_high_msg_;
  geometry_msgs::msg::Twist last_stop_msg_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr high_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr low_sub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr stop_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CmdMuxNode>());
  rclcpp::shutdown();
  return 0;
}
