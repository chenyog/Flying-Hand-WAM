#pragma once
#include "base.hpp"

namespace flygripper_common{

static constexpr Scalar RAD2DEG = 180.0 / M_PI;

// For plotting purpose
class Quat2RPY{
 public:
  Quat2RPY(bool wrap=true, bool deg=true):
                                                wrap_(wrap), deg_(deg){}
  Vector<3> convert(const Quaternion & quat){
    Quaternion q = quat.normalized();
    double q_x = q.x();
    double q_y = q.y();
    double q_z = q.z();
    double q_w = q.w();
    double roll, pitch, yaw;
    // roll (x-axis rotation)
    double sinr_cosp = 2 * (q_w * q_x + q_y * q_z);
    double cosr_cosp = 1 - 2 * (q_x * q_x + q_y * q_y);
    roll = std::atan2(sinr_cosp, cosr_cosp);

    // pitch (y-axis rotation)
    double sinp = 2 * (q_w * q_y - q_z * q_x);
    if (std::abs(sinp) >= 1)
    {
      pitch = std::copysign(M_PI_2, sinp);  // use 90 degrees if out of range
    }
    else
    {
      pitch = std::asin(sinp);
    }
    // yaw (z-axis rotation)
    double siny_cosp = 2 * (q_w * q_z + q_x * q_y);
    double cosy_cosp = 1 - 2 * (q_y * q_y + q_z * q_z);
    yaw = std::atan2(siny_cosp, cosy_cosp);

    const double WRAP_ANGLE = M_PI * 2.0;
    const double WRAP_THRESHOLD = M_PI * 1.95;

    //--------- wrap ------
    if (!first_ && wrap_)
    {
      first_ = false;
      if ((roll - prev_roll_) > WRAP_THRESHOLD)
      {
        roll_offset_ -= WRAP_ANGLE;
      }
      else if ((prev_roll_ - roll) > WRAP_THRESHOLD)
      {
        roll_offset_ += WRAP_ANGLE;
      }

      if ((pitch - prev_pitch_) > WRAP_THRESHOLD)
      {
        pitch_offset_ -= WRAP_ANGLE;
      }
      else if ((prev_pitch_ - pitch) > WRAP_THRESHOLD)
      {
        pitch_offset_ += WRAP_ANGLE;
      }

      if ((yaw - prev_yaw_) > WRAP_THRESHOLD)
      {
        yaw_offset_ -= WRAP_ANGLE;
      }
      else if ((prev_yaw_ - yaw) > WRAP_THRESHOLD)
      {
        yaw_offset_ += WRAP_ANGLE;
      }
    }

    prev_pitch_ = pitch;
    prev_roll_ = roll;
    prev_yaw_ = yaw;

    Vector<3> rpy(roll+roll_offset_,
                  pitch+pitch_offset_,
                  yaw+yaw_offset_);
    if (deg_)
    {
      return RAD2DEG * rpy;
    } else{
      return rpy;
    }
  }

 private:
  bool wrap_{true};
  bool deg_{true};
  bool first_{true};

  Scalar prev_roll_{0.0};
  Scalar prev_pitch_{0.0};
  Scalar prev_yaw_{0.0};

  Scalar roll_offset_{0.0};
  Scalar pitch_offset_{0.0};
  Scalar yaw_offset_{0.0};
};

inline Scalar quat2Yaw(const Quaternion& q)
{
  const Vector<3> B_x = q * Vector<3>::UnitX();
  const Vector<3> B_x_proj = Vector<3>(B_x.x(), B_x.y(), 0);
  if (B_x_proj.norm() < 1e-3) return 0;
  const Vector<3> B_x_proj_norm = B_x_proj.normalized();
  const Vector<3> cross = Vector<3>::UnitX().cross(B_x_proj_norm);
  const Scalar angle = asin(cross.z());
  if (B_x_proj_norm.x() >= 0.0) return angle;
  if (B_x_proj_norm.y() >= 0.0) return M_PI - angle;
  return -M_PI - angle;
}

inline Vector<3> quat2RPY(const Quaternion& q)
{
  Vector<3> euler_angles;
  euler_angles(0) = atan2(
      2.0 * q.w() * q.x() + 2.0 * q.y() * q.z(),
      q.w() * q.w() - q.x() * q.x() - q.y() * q.y() + q.z() * q.z());
  euler_angles(1) = -asin(2.0 * q.x() * q.z() - 2.0 * q.w() * q.y());
  euler_angles(2) = atan2(
      2.0 * q.w() * q.z() + 2.0 * q.x() * q.y(),
      q.w() * q.w() + q.x() * q.x() - q.y() * q.y() - q.z() * q.z());
  return euler_angles;
}


inline Eigen::Matrix2d angle2RotMatrix(const double& angle){
  Eigen::Rotation2Dd rot(angle);
  return rot.toRotationMatrix();
}

inline double RotMatrix2angle(const Eigen::Matrix2d& mat){
  Eigen::Rotation2Dd rot;
  rot.fromRotationMatrix(mat);
  return rot.angle();
}

// calculate (ang_s -> ang_t) on SO(2), return within -pi~pi, ang_s and ang_t can be arbitrary value
inline double getAngleError(const double& ang_s, const double& ang_t){
  Eigen::Matrix2d s = angle2RotMatrix(ang_s);
  Eigen::Matrix2d t = angle2RotMatrix(ang_t);
  Eigen::Matrix2d err = t*s.inverse();
  double err_angle = RotMatrix2angle(err); // -pi~pi
  return err_angle;
}

}
