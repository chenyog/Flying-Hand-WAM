#pragma once
#include <Eigen/Dense>

namespace flygripper_planner{

inline Eigen::MatrixXd cal_timebase_acc(const int order, const double& t){
  double s1, s2, s3;
  s1 = t;
  s2 = s1 * s1;
  s3 = s2 * s1;
  Eigen::Matrix<double, 4, 1> beta;
  switch(order){
  case 0:
    beta << 1.0, s1, s2, s3;
    break;
  case 1:
    beta << 0.0, 1.0, 2.0*s1, 3.0*s2;
    break;
  case 2:
    beta << 0.0, 0.0, 2.0, 6.0 * s1;
    break;
  default:
    std::cout << "[trajopt] cal_timebase error." << std::endl;
    break;
  }
  return beta;
}

inline Eigen::MatrixXd cal_timebase_jerk(const int order, const double& t){
  double s1, s2, s3, s4, s5;
  s1 = t;
  s2 = s1 * s1;
  s3 = s2 * s1;
  s4 = s2 * s2;
  s5 = s4 * s1;
  Eigen::Matrix<double, 6, 1> beta;
  switch(order){
  case 0:
    beta << 1.0, s1, s2, s3, s4, s5;
    break;
  case 1:
    beta << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4;
    break;
  case 2:
    beta << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3;
    break;
  case 3:
    beta << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2;
    break;
  default:
    std::cout << "[trajopt] cal_timebase error." << std::endl;
    break;
  }
  return beta;
}

inline Eigen::MatrixXd cal_timebase_snap(const int order, const double& t){
  double s1, s2, s3, s4, s5, s6, s7;
  s1 = t;
  s2 = s1 * s1;
  s3 = s2 * s1;
  s4 = s2 * s2;
  s5 = s4 * s1;
  s6 = s4 * s2;
  s7 = s4 * s3;
  Eigen::Matrix<double, 8, 1> beta;
  switch(order){
  case 0:
    beta << 1.0, s1, s2, s3, s4, s5, s6, s7;
    break;
  case 1:
    beta << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4, 6.0 * s5, 7.0 * s6;
    break;
  case 2:
    beta << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3, 30.0 * s4, 42.0 * s5;
    break;
  case 3:
    beta << 0.0, 0.0, 0.0, 6.0, 24.0 * s1, 60.0 * s2, 120.0 * s3, 210.0 * s4;
    break;
  case 4:
    beta << 0.0, 0.0, 0.0, 0.0, 24.0, 120.0 * s1, 360.0 * s2, 840.0 * s3;
    break;
  default:
    std::cout << "[trajopt] cal_timebase error." << std::endl;
    break;
  }
  return beta;
}

inline double smoothedL1(const double& x, double& grad) {
  static double mu = 0.01;
  if (x < 0.0) {
    return 0.0;
  } else if (x > mu) {
    grad = 1.0;
    return x - 0.5 * mu;
  } else {
    const double xdmu = x / mu;
    const double sqrxdmu = xdmu * xdmu;
    const double mumxd2 = mu - 0.5 * x;
    grad = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
    return mumxd2 * sqrxdmu * xdmu;
  }
}

inline double smoothed01(const double& x, double& grad, const double mu) {
  static double mu4 = mu * mu * mu * mu;
  static double mu4_1 = 1.0 / mu4;
  if (x < -mu) {
    grad = 0;
    return 0;
  } else if (x < 0) {
    double y = x + mu;
    double y2 = y * y;
    grad = y2 * (mu - 2 * x) * mu4_1;
    return 0.5 * y2 * y * (mu - x) * mu4_1;
  } else if (x < mu) {
    double y = x - mu;
    double y2 = y * y;
    grad = y2 * (mu + 2 * x) * mu4_1;
    return 0.5 * y2 * y * (mu + x) * mu4_1 + 1;
  } else {
    grad = 0;
    return 1;
  }
}

inline double penF(const double& x, double& grad) {
  static double eps = 0.05;
  static double eps2 = eps * eps;
  static double eps3 = eps * eps2;
  if (x < 2 * eps) {
    double x2 = x * x;
    double x3 = x * x2;
    double x4 = x2 * x2;
    grad = 12 / eps2 * x2 - 4 / eps3 * x3;
    return 4 / eps2 * x3 - x4 / eps3;
  } else {
    grad = 16;
    return 16 * (x - eps);
  }
}

inline double penF2(const double& x, double& grad) {
  double x2 = x * x;
  grad = 3 * x2;
  return x * x2;
}

//! t => T
inline double expC2(double t) {
  return t > 0.0 ? ((0.5 * t + 1.0) * t + 1.0)
                 : 1.0 / ((0.5 * t - 1.0) * t + 1.0);
}

//! T => t
inline double logC2(double T) {
  return T > 1.0 ? (sqrt(2.0 * T - 1.0) - 1.0) : (1.0 - sqrt(2.0 / T - 1.0));
}

//! element grad_T => grad_t
inline double gdT2t(double t) {
  if (t > 0) {
    return t + 1.0;
  } else {
    double denSqrt = (0.5 * t - 1.0) * t + 1.0;
    return (1.0 - t) / (denSqrt * denSqrt);
  }
}

//! non-uniform grad_T => grad_t
inline void addLayerTGrad(const Eigen::Ref<const Eigen::VectorXd>& t,
                   const Eigen::Ref<const Eigen::VectorXd>& grad_T,
                   Eigen::Ref<Eigen::VectorXd> grad_t) {
  int M = t.size();
  for (int i = 0; i < M; ++i) {
    grad_t(i) = grad_T(i) * gdT2t(t(i));
  }
  return;
}

//! uniform grad_T => grad_t
inline void addLayerTGrad(const Eigen::Ref<const Eigen::VectorXd>& t,
                   const double& sT,
                   const Eigen::Ref<const Eigen::VectorXd>& grad_T,
                   Eigen::Ref<Eigen::VectorXd> grad_t) {
  int Ms1 = t.size();
  Eigen::VectorXd gFree = sT * grad_T.head(Ms1);
  double gTail = sT * grad_T(Ms1);
  Eigen::VectorXd dExpTau(Ms1);
  double expTauSum = 0.0, gFreeDotExpTau = 0.0;
  double denSqrt, expTau;
  for (int i = 0; i < Ms1; i++) {
    if (t(i) > 0) {
      expTau = (0.5 * t(i) + 1.0) * t(i) + 1.0;
      dExpTau(i) = t(i) + 1.0;
      expTauSum += expTau;
      gFreeDotExpTau += expTau * gFree(i);
    } else {
      denSqrt = (0.5 * t(i) - 1.0) * t(i) + 1.0;
      expTau = 1.0 / denSqrt;
      dExpTau(i) = (1.0 - t(i)) / (denSqrt * denSqrt);
      expTauSum += expTau;
      gFreeDotExpTau += expTau * gFree(i);
    }
  }
  denSqrt = expTauSum + 1.0;
  grad_t = (gFree.array() - gTail) * dExpTau.array() / denSqrt -
      (gFreeDotExpTau - gTail * expTauSum) * dExpTau.array() / (denSqrt * denSqrt);
}

//! non-uniform t => T
inline void forwardT(const Eigen::Ref<const Eigen::VectorXd>& t, Eigen::Ref<Eigen::VectorXd> vecT) {
  int M = t.size();
  for (int i = 0; i < M; ++i) {
    vecT(i) = expC2(t(i));
  }
  return;
}

//! non-uniform T => t
inline void backwardT(const Eigen::Ref<const Eigen::VectorXd>& vecT, Eigen::Ref<Eigen::VectorXd> t) {
  int M = vecT.size();
  for (int i = 0; i < M; ++i) {
    t(i) = logC2(vecT(i));
  }
  return;
}

//! uniform t => T
inline void forwardT(const double& t, double& T){
  T = expC2(t);
}

//! uniform T => t
inline void backwardT(const double& T, double& t){
  t = logC2(T);
}


}