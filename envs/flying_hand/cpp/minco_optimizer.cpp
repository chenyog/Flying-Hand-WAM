// Python binding for the deployment Flying-Hand MINCO implementation.
//
// The polynomial solver, adjoint gradient propagation, positive-time mapping,
// smoothed penalties, and L-BFGS implementation below come directly from the
// deployment implementation vendored beside this file.  This wrapper only
// removes the ROS-facing TrajOpt shell and adapts its fixed-waypoint goal
// objective to NumPy arrays.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Eigen>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "lbfgs.hpp"
#include "minco.hpp"
#include "rotation.hpp"
#include "traj_opt_utils.hpp"

namespace py = pybind11;

namespace {

struct OptimizerConfig {
  double max_velocity;
  double max_acceleration;
  double max_yaw_rate;
  int integration_steps;
  double rho_time;
  double rho_velocity;
  double rho_acceleration;
  double rho_yaw_rate;
  double rho_yaw_alignment;
  int max_iterations;
};

template <typename T>
T required(const py::dict &config, const char *name) {
  if (!config.contains(name)) {
    throw std::invalid_argument(std::string("Missing MINCO parameter: ") + name);
  }
  return py::cast<T>(config[name]);
}

OptimizerConfig parseConfig(const py::dict &config) {
  OptimizerConfig result{
      required<double>(config, "plan_max_vel"),
      required<double>(config, "plan_max_acc"),
      required<double>(config, "plan_max_dyaw"),
      required<int>(config, "K"),
      required<double>(config, "rhoT"),
      required<double>(config, "rhoV"),
      required<double>(config, "rhoA"),
      required<double>(config, "rhoDYaw"),
      required<double>(config, "rhoYawAlignmentAngle"),
      required<int>(config, "max_iteration"),
  };
  if (result.integration_steps < 1 || result.max_iterations < 1) {
    throw std::invalid_argument("K and max_iteration must be positive");
  }
  return result;
}

Eigen::MatrixXd copyPoints(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &array) {
  if (array.ndim() != 2 || array.shape(1) != 3 || array.shape(0) < 2) {
    throw std::invalid_argument("points must have shape (piece_count + 1, 3)");
  }
  const auto input = array.unchecked<2>();
  Eigen::MatrixXd points(3, input.shape(0));
  for (py::ssize_t row = 0; row < input.shape(0); ++row) {
    for (py::ssize_t column = 0; column < 3; ++column) {
      points(column, row) = input(row, column);
    }
  }
  return points;
}

Eigen::VectorXd copyVector(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &array,
    const Eigen::Index expected_size,
    const char *name) {
  if (array.ndim() != 1 || array.shape(0) != expected_size) {
    throw std::invalid_argument(
        std::string(name) + " must be a one-dimensional array of size " +
        std::to_string(expected_size));
  }
  const auto input = array.unchecked<1>();
  Eigen::VectorXd values(expected_size);
  for (Eigen::Index index = 0; index < expected_size; ++index) {
    values(index) = input(index);
  }
  return values;
}

py::array_t<double> generatePositionCoefficients(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &points,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &times) {
  const Eigen::MatrixXd copied_points = copyPoints(points);
  const int piece_count = static_cast<int>(copied_points.cols()) - 1;
  const Eigen::VectorXd copied_times =
      copyVector(times, piece_count, "times");
  if ((copied_times.array() <= 0.0).any()) {
    throw std::invalid_argument("times must contain only positive durations");
  }

  Eigen::Matrix<double, 3, 3> initial_state;
  Eigen::Matrix<double, 3, 3> terminal_state;
  initial_state.setZero();
  terminal_state.setZero();
  initial_state.col(0) = copied_points.col(0);
  terminal_state.col(0) = copied_points.col(copied_points.cols() - 1);

  minco::MINCO<3, 3, false> trajectory;
  trajectory.reset(initial_state, terminal_state, piece_count);
  trajectory.generate(
      copied_points.middleCols(1, std::max(0, piece_count - 1)),
      copied_times);

  py::array_t<double> output(std::vector<py::ssize_t>{piece_count, 6, 3});
  auto output_view = output.mutable_unchecked<3>();
  for (int piece = 0; piece < piece_count; ++piece) {
    for (int coefficient = 0; coefficient < 6; ++coefficient) {
      for (int axis = 0; axis < 3; ++axis) {
        output_view(piece, coefficient, axis) =
            trajectory.b(6 * piece + coefficient, axis);
      }
    }
  }
  return output;
}

py::array_t<double> generateYawCoefficients(
    const py::array_t<double, py::array::c_style | py::array::forcecast>
        &yaw_points,
    const py::array_t<double, py::array::c_style | py::array::forcecast> &times,
    const double initial_yaw_rate, const double terminal_yaw_rate) {
  if (times.ndim() != 1 || times.shape(0) < 1) {
    throw std::invalid_argument("times must be a non-empty one-dimensional array");
  }
  const int piece_count = static_cast<int>(times.shape(0));
  const Eigen::VectorXd copied_times = copyVector(times, piece_count, "times");
  const Eigen::VectorXd copied_yaws =
      copyVector(yaw_points, piece_count + 1, "yaw_points");
  if ((copied_times.array() <= 0.0).any()) {
    throw std::invalid_argument("times must contain only positive durations");
  }

  Eigen::Matrix<double, 1, 2> initial_state;
  Eigen::Matrix<double, 1, 2> terminal_state;
  initial_state << copied_yaws(0), initial_yaw_rate;
  terminal_state << copied_yaws(copied_yaws.size() - 1), terminal_yaw_rate;

  minco::MINCO<1, 2, false> trajectory;
  trajectory.reset(initial_state, terminal_state, piece_count);
  trajectory.generate(
      copied_yaws.segment(1, std::max(0, piece_count - 1)).transpose(),
      copied_times);

  py::array_t<double> output(std::vector<py::ssize_t>{piece_count, 4});
  auto output_view = output.mutable_unchecked<2>();
  for (int piece = 0; piece < piece_count; ++piece) {
    for (int coefficient = 0; coefficient < 4; ++coefficient) {
      output_view(piece, coefficient) =
          trajectory.b(4 * piece + coefficient, 0);
    }
  }
  return output;
}

class FixedWaypointTimeOptimizer {
 public:
  FixedWaypointTimeOptimizer(const Eigen::MatrixXd &points,
                             const Eigen::VectorXd &yaw_points,
                             const double initial_yaw_rate,
                             const OptimizerConfig &config)
      : yaw_points_(yaw_points),
        config_(config),
        piece_count_(static_cast<int>(points.cols()) - 1),
        interior_points_(points.middleCols(1, std::max(0, piece_count_ - 1))),
        interior_yaws_(yaw_points.segment(1, std::max(0, piece_count_ - 1))) {
    if (yaw_points.size() != points.cols()) {
      throw std::invalid_argument(
          "yaw_points must contain one value per positional waypoint");
    }

    Eigen::Matrix<double, 3, 3> initial_state;
    Eigen::Matrix<double, 3, 3> terminal_state;
    initial_state.setZero();
    terminal_state.setZero();
    initial_state.col(0) = points.col(0);
    terminal_state.col(0) = points.col(points.cols() - 1);

    Eigen::Matrix<double, 1, 2> initial_yaw;
    Eigen::Matrix<double, 1, 2> terminal_yaw;
    initial_yaw << yaw_points(0), initial_yaw_rate;
    terminal_yaw << yaw_points(yaw_points.size() - 1), 0.0;

    xyz_.reset(initial_state, terminal_state, piece_count_);
    yaw_.reset(initial_yaw, terminal_yaw, piece_count_);
  }

  double evaluate(const Eigen::Ref<const Eigen::VectorXd> &raw_times,
                  Eigen::Ref<Eigen::VectorXd> gradient,
                  const bool count_evaluation = true) {
    if (raw_times.size() != piece_count_) {
      throw std::invalid_argument("raw time vector has the wrong size");
    }
    if (count_evaluation) {
      ++function_evaluations_;
    }

    Eigen::VectorXd times(piece_count_);
    flygripper_planner::forwardT(raw_times, times);
    xyz_.generate(interior_points_, times);
    yaw_.generate(interior_yaws_.transpose(), times);

    double jerk_energy = 0.0;
    double yaw_acceleration_energy = 0.0;
    xyz_.getEnergy(jerk_energy);
    yaw_.getEnergy(yaw_acceleration_energy);
    xyz_.calGrads_CT();
    yaw_.calGrads_CT();

    double cost = jerk_energy + yaw_acceleration_energy;
    addIntegratedPenalties(cost);

    // The waypoints are deliberately fixed, exactly as fix_p=true in the
    // deployment goal planner.  We still propagate coefficient gradients to
    // segment times through MINCO's adjoint banded solve.
    xyz_.calGrads_PT();
    yaw_.calGrads_PT();
    xyz_.gdT.array() += config_.rho_time;
    cost += config_.rho_time * times.sum();
    flygripper_planner::addLayerTGrad(
        raw_times, xyz_.gdT + yaw_.gdT, gradient);

    if (!std::isfinite(cost) || !gradient.allFinite()) {
      gradient.setZero();
      return std::numeric_limits<double>::max() / 16.0;
    }
    return cost;
  }

  py::dict optimize(const Eigen::VectorXd &initial_times) {
    if (initial_times.size() != piece_count_ ||
        (initial_times.array() <= 0.0).any()) {
      throw std::invalid_argument(
          "initial_times must contain one positive value per segment");
    }

    Eigen::VectorXd raw_times(piece_count_);
    flygripper_planner::backwardT(initial_times, raw_times);
    Eigen::VectorXd initial_gradient(piece_count_);
    const double initial_cost = evaluate(raw_times, initial_gradient, false);

    lbfgs::lbfgs_parameter_t parameters;
    lbfgs::lbfgs_load_default_parameters(&parameters);
    parameters.mem_size = 16;
    parameters.past = 3;
    parameters.g_epsilon = 0.0;
    parameters.min_step = 1.0e-16;
    parameters.delta = 1.0e-4;
    parameters.line_search_type = 0;
    parameters.max_iterations = config_.max_iterations;

    function_evaluations_ = 0;
    iterations_ = 0;
    double final_cost = initial_cost;
    const auto start = std::chrono::steady_clock::now();
    const int status = lbfgs::lbfgs_optimize(
        piece_count_, raw_times.data(), &final_cost, &objectiveCallback,
        nullptr, &progressCallback, this, &parameters);
    const auto stop = std::chrono::steady_clock::now();

    Eigen::VectorXd optimized_times(piece_count_);
    flygripper_planner::forwardT(raw_times, optimized_times);
    const double wall_time_seconds =
        std::chrono::duration<double>(stop - start).count();

    py::dict result;
    result["times"] = vectorToArray(optimized_times);
    result["initial_cost"] = initial_cost;
    result["final_cost"] = final_cost;
    result["status"] = status;
    result["message"] = std::string(lbfgs::lbfgs_strerror(status));
    result["success"] = status >= 0 && std::isfinite(final_cost) &&
                        optimized_times.allFinite() &&
                        (optimized_times.array() > 0.0).all();
    result["iterations"] = iterations_;
    result["function_evaluations"] = function_evaluations_;
    result["wall_time_seconds"] = wall_time_seconds;
    return result;
  }

 private:
  static double objectiveCallback(void *instance, const double *raw_times,
                                  double *gradient, const int count) {
    auto &optimizer = *static_cast<FixedWaypointTimeOptimizer *>(instance);
    const Eigen::Map<const Eigen::VectorXd> raw(raw_times, count);
    Eigen::Map<Eigen::VectorXd> grad(gradient, count);
    return optimizer.evaluate(raw, grad);
  }

  static int progressCallback(void *instance, const double *, const double *,
                              const double, const double, const double,
                              const double, int, int iteration, int) {
    auto &optimizer = *static_cast<FixedWaypointTimeOptimizer *>(instance);
    optimizer.iterations_ = iteration;
    return 0;
  }

  static py::array_t<double> vectorToArray(const Eigen::VectorXd &values) {
    py::array_t<double> result(values.size());
    auto output = result.mutable_unchecked<1>();
    for (Eigen::Index index = 0; index < values.size(); ++index) {
      output(index) = values(index);
    }
    return result;
  }

  static Eigen::Matrix<double, 6, 1> jerkTimeBase(const int derivative,
                                                   const double time) {
    return flygripper_planner::cal_timebase_jerk(derivative, time);
  }

  static Eigen::Matrix<double, 4, 1> yawTimeBase(const int derivative,
                                                 const double time) {
    return flygripper_planner::cal_timebase_acc(derivative, time);
  }

  static bool squaredNormPenalty(const Eigen::Vector3d &value,
                                 const double maximum,
                                 const double weight,
                                 Eigen::Vector3d &gradient,
                                 double &cost) {
    const double violation = value.squaredNorm() - maximum * maximum;
    gradient.setZero();
    cost = 0.0;
    if (violation <= 0.0) {
      return false;
    }
    double scalar_gradient = 0.0;
    cost = weight * flygripper_planner::smoothedL1(
                        violation, scalar_gradient);
    gradient = weight * scalar_gradient * 2.0 * value;
    return true;
  }

  static bool squaredScalarPenalty(const double value, const double maximum,
                                   const double weight, double &gradient,
                                   double &cost) {
    const double violation = value * value - maximum * maximum;
    gradient = 0.0;
    cost = 0.0;
    if (violation <= 0.0) {
      return false;
    }
    double scalar_gradient = 0.0;
    cost = weight * flygripper_planner::smoothedL1(
                        violation, scalar_gradient);
    gradient = weight * scalar_gradient * 2.0 * value;
    return true;
  }

  void yawAlignmentPenalty(const double current_yaw, double &gradient,
                           double &cost) const {
    // This is the deployment gradCostYawAlignmentAngle() expression.  The
    // negative SO(2) error is the derivative-compatible wrapped yaw delta.
    const double error = -flygripper_common::getAngleError(
        current_yaw, yaw_points_(yaw_points_.size() - 1));
    cost = config_.rho_yaw_alignment * error * error;
    gradient = config_.rho_yaw_alignment * 2.0 * error;
  }

  void addIntegratedPenalties(double &cost) {
    const int sample_count = config_.integration_steps + 1;
    for (int segment = 0; segment < piece_count_; ++segment) {
      const double duration = xyz_.T1(segment);
      const double step = duration / config_.integration_steps;
      const auto coefficients = xyz_.b.block<6, 3>(6 * segment, 0);
      const auto yaw_coefficients = yaw_.b.block<4, 1>(4 * segment, 0);

      for (int sample_index = 0; sample_index < sample_count; ++sample_index) {
        const double trapezoid_weight =
            (sample_index == 0 || sample_index == sample_count - 1) ? 0.5
                                                                    : 1.0;
        const double alpha =
            static_cast<double>(sample_index) / config_.integration_steps;
        const double sample_time = alpha * duration;

        const auto beta1 = jerkTimeBase(1, sample_time);
        const auto beta2 = jerkTimeBase(2, sample_time);
        const auto beta3 = jerkTimeBase(3, sample_time);
        const auto yaw_beta0 = yawTimeBase(0, sample_time);
        const auto yaw_beta1 = yawTimeBase(1, sample_time);
        const auto yaw_beta2 = yawTimeBase(2, sample_time);

        const Eigen::Vector3d velocity = coefficients.transpose() * beta1;
        const Eigen::Vector3d acceleration = coefficients.transpose() * beta2;
        const Eigen::Vector3d jerk = coefficients.transpose() * beta3;
        const double current_yaw =
            (yaw_coefficients.transpose() * yaw_beta0)(0, 0);
        const double yaw_rate =
            (yaw_coefficients.transpose() * yaw_beta1)(0, 0);
        const double yaw_acceleration =
            (yaw_coefficients.transpose() * yaw_beta2)(0, 0);

        Eigen::Vector3d velocity_gradient = Eigen::Vector3d::Zero();
        Eigen::Vector3d acceleration_gradient = Eigen::Vector3d::Zero();
        double yaw_gradient = 0.0;
        double yaw_rate_gradient = 0.0;
        double inner_cost = 0.0;

        Eigen::Vector3d term_vector_gradient;
        double term_scalar_gradient = 0.0;
        double term_cost = 0.0;
        if (squaredNormPenalty(velocity, config_.max_velocity,
                               config_.rho_velocity, term_vector_gradient,
                               term_cost)) {
          velocity_gradient += term_vector_gradient;
          inner_cost += term_cost;
        }
        if (squaredNormPenalty(acceleration, config_.max_acceleration,
                               config_.rho_acceleration, term_vector_gradient,
                               term_cost)) {
          acceleration_gradient += term_vector_gradient;
          inner_cost += term_cost;
        }
        if (squaredScalarPenalty(yaw_rate, config_.max_yaw_rate,
                                 config_.rho_yaw_rate, term_scalar_gradient,
                                 term_cost)) {
          yaw_rate_gradient += term_scalar_gradient;
          inner_cost += term_cost;
        }

        yawAlignmentPenalty(current_yaw, term_scalar_gradient, term_cost);
        yaw_gradient += term_scalar_gradient;
        inner_cost += term_cost;

        Eigen::Matrix<double, 6, 3> coefficient_gradient =
            beta1 * velocity_gradient.transpose();
        double direct_time_gradient = velocity_gradient.dot(acceleration);
        coefficient_gradient += beta2 * acceleration_gradient.transpose();
        direct_time_gradient += acceleration_gradient.dot(jerk);

        Eigen::Matrix<double, 4, 1> yaw_coefficient_gradient =
            yaw_beta0 * yaw_gradient + yaw_beta1 * yaw_rate_gradient;
        const double direct_yaw_time_gradient =
            yaw_gradient * yaw_rate +
            yaw_rate_gradient * yaw_acceleration;

        xyz_.gdC.block<6, 3>(6 * segment, 0) +=
            trapezoid_weight * step * coefficient_gradient;
        xyz_.gdT(segment) += trapezoid_weight *
            (inner_cost / config_.integration_steps +
             alpha * step *
                 (direct_time_gradient + direct_yaw_time_gradient));
        yaw_.gdC.block<4, 1>(4 * segment, 0) +=
            trapezoid_weight * step * yaw_coefficient_gradient;
        cost += trapezoid_weight * step * inner_cost;
      }
    }
  }

  Eigen::VectorXd yaw_points_;
  OptimizerConfig config_;
  int piece_count_;
  Eigen::MatrixXd interior_points_;
  Eigen::VectorXd interior_yaws_;
  minco::MINCO<3, 3, false> xyz_;
  minco::MINCO<1, 2, false> yaw_;
  int function_evaluations_{0};
  int iterations_{0};
};

FixedWaypointTimeOptimizer makeOptimizer(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &points,
    const py::array_t<double, py::array::c_style | py::array::forcecast>
        &yaw_points,
    const double initial_yaw_rate, const py::dict &config) {
  const Eigen::MatrixXd copied_points = copyPoints(points);
  const Eigen::VectorXd copied_yaws =
      copyVector(yaw_points, copied_points.cols(), "yaw_points");
  return FixedWaypointTimeOptimizer(copied_points, copied_yaws,
                                    initial_yaw_rate, parseConfig(config));
}

py::dict optimizeTimes(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &points,
    const py::array_t<double, py::array::c_style | py::array::forcecast>
        &yaw_points,
    const py::array_t<double, py::array::c_style | py::array::forcecast>
        &initial_times,
    const double initial_yaw_rate, const py::dict &config) {
  auto optimizer = makeOptimizer(points, yaw_points, initial_yaw_rate, config);
  const Eigen::VectorXd copied_times =
      copyVector(initial_times, points.shape(0) - 1, "initial_times");
  return optimizer.optimize(copied_times);
}

py::tuple evaluateRawTimes(
    const py::array_t<double, py::array::c_style | py::array::forcecast> &points,
    const py::array_t<double, py::array::c_style | py::array::forcecast>
        &yaw_points,
    const py::array_t<double, py::array::c_style | py::array::forcecast>
        &raw_times,
    const double initial_yaw_rate, const py::dict &config) {
  auto optimizer = makeOptimizer(points, yaw_points, initial_yaw_rate, config);
  const Eigen::VectorXd copied_raw =
      copyVector(raw_times, points.shape(0) - 1, "raw_times");
  Eigen::VectorXd gradient(copied_raw.size());
  const double cost = optimizer.evaluate(copied_raw, gradient, false);
  py::array_t<double> output(gradient.size());
  auto output_view = output.mutable_unchecked<1>();
  for (Eigen::Index index = 0; index < gradient.size(); ++index) {
    output_view(index) = gradient(index);
  }
  return py::make_tuple(cost, output);
}

}  // namespace

PYBIND11_MODULE(_minco_cpp, module) {
  module.doc() =
      "C++ MINCO fixed-waypoint time optimizer used by the Flying-Hand tasks";
  module.def("optimize_times", &optimizeTimes, py::arg("points"),
             py::arg("yaw_points"), py::arg("initial_times"),
             py::arg("initial_yaw_rate"), py::arg("config"));
  module.def("evaluate_raw_times", &evaluateRawTimes, py::arg("points"),
             py::arg("yaw_points"), py::arg("raw_times"),
             py::arg("initial_yaw_rate"), py::arg("config"));
  module.def("generate_position_coefficients", &generatePositionCoefficients,
             py::arg("points"), py::arg("times"));
  module.def("generate_yaw_coefficients", &generateYawCoefficients,
             py::arg("yaw_points"), py::arg("times"),
             py::arg("initial_yaw_rate") = 0.0,
             py::arg("terminal_yaw_rate") = 0.0);
  module.attr("backend_name") = "deployment_cpp_minco_lbfgs";
}
