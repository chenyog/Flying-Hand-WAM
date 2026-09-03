# Vendored Flying-Hand MINCO sources

The headers in this directory are a local snapshot of the original
Flying-Hand deployment planner implementation at commit
`126eca134ce7fa4ee0cea81c512c2434b67d19fc`.

Copied files:

- `minco.hpp` from `flygripper_planner/minco.hpp`
- `lbfgs.hpp` from `flygripper_planner/lbfgs.hpp`
- `traj_opt_utils.hpp` from `flygripper_planner/traj_opt_utils.hpp`
- `polynomial_trajectory.hpp` from
  `flygripper_common/trajectory/polynomial_trajectory.hpp`
- `root_finder.hpp` from `flygripper_common/trajectory/root_finder.hpp`
- `rotation.hpp` from `flygripper_common/utils/rotation.hpp`
- `base.hpp` from `flygripper_common/type/base.hpp`

Only three include paths in the copied headers were changed so that they
resolve to files in this directory. The optimizer can therefore be built
without access to the deployment repository or ROS source tree.

`minco_optimizer.cpp` is the simulation-specific pybind11 adapter. It retains
fixed task waypoints, uses the vendored MINCO adjoint time gradients and
L-BFGS implementation, and exposes the fixed-waypoint time optimizer plus
position/yaw polynomial coefficients to Python. Python only samples those
coefficients while stepping the simulator; it does not solve MINCO systems or
reimplement the optimization objective.
