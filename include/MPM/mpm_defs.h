#pragma once
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include <Eigen/StdVector>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <tuple>

#include "spdlog/fmt/ostr.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

using namespace Eigen;

namespace mpm {

struct GridAttr {
  float mass_i;
  Vector3f vel_in;
  Vector3f vel_i;
  Vector3f force_i;
  Vector3i Xi;
};

struct Particle {
  Vector3f pos_p;
  Vector3f vel_p;
  Matrix3f F; // deformation gradients
  Matrix3f Fe;
  Matrix3f Fp;
  Matrix3f Bp; // for APIC transfer

  float cp;    // fluids ratio
  Matrix3f Dp; // inertia tensor

  struct MPM_Material *material;
};

struct SimInfo {
  int particle_size = 0;
  int grid_size = 0;
  int grid_w = 0;
  int grid_h = 0;
  int grid_l = 0;

  // simulation factors
  // float E = 50.0f;     // Young's modules
  // float nu = 0.3f;     // Possion ratio
  float alpha = 0.95f; // 0.95 flip/pic

  // float particle_density;
  // float particle_mass;
  // std::string model_path;

  Vector3f gravity = Vector3f::Zero();
  Vector3f world_area = Vector3f::Zero();
  float h = 0.0f;
  unsigned int curr_step = 0;
};

// Matrix3f neohookean_piola(float E, float nu, const Matrix3f &F);

} // namespace mpm
