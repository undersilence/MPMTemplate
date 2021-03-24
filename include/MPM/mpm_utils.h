#pragma once
#include "MPM/mpm_defs.h"

namespace mpm {

template <class Tensor> std::string make_string(const Tensor v) {
  std::string output;
  for (int i = 0; i < v.rows(); i++) {
    for (int j = 0; j < v.cols(); j++) {
      output.append(std::to_string(v(i, j)) +
                    std::to_string(",]"[j + 1 == v.cols()]));
    }
    output.append("\n");
  }
}

bool read_particles(const std::string &model_path,
                    std::vector<Vector3f> &positions);
bool write_particles(const std::string &write_path,
                     const std::vector<Vector3f> &positions);

inline Vector3f calc_quadratic(float o, float x);

inline Vector3f calc_quadratic_grad(float o, float x);
// under gridspace coords
std::tuple<Vector3i, Matrix3f, Matrix3f>
quatratic_interpolation(const Vector3f &particle_pos);

// define logger
class MPMLog {
public:
  static void init();
  MPMLog() = default;
  virtual ~MPMLog() = default;

  inline static const std::shared_ptr<spdlog::logger> &get_logger() {
    return s_logger;
  }

private:
  static std::shared_ptr<spdlog::logger> s_logger;
};

class MPMProfiler {
public:
  MPMProfiler(const std::string &tag);
  virtual ~MPMProfiler();

private:
  std::string tag;
  std::chrono::high_resolution_clock::time_point start;
};

#ifndef DIST_VERSION
// Client log macros
#define MPM_FATAL(...) MPMLog::get_logger()->fatal(__VA_ARGS__)
#define MPM_ERROR(...) MPMLog::get_logger()->error(__VA_ARGS__)
#define MPM_WARN(...) MPMLog::get_logger()->warn(__VA_ARGS__)
#define MPM_INFO(...) MPMLog::get_logger()->info(__VA_ARGS__)
#define MPM_TRACE(...) MPMLog::get_logger()->trace(__VA_ARGS__)

#define MPM_ASSERT(condition, statement)                                       \
  do {                                                                         \
    if (!(condition)) {                                                        \
      mpm::MPM_ERROR(statement);                                               \
      assert(condition);                                                       \
    }                                                                          \
  } while (false)

#define MPM_FUNCTION_SIG __func__
#define MPM_PROFILE(tag) MPMProfiler timer##__LINE__(tag)
#define MPM_PROFILE_FUNCTION() MPM_PROFILE(MPM_FUNCTION_SIG)

#else

#define MPM_FATAL(...)
#define MPM_ERROR(...)
#define MPM_WARN(...)
#define MPM_INFO(...)
#define MPM_TRACE(...)

#define MPM_ASSERT(...)

#define MPM_FUNCTION_SIG
#define MPM_PROFILE(tag)
#define MPM_PROFILE_FUNCTION()

#endif

} // namespace mpm