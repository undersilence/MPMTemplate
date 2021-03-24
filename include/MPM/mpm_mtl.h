#pragma once

#include "MPM/mpm_defs.h"

namespace mpm {
struct MPM_Material {
  // lame params
  float E;  // young's modulus
  float nu; // possion ratio
  float mass;
  float density;

  float lambda;
  float mu;
  float K; // bulk modulus
  float volume;
  MPM_Material(float E, float nu, float mass, float density);
};

// Constitutive Model for simulation
class MPM_CM {
public:
  virtual Matrix3f calc_stress_tensor(const Particle &particle) = 0;
  // virtual Matrix3f calc_strain_tensor(const Particle &particle) = 0;
  virtual float calc_psi(const Particle &particle) = 0;
};

class NeoHookean_Piola : public MPM_CM {
  virtual Matrix3f calc_stress_tensor(const Particle &particle);
  virtual float calc_psi(const Particle &particle);
};

class QuatraticVolumePenalty : public MPM_CM {
  virtual Matrix3f calc_stress_tensor(const Particle &particle);
  virtual float calc_psi(const Particle &particle);
};

class NeoHookean_Fluid : public MPM_CM {
  virtual Matrix3f calc_stress_tensor(const Particle &particle);
  virtual float calc_psi(const Particle &particle);
};

class CDMPM_Fluid : public MPM_CM {
  virtual Matrix3f calc_stress_tensor(const Particle &particle);
  virtual float calc_psi(const Particle &particle);
};

} // namespace mpm
