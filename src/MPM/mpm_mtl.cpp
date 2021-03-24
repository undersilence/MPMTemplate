#include "MPM/mpm_mtl.h"

namespace mpm {
MPM_Material::MPM_Material(float E, float nu, float mass, float density)
    : E(E), nu(nu), mass(mass), density(density) {
  mu = 0.5f * E / (1 + nu);
  lambda = E * nu / (1 + nu) / (1 - 2 * nu);
  volume = mass / density;
  K = E / (1 - 2 * nu) / 3.0f;
}

Matrix3f NeoHookean_Piola::calc_stress_tensor(const Particle &particle) {
  auto m = particle.material;
  auto F = particle.F;
  auto J = F.determinant();
  auto piola = m->mu * (F - F.transpose().inverse()) +
               m->lambda * log(J) * F.transpose().inverse();
  return piola;
}

float NeoHookean_Piola::calc_psi(const Particle &particle) {
  auto m = particle.material;
  auto F = particle.F;
  auto J = F.determinant();
  auto log_J = log(J);

  auto psi = 0.5f * m->mu * ((F.transpose() * F).trace() - 3) - m->mu * log_J +
             0.5f * m->lambda * log_J * log_J;
  return psi;
}

Matrix3f QuatraticVolumePenalty::calc_stress_tensor(const Particle &particle) {
  auto m = particle.material;
  auto F = particle.F;
  auto J = F.determinant();

  return 0.5f * m->lambda * 2 * (J - 1) * J * F.inverse().transpose();
}

float QuatraticVolumePenalty::calc_psi(const Particle &particle) {
  auto m = particle.material;
  auto F = particle.F;
  auto J = F.determinant();

  return 0.5f * m->lambda * std::pow(J - 1, 2);
}

Matrix3f NeoHookean_Fluid::calc_stress_tensor(const Particle &particle) {
  auto m = particle.material;
  auto F = particle.F;
  auto J = F.determinant();
  auto piola = m->lambda * log(J) * F.transpose().inverse();
  return piola;
}

float NeoHookean_Fluid::calc_psi(const Particle &particle) {
  auto m = particle.material;
  auto F = particle.F;
  auto J = F.determinant();
  auto log_J = log(J);

  auto psi = 0.5f * m->mu * ((F.transpose() * F).trace() - 3) - m->mu * log_J +
             0.5f * m->lambda * log_J * log_J;
  return psi;
}

Matrix3f CDMPM_Fluid::calc_stress_tensor(const Particle &particle) {
  auto m = particle.material;
  auto F = particle.F;
  auto J = F.determinant();
  // d(psi)/d(J) * d(J)/d(F) =
  // 0.5 * K (J - 1 / J) * J * F^{-T}
  auto piola = 0.5f * m->K * (J - 1 / J) * J * F.inverse().transpose();
  return piola;
}

float CDMPM_Fluid::calc_psi(const Particle &particle) {
  auto m = particle.material;
  auto F = particle.F;
  auto J = F.determinant();
  auto log_J = log(J);
  // 0.5 * K ( 0.5 * (J^2 - 1) - Log(J))
  auto psi = 0.5f * m->K * (0.5f * (J * J - 1) - log_J);
  return psi;
}

} // namespace mpm