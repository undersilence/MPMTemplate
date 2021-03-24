#include "MPM/mpm_defs.h"

namespace mpm {

// Matrix3f neohookean_piola(float E, float nu, const Matrix3f &F) {
//   auto mu = 0.5f * E / (1.0f + nu);
//   auto lambda = E * nu / (1.0f + nu) / (1.0f - 2.0f * nu);
//   auto J = F.determinant();
//   Matrix3f piola = mu * (F - F.transpose().inverse()) +
//                    lambda * log(J) * F.transpose().inverse();
//   return piola;
// }

} // namespace mpm