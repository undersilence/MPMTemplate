#include "MPM/mpm_sim.h"

#include <tbb/parallel_for.h>
#include <tbb/spin_mutex.h>

#include "MPM/mpm_utils.h"

namespace mpm {

MPM_Simulator::MPM_Simulator() : sim_info() {}

MPM_Simulator::~MPM_Simulator() {
  // note: build a active_id <-> grid_id map in future
  //      for lower memory usage
  if (particles) {
    delete[] particles;
    particles = nullptr;
  }
  if (grid_attrs) {
    delete[] grid_attrs;
    grid_attrs = nullptr;
  }
  if (grid_mutexs) {
    delete[] grid_mutexs;
    grid_mutexs = nullptr;
  }
}

void MPM_Simulator::clear_simulation() {
  if (particles) {
    delete[] particles;
    particles = nullptr;
  }
  if (grid_attrs) {
    delete[] grid_attrs;
    grid_attrs = nullptr;
  }
  if (grid_mutexs) {
    delete[] grid_mutexs;
    grid_mutexs = nullptr;
  }
  sim_info = SimInfo();
}

void MPM_Simulator::mpm_demo(const std::shared_ptr<MPM_CM> &cm_demo,
                             const std::string &output_relative_path) {

  clear_simulation();
  Vector3f gravity{0.0f, -9.8f, 0.0f};
  Vector3f area{1.0f, 1.0f, 1.0f};
  Vector3f velocity{-0.5f, 0.5f, -0.3f};

  float h = 0.02f;
  auto mtl_jello = new MPM_Material(50.0f, 0.3f, 10.0f, 1.0f);
  auto mtl_water = new MPM_Material(50.0f, 0.3f, 8e-6f, 1.0f);

  mpm_initialize(gravity, area, h);
  cm = cm_demo;

  std::vector<Vector3f> positions;
  auto model_path = "../models/dense_cube.obj";

  if (read_particles(model_path, positions)) {
    MPM_INFO("read in particles from {} SUCCESS", model_path);
    add_object(positions, mtl_water);
  }

  int frame_rate = 60;
  float dt = 1e-3f;
  int total_frame = 300;
  int steps_per_frame = (int)ceil((1.0f / frame_rate) / dt);

  MPM_INFO("Simulation start, Meta Informations:\n"
           "\tframe_rate: {}\n"
           "\tdt: {}\n"
           "\tsteps_per_frame: {}\n",
           frame_rate, dt, steps_per_frame);

  // export frame#0 first
  auto output_dir = "../output/" + output_relative_path;
  write_particles(output_dir + "0.bgeo", get_positions());
  for (int frame = 0; frame < total_frame;) {
    {
      MPM_PROFILE("frame#" + std::to_string(frame + 1));
      for (int i = 0; i < steps_per_frame; i++) {
        // MPM_INFO("begin step {}", i);
        substep(dt);
      }
    }
    write_particles(output_dir + std::to_string(++frame) + ".bgeo",
                    get_positions());
    // export_result("../output/", ++frame);
  }
}

void MPM_Simulator::substep(float dt) {
  // MPM_PROFILE_FUNCTION();
  // TODO: add profiler later
  prestep();
  transfer_P2G();
  add_gravity();
  update_grid_force();
  update_grid_velocity(dt);
  solve_grid_boundary(2);
  update_F(dt);
  transfer_G2P();
  advection(dt);
  sim_info.curr_step++;
}

std::vector<Vector3f> MPM_Simulator::get_positions() const {
  std::vector<Vector3f> positions(sim_info.particle_size);
  tbb::parallel_for(0, sim_info.particle_size,
                    [&](int i) { positions[i] = particles[i].pos_p; });
  return positions;
}

// bool MPM_Simulator::export_result(const std::string &export_dir,
//                                   int curr_frame) {
//   // MPM_INFO("export frame_{}'s result.", curr_frame);
//   std::vector<Vector3f> positions(sim_info.particle_size);

//   tbb::parallel_for(0, sim_info.particle_size,
//                     [&](int i) { positions[i] = particles[i].pos_p; });

//   auto export_path = export_dir + std::to_string(curr_frame) + ".bgeo";

//   return write_particles(export_path, positions);
// }

void MPM_Simulator::mpm_initialize(const Vector3f &gravity,
                                   const Vector3f &world_area, float h) {
  sim_info.h = h;
  sim_info.gravity = gravity;
  sim_info.world_area = world_area;

  // grid_initialize
  int W = world_area[0] / h + 1;
  int H = world_area[1] / h + 1;
  int L = world_area[2] / h + 1;

  sim_info.grid_w = W;
  sim_info.grid_h = H;
  sim_info.grid_l = L;
  sim_info.grid_size = W * H * L;
  grid_attrs = new GridAttr[sim_info.grid_size];
  grid_mutexs = new tbb::spin_mutex[sim_info.grid_size];

  MPM_INFO("MPM simulation space info:\n"
           "\tgrid_size: {}->{}x{}x{}\n"
           "\tgrid dx: {}\n"
           "\tgrid gravity: {}\n"
           "\tworld area: {}",
           sim_info.grid_size, W, H, L, sim_info.h,
           sim_info.gravity.transpose(), sim_info.world_area.transpose());

  for (int i = 0; i < W; i++)
    for (int j = 0; j < H; j++)
      for (int k = 0; k < L; k++) {
        int index = i * H * L + j * L + k;
        grid_attrs[index].mass_i = 0;
        grid_attrs[index].force_i = Vector3f::Zero();
        grid_attrs[index].vel_i = Vector3f::Zero();
        grid_attrs[index].vel_in = Vector3f::Zero();
        grid_attrs[index].Xi = Vector3i(i, j, k);
      }

} // namespace mpm

void MPM_Simulator::add_object(const std::vector<Vector3f> &positions,
                               const std::vector<Vector3f> &velocities,
                               MPM_Material *material) {

  MPM_ASSERT(positions.size() == velocities.size() && material != nullptr,
             "PLEASE CHECK OBJECT's POSITION SIZE IFF EQUALS VELOCITY SIZE");
  auto new_size = sim_info.particle_size + positions.size();

  if (particles) {
    Particle *new_particles = new Particle[new_size];
    memcpy(new_particles, particles, sim_info.particle_size * sizeof(Particle));
    delete[] particles;
    particles = new_particles;
  } else {
    particles = new Particle[new_size];
  }

  for (auto i = sim_info.particle_size; i < new_size; i++) {
    particles[i].pos_p = positions[i - sim_info.particle_size];
    particles[i].vel_p = velocities[i - sim_info.particle_size];
    particles[i].F = Matrix3f::Identity();
    particles[i].Fe = Matrix3f::Identity();
    particles[i].Fp = Matrix3f::Identity();
    particles[i].Bp = Matrix3f::Zero();
    particles[i].material = material;
  }

  sim_info.particle_size = new_size;
}

void MPM_Simulator::add_object(const std::vector<Vector3f> &positions,
                               MPM_Material *material) {
  MPM_ASSERT(material != nullptr, "MATERIAL SHOULD NOT BE NULLPTR");
  auto new_size = sim_info.particle_size + positions.size();

  if (particles) {
    Particle *new_particles = new Particle[new_size];
    memcpy(new_particles, particles, sim_info.particle_size * sizeof(Particle));
    delete[] particles;
    particles = new_particles;
  } else {
    particles = new Particle[new_size];
  }

  for (auto i = sim_info.particle_size; i < new_size; i++) {
    particles[i].pos_p = positions[i - sim_info.particle_size];
    particles[i].F = Matrix3f::Identity();
    particles[i].Fe = Matrix3f::Identity();
    particles[i].Fp = Matrix3f::Identity();
    particles[i].Bp = Matrix3f::Zero();
    particles[i].material = material;
  }
  sim_info.particle_size = new_size;
}

void MPM_Simulator::set_constitutive_model(const std::shared_ptr<MPM_CM> &cm) {
  this->cm = cm;
}

void MPM_Simulator::set_transfer_scheme(TransferScheme ts) {
  this->transfer_scheme = ts;
  if (ts == TransferScheme::FLIP95) {
    sim_info.alpha = 0.95f;
  } else if (ts == TransferScheme::FLIP99) {
    sim_info.alpha = 0.99;
  }
}

void MPM_Simulator::prestep() {
  // MPM_PROFILE_FUNCTION();
  // tbb::parallel_for(0, (int)sim_info.particle_size, [&](int iter) {
  //   // for (int iter = 0; iter < sim_info.particle_size; iter++) {
  //   // convert particles position to grid space by divide h
  //   // particle position in grid space
  //   Vector3f gs_particle_pos = particles[iter].pos_p / sim_info.h;
  //   auto [base_node, wp, dwp] = quatratic_interpolation(gs_particle_pos);

  //   auto &particle = particles[iter];
  //   auto &mass_p = particle.material->mass;
  //   particle.Dp = Matrix3f::Zero();

  //   for (int i = 0; i < 3; i++)
  //     for (int j = 0; j < 3; j++)
  //       for (int k = 0; k < 3; k++) {
  //         // note: do not use auto here (cause error in release mode)
  //         Vector3i curr_node = base_node + Vector3i(i, j, k);
  //         int index = curr_node(0) * sim_info.grid_h * sim_info.grid_l +
  //                     curr_node(1) * sim_info.grid_l + curr_node(2);
  //         float wijk = wp(i, 0) * wp(j, 1) * wp(k, 2);
  //         Vector3f dxip = curr_node.cast<float>() - gs_particle_pos;
  //         particle.Dp += wijk * dxip * dxip.transpose();
  //       }

  //   MPM_INFO("particle {}'s postision = {}, Dp = \n{}", iter,
  //            particle.pos_p.transpose(), particle.Dp);
  // });

  tbb::parallel_for(0, sim_info.grid_size, [&](int i) {
    grid_attrs[i].mass_i = 0;
    grid_attrs[i].force_i = Vector3f::Zero();
    grid_attrs[i].vel_i = Vector3f::Zero();
    grid_attrs[i].vel_in = Vector3f::Zero();
  });
  active_nodes.resize(0);
}

void MPM_Simulator::transfer_P2G() {
  // MPM_PROFILE_FUNCTION();
  tbb::parallel_for(0, (int)sim_info.particle_size, [&](int iter) {
    // for (int iter = 0; iter < sim_info.particle_size; iter++) {
    // convert particles position to grid space by divide h
    // particle position in grid space
    Vector3f gs_particle_pos = particles[iter].pos_p / sim_info.h;
    auto [base_node, wp, dwp] = quatratic_interpolation(gs_particle_pos);

    auto particle = particles[iter];
    auto mass_p = particle.material->mass;

    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) {
          // note: do not use auto here (cause error in release mode)
          Vector3i curr_node = base_node + Vector3i(i, j, k);
          int index = curr_node(0) * sim_info.grid_h * sim_info.grid_l +
                      curr_node(1) * sim_info.grid_l + curr_node(2);

          // check if particles run out of boundaries
          MPM_ASSERT(0 <= index && index < sim_info.grid_size,
                     " PARTICLE OUT OF GRID at Transfer_P2G");

          float wijk = wp(i, 0) * wp(j, 1) * wp(k, 2);
          Vector3f plus = Vector3f::Zero();
          if (transfer_scheme == TransferScheme::APIC) {
            plus = 4 * particles[iter].Bp *
                   (curr_node.cast<float>() - gs_particle_pos);
          }

          {
            // critical section
            tbb::spin_mutex::scoped_lock lock(grid_mutexs[index]);

            // accumulate momentum at time n
            grid_attrs[index].vel_in +=
                wijk * mass_p * (particles[iter].vel_p + plus);
            grid_attrs[index].mass_i += wijk * mass_p;
          }
        }
  });

  tbb::parallel_for(0, (int)sim_info.grid_size, [&](int iter) {
    if (grid_attrs[iter].mass_i > 1e-15) {
      {
        // critical section
        active_nodes.push_back(iter);
      }
      grid_attrs[iter].vel_in =
          grid_attrs[iter].vel_in / grid_attrs[iter].mass_i;
    } else {
      grid_attrs[iter].vel_in = Vector3f::Zero();
    }
  });
} // namespace mpm

void MPM_Simulator::add_gravity() {
  // MPM_PROFILE_FUNCTION();
  // MPM_ASSERT(active_nodes.size() < sim_info.grid_size);
  tbb::parallel_for(0, (int)active_nodes.size(), [&](int i) {
    int index = active_nodes[i];
    grid_attrs[index].force_i += sim_info.gravity * grid_attrs[index].mass_i;
  });
}

void MPM_Simulator::update_grid_force() {
  // update grid forcing from particles F(deformation gradients)

  // MPM_PROFILE_FUNCTION();
  tbb::parallel_for(0, (int)sim_info.particle_size, [&](int iter) {
    // for (int iter = 0; iter < sim_info.particle_size; iter++) {
    auto F = particles[iter].F;
    auto vol_p = particles[iter].material->volume;
    auto h = sim_info.h;

    // use constitutive_model (may cause problems in multi-threads condition)
    Matrix3f piola = cm->calc_stress_tensor(particles[iter]);

    auto [base_node, wp, dwp] =
        quatratic_interpolation(particles[iter].pos_p / h);
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) {
          Vector3i curr_node = base_node + Vector3i(i, j, k);
          Vector3f grad_wip{dwp(i, 0) * wp(j, 1) * wp(k, 2) / h,
                            wp(i, 0) * dwp(j, 1) * wp(k, 2) / h,
                            wp(i, 0) * wp(j, 1) * dwp(k, 2) / h};

          auto index = curr_node.x() * sim_info.grid_h * sim_info.grid_l +
                       curr_node.y() * sim_info.grid_l + curr_node.z();
          MPM_ASSERT(0 <= index && index < sim_info.grid_size,
                     "PARTICLE OUT OF GRID");

          {
            // critical section
            tbb::spin_mutex::scoped_lock lock(grid_mutexs[index]);

            grid_attrs[index].force_i -=
                vol_p * (piola * F.transpose()) * grad_wip;
          }
        }
  });
}

void MPM_Simulator::update_grid_velocity(float dt) {
  // MPM_PROFILE_FUNCTION();
  tbb::parallel_for(0, (int)active_nodes.size(), [&](int i) {
    int index = active_nodes[i];
    // vel_n+1 = vel_n + f_i / m_i * dt
    grid_attrs[index].vel_i =
        grid_attrs[index].vel_in +
        dt * grid_attrs[index].force_i / grid_attrs[index].mass_i;
  });
}

void MPM_Simulator::update_F(float dt) {
  // MPM_PROFILE_FUNCTION();
  tbb::parallel_for(0, (int)sim_info.particle_size, [&](int iter) {
    // for (int iter = 0; iter < sim_info.particle_size; iter++) {
    auto F = particles[iter].F;
    auto h = sim_info.h;
    auto [base_node, wp, dwp] =
        quatratic_interpolation(particles[iter].pos_p / h);

    Matrix3f weight = Matrix3f::Zero();
    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) {
          Vector3i curr_node = base_node + Vector3i(i, j, k);
          Vector3f grad_wip{dwp(i, 0) * wp(j, 1) * wp(k, 2) / h,
                            wp(i, 0) * dwp(j, 1) * wp(k, 2) / h,
                            wp(i, 0) * wp(j, 1) * dwp(k, 2) / h};

          auto index = curr_node(0) * sim_info.grid_h * sim_info.grid_l +
                       curr_node(1) * sim_info.grid_l + curr_node(2);

          MPM_ASSERT(0 <= index && index < sim_info.grid_size,
                     "PARTICLE OUT OF GRID");

          weight += grid_attrs[index].vel_i * grad_wip.transpose();
        }

    particles[iter].F = F + dt * weight * F;

    if (particles[iter].F.determinant() < 0) {
      MPM_ERROR("particles[{}]'s determinat(F) is negative!\n{}", iter,
                particles[iter].F);
      assert(false);
    }
  });
  // MPM_INFO("particles[0]'s F:\n{}", particles[0].F);
}

void MPM_Simulator::transfer_G2P() {
  // MPM_PROFILE_FUNCTION();
  tbb::parallel_for(0, (int)sim_info.particle_size, [&](int iter) {
    // particle position in grid space
    Vector3f gs_particle_pos = particles[iter].pos_p / sim_info.h;
    auto [base_node, wp, dwp] = quatratic_interpolation(gs_particle_pos);

    Vector3f v_pic = Vector3f::Zero();
    Vector3f v_flip = particles[iter].vel_p;
    particles[iter].Bp = Matrix3f::Zero();

    for (int i = 0; i < 3; i++)
      for (int j = 0; j < 3; j++)
        for (int k = 0; k < 3; k++) {
          Vector3i curr_node = base_node + Vector3i(i, j, k);
          auto wijk = wp(i, 0) * wp(j, 1) * wp(k, 2);
          auto index = curr_node(0) * sim_info.grid_h * sim_info.grid_l +
                       curr_node(1) * sim_info.grid_l + curr_node(2);

          MPM_ASSERT(0 <= index && index < sim_info.grid_size,
                     "PARTICLE OUT OF GRID");

          v_pic += wijk * grid_attrs[index].vel_i;
          v_flip += wijk * (grid_attrs[index].vel_i - grid_attrs[index].vel_in);
          particles[iter].Bp +=
              wijk * grid_attrs[index].vel_i *
              (curr_node.cast<float>() - gs_particle_pos).transpose();
        }

    switch (transfer_scheme) {
    case TransferScheme::APIC:
      particles[iter].vel_p = v_pic;
      break;
    case TransferScheme::FLIP99:
    case TransferScheme::FLIP95:
      particles[iter].vel_p =
          (1 - sim_info.alpha) * v_pic + sim_info.alpha * v_flip;
    }
  });
}

void MPM_Simulator::advection(float dt) {
  // MPM_PROFILE_FUNCTION();
  tbb::parallel_for(0, sim_info.particle_size, [&](int i) {
    particles[i].pos_p += dt * particles[i].vel_p;
  });
}

void MPM_Simulator::solve_grid_boundary(int thickness) {
  // MPM_PROFILE_FUNCTION();
  // Sticky boundary
  auto [W, H, L] = std::tie(sim_info.grid_w, sim_info.grid_h, sim_info.grid_l);
  // check x-axis bound
  for (int i = 0; i < thickness; i++) {
    for (int j = 0; j < H; j++) {
      for (int k = 0; k < L; k++) {
        int index1 = i * H * L + j * L + k;
        int index2 = (W - i - 1) * H * L + j * L + k;
        if (grid_attrs[index1].vel_i[0] < 0) {
          grid_attrs[index1].vel_i[0] = 0.0f;
        }
        if (grid_attrs[index2].vel_i[0] > 0) {
          grid_attrs[index2].vel_i[0] = 0.0f;
        }
      }
    }
  }
  // check y-axis bound
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < thickness; j++) {
      for (int k = 0; k < L; k++) {
        int index1 = i * H * L + j * L + k;
        int index2 = i * H * L + (H - j - 1) * L + k;
        if (grid_attrs[index1].vel_i[1] < 0) {
          grid_attrs[index1].vel_i[1] = 0.0f;
        }
        if (grid_attrs[index2].vel_i[1] > 0) {
          grid_attrs[index2].vel_i[1] = 0.0f;
        }
      }
    }
  }
  // check z-axis bound
  for (int i = 0; i < W; i++) {
    for (int j = 0; j < H; j++) {
      for (int k = 0; k < thickness; k++) {
        int index1 = i * H * L + j * L + k;
        int index2 = i * H * L + j * L + (L - k - 1);
        if (grid_attrs[index1].vel_i[2] < 0) {
          grid_attrs[index1].vel_i[2] = 0.0f;
        }
        if (grid_attrs[index2].vel_i[2] > 0) {
          grid_attrs[index2].vel_i[2] = 0.0f;
        }
      }
    }
  }
}

} // namespace mpm