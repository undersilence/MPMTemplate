
#include <filesystem>
#include <iostream>
#include <memory>
#include <tbb/task_scheduler_init.h>

#include "MPM/mpm_sim.h"
#include "MPM/mpm_utils.h"
using namespace std;
using namespace Eigen;
using namespace mpm;
namespace fs = std::filesystem;

void quatratic_test() {
  MPM_PROFILE_FUNCTION();
  MPM_INFO("{} start", __func__);
  float h = 0.02f;
  int W = 1.0f / h + 1;
  int H = 1.0f / h + 1;
  int L = 1.0f / h + 1;
  Vector3f pos(0.648932, 0.121521, 0.265484);
  auto [base_node, wp, dwp] = mpm::quatratic_interpolation(pos / h);
  for (int i = 0; i < 3; i++)
    for (int j = 0; j < 3; j++)
      for (int k = 0; k < 3; k++) {
        float wi = wp(i, 0), wj = wp(j, 1), wk = wp(k, 2);
        float dwi = dwp(i, 0), dwj = dwp(j, 1), dwk = dwp(k, 2);
        float wijk = wi * wj * wk;
        Vector3i curr_node = base_node + Vector3i(i, j, k);

        int index = curr_node(0) * H * L + curr_node(1) * L + curr_node(2);
        Vector3f grad_wp(dwi * wj * wk / h, wi * dwj * wk / h,
                         wi * wj * dwk / h);

        MPM_INFO("offset: {}", Vector3i(i, j, k).transpose());
        MPM_INFO("weight_ijk: {}", wijk);
        MPM_INFO("grad_wp: {}", grad_wp.transpose());
      }
  MPM_INFO("{} end", __func__);
}

int main() {
  // tbb::task_scheduler_init init(1);
  // initialize logger
  mpm::MPMLog::init();
  // quatratic_test();
  auto sim = std::make_shared<mpm::MPM_Simulator>();

  auto mtl_jello = new MPM_Material(50.0f, 0.3f, 10.0f, 1.0f);
  auto mtl_water = new MPM_Material(500.0f, 0.40f, 0.001f, 1.0f);

  auto cm_solid = std::make_shared<mpm::NeoHookean_Piola>();
  auto cm_fluid = std::make_shared<mpm::NeoHookean_Fluid>();
  auto cm_fluid_1 = std::make_shared<mpm::QuatraticVolumePenalty>();
  auto cm_fluid_2 = std::make_shared<mpm::CDMPM_Fluid>();

  sim->clear_simulation();
  Vector3f gravity{0.0f, -9.8f, 0.0f};
  Vector3f area{1.0f, 1.0f, 1.0f};
  Vector3f velocity{-2.5f, 0.5f, -0.3f};
  float h = 0.02f;

  sim->mpm_initialize(gravity, area, h);
  sim->set_constitutive_model(cm_fluid_2);
  sim->set_transfer_scheme(mpm::MPM_Simulator::TransferScheme::FLIP99);

  std::vector<Vector3f> positions;
  auto model_path = "../models/dense_cube.obj";

  if (mpm::read_particles(model_path, positions)) {
    mpm::MPM_INFO("read in particles from {} SUCCESS", model_path);
    sim->add_object(positions,
                    std::vector<Vector3f>(positions.size(), velocity),
                    mtl_water);
  } else {
    return 0;
  }

  int frame_rate = 60;
  float dt = 1e-4f;
  int total_frame = 200;
  int steps_per_frame = (int)ceil((1.0f / frame_rate) / dt);

  mpm::MPM_INFO("Simulation start, Meta Informations:\n"
                "\tframe_rate: {}\n"
                "\tdt: {}\n"
                "\tsteps_per_frame: {}\n",
                frame_rate, dt, steps_per_frame);

  std::string output_dir("../output/test/");

  write_particles(output_dir + "0.bgeo", sim->get_positions());
  for (int frame = 0; frame < total_frame;) {
    // add another jello
    if (frame && frame % 50 == 0) {
      sim->add_object(positions,
                      std::vector<Vector3f>(positions.size(), velocity),
                      mtl_water);
    }

    {
      mpm::MPM_PROFILE("frame#" + std::to_string(frame + 1));
      for (int i = 0; i < steps_per_frame; i++) {
        sim->substep(dt);
      }
    }
    write_particles(output_dir + std::to_string(++frame) + ".bgeo",
                    sim->get_positions());
  }
  // sim->mpm_demo(cm_fluid, "neohookean_fluids/");

  printf("mpm finished!\n");
  return 0;
}