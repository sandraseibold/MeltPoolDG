#include <gtest/gtest.h>

#include <gmock/gmock.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <meltpooldg/particles/cohesive_forces.hpp>
#include <meltpooldg/particles/obstacle_data.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/obstacle_forces.hpp>
#include <meltpooldg/particles/particle.hpp>
#include <meltpooldg/utilities/distance_functions.hpp>

#include <array>
#include <memory>
#include <string>

/**
 * This test suite verifies the correct implementation of cohesive forces between a particle and a
 * wall, which are computed as described in Meier et al., “Modeling and characterization of cohesion
 * in fine metal powders with a focus on additive manufacturing process simulations”,
 * (DOI:10.1016/j.powtec.2018.11.072).
 *
 * To this end, a single particle is placed at different distances above a horizontal plane wall,
 * and the resulting cohesive forces are compared against precomputed reference values. See the
 * individual tests for further details.
 */
namespace
{
  constexpr int dim = 3;

  struct ParticleTestData
  {
    using particle_id_type = int;

    particle_id_type           id{};
    double                     radius{};
    dealii::Point<dim, double> location{};
  };

  struct TestData
  {
    std::string                    test_name;
    dealii::Tensor<1, dim, double> expected_force;
    double                         normal_distance;
  };

  class CohesiveWallForceTest : public ::testing::TestWithParam<TestData>
  {
  protected:
    using ObstacleType = MeltPoolDG::SphericalParticle<dim, double>;

    CohesiveWallForceTest()
      : mapping(1)
      , cohesive_force_data({.hamaker_constant                       = 40e-20,
                             .surface_energy                         = 1e-4,
                             .cut_off_relative_decline_van_der_waals = 1e-2})
      , wall_cohesive_force_data({.hamaker_constant                       = 20e-20,
                                  .surface_energy                         = 2e-4,
                                  .cut_off_relative_decline_van_der_waals = 1e-2})
      , cohesive_force(cohesive_force_data)
    {}

    void
    SetUp() override
    {
      dealii::GridGenerator::hyper_cube(triangulation);
      triangulation.refine_global(4);

      std::vector<std::vector<double>> particle_properties = {};
      std::vector<dealii::Point<dim>>  particle_locations  = {};

      obstacle_field = std::make_unique<MeltPoolDG::ObstacleField<dim, double, ObstacleType>>(
        obstacle_data, triangulation, mapping, particle_locations, particle_properties);

      const auto wall = std::make_shared<MeltPoolDG::Functions::Distance::Plane<dim, double>>(
        dealii::Point<dim>(0.5, 0.5, 0.0),
        dealii::Tensor<1, dim, double>({0.0, 0.0, 1.0}),
        std::array<double, dim - 1>{{1.0, 1.0}},
        dealii::Tensor<1, dim, double>({1.0, 0.0, 0.0}));

      cohesive_force.attach_wall(wall, wall_cohesive_force_data);
    }

    void
    insert_particle(const ParticleTestData &particle_data)
    {
      std::vector<double> properties(ObstacleType::n_obstacle_properties, 0);
      properties[ObstacleType::Properties::radius]      = particle_data.radius;
      properties[ObstacleType::Properties::particle_id] = particle_data.id;

      std::vector<dealii::Point<dim>>  particle_locations  = {particle_data.location};
      std::vector<std::vector<double>> particle_properties = {properties};

      obstacle_field->insert_obstacles(triangulation, {particle_locations}, particle_properties);
    }

    dealii::Triangulation<dim>                                            triangulation;
    dealii::MappingQ<dim>                                                 mapping;
    MeltPoolDG::ObstacleData<double>                                      obstacle_data;
    std::unique_ptr<MeltPoolDG::ObstacleField<dim, double, ObstacleType>> obstacle_field;
    MeltPoolDG::SphericalParticleCohesiveForceData<double>                cohesive_force_data;
    MeltPoolDG::SphericalParticleCohesiveForceData<double>                wall_cohesive_force_data;
    MeltPoolDG::SphericalParticleCohesiveForce<dim, double, ObstacleType> cohesive_force;
  };

  TEST_P(CohesiveWallForceTest, ParticleWallCohesiveForce_HorizontalWall)
  {
    const TestData &test_data = GetParam();

    constexpr double radius = 0.1;
    insert_particle({.id       = 0,
                     .radius   = radius,
                     .location = dealii::Point<dim>(0.5, 0.5, radius + test_data.normal_distance)});

    ASSERT_EQ(obstacle_field->n_global_particles(), 1)
      << "Test setup error: expected exactly one particle";

    cohesive_force.add_load_to_obstacles(*this->obstacle_field);

    for (const MeltPoolDG::DEMParticleAccessor<dim, double> &particle :
         this->obstacle_field->locally_owned_particle_range())
      {
        for (int d = 0; d < dim; ++d)
          {
            const double expected_force = test_data.expected_force[d];
            double       abs_err = expected_force == 0.0 ? 1e-50 : 1e-7 * std::abs(expected_force);

            SCOPED_TRACE("Component: " + std::to_string(d));
            EXPECT_NEAR(particle.force(d), expected_force, abs_err);
          }
      }
  }

  // Reference values for R_e = R = 0.1, A = 20e-20, gamma = 2e-4, c_FPO = 1e-2:
  //   F_po = 4 pi gamma R          = 2.51327412e-4
  //   g_0  = sqrt(A R / (6 F_po))  ~ 3.64e-9
  //   g*   = g_0 / sqrt(c_FPO)     ~ 3.64e-8
  // The force points from the particle towards the wall, i.e., in negative z-direction.
  INSTANTIATE_TEST_SUITE_P(
    ParticleWallCohesiveForce,
    CohesiveWallForceTest,
    ::testing::Values(
      // Gap is zero, so pull-off force is applied.
      TestData{
        .test_name       = "PullOffForce",
        .expected_force  = dealii::Tensor<1, dim, double>({0.0, 0.0, -2.51327412e-4}),
        .normal_distance = 0.,
      },
      // Gap g_N is chosen such that g_0 < g_N < g*, so van der Waals force is applied.
      TestData{
        .test_name       = "VanDerWaalsForce",
        .expected_force  = dealii::Tensor<1, dim, double>({0.0, 0.0, -8.33333333e-6}),
        .normal_distance = 2e-8,
      },
      // Gap g_N is larger than cut-off distance g*, so no force is applied.
      TestData{
        .test_name       = "ZeroCutOffForce",
        .expected_force  = dealii::Tensor<1, dim, double>({0.0, 0.0, 0.0}),
        .normal_distance = 1e-7,
      }),
    [](const testing::TestParamInfo<CohesiveWallForceTest::ParamType> &info) {
      return info.param.test_name;
    });
} // namespace
