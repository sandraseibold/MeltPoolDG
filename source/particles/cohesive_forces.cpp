
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/particles/cohesive_forces.hpp>
#include <meltpooldg/particles/obstacle_field.hpp>
#include <meltpooldg/particles/particle_accessor.hpp>

#include <numbers>

namespace MeltPoolDG
{
  template <int dim, typename number, typename ObstacleType>
  SphericalParticleCohesiveForce<dim, number, ObstacleType>::SphericalParticleCohesiveForce(
    const SphericalParticleCohesiveForceData<number> &cohesive_force_data)
    : cohesive_force_data(cohesive_force_data)
  {}

  template <int dim, typename number, typename ObstacleType>
  void
  SphericalParticleCohesiveForce<dim, number, ObstacleType>::attach_wall(
    std::shared_ptr<dealii::Function<dim, number>>    wall_distance_function,
    const SphericalParticleCohesiveForceData<number> &wall_cohesive_force_data)
  {
    walls.push_back({std::move(wall_distance_function), wall_cohesive_force_data});
  }

  template <int dim, typename number, typename ObstacleType>
  void
  SphericalParticleCohesiveForce<dim, number, ObstacleType>::add_load_to_obstacles(
    ObstacleField<dim, number, ObstacleType> &obstacle_field) const
  {
    for (DEMParticleAccessor<dim, number> &particle : obstacle_field.locally_owned_particle_range())
      {
        // TODO: The value passed as relative tolerance for the find particles in neighborhood
        // function should be a parameter of the cohesive force data structure. For now, we use a
        // hardcoded value of 0.1.
        for (DEMParticleAccessor<dim, number> &other :
             obstacle_field.find_particles_in_neighborhood(particle, 0.1))
          {
            if (particle.id() == other.id())
              continue;

            CohesiveContactConfiguration contact_configuration(particle,
                                                               other,
                                                               cohesive_force_data);

            if (contact_configuration.normal_distance < contact_configuration.cut_off_distance)
              {
                dealii::Tensor<1, dim, number> cohesive_force;
                if (contact_configuration.normal_distance <=
                    contact_configuration.pull_off_force_limit_distance)
                  {
                    cohesive_force = contact_configuration.pull_off_force_magnitude *
                                     contact_configuration.normal_vector;
                  }
                else
                  {
                    cohesive_force = cohesive_force_data.hamaker_constant *
                                     contact_configuration.effective_radius /
                                     (6. * contact_configuration.normal_distance *
                                      contact_configuration.normal_distance) *
                                     contact_configuration.normal_vector;
                  }
                particle.add_force(cohesive_force);
              }
          }

        for (const CohesiveWall &wall : walls)
          {
            CohesiveContactConfiguration contact_configuration(particle,
                                                               wall.wall_distance_function.get(),
                                                               wall.cohesive_force_data);

            if (contact_configuration.normal_distance < contact_configuration.cut_off_distance)
              {
                dealii::Tensor<1, dim, number> cohesive_force;
                if (contact_configuration.normal_distance <=
                    contact_configuration.pull_off_force_limit_distance)
                  {
                    cohesive_force = contact_configuration.pull_off_force_magnitude *
                                     contact_configuration.normal_vector;
                  }
                else
                  {
                    cohesive_force = wall.cohesive_force_data.hamaker_constant *
                                     contact_configuration.effective_radius /
                                     (6. * contact_configuration.normal_distance *
                                      contact_configuration.normal_distance) *
                                     contact_configuration.normal_vector;
                  }
                particle.add_force(cohesive_force);
              }
          }
      }
  }

  template <int dim, typename number, typename ObstacleType>
  SphericalParticleCohesiveForce<dim, number, ObstacleType>::CohesiveContactConfiguration::
    CohesiveContactConfiguration(
      const DEMParticleAccessor<dim, number>           &self,
      const DEMParticleAccessor<dim, number>           &other,
      const SphericalParticleCohesiveForceData<number> &cohesive_force_data)
  {
    const number distance = self.get_location().distance(other.get_location());
    normal_vector         = (other.get_location() - self.get_location()) / distance;

    normal_distance  = distance - (self.get_property(ObstacleType::Properties::radius) +
                                  other.get_property(ObstacleType::Properties::radius));
    effective_radius = (self.get_property(ObstacleType::Properties::radius) *
                        other.get_property(ObstacleType::Properties::radius)) /
                       (self.get_property(ObstacleType::Properties::radius) +
                        other.get_property(ObstacleType::Properties::radius));

    pull_off_force_magnitude =
      4. * std::numbers::pi_v<number> * cohesive_force_data.surface_energy * effective_radius;

    pull_off_force_limit_distance =
      std::sqrt(cohesive_force_data.hamaker_constant * effective_radius /
                (6. * std::abs(pull_off_force_magnitude)));

    cut_off_distance = pull_off_force_limit_distance /
                       std::sqrt(cohesive_force_data.cut_off_relative_decline_van_der_waals);
  }

  template <int dim, typename number, typename ObstacleType>
  SphericalParticleCohesiveForce<dim, number, ObstacleType>::CohesiveContactConfiguration::
    CohesiveContactConfiguration(
      const DEMParticleAccessor<dim, number>           &particle,
      const dealii::Function<dim, number>              *wall,
      const SphericalParticleCohesiveForceData<number> &cohesive_force_data)
  {
    effective_radius = particle.get_property(ObstacleType::Properties::radius);
    normal_distance  = std::abs(wall->value(particle.get_location())) - effective_radius;

    pull_off_force_magnitude =
      4. * std::numbers::pi_v<number> * cohesive_force_data.surface_energy * effective_radius;

    pull_off_force_limit_distance =
      std::sqrt(cohesive_force_data.hamaker_constant * effective_radius /
                (6. * std::abs(pull_off_force_magnitude)));

    cut_off_distance = pull_off_force_limit_distance /
                       std::sqrt(cohesive_force_data.cut_off_relative_decline_van_der_waals);

    // Evaluating the gradient only if the particle is within the cut-off distance.
    if (normal_distance < cut_off_distance)
      {
        // Normal vector points from particle to wall.
        normal_vector =
          -wall->gradient(particle.get_location()) / wall->gradient(particle.get_location()).norm();
      }
  }
} // namespace MeltPoolDG

template struct MeltPoolDG::SphericalParticleCohesiveForceData<double>;

template class MeltPoolDG::
  SphericalParticleCohesiveForce<1, double, MeltPoolDG::SphericalParticle<1, double>>;
template class MeltPoolDG::
  SphericalParticleCohesiveForce<2, double, MeltPoolDG::SphericalParticle<2, double>>;
template class MeltPoolDG::
  SphericalParticleCohesiveForce<3, double, MeltPoolDG::SphericalParticle<3, double>>;
