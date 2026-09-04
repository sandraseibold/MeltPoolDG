#pragma once

#include <deal.II/base/exceptions.h>
#include <deal.II/base/function.h>
#include <deal.II/base/point.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>

#include <algorithm>
#include <array>
#include <cmath>


/**
 * Collection of (unsigned) distance functions to geometric primitives, in analogy to
 * dealii::Functions::SignedDistance.
 */
namespace MeltPoolDG::Functions::Distance
{
  /**
   * Distance function of a finite (rectangular) wall segment:
   * described by a @p point_on_wall, its outward @p normal, and the in-plane tangent
   * direction(s) along which the wall extends. In 2D there is a single tangent direction,
   * perpendicular to the normal; in 3D there are two, spanned by @p first_tangent_direction and
   * its cross product with the normal. The @p half_extents give the size of the wall: its
   * half-width along each of these tangent directions.
   */
  template <int dim, typename number>
  class Plane : public dealii::Function<dim, number>
  {
  public:
    /**
     * @param point_on_wall Center of the rectangular wall plane.
     * @param normal        Outward-pointing unit normal of the wall plane.
     * @param half_extents  Half-widths of the wall along each in-plane tangent direction.
     * @param first_tangent_direction (3D only) Direction of the first in-plane tangent, to which
     *                      @p half_extents[0] corresponds. Must be perpendicular to @p normal.
     *                      The second tangent is computed as normal x first_tangent_direction.
     */
    Plane(const dealii::Point<dim>          &point_on_wall,
          const dealii::Tensor<1, dim>      &normal,
          const std::array<number, dim - 1> &half_extents,
          const dealii::Tensor<1, dim>      &first_tangent_direction = dealii::Tensor<1, dim>())
      : point_on_wall(point_on_wall)
      , normal(normal)
      , half_extents(half_extents)
    {
      // Building  tangent basis
      if constexpr (dim == 2)
        {
          tangents[0][0] = -normal[1];
          tangents[0][1] = normal[0];
        }
      else if constexpr (dim == 3)
        {
          const number first_tangent_norm = first_tangent_direction.norm();
          tangents[0]                     = first_tangent_direction / first_tangent_norm;
          tangents[1]                     = dealii::cross_product_3d(normal, tangents[0]);
        }
    }

    dealii::Tensor<1, dim>
    gradient(const dealii::Point<dim> &x, const unsigned int component = 0) const override
    {
      const dealii::Tensor<1, dim> r      = offset_from_wall(x);
      const number                 r_norm = r.norm();

      // At the wall surface, r vanishes and r/r_norm is
      // undefined (0/0). Fall back to the wall's outward unit normal as the
      // gradient direction.
      constexpr number small_tolerance = 1e-12;
      if (r_norm < small_tolerance) // guard against the on-surface singularity
        return normal;              // gradient replaced by outward wall normal

      return r / r_norm;
    }

    number
    value(const dealii::Point<dim> &x, const unsigned int component = 0) const override
    {
      return offset_from_wall(x).norm();
    }

  private:
    /**
     * Vector from the closest point on the (clamped, finite) wall patch to @p x.
     */
    dealii::Tensor<1, dim>
    offset_from_wall(const dealii::Point<dim> &x) const
    {
      const dealii::Tensor<1, dim> d = x - point_on_wall;
      const number                 s = d * normal;

      dealii::Tensor<1, dim> r = s * normal;
      for (unsigned int i = 0; i < dim - 1; ++i)
        {
          const number u   = d * tangents[i];
          const number u_c = std::clamp(u, -half_extents[i], half_extents[i]);
          r += (u - u_c) * tangents[i];
        }

      return r;
    }

    dealii::Point<dim>                          point_on_wall;
    dealii::Tensor<1, dim>                      normal;
    std::array<dealii::Tensor<1, dim>, dim - 1> tangents;
    std::array<number, dim - 1>                 half_extents;
  };
} // namespace MeltPoolDG::Functions::Distance
