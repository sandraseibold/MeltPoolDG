#include <gtest/gtest.h>

#include <deal.II/base/function_parser.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <meltpooldg/utilities/distance_functions.hpp>

#include <cmath>
#include <memory>
#include <string>

namespace
{
  constexpr double tolerance = 1e-12;

  /**
   * Creates a time-dependent displacement function u(t) with dim components from a
   * ';'-separated expression, e.g. "0; -0.5*t".
   */
  template <int dim>
  std::unique_ptr<dealii::Function<dim>>
  create_displacement(const std::string &expression)
  {
    auto displacement = std::make_unique<dealii::FunctionParser<dim>>(dim);
    displacement->initialize(dealii::FunctionParser<dim>::default_variable_names() + ",t",
                             expression,
                             typename dealii::FunctionParser<dim>::ConstMap(),
                             true /* time dependent */);
    return displacement;
  }
} // namespace

/**
 * The fixture tests below check the TranslatingWall wrapper in 2D. The wall at rest is a
 * horizontal platform at y = 0 with outward normal +y, spanning x in [-1, 1]. It is translated
 * downwards with u(t) = (0, -0.5 t), i.e. at time t = 2 the platform lies at y = -1.
 */
class TranslatingWallTest2D : public ::testing::Test
{
protected:
  TranslatingWallTest2D()
    : wall(std::make_unique<MeltPoolDG::Functions::Distance::Plane<2, double>>(
             dealii::Point<2>(0., 0.),
             dealii::Tensor<1, 2>{{0., 1.}},
             std::array<double, 1>{{1.0}}),
           create_displacement<2>("0; -0.5*t"))
  {}

  MeltPoolDG::Functions::Distance::TranslatingWall<2, double> wall;
};

/**
 * Without calling set_time(), the wall must be at its initial position, since the constructor
 * evaluates the displacement at the initial time.
 */
TEST_F(TranslatingWallTest2D, Value_AtInitialTime)
{
  EXPECT_NEAR(wall.value(dealii::Point<2>(0., 1.)), 1.0, tolerance);
}

/**
 * After moving the platform down to y = -1, the distance of a fixed point above it increases.
 */
TEST_F(TranslatingWallTest2D, Value_AfterTranslation)
{
  wall.set_time(2.0);

  EXPECT_NEAR(wall.value(dealii::Point<2>(0., 1.)), 2.0, tolerance);
}

/**
 * A point on the translated platform surface has zero distance.
 */
TEST_F(TranslatingWallTest2D, Value_OnTranslatedSurface)
{
  wall.set_time(2.0);

  EXPECT_NEAR(wall.value(dealii::Point<2>(0.5, -1.)), 0.0, tolerance);
}

/**
 * The finite extent of the platform is translated as well: a point at the height of the
 * translated platform but beyond its edge has the in-plane distance to the edge.
 */
TEST_F(TranslatingWallTest2D, Value_BeyondFiniteExtentAfterTranslation)
{
  wall.set_time(2.0);

  EXPECT_NEAR(wall.value(dealii::Point<2>(2., -1.)), 1.0, tolerance);
}

/**
 * The gradient is evaluated at the shifted point and thus points from the translated platform
 * towards the point.
 */
TEST_F(TranslatingWallTest2D, Gradient_AfterTranslation)
{
  wall.set_time(2.0);

  const auto gradient = wall.gradient(dealii::Point<2>(0., 1.));
  EXPECT_NEAR(gradient[0], 0.0, tolerance);
  EXPECT_NEAR(gradient[1], 1.0, tolerance);
}

/**
 * Setting the time back must restore the initial position, i.e. the displacement is not
 * accumulated over consecutive set_time() calls.
 */
TEST_F(TranslatingWallTest2D, Value_ResetTime)
{
  wall.set_time(2.0);
  wall.set_time(0.0);

  EXPECT_NEAR(wall.value(dealii::Point<2>(0., 1.)), 1.0, tolerance);
}

/**
 * The fixture tests below check the TranslatingWall wrapper in 3D. The wall at rest is a
 * horizontal platform at z = 0 with outward normal +z and half-extents 1.0 (x) and 1.5 (y). It is
 * translated downwards with u(t) = (0, 0, -0.5 t).
 */
class TranslatingWallTest3D : public ::testing::Test
{
protected:
  TranslatingWallTest3D()
    : wall(std::make_unique<MeltPoolDG::Functions::Distance::Plane<3, double>>(
             dealii::Point<3>(0., 0., 0.),
             dealii::Tensor<1, 3>{{0., 0., 1.}},
             std::array<double, 2>{{1.0, 1.5}},
             dealii::Tensor<1, 3>{{1., 0., 0.}}),
           create_displacement<3>("0; 0; -0.5*t"))
  {}

  MeltPoolDG::Functions::Distance::TranslatingWall<3, double> wall;
};

TEST_F(TranslatingWallTest3D, Value_AfterTranslation)
{
  wall.set_time(2.0);

  EXPECT_NEAR(wall.value(dealii::Point<3>(0., 0., 1.)), 2.0, tolerance);
}

TEST_F(TranslatingWallTest3D, Value_BeyondBothExtentsAfterTranslation)
{
  wall.set_time(2.0);

  // closest point on the translated platform is the corner (1, 1.5, -1)
  EXPECT_NEAR(wall.value(dealii::Point<3>(2., 2.5, 0.)), std::sqrt(3.0), tolerance);
}

TEST_F(TranslatingWallTest3D, Gradient_AfterTranslation)
{
  wall.set_time(2.0);

  const auto gradient = wall.gradient(dealii::Point<3>(0.5, 0.5, 1.));
  EXPECT_NEAR(gradient[0], 0.0, tolerance);
  EXPECT_NEAR(gradient[1], 0.0, tolerance);
  EXPECT_NEAR(gradient[2], 1.0, tolerance);
}
