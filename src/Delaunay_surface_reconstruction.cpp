// Copyright (c) 2026 Shengxian Zhao. All rights reserved.
//
// Author(s) : Shengxian Zhao

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/IO/read_points.h>
#include <CGAL/property_map.h>
#include <CGAL/Surface_mesh.h>

#include "Delaunay_surface_reconstruction.h"

#include <CGAL/Timer.h>
#include <fstream>
#include <iostream>

typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
typedef Kernel::Point_3  Point;
typedef Kernel::Vector_3 Vector;

typedef DSR::Delaunay_surface_reconstruction<Kernel> Delaunay_surface_reconstruction;

typedef CGAL::Surface_mesh<Point> Surface_mesh;

// Point with normal
typedef boost::tuple<Point, Vector> PN;
typedef CGAL::Nth_of_tuple_property_map<0, PN> Point_map;
typedef CGAL::Nth_of_tuple_property_map<1, PN> Normal_map;

/*
* The following example shows the reconstruction using
* user-provided unoriented normals stored in PLY format.
*/

int main(int argc, char* argv[])
{
  const std::string input_file = (argc > 1) ? argv[1] : CGAL::data_file_path("points_3/oni.ply");
  std::ifstream input_stream(input_file.c_str());
  if (!input_stream) {
    std::cerr << "Error: cannot open " << input_file << std::endl;
    return EXIT_FAILURE;
  }

  std::optional<double> alpha;
  if (argc > 2) alpha = std::stod(argv[2]);

  std::vector<PN> points; // store points

  std::cout << "Loading point cloud: " << input_file << "..." << std::endl;
  CGAL::Timer t;
  t.start();

  if (!CGAL::IO::read_PLY_with_properties(
        input_stream,
        std::back_inserter(points),
        CGAL::IO::make_ply_point_reader(Point_map()),
        CGAL::IO::make_ply_normal_reader(Normal_map())))
  {
    std::cerr << "Error: cannot read PLY with point+normal properties: " << input_file << std::endl;
    return EXIT_FAILURE;
  }
  else
    std::cout << "Done. " << points.size() << " points. Time: " << t.time() << " sec." << std::endl;

  //////////////////////////////////////////////////////////////////////////

  std::cout << "Building Delaunay + solving ILP..." << std::endl;
  t.reset();

  Delaunay_surface_reconstruction algo(points, Point_map(), Normal_map());

  std::cout << " Done. Time: " << t.time() << " sec." << std::endl;

  //////////////////////////////////////////////////////////////////////////

  Surface_mesh model;

  std::cout << "Reconstructing...";
  t.reset();

  if (!algo.reconstruct(model, alpha)) {
    std::cerr << "Failed: " << algo.error_message() << std::endl;
    return EXIT_FAILURE;
  }

  // Saves the mesh model
  const std::string output_file("delaunay_surface_reconstruction_result.off");
  if (CGAL::IO::write_OFF(output_file, model)) {
    std::cout << "Done. Saved to " << output_file << ". Time: " << t.time() << " sec." << std::endl;
  } else {
    std::cerr << "Failed saving OFF file." << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
