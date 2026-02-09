// Copyright (c) 2026 Shengxian Zhao. All rights reserved.
//
// Author(s) : Shengxian Zhao

#ifndef DELAUNAY_SURFACE_RECONSTRUCTION_H
#define DELAUNAY_SURFACE_RECONSTRUCTION_H

#include <CGAL/bounding_box.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_3.h>
#include <CGAL/Triangulation_data_structure_3.h>

#include <CGAL/Alpha_shape_3.h>
#include <CGAL/Alpha_shape_vertex_base_3.h>
#include <CGAL/Alpha_shape_cell_base_3.h>

#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>

#include <boost/functional/hash.hpp>

#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <optional>
#include <memory>
#include <unordered_map>

// ---- SCIP ----
#if defined(USE_SCIP)
  #include <scip/scip.h>
  #include <scip/scipdefplugins.h>
#endif

/*!
\file Delaunay_surface_reconstruction.h
*/

namespace DSR {

// ------------------ Hash keys ------------------

struct Facet_key {
  std::array<std::size_t, 3> v;
  bool operator==(const Facet_key& o) const { return v == o.v; }
};
struct Facet_key_hash {
  std::size_t operator()(const Facet_key& k) const noexcept {
    std::size_t seed = 0;
    boost::hash_combine(seed, k.v[0]);
    boost::hash_combine(seed, k.v[1]);
    boost::hash_combine(seed, k.v[2]);
    return seed;
  }
};

struct Edge_key {
  std::array<std::size_t, 2> v; // sorted
  bool operator==(const Edge_key& o) const { return v == o.v; }
};
struct Edge_key_hash {
  std::size_t operator()(const Edge_key& k) const noexcept {
    std::size_t seed = 0;
    boost::hash_combine(seed, k.v[0]);
    boost::hash_combine(seed, k.v[1]);
    return seed;
  }
};

// ------------------ Main class ------------------

/*!
\brief

Implementation of the Delaunay Surface Reconstruction method.

Given a set of 3D points with unoriented normals sampled from
the outer boundary of an object, the Delaunay Surface Reconstruction
method outputs a watertight surface mesh interpolating the input point set.

The method first computes an alpha shape, which contains the desired boundary/surface.
Finally an optimal subset of the candidate faces is selected through optimization
under hard constraints that enforce the final model to be manifold and watertight.

\tparam GeomTraits a geometric traits class, model of Kernel
*/

template <class GeomTraits>
class Delaunay_surface_reconstruction
{
public:
  using Point  = typename GeomTraits::Point_3;
  using Vector = typename GeomTraits::Vector_3;
  using Surface_mesh = CGAL::Surface_mesh<Point>;

  template <typename PointRange, typename PointMap, typename NormalMap>
  Delaunay_surface_reconstruction(const PointRange& pts, PointMap pm, NormalMap nm)
  {
    points_.reserve(pts.size());
    normals_.reserve(pts.size());
    for (const auto& it : pts) {
      points_.push_back(get(pm, it));
      normals_.push_back(get(nm, it));
    }
  }

  // If alpha is provided -> use it. Otherwise -> choose "optimal" alpha via CGAL (1 component) and use it.
  bool reconstruct(Surface_mesh& out_mesh, std::optional<double> alpha = std::nullopt);

  // Returns the error message if reconstruction failed.
  const std::string& error_message() const { return error_; }

private:
  // --- Underlying DT for Alpha_shape_3 with vertex->info() support ---
  using Vb0 = CGAL::Triangulation_vertex_base_with_info_3<std::size_t, GeomTraits>;
  using Vb  = CGAL::Alpha_shape_vertex_base_3<GeomTraits, Vb0>;

  using Cb0 = CGAL::Triangulation_cell_base_3<GeomTraits>;
  using Cb  = CGAL::Alpha_shape_cell_base_3<GeomTraits, Cb0>;

  using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
  using DT  = CGAL::Delaunay_triangulation_3<GeomTraits,Tds,CGAL::Fast_location>;

  using Alpha_shape_3 = CGAL::Alpha_shape_3<DT>;
  using FT = typename Alpha_shape_3::FT;
  using Alpha_iterator = typename Alpha_shape_3::Alpha_iterator;

  struct Facet_record {
    Facet_key key;                       // sorted vertex ids
    std::array<std::size_t,3> vids;      // geometric ordering (as read from a facet)
    double cost = 0.0;
  };

  std::vector<Point>  points_;
  std::vector<Vector> normals_;
  std::string error_;

  std::unique_ptr<Alpha_shape_3> as_;

  std::vector<Facet_record> facets_; // facet_id -> record
  std::unordered_map<Facet_key, std::size_t, Facet_key_hash> facet_id_;

  std::vector<Edge_key> edges_; // edge_id -> key
  std::unordered_map<Edge_key, std::size_t, Edge_key_hash> edge_id_;

  std::vector<std::vector<std::size_t>> incident_facets_; // edge_id -> list of facet_id

private:
  // weight for the area term
  double weight_area(double parameter_area = 4.0) const {
    CGAL::Bbox_3 bbox = CGAL::bbox_3(points_.begin(), points_.end());
    double dx = bbox.xmax() - bbox.xmin();
    double dy = bbox.ymax() - bbox.ymin();
    double dz = bbox.zmax() - bbox.zmin();
    double box_area = 2.0 * (dx*dx + dy*dy + dz*dz);
    return parameter_area / box_area;
  }

  // weight for the aspect ratio of every triangle
  double weight_aspect_ratio(double parameter_edge_ratio = 1.0) const {
    return parameter_edge_ratio;
  }

  bool build_alpha_shape(std::optional<double> alpha);
  bool enumerate_facets_and_edges_from_alpha_boundary();
  double facet_cost(const std::array<std::size_t,3>& vids) const;

  bool solve_binary_ilp_scip(std::vector<int>& x_solution) const;
  bool build_mesh_from_solution(const std::vector<int>& x_solution, Surface_mesh& out_mesh) const;

  static double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
  }

private: // Copying is not allowed
  Delaunay_surface_reconstruction(const Delaunay_surface_reconstruction& dsr);

}; // end of Delaunay_surface_reconstruction

// ------------------ Implementation ------------------

template <class GeomTraits>
double Delaunay_surface_reconstruction<GeomTraits>::facet_cost(const std::array<std::size_t,3>& vids) const
{
  const Point& p0 = points_[vids[0]];
  const Point& p1 = points_[vids[1]];
  const Point& p2 = points_[vids[2]];

  const Vector e1 = p1 - p0;
  const Vector e2 = p2 - p0;
  const Vector e3 = p2 - p1;
  const double l1n2 = CGAL::to_double(e1.squared_length());
  const double l2n2 = CGAL::to_double(e2.squared_length());
  const double l3n2 = CGAL::to_double(e3.squared_length());
  const double ln2_min = std::min({l1n2, l2n2, l3n2});
  const double ln2_max = std::max({l1n2, l2n2, l3n2});
  Vector nf = CGAL::cross_product(e1, e2);

  const double nfn2 = CGAL::to_double(nf.squared_length());

  const double eps = 5e-3;
  double c = points_.size() * weight_area() * std::sqrt(nfn2);

  if (!(nfn2 > 0.0)) return 1e6; // degenerate
  nf = nf / std::sqrt(nfn2);

  if (!(ln2_min > 0.0)) {
    return 1e6;
  } else {
    c += weight_aspect_ratio() * std::sqrt(ln2_max / ln2_min - 1);
  }

  for (int k = 0; k < 3; ++k) {
    Vector nv = normals_[vids[k]];
    const double nvn2 = CGAL::to_double(nv.squared_length());
    if (!(nvn2 > 0.0)) { continue; }
    nv = nv / std::sqrt(nvn2);

    double d = std::fabs(CGAL::to_double(nf * nv)); // |dot|
    d = clamp(d, eps, 1.0 - eps);

    c += (std::log(1.0 - d*d) - std::log(d*d));
  }
  return c;
}

template <class GeomTraits>
bool Delaunay_surface_reconstruction<GeomTraits>::build_alpha_shape(std::optional<double> alpha)
{
  if (points_.empty()) {
    error_ = "Empty input.";
    return false;
  }

  // 1) Build underlying Delaunay triangulation with info() set correctly
  DT dt;
  for (std::size_t i = 0; i < points_.size(); ++i) {
    auto vh = dt.insert(points_[i]);
    if (vh == typename DT::Vertex_handle()) {
      error_ = "Delaunay insertion failed while building alpha shape.";
      return false;
    }
    vh->info() = i;
  }

  // 2) Construct alpha shape from DT (this destroys 'dt' per CGAL docs)
  as_ = std::make_unique<Alpha_shape_3>(dt, FT(0), Alpha_shape_3::REGULARIZED);

  // 3) Choose alpha
  if (alpha.has_value()) {
    as_->set_alpha(FT(*alpha));
    return true;
  }

  // "Optimal" alpha per CGAL: smallest alpha giving <= nb_components solid components
  auto opt = as_->find_optimal_alpha(1);  // nb_components = 1
  if (opt == as_->alpha_end()) {
    error_ = "Alpha_shape_3::find_optimal_alpha(1) failed.";
    return false;
  }
  as_->set_alpha(*opt);
  return true;
}

template <class GeomTraits>
bool Delaunay_surface_reconstruction<GeomTraits>::enumerate_facets_and_edges_from_alpha_boundary()
{
  facets_.clear();
  facet_id_.clear();
  edges_.clear();
  edge_id_.clear();
  incident_facets_.clear();

  for (auto fit = as_->finite_facets_begin(); fit != as_->finite_facets_end(); ++fit) {
    auto facet = *fit;
    if (as_->classify(facet) != Alpha_shape_3::REGULAR && as_->classify(facet) != Alpha_shape_3::INTERIOR)
      continue;

    auto cell = facet.first;
    int opp = facet.second;

    std::array<std::size_t,3> vids;
    int t = 0;
    for (int i = 0; i < 4; ++i) {
      if (i == opp) continue;
      vids[t++] = cell->vertex(i)->info();
    }

    Facet_key key{vids};
    std::sort(key.v.begin(), key.v.end());

    if (facet_id_.find(key) == facet_id_.end()) {
      const std::size_t id = facets_.size();
      facet_id_[key] = id;

      Facet_record rec;
      rec.key  = key;
      rec.vids = vids;
      rec.cost = facet_cost(vids);
      facets_.push_back(rec);
    }
  }

  if (facets_.empty()) {
    error_ = "No finite facets in alpha shape for the chosen alpha.";
    return false;
  }

  // 2) Build edge set E and incidence N(e) from candidate facets directly (robust; avoids triangulation edge APIs)
  edges_.reserve(facets_.size() * 3);
  incident_facets_.clear();

  for (std::size_t fid = 0; fid < facets_.size(); ++fid) {
    const auto& tri = facets_[fid].vids;

    auto add_edge = [&](std::size_t u, std::size_t v) {
      if (u == v) return;
      if (u > v) std::swap(u, v);
      Edge_key ek{{u,v}};

      auto it = edge_id_.find(ek);
      if (it == edge_id_.end()) {
        const std::size_t eid = edges_.size();
        edge_id_[ek] = eid;
        edges_.push_back(ek);
        incident_facets_.push_back({});
        incident_facets_.back().push_back(fid);
      } else {
        incident_facets_[it->second].push_back(fid);
      }
    };

    add_edge(tri[0], tri[1]);
    add_edge(tri[1], tri[2]);
    add_edge(tri[2], tri[0]);
  }

  // Deduplicate incidence lists
  for (auto& list : incident_facets_) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }

  if (edges_.empty()) {
    error_ = "No edges derived from candidate facets.";
    return false;
  }

  return true;
}

template <class GeomTraits>
bool Delaunay_surface_reconstruction<GeomTraits>::solve_binary_ilp_scip(std::vector<int>& x_solution) const
{
  if (!error_.empty()) { // an error has occurred in the constructor
    return false;
  }

#if !defined(USE_SCIP)
  (void)x_solution;
  return false;
#else
  const int nF = static_cast<int>(facets_.size());
  const int nE = static_cast<int>(edges_.size());

  SCIP* scip = nullptr;
  SCIP_CALL_ABORT( SCIPcreate(&scip) );
  SCIP_CALL_ABORT( SCIPincludeDefaultPlugins(scip) );
  SCIP_CALL_ABORT( SCIPcreateProbBasic(scip, "alpha_surface_reconstruction") );
  SCIP_CALL_ABORT( SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE) );

  // Variables: x_f (binary), y_e (binary)
  std::vector<SCIP_VAR*> xvars(nF, nullptr);
  for (int f = 0; f < nF; ++f) {
    SCIP_VAR* var = nullptr;
    std::string name = "x_" + std::to_string(f);
    SCIP_CALL_ABORT(
      SCIPcreateVarBasic(
        scip, &var, name.c_str(),
        0.0, 1.0,
        facets_[f].cost,
        SCIP_VARTYPE_BINARY
      )
    );
    SCIP_CALL_ABORT( SCIPaddVar(scip, var) );
    xvars[f] = var;
  }

  std::vector<SCIP_VAR*> yvars(nE, nullptr);
  for (int e = 0; e < nE; ++e) {
    SCIP_VAR* var = nullptr;
    std::string name = "y_" + std::to_string(e);
    SCIP_CALL_ABORT(
      SCIPcreateVarBasic(
        scip, &var, name.c_str(),
        0.0, 1.0,
        0.0,
        SCIP_VARTYPE_BINARY
      )
    );
    SCIP_CALL_ABORT( SCIPaddVar(scip, var) );
    yvars[e] = var;
  }

  // Constraints: for each edge e: sum_{f in N(e)} x_f - 2 y_e = 0
  for (int e = 0; e < nE; ++e) {
    SCIP_CONS* cons = nullptr;
    std::string cname = "edge_" + std::to_string(e);

    SCIP_CALL_ABORT( SCIPcreateConsBasicLinear(scip, &cons, cname.c_str(),
                                              0, nullptr, nullptr,
                                              0.0, 0.0) );

    for (std::size_t idx = 0; idx < incident_facets_[e].size(); ++idx) {
      const int f = static_cast<int>(incident_facets_[e][idx]);
      SCIP_CALL_ABORT( SCIPaddCoefLinear(scip, cons, xvars[f], 1.0) );
    }
    SCIP_CALL_ABORT( SCIPaddCoefLinear(scip, cons, yvars[e], -2.0) );

    SCIP_CALL_ABORT( SCIPaddCons(scip, cons) );
    SCIP_CALL_ABORT( SCIPreleaseCons(scip, &cons) );
  }

  // Global non-emptiness constraint: sum_f x_f >= 1
  {
    SCIP_CONS* cons = nullptr;

    SCIP_CALL_ABORT(
      SCIPcreateConsBasicLinear(
        scip,
        &cons,
        "at_least_one_facet",
        0, nullptr, nullptr,
        1.0,                     // lhs = 1
        SCIPinfinity(scip)       // rhs = +∞
      )
    );

    for (int f = 0; f < nF; ++f) {
      SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, xvars[f], 1.0));
    }

    SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
    SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
  }


  SCIP_CALL_ABORT( SCIPsolve(scip) );

  SCIP_SOL* sol = SCIPgetBestSol(scip);
  if (!sol) {
    for (auto*& v : xvars) SCIP_CALL_ABORT( SCIPreleaseVar(scip, &v) );
    for (auto*& v : yvars) SCIP_CALL_ABORT( SCIPreleaseVar(scip, &v) );
    SCIP_CALL_ABORT( SCIPfree(&scip) );
    return false;
  }

  x_solution.assign(nF, 0);
  for (int f = 0; f < nF; ++f) {
    const double val = SCIPgetSolVal(scip, sol, xvars[f]);
    x_solution[f] = (val > 0.5) ? 1 : 0;
  }

  for (auto*& v : xvars) SCIP_CALL_ABORT( SCIPreleaseVar(scip, &v) );
  for (auto*& v : yvars) SCIP_CALL_ABORT( SCIPreleaseVar(scip, &v) );

  SCIP_CALL_ABORT( SCIPfree(&scip) );
  return true;
#endif
}

template <class GeomTraits>
bool Delaunay_surface_reconstruction<GeomTraits>::build_mesh_from_solution(
  const std::vector<int>& x_solution,
  Surface_mesh& out_mesh) const
{
  namespace PMP = CGAL::Polygon_mesh_processing;

  std::vector<Point> soup_points = points_;
  std::vector<std::array<std::size_t,3>> soup_faces;
  soup_faces.reserve(facets_.size());

  for (std::size_t f = 0; f < facets_.size(); ++f) {
    if (x_solution[f] == 0) continue;
    soup_faces.push_back(facets_[f].vids);
  }
  if (soup_faces.empty())
    return false;

  PMP::orient_polygon_soup(soup_points, soup_faces);
  out_mesh.clear();
  PMP::polygon_soup_to_polygon_mesh(soup_points, soup_faces, out_mesh);

  return !out_mesh.is_empty();
}

template <class GeomTraits>
bool Delaunay_surface_reconstruction<GeomTraits>::reconstruct(Surface_mesh& out_mesh, std::optional<double> alpha)
{
  error_.clear();

  if (!build_alpha_shape(alpha)) return false;
  if (!enumerate_facets_and_edges_from_alpha_boundary()) return false;

#if defined(USE_SCIP)
  std::vector<int> x_sol;
  if (!solve_binary_ilp_scip(x_sol)) {
    error_ = "SCIP failed to solve the binary ILP (no feasible/optimal solution returned).";
    return false;
  }
  if (!build_mesh_from_solution(x_sol, out_mesh)) {
    error_ = "ILP solved, but selected facet set did not yield a valid polygon mesh.";
    return false;
  }
  return true;
#else
  error_ = "This build does not enable SCIP (compile with -DUSE_SCIP and link SCIP).";
  return false;
#endif
}

} // namespace DSR

#endif // DELAUNAY_SURFACE_RECONSTRUCTION_H
// Copyright (c) 2026 Shengxian Zhao. All rights reserved.
//
// Author(s) : Shengxian Zhao

#ifndef DELAUNAY_SURFACE_RECONSTRUCTION_H
#define DELAUNAY_SURFACE_RECONSTRUCTION_H

#include <CGAL/bounding_box.h>
#include <CGAL/Delaunay_triangulation_3.h>
#include <CGAL/Triangulation_vertex_base_with_info_3.h>
#include <CGAL/Triangulation_cell_base_3.h>
#include <CGAL/Triangulation_data_structure_3.h>

#include <CGAL/Alpha_shape_3.h>
#include <CGAL/Alpha_shape_vertex_base_3.h>
#include <CGAL/Alpha_shape_cell_base_3.h>

#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/orient_polygon_soup.h>
#include <CGAL/Polygon_mesh_processing/polygon_soup_to_polygon_mesh.h>

#include <boost/functional/hash.hpp>

#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <limits>
#include <algorithm>
#include <optional>
#include <memory>
#include <unordered_map>

// ---- SCIP ----
#if defined(USE_SCIP)
  #include <scip/scip.h>
  #include <scip/scipdefplugins.h>
#endif

/*!
\file Delaunay_surface_reconstruction.h
*/

namespace DSR {

// ------------------ Hash keys ------------------

struct Facet_key {
  std::array<std::size_t, 3> v;
  bool operator==(const Facet_key& o) const { return v == o.v; }
};
struct Facet_key_hash {
  std::size_t operator()(const Facet_key& k) const noexcept {
    std::size_t seed = 0;
    boost::hash_combine(seed, k.v[0]);
    boost::hash_combine(seed, k.v[1]);
    boost::hash_combine(seed, k.v[2]);
    return seed;
  }
};

struct Edge_key {
  std::array<std::size_t, 2> v; // sorted
  bool operator==(const Edge_key& o) const { return v == o.v; }
};
struct Edge_key_hash {
  std::size_t operator()(const Edge_key& k) const noexcept {
    std::size_t seed = 0;
    boost::hash_combine(seed, k.v[0]);
    boost::hash_combine(seed, k.v[1]);
    return seed;
  }
};

// ------------------ Main class ------------------

/*!
\brief

Implementation of the Delaunay Surface Reconstruction method.

Given a set of 3D points with unoriented normals sampled from
the outer boundary of an object, the Delaunay Surface Reconstruction
method outputs a watertight surface mesh interpolating the input point set.

The method first reduces the size of the point set by voxel downsampling.
Then an alpha shape is computed, which contains the desired boundary/surface.
Finally an optimal subset of the candidate faces is selected through optimization
under hard constraints that enforce the final model to be manifold and watertight.

\tparam GeomTraits a geometric traits class, model of Kernel
*/

template <class GeomTraits>
class Delaunay_surface_reconstruction
{
public:
  using Point  = typename GeomTraits::Point_3;
  using Vector = typename GeomTraits::Vector_3;
  using Surface_mesh = CGAL::Surface_mesh<Point>;

  template <typename PointRange, typename PointMap, typename NormalMap>
  Delaunay_surface_reconstruction(const PointRange& pts, PointMap pm, NormalMap nm)
  {
    points_.reserve(pts.size());
    normals_.reserve(pts.size());
    for (const auto& it : pts) {
      points_.push_back(get(pm, it));
      normals_.push_back(get(nm, it));
    }
  }

  // If alpha is provided -> use it. Otherwise -> choose "optimal" alpha via CGAL (1 component) and use it.
  bool reconstruct(Surface_mesh& out_mesh, std::optional<double> alpha = std::nullopt);

  // Returns the error message if reconstruction failed.
  const std::string& error_message() const { return error_; }

private:
  // --- Underlying DT for Alpha_shape_3 with vertex->info() support ---
  using Vb0 = CGAL::Triangulation_vertex_base_with_info_3<std::size_t, GeomTraits>;
  using Vb  = CGAL::Alpha_shape_vertex_base_3<GeomTraits, Vb0>;

  using Cb0 = CGAL::Triangulation_cell_base_3<GeomTraits>;
  using Cb  = CGAL::Alpha_shape_cell_base_3<GeomTraits, Cb0>;

  using Tds = CGAL::Triangulation_data_structure_3<Vb, Cb>;
  using DT  = CGAL::Delaunay_triangulation_3<GeomTraits,Tds,CGAL::Fast_location>;

  using Alpha_shape_3 = CGAL::Alpha_shape_3<DT>;
  using FT = typename Alpha_shape_3::FT;
  using Alpha_iterator = typename Alpha_shape_3::Alpha_iterator;

  struct Facet_record {
    Facet_key key;                       // sorted vertex ids
    std::array<std::size_t,3> vids;      // geometric ordering (as read from a facet)
    double cost = 0.0;
  };

  std::vector<Point>  points_;
  std::vector<Vector> normals_;
  std::string error_;

  std::unique_ptr<Alpha_shape_3> as_;

  std::vector<Facet_record> facets_; // facet_id -> record
  std::unordered_map<Facet_key, std::size_t, Facet_key_hash> facet_id_;

  std::vector<Edge_key> edges_; // edge_id -> key
  std::unordered_map<Edge_key, std::size_t, Edge_key_hash> edge_id_;

  std::vector<std::vector<std::size_t>> incident_facets_; // edge_id -> list of facet_id

private:
  // weight for the area term
  double weight_area(double parameter_area = 4.0) const {
    CGAL::Bbox_3 bbox = CGAL::bbox_3(points_.begin(), points_.end());
    double dx = bbox.xmax() - bbox.xmin();
    double dy = bbox.ymax() - bbox.ymin();
    double dz = bbox.zmax() - bbox.zmin();
    double box_area = 2.0 * (dx*dx + dy*dy + dz*dz);
    return parameter_area / box_area;
  }

  // weight for the aspect ratio of every triangle
  double weight_aspect_ratio(double parameter_edge_ratio = 1.0) const {
    return parameter_edge_ratio;
  }

  bool build_alpha_shape(std::optional<double> alpha);
  bool enumerate_facets_and_edges_from_alpha_boundary();
  double facet_cost(const std::array<std::size_t,3>& vids) const;

  bool solve_binary_ilp_scip(std::vector<int>& x_solution) const;
  bool build_mesh_from_solution(const std::vector<int>& x_solution, Surface_mesh& out_mesh) const;

  static double clamp(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
  }

private: // Copying is not allowed
  Delaunay_surface_reconstruction(const Delaunay_surface_reconstruction& dsr);

}; // end of Delaunay_surface_reconstruction

// ------------------ Implementation ------------------

template <class GeomTraits>
double Delaunay_surface_reconstruction<GeomTraits>::facet_cost(const std::array<std::size_t,3>& vids) const
{
  const Point& p0 = points_[vids[0]];
  const Point& p1 = points_[vids[1]];
  const Point& p2 = points_[vids[2]];

  const Vector e1 = p1 - p0;
  const Vector e2 = p2 - p0;
  const Vector e3 = p2 - p1;
  const double l1n2 = CGAL::to_double(e1.squared_length());
  const double l2n2 = CGAL::to_double(e2.squared_length());
  const double l3n2 = CGAL::to_double(e3.squared_length());
  const double ln2_min = std::min({l1n2, l2n2, l3n2});
  const double ln2_max = std::max({l1n2, l2n2, l3n2});
  Vector nf = CGAL::cross_product(e1, e2);

  const double nfn2 = CGAL::to_double(nf.squared_length());

  const double eps = 5e-3;
  double c = points_.size() * weight_area() * std::sqrt(nfn2);

  if (!(nfn2 > 0.0)) return 1e6; // degenerate
  nf = nf / std::sqrt(nfn2);

  if (!(ln2_min > 0.0)) {
    return 1e6;
  } else {
    c += weight_aspect_ratio() * std::sqrt(ln2_max / ln2_min - 1);
  }

  for (int k = 0; k < 3; ++k) {
    Vector nv = normals_[vids[k]];
    const double nvn2 = CGAL::to_double(nv.squared_length());
    if (!(nvn2 > 0.0)) { continue; }
    nv = nv / std::sqrt(nvn2);

    double d = std::fabs(CGAL::to_double(nf * nv)); // |dot|
    d = clamp(d, eps, 1.0 - eps);

    c += (std::log(1.0 - d*d) - std::log(d*d));
  }
  return c;
}

template <class GeomTraits>
bool Delaunay_surface_reconstruction<GeomTraits>::build_alpha_shape(std::optional<double> alpha)
{
  if (points_.empty()) {
    error_ = "Empty input.";
    return false;
  }

  // 1) Build underlying Delaunay triangulation with info() set correctly
  DT dt;
  for (std::size_t i = 0; i < points_.size(); ++i) {
    auto vh = dt.insert(points_[i]);
    if (vh == typename DT::Vertex_handle()) {
      error_ = "Delaunay insertion failed while building alpha shape.";
      return false;
    }
    vh->info() = i;
  }

  // 2) Construct alpha shape from DT (this destroys 'dt' per CGAL docs)
  as_ = std::make_unique<Alpha_shape_3>(dt, FT(0), Alpha_shape_3::REGULARIZED);

  // 3) Choose alpha
  if (alpha.has_value()) {
    as_->set_alpha(FT(*alpha));
    return true;
  }

  // "Optimal" alpha per CGAL: smallest alpha giving <= nb_components solid components
  auto opt = as_->find_optimal_alpha(1);  // nb_components = 1
  if (opt == as_->alpha_end()) {
    error_ = "Alpha_shape_3::find_optimal_alpha(1) failed.";
    return false;
  }
  as_->set_alpha(*opt);
  return true;
}

template <class GeomTraits>
bool Delaunay_surface_reconstruction<GeomTraits>::enumerate_facets_and_edges_from_alpha_boundary()
{
  facets_.clear();
  facet_id_.clear();
  edges_.clear();
  edge_id_.clear();
  incident_facets_.clear();

  for (auto fit = as_->finite_facets_begin(); fit != as_->finite_facets_end(); ++fit) {
    auto facet = *fit;
    if (as_->classify(facet) != Alpha_shape_3::REGULAR && as_->classify(facet) != Alpha_shape_3::INTERIOR)
      continue;

    auto cell = facet.first;
    int opp = facet.second;

    std::array<std::size_t,3> vids;
    int t = 0;
    for (int i = 0; i < 4; ++i) {
      if (i == opp) continue;
      vids[t++] = cell->vertex(i)->info();
    }

    Facet_key key{vids};
    std::sort(key.v.begin(), key.v.end());

    if (facet_id_.find(key) == facet_id_.end()) {
      const std::size_t id = facets_.size();
      facet_id_[key] = id;

      Facet_record rec;
      rec.key  = key;
      rec.vids = vids;
      rec.cost = facet_cost(vids);
      facets_.push_back(rec);
    }
  }

  if (facets_.empty()) {
    error_ = "No finite facets in alpha shape for the chosen alpha.";
    return false;
  }

  // 2) Build edge set E and incidence N(e) from candidate facets directly (robust; avoids triangulation edge APIs)
  edges_.reserve(facets_.size() * 3);
  incident_facets_.clear();

  for (std::size_t fid = 0; fid < facets_.size(); ++fid) {
    const auto& tri = facets_[fid].vids;

    auto add_edge = [&](std::size_t u, std::size_t v) {
      if (u == v) return;
      if (u > v) std::swap(u, v);
      Edge_key ek{{u,v}};

      auto it = edge_id_.find(ek);
      if (it == edge_id_.end()) {
        const std::size_t eid = edges_.size();
        edge_id_[ek] = eid;
        edges_.push_back(ek);
        incident_facets_.push_back({});
        incident_facets_.back().push_back(fid);
      } else {
        incident_facets_[it->second].push_back(fid);
      }
    };

    add_edge(tri[0], tri[1]);
    add_edge(tri[1], tri[2]);
    add_edge(tri[2], tri[0]);
  }

  // Deduplicate incidence lists
  for (auto& list : incident_facets_) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }

  if (edges_.empty()) {
    error_ = "No edges derived from candidate facets.";
    return false;
  }

  return true;
}

template <class GeomTraits>
bool Delaunay_surface_reconstruction<GeomTraits>::solve_binary_ilp_scip(std::vector<int>& x_solution) const
{
  if (!error_.empty()) { // an error has occurred in the constructor
    return false;
  }

#if !defined(USE_SCIP)
  (void)x_solution;
  return false;
#else
  const int nF = static_cast<int>(facets_.size());
  const int nE = static_cast<int>(edges_.size());

  SCIP* scip = nullptr;
  SCIP_CALL_ABORT( SCIPcreate(&scip) );
  SCIP_CALL_ABORT( SCIPincludeDefaultPlugins(scip) );
  SCIP_CALL_ABORT( SCIPcreateProbBasic(scip, "alpha_surface_reconstruction") );
  SCIP_CALL_ABORT( SCIPsetObjsense(scip, SCIP_OBJSENSE_MINIMIZE) );

  // Variables: x_f (binary), y_e (binary)
  std::vector<SCIP_VAR*> xvars(nF, nullptr);
  for (int f = 0; f < nF; ++f) {
    SCIP_VAR* var = nullptr;
    std::string name = "x_" + std::to_string(f);
    SCIP_CALL_ABORT(
      SCIPcreateVarBasic(
        scip, &var, name.c_str(),
        0.0, 1.0,
        facets_[f].cost,
        SCIP_VARTYPE_BINARY
      )
    );
    SCIP_CALL_ABORT( SCIPaddVar(scip, var) );
    xvars[f] = var;
  }

  std::vector<SCIP_VAR*> yvars(nE, nullptr);
  for (int e = 0; e < nE; ++e) {
    SCIP_VAR* var = nullptr;
    std::string name = "y_" + std::to_string(e);
    SCIP_CALL_ABORT(
      SCIPcreateVarBasic(
        scip, &var, name.c_str(),
        0.0, 1.0,
        0.0,
        SCIP_VARTYPE_BINARY
      )
    );
    SCIP_CALL_ABORT( SCIPaddVar(scip, var) );
    yvars[e] = var;
  }

  // Constraints: for each edge e: sum_{f in N(e)} x_f - 2 y_e = 0
  for (int e = 0; e < nE; ++e) {
    SCIP_CONS* cons = nullptr;
    std::string cname = "edge_" + std::to_string(e);

    SCIP_CALL_ABORT( SCIPcreateConsBasicLinear(scip, &cons, cname.c_str(),
                                              0, nullptr, nullptr,
                                              0.0, 0.0) );

    for (std::size_t idx = 0; idx < incident_facets_[e].size(); ++idx) {
      const int f = static_cast<int>(incident_facets_[e][idx]);
      SCIP_CALL_ABORT( SCIPaddCoefLinear(scip, cons, xvars[f], 1.0) );
    }
    SCIP_CALL_ABORT( SCIPaddCoefLinear(scip, cons, yvars[e], -2.0) );

    SCIP_CALL_ABORT( SCIPaddCons(scip, cons) );
    SCIP_CALL_ABORT( SCIPreleaseCons(scip, &cons) );
  }

  // Global non-emptiness constraint: sum_f x_f >= 1
  {
    SCIP_CONS* cons = nullptr;

    SCIP_CALL_ABORT(
      SCIPcreateConsBasicLinear(
        scip,
        &cons,
        "at_least_one_facet",
        0, nullptr, nullptr,
        1.0,                     // lhs = 1
        SCIPinfinity(scip)       // rhs = +∞
      )
    );

    for (int f = 0; f < nF; ++f) {
      SCIP_CALL_ABORT(SCIPaddCoefLinear(scip, cons, xvars[f], 1.0));
    }

    SCIP_CALL_ABORT(SCIPaddCons(scip, cons));
    SCIP_CALL_ABORT(SCIPreleaseCons(scip, &cons));
  }


  SCIP_CALL_ABORT( SCIPsolve(scip) );

  SCIP_SOL* sol = SCIPgetBestSol(scip);
  if (!sol) {
    for (auto*& v : xvars) SCIP_CALL_ABORT( SCIPreleaseVar(scip, &v) );
    for (auto*& v : yvars) SCIP_CALL_ABORT( SCIPreleaseVar(scip, &v) );
    SCIP_CALL_ABORT( SCIPfree(&scip) );
    return false;
  }

  x_solution.assign(nF, 0);
  for (int f = 0; f < nF; ++f) {
    const double val = SCIPgetSolVal(scip, sol, xvars[f]);
    x_solution[f] = (val > 0.5) ? 1 : 0;
  }

  for (auto*& v : xvars) SCIP_CALL_ABORT( SCIPreleaseVar(scip, &v) );
  for (auto*& v : yvars) SCIP_CALL_ABORT( SCIPreleaseVar(scip, &v) );

  SCIP_CALL_ABORT( SCIPfree(&scip) );
  return true;
#endif
}

template <class GeomTraits>
bool Delaunay_surface_reconstruction<GeomTraits>::build_mesh_from_solution(
  const std::vector<int>& x_solution,
  Surface_mesh& out_mesh) const
{
  namespace PMP = CGAL::Polygon_mesh_processing;

  std::vector<Point> soup_points = points_;
  std::vector<std::array<std::size_t,3>> soup_faces;
  soup_faces.reserve(facets_.size());

  for (std::size_t f = 0; f < facets_.size(); ++f) {
    if (x_solution[f] == 0) continue;
    soup_faces.push_back(facets_[f].vids);
  }
  if (soup_faces.empty())
    return false;

  PMP::orient_polygon_soup(soup_points, soup_faces);
  out_mesh.clear();
  PMP::polygon_soup_to_polygon_mesh(soup_points, soup_faces, out_mesh);

  return !out_mesh.is_empty();
}

template <class GeomTraits>
bool Delaunay_surface_reconstruction<GeomTraits>::reconstruct(Surface_mesh& out_mesh, std::optional<double> alpha)
{
  error_.clear();

  if (!build_alpha_shape(alpha)) return false;
  if (!enumerate_facets_and_edges_from_alpha_boundary()) return false;

#if defined(USE_SCIP)
  std::vector<int> x_sol;
  if (!solve_binary_ilp_scip(x_sol)) {
    error_ = "SCIP failed to solve the binary ILP (no feasible/optimal solution returned).";
    return false;
  }
  if (!build_mesh_from_solution(x_sol, out_mesh)) {
    error_ = "ILP solved, but selected facet set did not yield a valid polygon mesh.";
    return false;
  }
  return true;
#else
  error_ = "This build does not enable SCIP (compile with -DUSE_SCIP and link SCIP).";
  return false;
#endif
}

} // namespace DSR

#endif // DELAUNAY_SURFACE_RECONSTRUCTION_H
