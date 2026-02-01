# Polygonal Surface Reconstruction from Sparse Point Clouds

This project implements a **surface reconstruction algorithm based on alpha complex** for sparse 3D point clouds.  
The goal is to reconstruct a surface by selecting appropriate simplices from the Delaunay complex using geometric and optimization-based criteria.

The implementation is written in **C++**, relies on **CGAL** for geometric computations, and uses open-source solver **SCIP** as an optimization backend.

---

## 0. Prerequisites

Required software and packages: [CGAL](https://www.cgal.org/) and [SCIP](https://www.scipopt.org/).

## 1. Installing

Download the [source code](https://github.com/sh-zhao/Polygonal-Surface-Reconstruction-from-Sparse-Point-Clouds) or clone the repository:

    git clone https://github.com/sh-zhao/Polygonal-Surface-Reconstruction-from-Sparse-Point-Clouds.git
    cd Polygonal-Surface-Reconstruction-from-Sparse-Point-Clouds

Create a build directory and compile:

    mkdir build
    cd build
    cmake -DCMAKE_BUILD_TYPE=Release ..
    cmake --build .

The executable will be generated in:

    ./bin/delaunay_surface_reconstruction


## 2. Run the project

You can directly start reconstructing by running the following command for the PLY file rabbit.ply in the example folder.

Go to the root folder.

    cd ..

... and reconstruct.

    ./bin/delaunay_surface_reconstruction ./example/rabbit.ply

You can modify the source code to load point clouds from the example directory and experiment with different reconstruction parameters.
For example, if you want to run the code with a customized alpha value 1000, then

    // alpha=1000
    ./bin/delaunay_surface_reconstruction ./example/rabbit.ply 1000

## Notes
The current solver SCIP can handle only sparse point clouds (~1000 points), a better choice is the Gurobi solver, which more efficient and reliable.
To use Gurobi, you must install the software and [obtain a license](https://www.gurobi.com/downloads/end-user-license-agreement-academic/), which is available at no cost for academic purposes. Additionally, you may need to update the directory paths in FindGUROBI.cmake so that CMake can correctly locate the installation.

Acknowledgments

We thank the data provider: Huang, Zhangjin, et al. "Surface reconstruction from point clouds: A survey and a benchmark."
