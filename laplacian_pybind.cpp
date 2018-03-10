#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/iostream.h>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <Eigen/SparseLU>
#include <Eigen/SparseQR>
#include <Eigen/OrderingMethods>
#include <algorithm>
#include <ctime>

namespace py = pybind11;

using spMatrix = Eigen::SparseMatrix<float>; //default in Eigen
using Vector = Eigen::VectorXf;
using Vector3i = Eigen::Vector3i;
using iterator_vec = Eigen::VectorXf::InnerIterator;
using Matrix = Eigen::MatrixXf;
using Matrixi = Eigen::MatrixXi;
        
Matrix Laplacian(Eigen::Ref<Matrixi> &triangles_, 
        Eigen::Ref<Matrix> &vertices_, 
        int ntri, int nvert, float step = 0.5, int iter = 10, 
        bool euler_step_backward = false) {

    static spMatrix adjacency(nvert, nvert);
    spMatrix laplacian(nvert, nvert);
    spMatrix B(nvert, nvert);
    adjacency.reserve(6 * nvert);
    laplacian.reserve(6 * nvert);
    B.reserve(6 * nvert);
    
    clock_t t1 = clock();
    Vector3i vec3 = Vector3i(3);
    for(int row = 0; row < ntri; row ++) {
        vec3 = triangles_.row(row);
        adjacency.insert(vec3[0], vec3[1]) = 1.0;
        adjacency.insert(vec3[1], vec3[2]) = 1.0;
        adjacency.insert(vec3[2], vec3[0]) = 1.0;
    }

    clock_t t2 = clock();
    double elapsed = double(t2 - t1) / CLOCKS_PER_SEC;
    std::cout << "time elapsed construction adjacency matrix: " << elapsed << std::endl;

    std::vector<float> weights(nvert);
    const auto& ref_adjacency = adjacency;
    clock_t t = clock(); 
    Vector vec = Vector(nvert);
    std::generate(
            weights.begin(), 
            weights.end(), 
            [col = 0, ref_adjacency, &weights, &vec, nvert] () mutable {
                int sum = 0;
                vec = ref_adjacency.innerVector(col); 
                for(int i = 0; i < nvert; i++) {
                    sum += vec[i];
                }
                col++;
                return(-1. * sum);
            }
    );


    clock_t end = clock();
    double time_ = double(end - t) / CLOCKS_PER_SEC;
    std::cout << "time elapsed construction laplacian Matrix: " << time_ << std::endl;
    float *pointer = &weights[0];

    
    Eigen::Map<Vector> diag(pointer, nvert); 
    laplacian = diag.asDiagonal();
    laplacian += adjacency;    
    //Normalizing along rows
    laplacian = -1. * laplacian.diagonal().asDiagonal().inverse() * laplacian;
    laplacian *= step;

    spMatrix vertices = vertices_.sparseView();

    B = Matrix::Identity(nvert, nvert).sparseView() + laplacian;

    Eigen::SparseLU<spMatrix, Eigen::COLAMDOrdering<int> > solver;
    //solver.compute(B);
    solver.analyzePattern(B);
    solver.factorize(B);

    clock_t time_step_start = clock();
    if(euler_step_backward) {
        for(int i = 0; i < iter; i++)
        {
            vertices = solver.solve(vertices);
        }

    }
    else {
        for(int i = 0; i < iter; i++)
        {
            vertices = B * vertices;
        }
    }
    clock_t time_step_stop = clock();
    double step_time = double(time_step_stop - time_step_start) / CLOCKS_PER_SEC;
    std::cout << "time elapsed for steps: " << step_time << std::endl;

    double step_time_per = (double(time_step_stop - time_step_start) / CLOCKS_PER_SEC)/double(iter);
    std::cout << "step_time/iter pybind11 : \n" << step_time_per << std::endl;
    std::cout << "error + iterations" << std::endl;
    return Matrix(vertices);
};


PYBIND11_MODULE(laplacian_pybind, m) {
    m.def("euler_step_backward", [] (Eigen::Ref<Matrix> &laplacian, Eigen::Ref<Matrix> &vertices, int nvert, double step)
        -> Matrix {
            laplacian *= step;
            Matrix I = Matrix::Identity(nvert, nvert);
            laplacian += I;
            return laplacian * vertices;
    }, "backward euler step", py::return_value_policy::reference, 
        py::arg("laplacian").noconvert(), 
        py::arg("vertices").noconvert(),
        py::arg("nvert").noconvert(),
        py::arg("step").noconvert()
        ); 
    
    //Backward les resultat sont weird
    m.def("euler_step_forward", [] (Eigen::Ref<Matrix> laplacian, Eigen::Ref<Matrix> vertices, int nvert, double step)
        -> Matrix {
            Eigen::ConjugateGradient<spMatrix> solver;
            laplacian *= step;
            spMatrix I = Matrix::Identity(nvert, nvert).sparseView();
            I.reserve(nvert);
            laplacian += I;
            return (solver.compute(laplacian.sparseView()).solve(vertices.sparseView())); 
    }, "forward euler step", py::return_value_policy::reference,
        py::arg("laplacian").noconvert(), 
        py::arg("vertices").noconvert(),
        py::arg("nvert").noconvert(),
        py::arg("step").noconvert()
        );
    
    //take::ownership:
    //Reference an existing object (i.e. do not create a new copy) and take ownership. 
    //Python will call the destructor and delete operator when the object’s reference count reaches zero. 
    //Undefined behavior ensues when the C++ side does the same, or when the data was not dynamically allocated.
    m.def("Laplacian", &Laplacian, "perform laplacian smooth", py::return_value_policy::take_ownership,
        py::arg("triangles_").noconvert(), 
        py::arg("vertices_").noconvert(),
        py::arg("nvert"),
        py::arg("ntri"),
        py::arg("step") = 0.5,
        py::arg("iter") = 10,
        py::arg("euler_step_forward") = true
    );
}

