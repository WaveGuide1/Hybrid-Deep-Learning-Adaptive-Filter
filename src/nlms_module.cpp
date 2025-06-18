#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>

namespace py = pybind11;

py::array_t<double> nlms_filter(py::array_t<double> input_array, 
                                 py::array_t<double> desired_array, 
                                 double mu=0.01, 
                                 double epsilon=1e-8, 
                                 int filter_order=16) 
{
    // Extract buffers
    auto buf_in = input_array.request();
    auto buf_desired = desired_array.request();

    size_t N = buf_in.shape[0];
    std::vector<double> input(buf_in.ptr, buf_in.ptr + N);
    std::vector<double> desired(buf_desired.ptr, buf_desired.ptr + N);

    std::vector<double> weights(filter_order, 0.0);
    std::vector<double> output(N, 0.0);

    for (size_t n = filter_order; n < N; n++) {
        // Build input vector
        std::vector<double> x(filter_order);
        for (int k = 0; k < filter_order; k++) {
            x[k] = input[n - k - 1];
        }

        // Calculate filter output
        double y = 0.0;
        for (int k = 0; k < filter_order; k++) {
            y += weights[k] * x[k];
        }
        output[n] = y;

        // Compute error
        double e = desired[n] - y;

        // Norm of input vector
        double norm_x = epsilon;
        for (int k = 0; k < filter_order; k++) {
            norm_x += x[k] * x[k];
        }

        // Update weight
        for (int k = 0; k < filter_order; k++) {
            weights[k] += (mu * e * x[k]) / norm_x;
        }
    }

    // Return output as numpy array
    py::array_t<double> result(N);
    auto buf_out = result.request();
    double* ptr_out = static_cast<double*>(buf_out.ptr);
    for (size_t i = 0; i < N; i++) {
        ptr_out[i] = output[i];
    }

    return result;
}

PYBIND11_MODULE(nlms_cpp, m) {
    m.def("nlms_filter", &nlms_filter, 
          "Apply NLMS Adaptive Filter",
          py::arg("input_array"), 
          py::arg("desired_array"), 
          py::arg("mu") = 0.01, 
          py::arg("epsilon") = 1e-8, 
          py::arg("filter_order") = 16);
}
