#include "marlin_w4a8.h"
#include <pybind11/pybind11.h>
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("moe_w4a8_marlin_gemm", &moe_w4a8_marlin_gemm,
        py::arg("a"),
        py::arg("b_q_weight"),
        py::arg("d_or_none"),
        py::arg("s1"),
        py::arg("s2"),
        py::arg("s3"),
        py::arg("sorted_token_ids"),
        py::arg("expert_ids"),
        py::arg("num_tokens_past_padded"),
        py::arg("topk_weights"),
        py::arg("moe_block_size"),
        py::arg("top_k"),
        py::arg("mul_topk_weights"),
        py::arg("is_ep"),
        py::arg("workspace"),
        py::arg("prob_m"),
        py::arg("prob_n"),
        py::arg("prob_k")
  );
}