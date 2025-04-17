/* 
 * Adapted from https://github.com/IST-DASLab/marlin/blob/master/marlin/marlin_cuda.cpp
 * Modified by HandH1998
 * Copyright (C) 2024 HandH1998
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

 #include <torch/extension.h>

 torch::Tensor moe_w4a8_marlin_gemm(
   const torch::Tensor& a,
   const torch::Tensor& b,
   std::optional<torch::Tensor> const& d_or_none,
   const torch::Tensor& s1,
   const torch::Tensor& s2,
   const torch::Tensor& s3,
   torch::Tensor& sorted_token_ids,           //moe
   torch::Tensor& expert_ids,
   torch::Tensor& num_tokens_past_padded,
   torch::Tensor& topk_weights,
   int64_t moe_block_size, 
   int64_t top_k,
   bool mul_topk_weights,
   bool is_ep,                                //moe
   torch::Tensor& workspace,
   int64_t prob_m,
   int64_t prob_n,
   int64_t prob_k
 );
 