# SPDX-License-Identifier: Apache-2.0
"""Tests for the MOE layers.

Run `pytest tests/kernels/test_moe.py`.
"""
import pytest
import torch
from typing import Optional
import functools

from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

from vllm.model_executor.layers.fused_moe.fused_moe import (
    fused_topk, moe_align_block_size, try_get_optimal_moe_config)
from vllm.model_executor.layers.quantization.utils.marlin_utils_test_qqq import (  # noqa: E501
    marlin_qqq_quantize)

from QQQ._CUDA import moe_w4a8_marlin_gemm
# from sgl_kernel import moe_w4a8_marlin_gemm


NUM_EXPERTS = [8, 64]
EP_SIZE = [1, 4]
TOP_KS = [2, 6]

def stack_and_dev(tensors: list[torch.Tensor]):
    dev = tensors[0].device
    return torch.stack(tensors, dim=0).to(dev)


def torch_moe_single(a, w, score, topk):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    _, topk_ids = torch.topk(score, topk)
    topk_ids = topk_ids.view(-1)
    for i in range(w.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = a[mask] @ w[i].transpose(0, 1)
    return (out.view(B, -1, w.shape[1])).sum(dim=1)

# @pytest.mark.skip("This test is here for the sake of debugging, "
#                   "don't run it in automated tests.")
# @pytest.mark.parametrize("m", [64])
# @pytest.mark.parametrize("n", [128, 1024])
# @pytest.mark.parametrize("k", [256, 2048])
# @pytest.mark.parametrize("e", [4, 12])
# @pytest.mark.parametrize("topk", [2, 3])
# @pytest.mark.parametrize("dtype", [torch.float16])
# @pytest.mark.parametrize("group_size", [-1, 128])
# @pytest.mark.parametrize("num_bits", [4])
def test_single_marlin_moe_multiply(m: int, n: int, k: int, e: int, topk: int,
                                    dtype: torch.dtype, group_size: int,
                                    num_bits: int):

    int8_traits = torch.iinfo(torch.int8)

    a_input = torch.randn((m, k), device="cuda", dtype=dtype) / 10
    w = torch.randn((e, n, k), device="cuda", dtype=dtype) / 10

    # Quantize activations
    s_tok = a_input.abs().max(dim=-1, keepdim=True)[0].div(int8_traits.max).to(
        torch.float)
    q_a = (a_input / s_tok).round().clamp(int8_traits.min,
                                        int8_traits.max).to(torch.int8)

    # Quantize weights
    w_ref_l = []
    qweight_l = []
    s_group_l = []
    s_ch_l = []

    for i in range(w.shape[0]):
        w_ref, qweight, s_group, s_ch = marlin_qqq_quantize(
                  w[i].transpose(1, 0), num_bits, group_size)

        print("w_ref",w_ref.shape)
        print("qweight",qweight.shape)
        w_ref_l.append(w_ref.T)
        qweight_l.append(qweight)
        s_group_l.append(s_group)
        s_ch_l.append(s_ch)

    w_ref = stack_and_dev(w_ref_l)
    qweight = stack_and_dev(qweight_l).contiguous()
    s_ch = stack_and_dev(s_ch_l)
    s_group = stack_and_dev(s_group_l) if s_group_l else None

    score = torch.randn((m, e), device="cuda", dtype=dtype)
    marlin_output = single_marlin_moe(
        q_a,
        qweight,
        s_tok,
        s_ch,
        s_group,
        score,
        topk,
        renormalize=False,
        num_bits=num_bits
    )

    torch_output = torch_moe_single(a_input, w_ref, score, topk)

    torch.testing.assert_close(marlin_output, torch_output, atol=2e-2, rtol=0)


def single_marlin_moe(
    hidden_states: torch.Tensor,
    w: torch.Tensor,
    s_tok: torch.Tensor,
    s_ch: torch.Tensor,
    s_group: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool,
    global_num_experts: int = -1,
    expert_map: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    num_bits: int = 4
) -> torch.Tensor:
    """
    This function computes the multiplication of hidden_states with expert
    weights used in Marlin MoE, using weights w and top-k gating mechanism.
    Its purpose is testing and debugging the fused MoE kernel.

    Parameters:
    - hidden_states (torch.Tensor): The input tensor to the Marlin Mul.
    - w (torch.Tensor): The set of expert weights.
    - scales (torch.Tensor): The quantization scales.
    - gating_output (torch.Tensor): The output of the gating operation
        (before softmax).
    - topk (int): The number of top-k experts to select.
    - renormalize (bool): If True, renormalize the top-k weights to sum to 1.
    - num_bits (bool): The number of bits in expert weights quantization.

    Returns:
    - torch.Tensor: The output tensor after applying the MoE layer.
    """
    # Check constraints.
    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")
    assert hidden_states.shape[1] == w.shape[1] * 16, "Hidden size mismatch"
    assert gating_output.shape[1] == w.shape[0], "Number of experts mismatch"
    assert hidden_states.is_contiguous(), "Hidden_states must be contiguous"
    assert w.is_contiguous(), "Expert weights must be contiguous"
    # assert hidden_states.dtype in [torch.float16, torch.bfloat16]
    assert num_bits in [4, 8]

    M, K = hidden_states.shape
    E = w.shape[0]
    N = w.shape[2] // (num_bits // 2)

    topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk,
                                        renormalize)

    # This might not be an optimal config for a single MMM
    get_config_func = functools.partial(try_get_optimal_moe_config,
                                        w.shape,
                                        w.shape,
                                        topk_ids.shape[1],
                                        None,
                                        is_marlin=True)
    config = get_config_func(M)

    block_size_m = config['BLOCK_SIZE_M']

    if global_num_experts == -1:
        global_num_experts = E
    sorted_token_ids, expert_ids, num_tokens_post_padded = \
        moe_align_block_size(topk_ids, block_size_m, E)

    if workspace is None:
        max_workspace_size = (max(2 * N, K) // 64) * \
            (sorted_token_ids.size(0) // block_size_m)
        device = hidden_states.device
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        max_workspace_size = min(max_workspace_size, sms)
        workspace = torch.zeros(max_workspace_size,
                                dtype=torch.int,
                                device=device,
                                requires_grad=False)


    intermediate_cache = torch.empty(
        (M * topk_ids.shape[1], N),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )

    moe_w4a8_marlin_gemm(hidden_states,
                        w,
                        intermediate_cache,
                        s_tok,
                        s_ch,
                        s_group,
                        sorted_token_ids,
                        expert_ids,
                        num_tokens_post_padded,
                        topk_weights,
                        moe_block_size=block_size_m,
                        top_k=topk,
                        mul_topk_weights=False,
                        is_ep=expert_map is not None,
                        workspace=workspace,
                        prob_m=M,
                        prob_n=N,
                        prob_k=K)
    intermediate_cache = intermediate_cache.view(-1, topk, N)

    return torch.sum(intermediate_cache.view(*intermediate_cache.shape), dim=1)


if __name__ == '__main__':
    m,n,k = 64,1024,2048
    e = 8
    topk = 2
    dtype = torch.float16
    group_size = 128
    num_bits = 4
    test_single_marlin_moe_multiply(m,n,k,e,topk,dtype,group_size,num_bits)