# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in inference acceleration for Alpamayo 2 Edge.

``use_linear_patch_embedding`` computes the vision patch embedding as a GEMM.
Nothing here changes the default inference path.
"""

from alpamayo2_edge.accel.patch_embed import use_linear_patch_embedding

__all__ = ["use_linear_patch_embedding"]
