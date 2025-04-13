#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# Third Party
import torch

# CuRobo
from curobo.util.torch_utils import get_torch_jit_decorator

# Local Folder
from .zero_cost import ZeroCost

@get_torch_jit_decorator()
def quat2mat(quaternion):
    """
    Convert quaternions into rotation matrices.

    Args:
        quaternion (torch.Tensor): A tensor of shape (..., 4) representing batches of quaternions (w, x, y, z).

    Returns:
        torch.Tensor: A tensor of shape (..., 3, 3) representing batches of rotation matrices.
    """
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

    outer = quaternion.unsqueeze(-1) * quaternion.unsqueeze(-2)

    # Extract the necessary components
    xx = outer[..., 0, 0]
    yy = outer[..., 1, 1]
    zz = outer[..., 2, 2]
    xy = outer[..., 0, 1]
    xz = outer[..., 0, 2]
    yz = outer[..., 1, 2]
    xw = outer[..., 0, 3]
    yw = outer[..., 1, 3]
    zw = outer[..., 2, 3]

    rmat = torch.empty(quaternion.shape[:-1] + (3, 3), dtype=quaternion.dtype, device=quaternion.device)

    rmat[..., 0, 0] = 1 - 2 * (yy + zz)
    rmat[..., 0, 1] = 2 * (xy - zw)
    rmat[..., 0, 2] = 2 * (xz + yw)

    rmat[..., 1, 0] = 2 * (xy + zw)
    rmat[..., 1, 1] = 1 - 2 * (xx + zz)
    rmat[..., 1, 2] = 2 * (yz - xw)

    rmat[..., 2, 0] = 2 * (xz - yw)
    rmat[..., 2, 1] = 2 * (yz + xw)
    rmat[..., 2, 2] = 1 - 2 * (xx + yy)

    return rmat

class VisibilityCost(ZeroCost):
    """Visibility Cost"""

    def __init__(self, config):
        super().__init__(config)
        self._z_neg_vec = torch.tensor(
            [0.0, 0.0, -1.0], device=self.tensor_args.device, dtype=self.tensor_args.dtype
        )

    def forward(self, current_eye_pos, current_eye_quat, eyes_targets_pos, g_dist):
        desired_vector = eyes_targets_pos[0] - current_eye_pos
        desired_unit_vector = desired_vector / desired_vector.norm(dim=-1, keepdim=True)

        # wxyz -> xyzw
        current_eye_quat_xyzw = current_eye_quat.clone()
        current_eye_quat_xyzw[..., 0] = current_eye_quat[..., 1]
        current_eye_quat_xyzw[..., 1] = current_eye_quat[..., 2]
        current_eye_quat_xyzw[..., 2] = current_eye_quat[..., 3]
        current_eye_quat_xyzw[..., 3] = current_eye_quat[..., 0]
        current_eye_mat = quat2mat(current_eye_quat_xyzw)
        current_unit_vector = current_eye_mat @ self._z_neg_vec
        assert desired_unit_vector.shape == current_unit_vector.shape
        # cosine similarity is [-1, 1]
        cosine_similarity = torch.sum(desired_unit_vector * current_unit_vector, dim=-1)
        # cosine distance is [0, 2]
        cosine_distance = 1.0 - cosine_similarity
        sqrt_cosine_distance = torch.sqrt(cosine_distance).unsqueeze(-1)
        return super().forward(sqrt_cosine_distance, g_dist)