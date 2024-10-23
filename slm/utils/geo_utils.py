"""
Utility functions for geometric operations (torch only).
"""
import torch

def rots_mul_vecs(m, v):
  """(Batch) Apply rotations 'm' to vectors 'v'."""
  return torch.stack([
        m[..., 0, 0] * v[..., 0] + m[..., 0, 1] * v[..., 1] + m[..., 0, 2] * v[..., 2],
        m[..., 1, 0] * v[..., 0] + m[..., 1, 1] * v[..., 1] + m[..., 1, 2] * v[..., 2],
        m[..., 2, 0] * v[..., 0] + m[..., 2, 1] * v[..., 1] + m[..., 2, 2] * v[..., 2],
  ], dim=-1)
  
def distance(p, eps=1e-10):
    """Calculate distance between a pair of points (dim=-2)."""
    # [*, 2, 3]
    return (eps + torch.sum((p[..., 0, :] - p[..., 1, :]) ** 2, dim=-1)) ** 0.5

def dihedral(p, eps=1e-10):
    """Calculate dihedral angle between a quadruple of points (dim=-2)."""
    # p: [*, 4, 3]

    # [*, 3]
    u1 = p[..., 1, :] - p[..., 0, :]
    u2 = p[..., 2, :] - p[..., 1, :]
    u3 = p[..., 3, :] - p[..., 2, :]

    # [*, 3]
    u1xu2 = torch.cross(u1, u2, dim=-1)
    u2xu3 = torch.cross(u2, u3, dim=-1)

    # [*]
    u2_norm = (eps + torch.sum(u2 ** 2, dim=-1)) ** 0.5
    u1xu2_norm = (eps + torch.sum(u1xu2 ** 2, dim=-1)) ** 0.5
    u2xu3_norm = (eps + torch.sum(u2xu3 ** 2, dim=-1)) ** 0.5

    # [*]
    cos_enc = torch.einsum('...d,...d->...', u1xu2, u2xu3)/ (u1xu2_norm * u2xu3_norm)
    sin_enc = torch.einsum('...d,...d->...', u2, torch.cross(u1xu2, u2xu3, dim=-1)) /  (u2_norm * u1xu2_norm * u2xu3_norm)

    return torch.stack([cos_enc, sin_enc], dim=-1)

def calc_distogram(pos: torch.Tensor, min_bin: float, max_bin: float, num_bins: int):
    # pos: [*, L, 3]
    dists_2d = torch.linalg.norm(
        pos[..., :, None, :] - pos[..., None, :, :], axis=-1
    )[..., None]
    lower = torch.linspace(
        min_bin,
        max_bin,
        num_bins,
        device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    distogram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return distogram


def squared_deviation(xyz1, xyz2, reduction='none'):
    """Squared point-wise deviation between two point clouds after alignment.
    
    Args:
        xyz1: (*, L, 3), to be transformed
        xyz2: (*, L, 3), the reference 
    
    Returns:
        rmsd: (*, ) or none: (*, L)
    """
    map_to_np = False
    if not torch.is_tensor(xyz1):
        map_to_np = True
        xyz1 = torch.as_tensor(xyz1)
        xyz2 = torch.as_tensor(xyz2)
    
    R, t = _find_rigid_alignment(xyz1, xyz2)

    # print(R.shape, t.shape) # B, 3, 3 & B, 3
    xyz1_aligned = (R.bmm(xyz1.transpose(-2,-1))).transpose(-2,-1) + t.unsqueeze(1)
    sd = ((xyz1_aligned - xyz2)**2).sum(dim=-1)    # (*, L)
    
    assert sd.shape == xyz1.shape[:-1]  
    if reduction == 'none':
        pass
    elif reduction == 'rmsd':
        sd = torch.sqrt(sd.mean(dim=-1))
    else:
        raise NotImplementedError()
    
    sd = sd.numpy() if map_to_np else sd
    return sd

def _find_rigid_alignment(src, tgt):
    """Inspired by https://research.pasteur.fr/en/member/guillaume-bouvier/;
        https://gist.github.com/bougui505/e392a371f5bab095a3673ea6f4976cc8
    
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
    
    Args:
        src: Torch tensor of shape (*, L, 3) -- Point Cloud to Align (source)
        tgt: Torch tensor of shape (*, L, 3) -- Reference Point Cloud (target)
    Returns:
        R: optimal rotation (*, 3, 3)
        t: optimal translation (*, 3)
    """
    assert src.shape[-2] > 1
    src_com = src.mean(dim=-2, keepdim=True)
    tgt_com = tgt.mean(dim=-2, keepdim=True)
    src_centered = src - src_com
    tgt_centered = tgt - tgt_com

    # Covariance matrix
    H = src_centered.transpose(-2,-1).bmm(tgt_centered)    # *, 3, 3
    U, _, V = torch.svd(H)
    
    # Rotation matrix
    R = V.bmm(U.transpose(-2,-1))
    # Translation vector
    t = tgt_com - R.bmm(src_com.transpose(-2,-1)).transpose(-2,-1)
    return R, t.squeeze(-2) # (B, 3, 3), (B, 3)


def get_center_by_batch_idx(xyz: torch.Tensor, batch: torch.LongTensor) -> torch.Tensor:
    """
    Move the molecule center to zero for sparse position tensors.

    Args:
        xyz: [N, 3] batch positions of atoms in the molecule in sparse batch format.
        batch: [N, ] batch index for each atom in sparse batch format.

    Returns:
        xyz: [N, 3] zero-centered batch positions of atoms in the molecule in sparse batch format.
    """
    last_dim = 3
    assert len(xyz.shape) == 2 and xyz.shape[-1] == last_dim, f"xyz must have shape [N, {last_dim}] but got {xyz.shape}"
    n_samples = torch.unique(batch, return_counts=True)[1]
    means = torch.full((n_samples.shape[0], last_dim), fill_value=0.0, 
        dtype=xyz.dtype, device=xyz.device).scatter_add_(0, batch[:, None].expand_as(xyz), xyz)
    means = means / n_samples[:, None]
    # return xyz - means[batch]
    return means


@torch.no_grad()
def align_structures(
    src_xyz: torch.Tensor,  # (N, 3)
    tgt_xyz: torch.Tensor,  # (N, 3)
    batch: torch.Tensor,    # (N, )
):
    # Minimize || Q @ R.T - P ||, which is the same as || Q - P @ R ||
    assert src_xyz.shape == tgt_xyz.shape, \
        f"src_xyz and tgt_xyz must have the same shape but got {src_xyz.shape} and {tgt_xyz.shape}"
    last_dim = 3
    
    # (B, 3)
    src_center = get_center_by_batch_idx(src_xyz, batch)
    tgt_center = get_center_by_batch_idx(tgt_xyz, batch)
    src_xyz = src_xyz - src_center[batch]
    tgt_xyz = tgt_xyz - tgt_center[batch]
    B = torch.unique(batch, return_counts=True)[1].shape[0]
    
    # Compute covariance matrix for optimal rotation (Q.T @ P) -> [B, 3, 3].
    _cov = src_xyz[:, None, :] * tgt_xyz[:, :, None]    # [N, 3, 3]
    cov = torch.full((B, last_dim, last_dim), fill_value=0.0, dtype=src_xyz.dtype, device=src_xyz.device)
    # equiv to bmm 
    cov.scatter_add_(0, batch[:, None, None].expand_as(_cov), _cov)  # [B, 3, 3]

    # Perform singular value decomposition. (all [B x 3 x 3])
    u, _, v_t = torch.linalg.svd(cov)
    
    # Convenience transposes.
    u_t = u.transpose(1, 2)
    v = v_t.transpose(1, 2)

    # Compute rotation matrix correction for ensuring right-handed coordinate system.
    # det(AB) = det(A)*det(B) and det(A) = det(A.T)
    sign_correction = torch.sign(torch.linalg.det(torch.bmm(v, u_t)))
    # Correct transpose of U: diag(1, 1, sign_correction) @ U.T (row-wise)
    u_t[:, 2, :] = u_t[:, 2, :] * sign_correction[:, None]

    # Compute optimal rotation matrix: (R = V @ diag(1, 1, sign_correction) @ U.T).
    rot_src_to_tgt = torch.bmm(v, u_t) # [B, 3, 3]
    rot_src_to_tgt = rot_src_to_tgt.transpose(-2, -1)   # ready to left apply on pts
    
    # Rotate batch positions P to optimal alignment with Q: (P @ R)
    # src_rotated = torch.bmm(
    #     src_xyz[:, None, :],
    #     rot_src_to_tgt[batch],
    # ).squeeze(1)
    src_rotated = rots_mul_vecs(
        rot_src_to_tgt[batch], src_xyz
    )
    
    # print("debug:", torch.abs(src_rotated-tgt_xyz))
    
    # return:
    # src is aligned to tgt
    # tgt is zero-centered
    return src_rotated + tgt_center[batch], tgt_xyz, rot_src_to_tgt   # (N, 3), (N, 3), (B, 3, 3)


def align_batched_structures(
    batch_src_xyz,  # (B, N, 3)
    batch_tgt_xyz,  # (B, N, 3)
    mask: torch.Tensor = None,
):
    mask = mask if mask is not None \
       else torch.ones_like(batch_src_xyz[..., 0])
    assert batch_src_xyz.shape == batch_tgt_xyz.shape, \
        f"batch_src_xyz and batch_tgt_xyz must have the same shape but got {batch_src_xyz.shape} and {batch_tgt_xyz.shape}"
    B, N, D = batch_src_xyz.shape
    
    device = batch_src_xyz.device
    batch = torch.arange(B, device=device).repeat_interleave(N)
    mask = mask.bool()
    flat_src = batch_src_xyz.masked_select(mask[..., None]).view(-1, 3)
    flat_tgt = batch_tgt_xyz.masked_select(mask[..., None]).view(-1, 3)
    batch = batch.masked_select(mask.view(-1))

    aligned_src, centered_tgt, rot_src_to_tgt = align_structures(
        flat_src, flat_tgt, batch
    )   # (N, 3), (N, 3), (B, 3, 3)
    
    # scatter back to batched format
    src, tgt = torch.zeros_like(batch_src_xyz).view(-1, D), torch.zeros_like(batch_tgt_xyz).view(-1, D)
    src.masked_scatter_(mask.view(-1, 1), aligned_src)
    tgt.masked_scatter_(mask.view(-1, 1), centered_tgt)
    
    return src.view(batch_src_xyz.shape), tgt.view(batch_tgt_xyz.shape), rot_src_to_tgt




if __name__ == '__main__':
    import math
    from rotation3d import random_rotations

    torch.set_default_dtype(torch.double)
    B, L = 4, 5

    tgt = torch.randn(B, L, 3)
    rand_rotmats = random_rotations(B).view(B, 1, 3, 3)
    src = rots_mul_vecs(rand_rotmats, tgt).view(B, L, 3) + torch.randn(B, 1, 3)
    
    # mask = torch.randint(0, 2, (B, L)).bool()
    mask = torch.ones_like(src[..., 0]).bool()
    
    batch = torch.arange(B).repeat_interleave(L)
    batch = batch.masked_select(mask.view(-1))
    print(batch)
    
    src_aligned, tgt_aligned, _ = align_batched_structures(src, tgt, mask)
    src_aligned_list, tgt_aligned_list = [], []
    for i in range(B):
        tmp_src, tmp_tgt, _ = align_structures(src[i], tgt[i], torch.zeros(L, dtype=torch.long))
        src_aligned_list.append(tmp_src)
        tgt_aligned_list.append(tmp_tgt)
    src_aligned_list = torch.stack(src_aligned_list)
    tgt_aligned_list = torch.stack(tgt_aligned_list)

    rot, trans  = _find_rigid_alignment(src, tgt)
    src_aln = (rot.bmm(src.transpose(-2,-1))).transpose(-2,-1) + trans.unsqueeze(1)
    
    # flat_src = src_aligned.masked_select(mask[..., None]).view(-1, 3)
    # flat_tgt = tgt_aligned.masked_select(mask[..., None]).view(-1, 3)
    # print("The mean of src_aligned and tgt_aligned should be zero:")
    # print(
    #     get_center_by_batch_idx(flat_src, batch), '\n',
    #     get_center_by_batch_idx(flat_tgt, batch)
    # )
    print("align_structures():", torch.abs(tgt-src_aligned).sum())
    print("find_rigid_alignment():", torch.abs(tgt-src_aln).sum())
    print("individual align_structures()", torch.abs(src_aligned_list-tgt).sum())