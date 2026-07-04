import torch

def rot_matmul(a, b):
    row_1 = torch.stack([a[..., 0, 0]*b[..., 0, 0] + a[..., 0, 1]*b[..., 1, 0] + a[..., 0, 2]*b[..., 2, 0],
                         a[..., 0, 0]*b[..., 0, 1] + a[..., 0, 1]*b[..., 1, 1] + a[..., 0, 2]*b[..., 2, 1],
                         a[..., 0, 0]*b[..., 0, 2] + a[..., 0, 1]*b[..., 1, 2] + a[..., 0, 2]*b[..., 2, 2],], dim=-1)
    
    row_2 = torch.stack([a[..., 1, 0]*b[..., 0, 0] + a[..., 1, 1]*b[..., 1, 0] + a[..., 1, 2]*b[..., 2, 0],
                         a[..., 1, 0]*b[..., 0, 1] + a[..., 1, 1]*b[..., 1, 1] + a[..., 1, 2]*b[..., 2, 1],
                         a[..., 1, 0]*b[..., 0, 2] + a[..., 1, 1]*b[..., 1, 2] + a[..., 1, 2]*b[..., 2, 2],], dim=-1)

    row_3 = torch.stack([a[..., 2, 0]*b[..., 0, 0] + a[..., 2, 1]*b[..., 1, 0] + a[..., 2, 2]*b[..., 2, 0],
                         a[..., 2, 0]*b[..., 0, 1] + a[..., 2, 1]*b[..., 1, 1] + a[..., 2, 2]*b[..., 2, 1],
                         a[..., 2, 0]*b[..., 0, 2] + a[..., 2, 1]*b[..., 1, 2] + a[..., 2, 2]*b[..., 2, 2],], dim=-1)

    return torch.stack([row_1, row_2, row_3], dim=-2)

def rot_vec_mul(r, t):
    return torch.stack([r[..., 0, 0]*t[..., 0] + r[..., 0, 1]*t[..., 1] + r[..., 0, 2]*t[..., 2],
                        r[..., 1, 0]*t[..., 0] + r[..., 1, 1]*t[..., 1] + r[..., 1, 2]*t[..., 2],
                        r[..., 2, 0]*t[..., 0] + r[..., 2, 1]*t[..., 1] + r[..., 2, 2]*t[..., 2],], dim=-1)

class Frames:
    def __init__(self, R, t):
        # [N_Res, (rot) 3, (xyz) 3]
        self.R = R
        # [N_Res, (xyz) 3]
        self.t = t

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        return Frames(self.R[index + (slice(None), slice(None))], self.t[index + (slice(None),)])

    def __mul__(self, right):
        # Defensive shape check
        if self.t.shape[:-1] != right.shape:
            raise ValueError(f"Shape mismatch! Cannot multiply Frames with shape {self.t.shape[:-1]} "
                             f"using a tensor of shape {right.shape}. Leading dimensions must match exactly.")

        # Broadcast the mask over the 3x3 rotation matrix
        R = self.R * right[..., None, None]
        # Broadcast the mask over the 3D translation vector
        t = self.t * right[..., None]
        return Frames(R, t)

    def __rmul__(self, left):
        return self.__mul__(left)
    
    @property
    def shape(self):
        return self.t.shape[:-1]

    @staticmethod
    def cat(frames_list, dim):
        if not frames_list:
            raise ValueError("Cannot concatenate an empty list of Frames.")

        expected_shape = list(frames_list[0].shape)
        expected_shape.pop(dim)

        for f in frames_list[1:]:
            curr_shape = list(f.shape)
            curr_shape.pop(dim)

            if curr_shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch.\n"
                    f"Expected matching base shape (ignoring concat dim) to be {expected_shape}\n"
                    f"Got mismatched base shape -> {curr_shape}"
                )

        # Skips the final dimensions for rotations and translations, when using reverse
        dim_R = dim if dim >= 0 else dim - 2
        dim_t = dim if dim >= 0 else dim - 1

        R = torch.cat([f.R for f in frames_list], dim=dim_R)
        t = torch.cat([f.t for f in frames_list], dim=dim_t)

        return Frames(R, t)

    @staticmethod
    def from_gram_schmidt(p_neg_x_axis, origin, p_xy_plane, eps=1e-8):
        v1 = origin - p_neg_x_axis
        v2 = p_xy_plane - origin

        norm_v1 = torch.sqrt(torch.sum(v1 ** 2, dim=-1, keepdim=True) + eps)
        e1 = v1 / norm_v1
        
        e2 = v2 - e1 * torch.linalg.vecdot(e1, v2, dim=-1)[..., None]
        norm_e2 = torch.sqrt(torch.sum(e2 ** 2, dim=-1, keepdim=True) + eps)
        e2 = e2 / norm_e2

        e3 = torch.cross(e1, e2, dim=-1)

        rot = torch.stack([e1, e2, e3], dim=-1)
        return Frames(rot, origin)

    @staticmethod
    def from_frenet_serret(x, eps=1e-8):
        S = x.shape[-2]
        if S < 3:
            raise ValueError("Sequence length must be at least 3.")
            
        rots = torch.zeros((*x.shape[:-1], 3, 3), device=x.device, dtype=x.dtype)
        
        # Calculate backward and forward tangent vectors
        t_backward = x[..., 1:-1, :] - x[..., :-2, :]
        t_forward = x[..., 2:, :] - x[..., 1:-1, :]
        
        # Tangent (T)
        t_norm = torch.sqrt(torch.sum(t_forward ** 2, dim=-1, keepdim=True) + eps)
        t = t_forward / t_norm
        
        # Binormal (B) = t_backward x t_forward
        B_unnorm = torch.cross(t_backward, t_forward, dim=-1)
        b_norm = torch.sqrt(torch.sum(B_unnorm ** 2, dim=-1, keepdim=True) + eps)
        b = B_unnorm / b_norm
        
        # Normal (N) = B x T
        n = torch.cross(b, t, dim=-1)
        
        # Stack as [T, N, B] for a valid +1 determinant rotation matrix
        rots[..., 1:-1, :, :] = torch.stack([t, n, b], dim=-1)
        
        # Pad ends (simplistic padding, matching your original logic)
        rots[..., 0, :, :] = rots[..., 1, :, :]
        rots[..., -1, :, :] = rots[..., -2, :, :]
        
        return Frames(rots, x)

    def to_4x4(self):
        rotx = torch.zeros((*self.t.shape[:-1], 4, 4), dtype=self.t.dtype, device=self.t.device)
        rotx[..., :3, :3] = self.R
        rotx[..., :3, 3] = self.t
        rotx[..., 3, 3] = 1
        return rotx

    @staticmethod
    def from_4x4(tensor):
        R = tensor[..., :3, :3]
        t = tensor[..., :3, 3]
        return Frames(R, t)

    @staticmethod
    def identity(shape, dtype, device, requires_grad=False):
        R = torch.eye(3, dtype=dtype, device=device).expand(*shape, 3, 3)
        t = torch.zeros((*shape, 3), dtype=dtype, device=device)

        if requires_grad:
            R = R.clone()
            R.requires_grad_(True)
            t.requires_grad_(True)

        return Frames(R, t)

    def apply(self, pts):
        """ 
        Projects points from the local frame into the global frame.
        """
        return rot_vec_mul(self.R, pts) + self.t

    def invert(self):
        R_inv = self.R.transpose(-1, -2)
        t_inv = rot_vec_mul(R_inv, self.t)
        return Frames(R_inv, -t_inv)
    
    def invert_apply(self, pts):
        """ 
        Projects points from the global frame into the local frame.
        """
        shifted_pts = pts - self.t
        R_inv = self.R.transpose(-1, -2)
        return rot_vec_mul(R_inv, shifted_pts)
    
    def compose(self, update_frame):
        """ 
        Updates the current frame with another frame
        """
        return Frames(R = rot_matmul(self.R, update_frame.R),
                      t = rot_vec_mul(self.R, update_frame.t) + self.t)
    
