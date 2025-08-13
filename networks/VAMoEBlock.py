import math
import numpy as np
import torch
import torch.nn as nn
import torch.fft
import torch.nn.functional as F

from .moe_layer import Mlp, MoE
# from moe_layer import Mlp, MoE


def legendre_gauss_weights(n, a=-1.0, b=1.0):
    r"""
    Helper routine which returns the Legendre-Gauss nodes and weights
    on the interval [a, b]
    """

    xlg, wlg = np.polynomial.legendre.leggauss(n)
    xlg = (b - a) * 0.5 * xlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return xlg, wlg

def lobatto_weights(n, a=-1.0, b=1.0, tol=1e-16, maxiter=100):
    r"""
    Helper routine which returns the Legendre-Gauss-Lobatto nodes and weights
    on the interval [a, b]
    """

    wlg = np.zeros((n,))
    tlg = np.zeros((n,))
    tmp = np.zeros((n,))

    # Vandermonde Matrix
    vdm = np.zeros((n, n))
  
    # initialize Chebyshev nodes as first guess
    for i in range(n): 
        tlg[i] = -np.cos(np.pi*i / (n-1))
    
    tmp = 2.0
    
    for i in range(maxiter):
        tmp = tlg
       
        vdm[:,0] = 1.0 
        vdm[:,1] = tlg
       
        for k in range(2, n):
            vdm[:, k] = ( (2*k-1) * tlg * vdm[:, k-1] - (k-1) * vdm[:, k-2] ) / k
       
        tlg = tmp - ( tlg*vdm[:, n-1] - vdm[:, n-2] ) / ( n * vdm[:, n-1]) 
        
        if (max(abs(tlg - tmp).flatten()) < tol ):
            break 
    
    wlg = 2.0 / ( (n*(n-1))*(vdm[:, n-1]**2))

    # rescale
    tlg = (b - a) * 0.5 * tlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5
    
    return tlg, wlg


def clenshaw_curtiss_weights(n, a=-1.0, b=1.0):
    r"""
    Computation of the Clenshaw-Curtis quadrature nodes and weights.
    This implementation follows

    [1] Joerg Waldvogel, Fast Construction of the Fejer and Clenshaw-Curtis Quadrature Rules; BIT Numerical Mathematics, Vol. 43, No. 1, pp. 001–018.
    """

    assert(n > 1)

    tcc = np.cos(np.linspace(np.pi, 0, n))

    if n == 2:
        wcc = np.array([1., 1.])
    else:

        n1 = n - 1
        N = np.arange(1, n1, 2)
        l = len(N)
        m = n1 - l

        v = np.concatenate([2 / N / (N-2), 1 / N[-1:], np.zeros(m)])
        v = 0 - v[:-1] - v[-1:0:-1]

        g0 = -np.ones(n1)
        g0[l] = g0[l] + n1
        g0[m] = g0[m] + n1
        g = g0 / (n1**2 - 1 + (n1%2))
        wcc = np.fft.ifft(v + g).real
        wcc = np.concatenate((wcc, wcc[:1]))

    # rescale
    tcc = (b - a) * 0.5 * tcc + (b + a) * 0.5
    wcc = wcc * (b - a) * 0.5

    return tcc, wcc


def legpoly(mmax, lmax, x, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(x) at the positions specified by x.
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.
    """

    # compute the tensor P^m_n:
    nmax = max(mmax,lmax)
    vdm = np.zeros((nmax, nmax, len(x)), dtype=np.float64)
        
    norm_factor = 1. if norm == "ortho" else np.sqrt(4 * np.pi)
    norm_factor = 1. / norm_factor if inverse else norm_factor

    # initial values to start the recursion
    vdm[0,0,:] = norm_factor / np.sqrt(4 * np.pi)

    # fill the diagonal and the lower diagonal
    for l in range(1, nmax):
        vdm[l-1, l, :] = np.sqrt(2*l + 1) * x * vdm[l-1, l-1, :]
        vdm[l, l, :] = np.sqrt( np.abs((2*l + 1) * (1 + x) * (1 - x) / 2 / l) ) * vdm[l-1, l-1, :]

    # fill the remaining values on the upper triangle and multiply b
    for l in range(2, nmax):
        for m in range(0, l-1):
            vdm[m, l, :] = x * np.sqrt((2*l - 1) / (l - m) * (2*l + 1) / (l + m)) * vdm[m, l-1, :] \
                            - np.sqrt((l + m - 1) / (l - m) * (2*l + 1) / (2*l - 3) * (l - m - 1) / (l + m)) * vdm[m, l-2, :]

    if norm == "schmidt":
        for l in range(0, nmax):
            if inverse:
                vdm[:, l, : ] = vdm[:, l, : ] * np.sqrt(2*l + 1)
            else:
                vdm[:, l, : ] = vdm[:, l, : ] / np.sqrt(2*l + 1)

    vdm = vdm[:mmax, :lmax]

    if csphase:
        for m in range(1, mmax, 2):
            vdm[m] *= -1

    return vdm


def _precompute_legpoly(mmax, lmax, t, norm="ortho", inverse=False, csphase=True):

    return legpoly(mmax, lmax, np.cos(t), norm=norm, inverse=inverse, csphase=csphase)


class RealSHT(nn.Module):
    r"""
    Defines a module for computing the forward (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    The SHT is applied to the last two dimensions of the input

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):
        r"""
        Initializes the SHT Layer, precomputing the necessary quadrature weights

        Parameters:
        nlat: input grid resolution in the latitudinal direction
        nlon: input grid resolution in the longitudinal direction
        grid: grid in the latitude direction (for now only tensor product grids are supported)
        """

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # TODO: include assertions regarding the dimensions

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, w = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, w = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, w = clenshaw_curtiss_weights(nlat, -1, 1)
            # cost, w = fejer2_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        tq = np.flip(np.arccos(cost))

        # determine the dimensions 
        self.mmax = mmax or self.nlon // 2 + 1

        # combine quadrature weights with the legendre weights
        weights = torch.from_numpy(w)
        pct = _precompute_legpoly(self.mmax, self.lmax, tq, norm=self.norm, csphase=self.csphase)
        pct = torch.from_numpy(pct)
        weights = torch.einsum('mlk,k->mlk', pct, weights)
        weights = torch.tensor(weights, dtype = torch.float32)

        # remember quadrature weights
        # self.register_buffer('weights', weights, persistent=False)
        self.weights = nn.Parameter(weights)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        assert(x.shape[-2] == self.nlat)
        assert(x.shape[-1] == self.nlon)

        # apply real fft in the longitudinal direction
        x = 2.0 * torch.pi * torch.fft.rfft(x, dim=-1, norm=self.norm)   # 'forward'
        
        # do the Legendre-Gauss quadrature
        x = torch.view_as_real(x)
        
        # distributed contraction: fork
        out_shape = list(x.size())
        out_shape[-3] = self.lmax
        out_shape[-2] = self.mmax
        xout = torch.zeros(out_shape, dtype=x.dtype, device=x.device)
        
        # contraction
        xout[..., 0] = torch.einsum('...km,mlk->...lm', x[..., :self.mmax, 0], self.weights.to(x.device, x.dtype) )
        xout[..., 1] = torch.einsum('...km,mlk->...lm', x[..., :self.mmax, 1], self.weights.to(x.device, x.dtype) )
        x = torch.view_as_complex(xout)
        
        return x

class InverseRealSHT(nn.Module):
    r"""
    Defines a module for computing the inverse (real-valued) SHT.
    Precomputes Legendre Gauss nodes, weights and associated Legendre polynomials on these nodes.
    nlat, nlon: Output dimensions
    lmax, mmax: Input dimensions (spherical coefficients). For convenience, these are inferred from the output dimensions

    [1] Schaeffer, N. Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Wang, B., Wang, L., Xie, Z.; Accurate calculation of spherical and vector spherical harmonic expansions via spectral element grids; Adv Comput Math.
    """

    def __init__(self, nlat, nlon, lmax=None, mmax=None, grid="lobatto", norm="ortho", csphase=True):

        super().__init__()

        self.nlat = nlat
        self.nlon = nlon
        self.grid = grid
        self.norm = norm
        self.csphase = csphase

        # compute quadrature points
        if self.grid == "legendre-gauss":
            cost, _ = legendre_gauss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        elif self.grid == "lobatto":
            cost, _ = lobatto_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat-1
        elif self.grid == "equiangular":
            cost, _ = clenshaw_curtiss_weights(nlat, -1, 1)
            self.lmax = lmax or self.nlat
        else:
            raise(ValueError("Unknown quadrature mode"))

        # apply cosine transform and flip them
        t = np.flip(np.arccos(cost))

        # determine the dimensions 
        self.mmax = mmax or self.nlon // 2 + 1

        pct = _precompute_legpoly(self.mmax, self.lmax, t, norm=self.norm, inverse=True, csphase=self.csphase)
        # pct = torch.from_numpy(pct)
        pct = torch.tensor(pct, dtype = torch.float32)

        # register buffer
        # self.register_buffer('pct', pct, persistent=False)
        self.weights = nn.Parameter(pct)

    def extra_repr(self):
        r"""
        Pretty print module
        """
        return f'nlat={self.nlat}, nlon={self.nlon},\n lmax={self.lmax}, mmax={self.mmax},\n grid={self.grid}, csphase={self.csphase}'

    def forward(self, x: torch.Tensor):

        # assert(x.shape[-2] == self.lmax)
        # assert(x.shape[-1] == self.mmax)
        
        # Evaluate associated Legendre functions on the output nodes
        x = torch.view_as_real(x)
        
        rl = torch.einsum('...lm, mlk->...km', x[..., 0], self.weights.to(x.device, x.dtype) )
        im = torch.einsum('...lm, mlk->...km', x[..., 1], self.weights.to(x.device, x.dtype) )
        xs = torch.stack((rl, im), -1)

        # apply the inverse (real) FFT
        x = torch.view_as_complex(xs)
        x = torch.fft.irfft(x, n=self.nlon, dim=-1, norm=self.norm)   # 'forward'

        return x
    

@torch.jit.script
def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1. - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2d ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training) 


# class Mlp(nn.Module):
#     def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


class VAMoE2D(nn.Module):
    def __init__(self, h_size=32, w_size=64, hidden_size=768, num_blocks=16, high_freq=True, sparsity_threshold=0.01, hard_thresholding_fraction=1, hidden_size_factor=1, use_sht=True):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.use_sht = use_sht
        self.scale = 0.02

        self.h_size = h_size
        self.w_size = w_size
        self.high_freq = high_freq

        if self.use_sht:
            self.sht_cell = RealSHT(nlat=h_size, nlon=w_size, lmax=h_size, mmax=w_size // 2 + 1)
            self.isht_cell = InverseRealSHT(nlat=h_size, nlon=w_size, lmax=h_size, mmax=w_size // 2 + 1)

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor, dtype = torch.float32))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, dtype = torch.float32))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size, dtype = torch.float32))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, dtype = torch.float32))

        if self.high_freq:
            self.w = nn.Conv2d(hidden_size, hidden_size, 1, bias=False)  # High Frequency

    def forward(self, x):
        if self.high_freq:
            bias = x.permute(0,3,1,2)
            bias = self.w(bias)
            bias = bias.permute(0,2,3,1)
        else:
            bias = x

        dtype = x.dtype
        x = x.float()
        B, H, W, C = x.shape

        if self.use_sht:
            x = x.permute(0, 3, 1, 2)
            x = self.sht_cell(x).permute(0, 2, 3, 1)
        else:
            x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")

        x = x.reshape(B, H, W // 2 + 1, self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, H, W // 2 + 1, self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)


        total_modes = H // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes]  = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, total_modes-kept_modes:total_modes+kept_modes, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, H, W // 2 + 1, C)

        if self.use_sht:
            x = x.permute(0, 3, 1, 2)
            x = self.isht_cell(x).permute(0, 2, 3, 1)
        else:
            x = torch.fft.irfft2(x, s=(H, W), dim=(1,2), norm="ortho")
        
        x = x.type(dtype)

        return x + bias
    

class VAMoEBlock(nn.Module):
    def __init__(
            self,
            h_size,
            w_size,
            dim,
            mlp_ratio=4.,
            drop=0.,
            drop_path=0.,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
            double_skip=True,
            num_blocks=16,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
            use_moe=False,
            num_exports=69,
            noisy_gating=True,
            k_element=2
        ):
        super().__init__()
        self.use_moe = use_moe
        self.h_size = h_size
        self.w_size = w_size
        self.dim = dim
        self.layer_norm = norm_layer(dim)
        self.filter = VAMoE2D(h_size, w_size, dim, num_blocks, high_freq=True, sparsity_threshold=sparsity_threshold, hard_thresholding_fraction=hard_thresholding_fraction) 
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ffn_norm = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        if self.use_moe:
            self.moe = MoE(input_size=dim, output_size=dim, num_experts=num_exports, hidden_size=mlp_hidden_dim, noisy_gating=noisy_gating, k=k_element)
        else:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.double_skip = double_skip
        

    def forward(self, x):
        residual = x
        x = self.layer_norm(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.ffn_norm(x)

        if self.use_moe:
            x = x.view(-1, self.dim)
            x, loss = self.moe(x)
            x = x.view(-1, self.h_size, self.w_size, self.dim)
        else:
            x = self.mlp(x)

        x = self.drop_path(x)
        x = x + residual

        if self.use_moe:
            return x, loss
        else:
            return x

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

if __name__ == "__main__":
    # model = VAMoE2D(h_size=32, w_size=64, hidden_size=768)
    model = VAMoEBlock(h_size=32, w_size=64, dim=768, use_moe=True)
    outputs, _ = model(torch.randn(1, 32, 64, 768))
    print(outputs.shape)

    # inputs = torch.randn(1, 20, 32, 64)
    # sht = RealSHT(nlat=32, nlon=64, lmax=32, mmax=33)
    # mid_feature = sht(inputs)

    # isht = InverseRealSHT(nlat=32, nlon=64, lmax=32, mmax=33)
    # output = isht(mid_feature)
    # print(output.shape, inputs[0,:10,0,0], output[0,:10,0,0])