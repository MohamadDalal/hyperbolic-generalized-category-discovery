r"""
:math:`\kappa`-Stereographic math module.
The functions for the mathematics in gyrovector spaces are taken from the
following resources:
    [1] Ganea, Octavian, Gary Bécigneul, and Thomas Hofmann. "Hyperbolic
           neural networks." Advances in neural information processing systems.
           2018.
    [2] Bachmann, Gregor, Gary Bécigneul, and Octavian-Eugen Ganea. "Constant
           Curvature Graph Convolutional Networks." arXiv preprint
           arXiv:1911.05076 (2019).
    [3] Skopek, Ondrej, Octavian-Eugen Ganea, and Gary Bécigneul.
           "Mixed-curvature Variational Autoencoders." arXiv preprint
           arXiv:1911.08411 (2019).
    [4] Ungar, Abraham A. Analytic hyperbolic geometry: Mathematical
           foundations and applications. World Scientific, 2005.
    [5] Albert, Ungar Abraham. Barycentric calculus in Euclidean and
           hyperbolic geometry: A comparative introduction. World Scientific,
           2010.
"""

import functools
from typing import List, Optional

import torch
import torch.jit
from torch import Tensor

@torch.jit.script
def sign(x):
    return torch.sign(x.sign() + 0.5)

@torch.jit.script
def sabs(x, eps: float = 1e-15):
    #return x.abs().add_(eps)
    return x.abs().clamp_min(eps)

@torch.jit.script
def clamp_abs(x, eps: float = 1e-15):
    s = sign(x)
    return s * sabs(x, eps=eps)

@torch.jit.script
def abs_zero_grad(x):
    # this op has derivative equal to 1 at zero
    return x * sign(x)

@torch.jit.script
def arsinh(x: torch.Tensor):
    return (x + torch.sqrt(1 + x.pow(2))).clamp_min(1e-15).log().to(x.dtype)

@torch.jit.script
def tanh(x):
    return x.clamp(-15, 15).tanh()

@torch.jit.script
def artanh(x: torch.Tensor):
    x = x.clamp(-1 + 1e-7, 1 - 1e-7)
    return (torch.log(1 + x).sub(torch.log(1 - x))).mul(0.5)

@torch.jit.script
def tan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            + 1 / 3 * k * x**3
            + 2 / 15 * k**2 * x**5
            + 17 / 315 * k**3 * x**7
            + 62 / 2835 * k**4 * x**9
            + 1382 / 155925 * k**5 * x**11
            # + o(k**6)
        )
    elif order == 1:
        return x + 1 / 3 * k * x**3
    elif order == 2:
        return x + 1 / 3 * k * x**3 + 2 / 15 * k**2 * x**5
    elif order == 3:
        return x + 1 / 3 * k * x**3 + 2 / 15 * k**2 * x**5 + 17 / 315 * k**3 * x**7
    elif order == 4:
        return (
            x
            + 1 / 3 * k * x**3
            + 2 / 15 * k**2 * x**5
            + 17 / 315 * k**3 * x**7
            + 62 / 2835 * k**4 * x**9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")
    
@torch.jit.script
def artan_k_zero_taylor(x: torch.Tensor, k: torch.Tensor, order: int = -1):
    if order == 0:
        return x
    k = abs_zero_grad(k)
    if order == -1 or order == 5:
        return (
            x
            - 1 / 3 * k * x**3
            + 1 / 5 * k**2 * x**5
            - 1 / 7 * k**3 * x**7
            + 1 / 9 * k**4 * x**9
            - 1 / 11 * k**5 * x**11
            # + o(k**6)
        )
    elif order == 1:
        return x - 1 / 3 * k * x**3
    elif order == 2:
        return x - 1 / 3 * k * x**3 + 1 / 5 * k**2 * x**5
    elif order == 3:
        return x - 1 / 3 * k * x**3 + 1 / 5 * k**2 * x**5 - 1 / 7 * k**3 * x**7
    elif order == 4:
        return (
            x
            - 1 / 3 * k * x**3
            + 1 / 5 * k**2 * x**5
            - 1 / 7 * k**3 * x**7
            + 1 / 9 * k**4 * x**9
        )
    else:
        raise RuntimeError("order not in [-1, 5]")

@torch.jit.script
def tan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return tan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * tanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.clamp_max(1e38).tan()
    else:
        tan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.clamp_max(1e38).tan(), tanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, tan_k_zero_taylor(x, k, order=1), tan_k_nonzero)
    
@torch.jit.script
def artan_k(x: torch.Tensor, k: torch.Tensor):
    k_sign = k.sign()
    zero = torch.zeros((), device=k.device, dtype=k.dtype)
    k_zero = k.isclose(zero)
    # shrink sign
    k_sign = torch.masked_fill(k_sign, k_zero, zero.to(k_sign.dtype))
    if torch.all(k_zero):
        return artan_k_zero_taylor(x, k, order=1)
    k_sqrt = sabs(k).sqrt()
    scaled_x = x * k_sqrt

    if torch.all(k_sign.lt(0)):
        return k_sqrt.reciprocal() * artanh(scaled_x)
    elif torch.all(k_sign.gt(0)):
        return k_sqrt.reciprocal() * scaled_x.atan()
    else:
        artan_k_nonzero = (
            torch.where(k_sign.gt(0), scaled_x.atan(), artanh(scaled_x))
            * k_sqrt.reciprocal()
        )
        return torch.where(k_zero, artan_k_zero_taylor(x, k, order=1), artan_k_nonzero)

def project(x: torch.Tensor, k: torch.Tensor, dim=-1, eps=-1):
    r"""
    Safe projection on the manifold for numerical stability.

    Parameters
    ----------
    x : tensor
        point on the Poincare ball
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension to compute norm
    eps : float
        stability parameter, uses default for dtype if not provided

    Returns
    -------
    tensor
        projected vector on the manifold
    """
    return _project(x, k, dim, eps)

@torch.jit.script
def _project(x, k, dim: int = -1, eps: float = -1.0):
    if eps < 0:
        if x.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5
    maxnorm = (1 - eps) / (sabs(k) ** 0.5)
    maxnorm = torch.where(k.lt(0), maxnorm, k.new_full((), 1e15))
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

@torch.jit.script
def _expmap0(u: torch.Tensor, k: torch.Tensor, dim: int = -1):
    u_norm = u.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
    #print(k)
    #print("tan_k output", tan_k(u_norm, k))
    gamma_1 = tan_k(u_norm, k) * (u / u_norm)
    return gamma_1

def expmap0(u: torch.Tensor, k: torch.Tensor, dim=-1, project=True):
    r"""
    Compute the exponential map of :math:`u` at the origin :math:`0`.

    .. math::

        \operatorname{exp}^\kappa_0(u)
        =
        \tan_\kappa(\|u\|_2/2) \frac{u}{\|u\|_2}

    Parameters
    ----------
    u : tensor
        speed vector on manifold
    k : tensor
        sectional curvature of manifold
    project : bool
        whether to project the result on the manifold TODO: Investigate what this really means
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        :math:`\gamma_{0, u}(1)` end point
    """
    res = _expmap0(u, k, dim=dim)
    if project:
        return project(res, k, dim=dim)
    else:
        return res

def logmap0(y: torch.Tensor, k: torch.Tensor, dim=-1):
    r"""
    Compute the logarithmic map of :math:`y` at the origin :math:`0`.

    .. math::

        \operatorname{log}^\kappa_0(y)
        =
        \tan_\kappa^{-1}(\|y\|_2) \frac{y}{\|y\|_2}

    The result of the logmap at the origin is a vector :math:`u` in the tangent
    space of the origin :math:`0` such that

    .. math::

        y = \operatorname{exp}^\kappa_0(\operatorname{log}^\kappa_0(y))

    Parameters
    ----------
    y : tensor
        target point on manifold
    k : tensor
        sectional curvature of manifold
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        tangent vector :math:`u\in T_0 M` that transports :math:`0` to :math:`y`
    """
    return _logmap0(y, k, dim=dim)


@torch.jit.script
def _logmap0(y: torch.Tensor, k, dim: int = -1):
    y_norm = y.norm(dim=dim, p=2, keepdim=True).clamp_min(1e-15)
    return (y / y_norm) * artan_k(y_norm, k)

def pairwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Compute the pairwise geodesic distance between two batches of points on
    the Poincar disc.

    Args:
        x: Tensor of shape `(B1, D)` giving a space components of a batch
            of point on the Poincare disc.
        y: Tensor of shape `(B2, D)` giving a space components of another
            batch of points on the Poincare disc.
        curv: Positive scalar denoting negative disc curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B1, B2)` giving pairwise distance along the geodesics
        connecting the input points.
    """
    
    curv = -curv # To be consistent with the rest of the functions, we expect c<0
    x2 = x.pow(2).sum(dim=-1, keepdim=False)
    y2 = y.pow(2).sum(dim=-1, keepdim=False)
    xy = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(dim=-1, keepdim=False)

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = 1+((2*xy)/((1-x2).unsqueeze(1) * (1-y2).unsqueeze(0)))
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5

def elementwise_dist(
    x: Tensor, y: Tensor, curv: float | Tensor = 1.0, eps: float = 1e-8
) -> Tensor:
    """
    Compute the elementwise geodesic distance between two batches of points on
    the Poincare disc.

    Args:
        x: Tensor of shape `(B, D)` giving a space components of a batch
            of points on the Poincare disc.
        y: Tensor of same shape as `x` giving another batch of points.
        curv: Positive scalar denoting negative disc curvature.
        eps: Small float number to avoid numerical instability.

    Returns:
        Tensor of shape `(B, )` giving elemntwise distance along the geodesics
        connecting the input points.
    """
    
    curv = -curv # To be consistent with the rest of the functions, we expect c<0
    x2 = x.pow(2).sum(dim=-1, keepdim=False)
    y2 = y.pow(2).sum(dim=-1, keepdim=False)
    xy = (x - y).pow(2).sum(dim=-1, keepdim=False)

    # Ensure numerical stability in arc-cosh by clamping input.
    c_xyl = 1+((2*xy)/((1-x2) * (1-y2)))
    _distance = torch.acosh(torch.clamp(c_xyl, min=1 + eps))
    return _distance / curv**0.5

# TODO: Consider seperating into batches to avoid overflow in einstein midpoint

def einstein_midpoint(x: Tensor, curv: float | Tensor = 1.0) -> Tensor:
    """
    Compute the Einstein midpoint of multiple points on the hyperboloid. The Einstein
    midpoint is the point centroid of points in the Klein model.
    This is the transformed version for the Lorentz model.

    Args:
        x: Tensor of shape `(B, D)` giving a batch of space components of
            vectors on the hyperboloid.
        curv: Positive scalar denoting negative hyperboloid curvature.

    Returns:
        Tensor of shape `(1, D)` giving the Einstein midpoint of input vectors.
    """
    
    curv = -curv # To be consistent with the rest of the functions, we expect c<0
    x_klein = 2*x / (1 + curv*torch.sum(x**2, dim=-1, keepdim=True))
    gamma = 1 / torch.sqrt(1 - curv*torch.sum(x_klein**2, dim=-1, keepdim=True))
    numerator = torch.sum(x_klein * gamma, dim=0)
    denominator = torch.sum(gamma, dim=0)
    midpoint_klein = numerator / denominator
    midpoint = midpoint_klein / (1+torch.sqrt(1-curv*torch.sum(midpoint_klein**2, dim=-1, keepdim=True)))
    return midpoint

def lorentz_to_poincare(x, k, dim=-1):
    r"""
    Diffeomorphism that maps from Hyperboloid to Poincare disc.

    .. math::

        \Pi_{\mathbb{H}^{d, 1} \rightarrow \mathbb{D}^{d, 1}\left(x_{0}, \ldots, x_{d}\right)}=\frac{\left(x_{1}, \ldots, x_{d}\right)}{x_{0}+\sqrt{k}}

    Parameters
    ----------
    x : tensor
        points in Lorentz space (Without x_time)
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points on the Poincare disc
    """
    x_time = torch.sqrt(1 / k + torch.sum(x**2, dim=-1, keepdim=True))
    return x / (x_time + torch.sqrt(k))

def poincare_to_lorentz(x, k, dim=-1, eps=1e-6):
    r"""
    Diffeomorphism that maps from Poincare disc to Hyperboloid.

    .. math::

        \Pi_{\mathbb{D}^{d, k} \rightarrow \mathbb{H}^{d d, 1}}\left(x_{1}, \ldots, x_{d}\right)=\frac{\sqrt{k} \left(1+|| \mathbf{x}||_{2}^{2}, 2 x_{1}, \ldots, 2 x_{d}\right)}{1-\|\mathbf{x}\|_{2}^{2}}

    Parameters
    ----------
    x : tensor
        point on Poincare ball
    k : tensor
        manifold negative curvature
    dim : int
        reduction dimension for operations

    Returns
    -------
    tensor
        points in the Lorentz space
    """
    x_norm_square = torch.sum(x * x, dim=dim, keepdim=True)
    res = torch.sqrt(k) * 2 * x / (1.0 - x_norm_square + eps)
    return res