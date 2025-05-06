import torch
import numpy as np
import math

class SphericalHarmonics:
    """
    SphericalHarmonics class for converting between RGB and SH coefficients.
    The order of harmonics is fixed during initialization.
    """
    
    def __init__(self, order=3):
        """
        Initialize the SphericalHarmonics class.
        
        Args:
            order (int): Order of spherical harmonics. Default is 3.
                         This results in (order + 1)^2 coefficients per color channel.
        """
        self.order = order
        self.n_coeffs = (order + 1) ** 2
        
        # Precompute SH basis normalization factors
        self.norm_factors = self._compute_normalization_factors()
    
    def _compute_normalization_factors(self):
        """
        Compute normalization factors for the SH basis functions.
        
        Returns:
            List of normalization factors for each SH basis function.
        """
        factors = []
        for l in range(self.order + 1):
            for m in range(-l, l + 1):
                # Normalization factor for SH basis
                # K_l^m = sqrt((2l+1)/(4π) * (l-|m|)!/(l+|m|)!)
                term1 = (2 * l + 1) / (4 * math.pi)
                term2 = math.factorial(l - abs(m)) / math.factorial(l + abs(m))
                factors.append(math.sqrt(term1 * term2))
                
        return torch.tensor(factors)
    
    def _evaluate_sh_basis(self, directions):
        """
        Evaluate the spherical harmonics basis functions for the given directions.
        
        Args:
            directions (torch.Tensor): Directions of shape [N, 3] representing 
                                       unit vectors in Cartesian coordinates.
                                       
        Returns:
            torch.Tensor: SH basis evaluations of shape [N, n_coeffs]
        """
        x, y, z = directions[:, 0], directions[:, 1], directions[:, 2]
        
        # Convert to spherical coordinates
        # theta: polar angle (0 to pi)
        # phi: azimuthal angle (0 to 2pi)
        theta = torch.acos(torch.clamp(z, -1.0, 1.0))
        phi = torch.atan2(y, x)
        
        result = torch.zeros((directions.shape[0], self.n_coeffs), device=directions.device)
        
        # l=0, m=0 (constant basis)
        idx = 0
        result[:, idx] = 0.5 / torch.sqrt(torch.tensor(math.pi))
        
        if self.order >= 1:
            # l=1, m=-1, 0, 1
            idx += 1
            result[:, idx] = self.norm_factors[idx] * y  # l=1, m=-1
            idx += 1
            result[:, idx] = self.norm_factors[idx] * z  # l=1, m=0
            idx += 1
            result[:, idx] = self.norm_factors[idx] * x  # l=1, m=1
            
        if self.order >= 2:
            # l=2, m=-2, -1, 0, 1, 2
            idx += 1
            result[:, idx] = self.norm_factors[idx] * (x * y)  # l=2, m=-2
            idx += 1
            result[:, idx] = self.norm_factors[idx] * (y * z)  # l=2, m=-1
            idx += 1
            result[:, idx] = self.norm_factors[idx] * (3 * z * z - 1)  # l=2, m=0
            idx += 1
            result[:, idx] = self.norm_factors[idx] * (x * z)  # l=2, m=1
            idx += 1
            result[:, idx] = self.norm_factors[idx] * (x * x - y * y)  # l=2, m=2
        
        # For higher orders, we would need to implement more basis functions
        # or use recurrence relations for Associated Legendre Polynomials
        if self.order >= 3:
            # NOTE: For higher order SH, additional basis functions would be implemented here
            # This sample implementation supports up to order 2 explicitly
            # Higher orders would require more complex implementation of the basis functions
            raise NotImplementedError(f"SH basis for order {self.order} > 2 not fully implemented")
            
        return result
    
    def rgb2sh(self, rgb_values, directions):
        """
        Convert RGB values to SH coefficients.
        
        Args:
            rgb_values (torch.Tensor): RGB values of shape [N, 3]
            directions (torch.Tensor): Direction vectors of shape [N, 3]
            
        Returns:
            torch.Tensor: SH coefficients of shape [n_coeffs, 3]
        """
        assert rgb_values.shape[0] == directions.shape[0], "Number of RGB values must match number of directions"
        assert rgb_values.shape[1] == 3, "RGB values must have 3 channels"
        assert directions.shape[1] == 3, "Directions must be 3D vectors"
        
        # Normalize directions to unit vectors
        directions_norm = torch.nn.functional.normalize(directions, dim=1)
        
        # Evaluate SH basis functions
        basis = self._evaluate_sh_basis(directions_norm)  # [N, n_coeffs]
        
        # Project RGB values onto SH basis
        # For each color channel, compute the projection
        sh_coeffs = torch.zeros((self.n_coeffs, 3), device=rgb_values.device)
        
        for c in range(3):  # For each color channel (R, G, B)
            for i in range(self.n_coeffs):
                # Projection: Integrate RGB * SH basis over the sphere
                # In discrete form, this is a weighted average
                sh_coeffs[i, c] = torch.sum(rgb_values[:, c] * basis[:, i]) * (4 * math.pi / rgb_values.shape[0])
        
        return sh_coeffs
    
    def sh2rgb(self, sh_coeffs, directions):
        """
        Convert SH coefficients to RGB values.
        
        Args:
            sh_coeffs (torch.Tensor): SH coefficients of shape [n_coeffs, 3]
            directions (torch.Tensor): Direction vectors of shape [N, 3]
            
        Returns:
            torch.Tensor: RGB values of shape [N, 3]
        """
        assert sh_coeffs.shape[0] == self.n_coeffs, f"Expected {self.n_coeffs} SH coefficients, got {sh_coeffs.shape[0]}"
        assert sh_coeffs.shape[1] == 3, "SH coefficients must have 3 channels (RGB)"
        assert directions.shape[1] == 3, "Directions must be 3D vectors"
        
        # Normalize directions to unit vectors
        directions_norm = torch.nn.functional.normalize(directions, dim=1)
        
        # Evaluate SH basis functions
        basis = self._evaluate_sh_basis(directions_norm)  # [N, n_coeffs]
        
        # Reconstruct RGB values from SH coefficients
        # rgb = sum(sh_coeff[i] * basis[i]) for all i
        rgb_values = torch.zeros((directions.shape[0], 3), device=directions.device)
        
        for c in range(3):  # For each color channel (R, G, B)
            # Matrix multiplication: basis [N, n_coeffs] @ sh_coeffs[:, c] [n_coeffs] = rgb [:, c] [N]
            rgb_values[:, c] = torch.matmul(basis, sh_coeffs[:, c])
        
        return rgb_values