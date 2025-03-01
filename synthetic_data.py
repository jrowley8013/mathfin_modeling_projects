"""Synthetic data generator for factor model simulations."""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union


@dataclass
class FactorModelParams:
    """Parameters for the factor model simulation."""
    
    N: int  # Number of assets
    T: int  # Time series length
    p: int  # Number of factors
    inv_snr: float  # 1/SNR (noise level)
    rho: float  # Auto-correlation decay rate of residuals
    beta: float  # Cross-correlation magnitude
    J: Optional[int] = None  # Cross-correlation affecting range
    
    def __post_init__(self):
        """Set default J value if not provided."""
        if self.J is None:
            self.J = self.N // 10


class FactorModelGenerator:
    """Generator for synthetic data based on a factor model.
    
    The model follows:
        X_it = sum_{j=1}^p L_ij F_jt + sqrt(θ) U_it
    
    where:
        U_it = sqrt((1-ρ²)/(1+2Jβ²)) * e_it
        e_it = ρe_{i,t-1} + v_it + sum_{h=max(i-J,1)}^{i-1} βv_ht + sum_{h=i+1}^{min(i+J,N)} βv_ht
        v_ht, L_it, F_jt ~ N(0,1)
    """
    
    def __init__(self, params: FactorModelParams):
        """Initialize the generator with model parameters."""
        self.params = params
        self.theta = self.params.inv_snr * self.params.p  # θ = (1/SNR) * p
        
    def generate_data(self, seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data based on the factor model.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            X: Observed data matrix of shape (N, T)
            L: Factor loadings matrix of shape (N, p)
            F: Factor matrix of shape (p, T)
        """
        if seed is not None:
            np.random.seed(seed)
            
        N, T, p = self.params.N, self.params.T, self.params.p
        rho, beta, J = self.params.rho, self.params.beta, self.params.J
        
        # Generate factor loadings and factors from N(0,1)
        L = np.random.normal(0, 1, size=(N, p))
        F = np.random.normal(0, 1, size=(p, T))
        
        # Generate the idiosyncratic component U
        U = np.zeros((N, T))
        
        # First generate v_it ~ N(0,1)
        v = np.random.normal(0, 1, size=(N, T))
        
        # Generate e_it using the auto-regressive and cross-correlation structure
        e = np.zeros((N, T))
        
        # Calculate the scaling factor for U_it
        scaling_factor = np.sqrt((1 - rho**2) / (1 + 2 * J * beta**2))
        
        # First time point (t=0) has no auto-correlation
        for i in range(N):
            # Cross-correlation terms
            cross_corr_sum = 0
            for h in range(max(i-J, 0), i):
                cross_corr_sum += beta * v[h, 0]
            for h in range(i+1, min(i+J+1, N)):
                cross_corr_sum += beta * v[h, 0]
                
            e[i, 0] = v[i, 0] + cross_corr_sum
            U[i, 0] = scaling_factor * e[i, 0]
        
        # Remaining time points
        for t in range(1, T):
            for i in range(N):
                # Auto-correlation term
                auto_corr = rho * e[i, t-1]
                
                # Cross-correlation terms
                cross_corr_sum = 0
                for h in range(max(i-J, 0), i):
                    cross_corr_sum += beta * v[h, t]
                for h in range(i+1, min(i+J+1, N)):
                    cross_corr_sum += beta * v[h, t]
                    
                e[i, t] = auto_corr + v[i, t] + cross_corr_sum
                U[i, t] = scaling_factor * e[i, t]
        
        # Generate the observed data X
        X = L @ F + np.sqrt(self.theta) * U
        
        return X, L, F
    
    def generate_panel_data(self, seed: Optional[int] = None) -> np.ndarray:
        """Generate synthetic panel data and return as a DataFrame-compatible array.
        
        Returns:
            X: Observed data matrix of shape (T, N) for easy conversion to pandas DataFrame
        """
        X, _, _ = self.generate_data(seed)
        return X.T  # Transpose to (T, N) for DataFrame compatibility


def create_factor_model(
    N: int, 
    T: int, 
    p: int, 
    inv_snr: float,
    rho: float = 0.0,
    beta: float = 0.0,
    J: Optional[int] = None
) -> FactorModelGenerator:
    """Create a factor model generator with parameters.
    
    Args:
        N: Number of assets
        T: Time series length
        p: Number of factors
        inv_snr: Inverse signal-to-noise ratio (noise level)
        rho: Auto-correlation decay rate of residuals (default: 0.0)
        beta: Cross-correlation magnitude (default: 0.0)
        J: Cross-correlation affecting range (default: N//10)
    
    Returns:
        A configured FactorModelGenerator
    """
    params = FactorModelParams(
        N=N,
        T=T,
        p=p,
        inv_snr=inv_snr,
        rho=rho,
        beta=beta,
        J=J if J is not None else N//10
    )
    
    return FactorModelGenerator(params)