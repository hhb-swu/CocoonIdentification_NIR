"""
Recurrence Plot (RP) Generation
Author: hhb_swu
Created: 2025
"""

import numpy as np
from typing import Tuple


def phase_space_reconstruction(
        time_series: np.ndarray,
        embedding_dim: int = 2,
        delay: int = 6
) -> np.ndarray:
    """
    Perform phase space reconstruction using delay embedding method.

    Parameters
    ----------
    time_series : np.ndarray
        1D time series data of shape (N,)
    embedding_dim : int, default=2
        Embedding dimension (m)
    delay : int, default=6
        Delay step (τ)

    Returns
    -------
    np.ndarray
        Reconstructed phase space matrix of shape (M, embedding_dim)
        where M = N - (embedding_dim - 1) * delay

    Notes
    -----
    Based on Takens' embedding theorem for dynamical systems.
    """
    # Input validation
    if not isinstance(time_series, np.ndarray):
        time_series = np.array(time_series)

    if time_series.ndim != 1:
        raise ValueError("Input time_series must be 1-dimensional")

    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive integer")

    if delay <= 0:
        raise ValueError("delay must be positive integer")

    N = len(time_series)

    # Check if time series is long enough
    min_length = embedding_dim * delay
    if N < min_length:
        raise ValueError(
            f"Time series too short. Need at least {min_length} points, "
            f"but got {N} points."
        )

    # Phase space reconstruction using vectorized operations
    M = N - (embedding_dim - 1) * delay
    indices = np.arange(M)[:, np.newaxis] + np.arange(embedding_dim) * delay
    phase_space = time_series[indices]

    return phase_space


def compute_distance_matrix(phase_space: np.ndarray) -> np.ndarray:
    """
    Compute pairwise Euclidean distance matrix in phase space.

    Parameters
    ----------
    phase_space : np.ndarray
        Phase space matrix of shape (M, embedding_dim)

    Returns
    -------
    np.ndarray
        Distance matrix of shape (M, M)
    """
    # Input validation
    if phase_space.ndim != 2:
        raise ValueError("phase_space must be 2-dimensional")

    M, dim = phase_space.shape

    # Efficient computation using broadcasting
    # Reshape for broadcasting: (M, 1, dim) and (1, M, dim)
    diff = phase_space[:, np.newaxis, :] - phase_space[np.newaxis, :, :]

    # Sum of squared differences along the embedding dimension
    squared_diff = np.sum(diff ** 2, axis=2)

    # Euclidean distance
    distance_matrix = np.sqrt(squared_diff + 1e-12)  # Small epsilon for numerical stability

    # Ensure symmetry (should already be symmetric, but safe check)
    np.fill_diagonal(distance_matrix, 0)

    return distance_matrix


def generate_recurrence_plot(
        time_series: np.ndarray,
        embedding_dim: int = 2,
        delay: int = 6,
        return_normalized: bool = True
) -> np.ndarray:
    """
    Generate recurrence plot from time series data.

    Parameters
    ----------
    time_series : np.ndarray
        1D time series data
    embedding_dim : int, default=2
        Embedding dimension for phase space reconstruction
    delay : int, default=6
        Delay step for phase space reconstruction
    return_normalized : bool, default=True
        If True, return normalized distance matrix [0, 1]
        If False, return raw distance matrix

    Returns
    -------
    np.ndarray
        Recurrence plot matrix (normalized or raw distance matrix)

    Examples
    --------
    >>> time_series = np.random.randn(1000)
    >>> rp = generate_recurrence_plot(time_series, embedding_dim=2, delay=6)
    >>> print(rp.shape)
    (989, 989)
    """
    # Phase space reconstruction
    phase_space = phase_space_reconstruction(time_series, embedding_dim, delay)

    # Compute distance matrix
    distance_matrix = compute_distance_matrix(phase_space)

    if return_normalized:
        # Normalize to [0, 1]
        d_min = distance_matrix.min()
        d_max = distance_matrix.max()

        # Handle case where all distances are equal
        if np.isclose(d_max, d_min, atol=1e-12):
            normalized_matrix = np.zeros_like(distance_matrix)
        else:
            normalized_matrix = (distance_matrix - d_min) / (d_max - d_min)

        return normalized_matrix
    else:
        return distance_matrix


# Alias for backward compatibility with original function name
RP = generate_recurrence_plot


def visualize_recurrence_plot(
        rp_matrix: np.ndarray,
        title: str = "Recurrence Plot",
        cmap: str = "viridis",
        save_path: str = None
) -> None:
    """
    Visualize recurrence plot matrix.

    Parameters
    ----------
    rp_matrix : np.ndarray
        Recurrence plot matrix (2D array)
    title : str, default="Recurrence Plot"
        Plot title
    cmap : str, default="viridis"
        Colormap for visualization
    save_path : str, optional
        If provided, save the figure to this path

    Returns
    -------
    None
        Displays the plot
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.imshow(rp_matrix, cmap=cmap, origin='lower', aspect='auto')
        plt.colorbar(label='Distance (normalized)' if rp_matrix.max() <= 1 else 'Distance')
        plt.title(title)
        plt.xlabel("Time index i")
        plt.ylabel("Time index j")

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")

        plt.show()

    except ImportError:
        print("Matplotlib not installed. Please install with: pip install matplotlib")


def test_rp_generation() -> Tuple[bool, str]:
    """
    Test function to verify RP generation works correctly.

    Returns
    -------
    Tuple[bool, str]
        (success, message) indicating test result
    """
    try:
        # Create synthetic test data
        t = np.linspace(0, 20 * np.pi, 1000)
        test_series = np.sin(t) + 0.1 * np.random.randn(len(t))

        # Generate recurrence plot
        rp = generate_recurrence_plot(test_series, embedding_dim=2, delay=6)

        # Validate output
        assert isinstance(rp, np.ndarray), "Output should be numpy array"
        assert rp.ndim == 2, "Output should be 2D matrix"
        assert rp.shape[0] == rp.shape[1], "Output should be square matrix"
        assert rp.min() >= 0, "Normalized values should be >= 0"
        assert rp.max() <= 1, "Normalized values should be <= 1"

        # Check symmetry (should be symmetric due to distance matrix)
        symmetric_diff = np.max(np.abs(rp - rp.T))
        assert symmetric_diff < 1e-10, f"Matrix should be symmetric, but max diff={symmetric_diff}"

        return True, "All tests passed successfully!"

    except Exception as e:
        return False, f"Test failed with error: {str(e)}"


if __name__ == "__main__":
    """
    Main execution block for demonstration and testing.
    """
    print("=" * 60)
    print("Recurrence Plot (RP) Generation Module")
    print("=" * 60)

    # Run tests
    print("\nRunning tests...")
    success, message = test_rp_generation()
    if success:
        print(f"✓ {message}")
    else:
        print(f"✗ {message}")

    # Example usage
    print("\nExample: Generating RP for synthetic data...")

    # Create example time series
    t = np.linspace(0, 10 * np.pi, 500)
    example_series = np.sin(t) + 0.05 * np.random.randn(len(t))

    # Generate recurrence plot
    rp_example = generate_recurrence_plot(example_series)

    # Display information
    print(f"Input time series length: {len(example_series)}")
    print(f"RP matrix shape: {rp_example.shape}")
    print(f"RP value range: [{rp_example.min():.4f}, {rp_example.max():.4f}]")
    print(f"RP diagonal values: {np.diag(rp_example)[:5]}... (first 5)")

    # Try to visualize if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        visualize_recurrence_plot(
            rp_example,
            title="Example Recurrence Plot",
            save_path="example_recurrence_plot.png"
        )
    except ImportError:
        print("\nNote: Install matplotlib for visualization: pip install matplotlib")

    print("\nDone!")