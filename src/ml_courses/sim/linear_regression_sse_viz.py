"""
Linear Regression SSE Visualization for parameter fitting.

This module provides the LinearRegressionSSEVisualizer class, which creates an interactive
3D visualization of the Sum of Squared Errors (SSE) surface for linear regression parameters.
The visualization shows how the SSE varies across the 2D parameter space (bias and slope)
and can overlay optimisation paths to demonstrate learning algorithms.
"""

from typing import Any

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray


class LinearRegressionSSEVisualizer:
    """
    Interactive 3D visualization of SSE surface for linear regression parameter fitting.

    Creates a 3D surface plot showing:
    - X-axis: bias (intercept parameter)
    - Y-axis: slope (slope parameter)
    - Z-axis: SSE (Sum of Squared Errors)

    Can overlay optimisation paths and highlight minima.
    """

    def __init__(
        self,
        x_data: NDArray[np.floating[Any]],
        y_data: NDArray[np.floating[Any]],
        true_bias: float | None = None,
        true_slope: float | None = None,
    ):
        """
        Initialize the LinearRegressionSSEVisualizer.

        Args:
            x_data: Input features (e.g., order totals)
            y_data: Target values (e.g., tip amounts)
            true_bias: True bias parameter (if known)
            true_slope: True slope parameter (if known)
        """
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.true_bias = true_bias
        self.true_slope = true_slope

        # Calculate true SSE if true parameters are provided
        self.true_sse = None
        if self.true_bias is not None and self.true_slope is not None:
            self.true_sse = self._calculate_sse(self.true_bias, self.true_slope)

    def _calculate_sse(self, bias: float, slope: float) -> float:
        """
        Calculate Sum of Squared Errors for given parameters.

        Args:
            bias: Bias/intercept parameter
            slope: Slope parameter

        Returns
        -------
            Sum of squared errors
        """
        y_pred = bias + slope * self.x_data
        residuals = self.y_data - y_pred
        return float(np.sum(residuals**2))

    def calculate_sse_surface(
        self,
        bias_range: tuple[float, float] | None = None,
        slope_range: tuple[float, float] | None = None,
        resolution: int = 50,
        scale_factor: float = 3.0,
    ) -> tuple[NDArray[np.floating[Any]], NDArray[np.floating[Any]], NDArray[np.floating[Any]]]:
        """
        Calculate SSE surface over a grid of parameter values.

        Args:
            bias_range: Range for bias parameter as (min, max). Auto-detected if None.
            slope_range: Range for slope parameter as (min, max). Auto-detected if None.
            resolution: Number of grid points along each axis
            cscale_factor: Multiplier for parameter ranges

        Returns
        -------
            Tuple of (bias_mesh, slope_mesh, sse_mesh) for 3D plotting
        """
        # Auto-detect reasonable ranges if not provided, with enhanced ranges for convexity
        if bias_range is None:
            if self.true_bias is not None:
                bias_center = self.true_bias
                bias_range = (
                    float(bias_center - bias_center * scale_factor),
                    float(bias_center + bias_center * scale_factor),
                )
            else:
                # Use data-driven heuristic
                y_mean = np.mean(self.y_data)
                bias_range = (
                    float(y_mean - y_mean * scale_factor),
                    float(y_mean + y_mean * scale_factor),
                )

        if slope_range is None:
            if self.true_slope is not None:
                slope_center = self.true_slope
                slope_range = (
                    float(slope_center - slope_center * scale_factor),
                    float(slope_center + slope_center * scale_factor),
                )
            else:
                # Use data-driven heuristic
                x_range = np.max(self.x_data) - np.min(self.x_data)
                y_range = np.max(self.y_data) - np.min(self.y_data)
                slope_estimate = y_range / x_range
                slope_range = (
                    float(slope_estimate - slope_estimate * scale_factor),
                    float(slope_estimate + slope_estimate * scale_factor),
                )

        # Create parameter grids
        assert bias_range is not None
        assert slope_range is not None
        bias_vals = np.linspace(bias_range[0], bias_range[1], resolution)
        slope_vals = np.linspace(slope_range[0], slope_range[1], resolution)
        bias_mesh, slope_mesh = np.meshgrid(bias_vals, slope_vals)

        # Calculate SSE for each parameter combination
        sse_mesh = np.zeros_like(bias_mesh)
        for i in range(resolution):
            for j in range(resolution):
                sse_mesh[i, j] = self._calculate_sse(bias_mesh[i, j], slope_mesh[i, j])

        return bias_mesh, slope_mesh, sse_mesh

    def create_3d_surface_plot(  # noqa: PLR0913
        self,
        bias_samples: NDArray[np.floating[Any]] | None = None,
        slope_samples: NDArray[np.floating[Any]] | None = None,
        loss_samples: NDArray[np.floating[Any]] | None = None,
        bias_range: tuple[float, float] | None = None,
        slope_range: tuple[float, float] | None = None,
        resolution: int = 50,
        colorscale: str = "RdBu_r",
        title: str = "Linear Regression SSE Surface",
        scale_factor: float = 3.0,
        show_contours: bool = True,
    ) -> go.Figure:
        """
        Create interactive 3D surface plot of SSE landscape.

        Args:
            bias_samples: Array of bias values from optimization algorithm
            slope_samples: Array of slope values from optimization algorithm
            loss_samples: Array of SSE values from optimization algorithm
            bias_range: Range for bias parameter as (min, max)
            slope_range: Range for slope parameter as (min, max)
            resolution: Grid resolution for surface calculation
            colorscale: Plotly colorscale name (default: RdBu_r for blue=high, red=low)
            title: Plot title
            scale_factor: Multiplier for parameter ranges
            show_contours: Whether to show contour lines on the surface

        Returns
        -------
            Plotly Figure object
        """
        # Adjust ranges to include starting coordinate if sampling trace is shown
        adjusted_bias_range = bias_range
        adjusted_slope_range = slope_range

        if bias_samples is not None and slope_samples is not None:
            # Ensure surface includes the starting coordinate
            start_bias, start_slope = bias_samples[0], slope_samples[0]

            if adjusted_bias_range is None:
                if self.true_bias is not None:
                    bias_center = self.true_bias
                    bias_min = min(bias_center - abs(bias_center) * scale_factor, start_bias)
                    bias_max = max(bias_center + abs(bias_center) * scale_factor, start_bias)
                    adjusted_bias_range = (bias_min, bias_max)
                else:
                    y_mean = np.mean(self.y_data)
                    bias_min = min(y_mean - abs(y_mean) * scale_factor, start_bias)
                    bias_max = max(y_mean + abs(y_mean) * scale_factor, start_bias)
                    adjusted_bias_range = (bias_min, bias_max)
            else:
                # Expand existing range if needed
                adjusted_bias_range = (
                    min(adjusted_bias_range[0], start_bias),
                    max(adjusted_bias_range[1], start_bias),
                )

            if adjusted_slope_range is None:
                if self.true_slope is not None:
                    slope_center = self.true_slope
                    slope_min = min(slope_center - abs(slope_center) * scale_factor, start_slope)
                    slope_max = max(slope_center + abs(slope_center) * scale_factor, start_slope)
                    adjusted_slope_range = (slope_min, slope_max)
                else:
                    x_range = np.max(self.x_data) - np.min(self.x_data)
                    y_range = np.max(self.y_data) - np.min(self.y_data)
                    slope_estimate = y_range / x_range
                    slope_min = min(
                        slope_estimate - abs(slope_estimate) * scale_factor, start_slope
                    )
                    slope_max = max(
                        slope_estimate + abs(slope_estimate) * scale_factor, start_slope
                    )
                    adjusted_slope_range = (slope_min, slope_max)
            else:
                # Expand existing range if needed
                adjusted_slope_range = (
                    min(adjusted_slope_range[0], start_slope),
                    max(adjusted_slope_range[1], start_slope),
                )

        # Calculate SSE surface with adjusted ranges
        bias_mesh, slope_mesh, sse_mesh = self.calculate_sse_surface(
            bias_range=adjusted_bias_range,
            slope_range=adjusted_slope_range,
            resolution=resolution,
            scale_factor=scale_factor,
        )

        # Create the main surface with contours
        surface_kwargs = {
            "x": bias_mesh,
            "y": slope_mesh,
            "z": sse_mesh,
            "colorscale": colorscale,
            "name": "SSE Surface",
            "hovertemplate": (
                "<b>Parameters:</b><br>"
                "bias: %{x:.3f}<br>"
                "slope: %{y:.3f}<br>"
                "SSE: %{z:.3f}<br>"
                "<extra></extra>"
            ),
            "showscale": True,
            "colorbar": {"title": "SSE", "x": -0.2},  # Position colorbar further left
        }

        # Add contour lines if requested
        if show_contours:
            surface_kwargs["contours"] = {
                "z": {
                    "show": True,
                    "usecolormap": True,
                    "highlightcolor": "limegreen",
                    "project": {"z": True},
                }
            }

        surface = go.Surface(**surface_kwargs)

        # Create figure
        fig = go.Figure(data=[surface])

        # Add optimization path if provided
        if bias_samples is not None and slope_samples is not None and loss_samples is not None:
            self._add_optimization_trace(fig, bias_samples, slope_samples, loss_samples)

        # Add minima markers
        self._add_minima_markers(fig)

        # Configure layout
        fig.update_layout(
            title={"text": title, "x": 0.5, "font": {"size": 16}},
            scene={
                "xaxis_title": "bias",
                "yaxis_title": "slope",
                "zaxis_title": "SSE",
                "camera": {
                    "eye": {"x": 1.5, "y": 1.5, "z": 1.5}  # Good initial viewing angle
                },
            },
            width=800,
            height=600,
            margin={"l": 80, "r": 0, "t": 40, "b": 0},  # Increase left margin for colorbar
        )

        return fig

    def _add_optimization_trace(
        self,
        fig: go.Figure,
        bias_samples: NDArray[np.floating[Any]],
        slope_samples: NDArray[np.floating[Any]],
        loss_samples: NDArray[np.floating[Any]],
    ) -> None:
        """
        Add optimization path as 3D scatter trace.

        Args:
            fig: Plotly figure to add trace to
            bias_samples: Array of bias values from optimization
            slope_samples: Array of slope values from optimization
            loss_samples: Array of SSE values from optimization
        """
        # Subsample for better performance (show every 10th point)
        step = max(1, len(bias_samples) // 1000)
        bias_sub = bias_samples[::step]
        slope_sub = slope_samples[::step]
        loss_sub = loss_samples[::step]

        # Create color scale based on iteration number (early = red, late = blue)
        colors = np.linspace(0, 1, len(bias_sub))

        # Add optimization path as scatter points
        fig.add_trace(
            go.Scatter3d(
                x=bias_sub,
                y=slope_sub,
                z=loss_sub,
                mode="markers+lines",
                marker={
                    "size": 3,
                    "color": colors,
                    "colorscale": "RdYlBu_r",
                    "showscale": False,
                    "opacity": 0.7,
                },
                line={"color": "rgba(100, 100, 100, 0.5)", "width": 2},
                name="Optimization Path",
                hovertemplate=(
                    "<b>Optimization Step:</b><br>"
                    "bias: %{x:.3f}<br>"
                    "slope: %{y:.3f}<br>"
                    "SSE: %{z:.3f}<br>"
                    "<extra></extra>"
                ),
            )
        )

        # Highlight start and end points
        fig.add_trace(
            go.Scatter3d(
                x=[bias_samples[0]],
                y=[slope_samples[0]],
                z=[loss_samples[0]],
                mode="markers",
                marker={"size": 8, "color": "red", "symbol": "diamond"},
                name="Start Point",
                hovertemplate=(
                    "<b>Starting Point:</b><br>"
                    "bias: %{x:.3f}<br>"
                    "slope: %{y:.3f}<br>"
                    "SSE: %{z:.3f}<br>"
                    "<extra></extra>"
                ),
            )
        )

        fig.add_trace(
            go.Scatter3d(
                x=[bias_samples[-1]],
                y=[slope_samples[-1]],
                z=[loss_samples[-1]],
                mode="markers",
                marker={"size": 8, "color": "blue", "symbol": "diamond"},
                name="End Point",
                hovertemplate=(
                    "<b>Final Point:</b><br>"
                    "bias: %{x:.3f}<br>"
                    "slope: %{y:.3f}<br>"
                    "SSE: %{z:.3f}<br>"
                    "<extra></extra>"
                ),
            )
        )

    def _add_minima_markers(self, fig: go.Figure) -> None:
        """
        Add markers for true minimum and other notable points.

        Args:
            fig: Plotly figure to add markers to
        """
        if self.true_bias is not None and self.true_slope is not None and self.true_sse is not None:
            # Add true minimum marker
            fig.add_trace(
                go.Scatter3d(
                    x=[self.true_bias],
                    y=[self.true_slope],
                    z=[self.true_sse],
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": "gold",
                        "symbol": "diamond",
                        "line": {"color": "black", "width": 2},
                    },
                    name="True Minimum",
                    hovertemplate=(
                        "<b>True Minimum:</b><br>"
                        "bias: %{x:.3f}<br>"
                        "slope: %{y:.3f}<br>"
                        "SSE: %{z:.3f}<br>"
                        "<extra></extra>"
                    ),
                )
            )
