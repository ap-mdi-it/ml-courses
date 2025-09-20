"""
Derivative visualization for quadratic functions.

This module provides the DerivativeVisualization class, which illustrates the principle
of derivatives for the quadratic function y = x² through three progressive visualizations
showing how the secant line approaches the tangent line as Δx approaches zero.
"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from numpy.typing import NDArray


class DerivativeVisualization:
    """
    Derivative visualization for the quadratic function y = x².

    Creates a three-pane visualization showing:
    1. Large Δx with secant line and Δx/Δy square
    2. Medium Δx (half of the first) with secant line and smaller square
    3. Infinitesimal Δx showing the tangent line and derivative 2x
    """

    def __init__(
        self,
        x_point: float = 1.0,
        initial_delta: float = 1.0,
        x_range: tuple[float, float] = (-2.0, 4.0),
        y_range: tuple[float, float] = (-1.0, 16.0),
    ):
        """
        Initialize the DerivativeVisualization.

        Args:
            x_point: The x-coordinate where to calculate the derivative
            initial_delta: The initial Δx value for the first pane
            x_range: The x-axis range for plotting as (min, max)
            y_range: The y-axis range for plotting as (min, max)
        """
        self.x_point = x_point
        self.initial_delta = initial_delta
        self.x_range = x_range
        self.y_range = y_range

        # Calculate function values
        self.y_point = self._quadratic_function_scalar(self.x_point)

        # Calculate deltas for the three panes
        self.deltas = [
            self.initial_delta,  # Pane 1: full delta
            self.initial_delta / 2,  # Pane 2: half delta
            0.01,  # Pane 3: very small delta (approaching 0)
        ]

    def _quadratic_function_scalar(self, x: float) -> float:
        """Calculate y = x² for a single x value."""
        return float(x**2)

    def _quadratic_function_array(self, x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]:
        """Calculate y = x² for an array of x values."""
        return x**2

    def _calculate_secant_slope(self, x: float, delta_x: float) -> float:
        """Calculate the slope of the secant line."""
        y1 = self._quadratic_function_scalar(x)
        y2 = self._quadratic_function_scalar(x + delta_x)
        return (y2 - y1) / delta_x

    def _calculate_derivative(self, x: float) -> float:
        """Calculate the analytical derivative 2x."""
        return 2 * x

    def _plot_parabola(self, ax: plt.Axes) -> None:
        """Plot the parabola y = x² on the given axes."""
        x_vals = np.linspace(self.x_range[0], self.x_range[1], 200)
        y_vals = self._quadratic_function_array(x_vals)
        ax.plot(x_vals, y_vals, "b-", linewidth=2, label="y = x²")

    def _plot_points_and_secant(
        self, ax: plt.Axes, delta_x: float, pane_num: int
    ) -> tuple[float, float]:
        """Plot the two points and secant line for a given delta_x."""
        # Calculate points
        x1, x2 = self.x_point, self.x_point + delta_x
        y1, y2 = self._quadratic_function_scalar(x1), self._quadratic_function_scalar(x2)

        # Plot points
        ax.plot([x1, x2], [y1, y2], "ro", markersize=8, zorder=5)

        # Plot secant line (extend significantly beyond the points for better visibility)
        x_extend = 1.0
        x_line = np.array([x1 - x_extend, x2 + x_extend])
        slope = self._calculate_secant_slope(x1, delta_x)
        y_line = y1 + slope * (x_line - x1)

        color = "red" if pane_num < 3 else "green"
        label = (
            f"Secant line (slope ≈ {slope:.2f})"
            if pane_num < 3
            else f"Tangent line (slope = {slope:.2f})"
        )
        ax.plot(x_line, y_line, color=color, linewidth=2, linestyle="--", label=label)

        return x2, y2

    def _draw_delta_square(self, ax: plt.Axes, delta_x: float, pane_num: int) -> None:
        """Draw the square showing Δx and Δy."""
        x1, x2 = self.x_point, self.x_point + delta_x
        y1, y2 = self._quadratic_function_scalar(x1), self._quadratic_function_scalar(x2)

        # Only draw square for first two panes
        if pane_num < 3:
            # Create rectangle for Δx and Δy
            rect = Rectangle(
                (x1, y1),
                delta_x,
                y2 - y1,
                linewidth=2,
                edgecolor="orange",
                facecolor="orange",
                alpha=0.3,
            )
            ax.add_patch(rect)

            # Add labels for Δx and Δy
            ax.annotate(
                f"Δx = {delta_x:.2f}",
                xy=(x1 + delta_x / 2, y1 - 0.3),
                ha="center",
                va="top",
                fontsize=10,
                color="orange",
                weight="bold",
            )
            ax.annotate(
                f"Δy = {y2 - y1:.2f}",
                xy=(x1 - 0.1, y1 + (y2 - y1) / 2),
                ha="right",
                va="center",
                fontsize=10,
                color="orange",
                weight="bold",
            )

    def _add_slope_annotation(self, ax: plt.Axes, delta_x: float, pane_num: int) -> None:
        """Add slope calculation annotation."""
        slope = self._calculate_secant_slope(self.x_point, delta_x)
        y1, y2 = (
            self._quadratic_function_scalar(self.x_point),
            self._quadratic_function_scalar(self.x_point + delta_x),
        )

        if pane_num < 3:
            # Secant slope annotation - positioned at bottom-right to avoid legend overlap
            annotation = f"Slope = Δy/Δx = {y2 - y1:.2f}/{delta_x:.2f} = {slope:.2f}"
            ax.text(
                0.95,
                0.05,
                annotation,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
            )
        else:
            # Derivative annotation - positioned at bottom-right to avoid legend overlap
            derivative = self._calculate_derivative(self.x_point)
            annotation = f"lim(Δx→0) Δy/Δx = dy/dx = 2x = {derivative:.2f}"
            ax.text(
                0.95,
                0.05,
                annotation,
                transform=ax.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                fontsize=10,
                verticalalignment="bottom",
                horizontalalignment="right",
            )

    def plot(self, figsize: tuple[float, float] = (15, 5)) -> None:
        """
        Create the three-pane derivative visualization.

        Args:
            figsize: Figure size as (width, height)
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        titles = [
            f"Large Δx = {self.deltas[0]:.2f}",
            f"Medium Δx = {self.deltas[1]:.2f}",
            "Tangent Line (Δx → 0)",
        ]

        for i, (ax, delta_x, title) in enumerate(zip(axes, self.deltas, titles, strict=False)):
            # Plot the parabola
            self._plot_parabola(ax)

            # Plot points and secant/tangent line
            self._plot_points_and_secant(ax, delta_x, i + 1)

            # Draw the delta square (only for first two panes)
            self._draw_delta_square(ax, delta_x, i + 1)

            # Add slope annotation
            self._add_slope_annotation(ax, delta_x, i + 1)

            # Set axis properties
            ax.set_xlim(self.x_range)
            ax.set_ylim(self.y_range)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title(title, fontsize=12, weight="bold")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")

            # Add point label
            ax.annotate(
                f"({self.x_point:.1f}, {self.y_point:.1f})",
                xy=(self.x_point, self.y_point),
                xytext=(10, 10),
                textcoords="offset points",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
                fontsize=9,
            )

        plt.suptitle(
            "Derivative of y = x²: From Secant to Tangent Line", fontsize=16, weight="bold", y=0.98
        )
        plt.tight_layout()
        plt.show()

    def get_parameters(self) -> dict[str, Any]:
        """
        Get all visualization parameters.

        Returns
        -------
            Dictionary with all configuration parameters
        """
        return {
            "x_point": self.x_point,
            "initial_delta": self.initial_delta,
            "x_range": self.x_range,
            "y_range": self.y_range,
            "deltas": self.deltas,
        }

    def get_slopes(self) -> dict[str, float]:
        """
        Get the slopes for all three panes.

        Returns
        -------
            Dictionary with slopes for each pane and the analytical derivative
        """
        return {
            "pane_1_slope": self._calculate_secant_slope(self.x_point, self.deltas[0]),
            "pane_2_slope": self._calculate_secant_slope(self.x_point, self.deltas[1]),
            "pane_3_slope": self._calculate_secant_slope(self.x_point, self.deltas[2]),
            "analytical_derivative": self._calculate_derivative(self.x_point),
        }

    def __repr__(self) -> str:
        """Get string representation of the visualization."""
        return (
            f"DerivativeVisualization(x_point={self.x_point}, "
            f"initial_delta={self.initial_delta}, "
            f"derivative_at_point={self._calculate_derivative(self.x_point):.2f})"
        )
