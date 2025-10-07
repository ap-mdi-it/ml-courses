"""CNN visualization functions for educational purposes.

This module provides visualization functions for demonstrating convolutional neural network
concepts including convolution operations, padding, stride, activation functions, and pooling.
"""

from dataclasses import dataclass

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from numpy.typing import NDArray
from scipy.special import erf


@dataclass
class PoolingRegions:
    """Container for pooling visualization parameters."""

    regions: list[list[tuple[int, int]]]
    colors: list[str]
    max_positions: set[tuple[int, int]]


def visualize_convolution_steps() -> None:
    """Visualize the step-by-step convolution operation.

    Shows how a 2x2 filter slides over a 4x4 input matrix with stride=1,
    displaying all 5 steps of the convolution process.
    """
    # Define the input matrix and filter
    input_matrix = np.array([[1, 0, -2, 1], [-1, 0, 1, 2], [0, 2, 1, 0], [1, 0, 0, 1]])
    filter_matrix = np.array([[0, 1], [-1, 2]])

    # Create figure with subplots
    fig = plt.figure(figsize=(10, 12))

    # Define positions for highlighting at each step
    highlight_positions = [
        [(0, 0), (0, 1), (1, 0), (1, 1)],  # Step 1
        [(0, 1), (0, 2), (1, 1), (1, 2)],  # Step 2
        [(0, 2), (0, 3), (1, 2), (1, 3)],  # Step 3
        [(1, 0), (1, 1), (2, 0), (2, 1)],  # Step 4
        [(1, 1), (1, 2), (2, 1), (2, 2)],  # Step 5
    ]

    # Output values for each step
    output_values = [1, 0, 4, 4, 1]
    output_positions = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]

    # Draw each step
    for step in range(5):
        row = step

        # Input matrix
        ax1 = plt.subplot(5, 5, row * 5 + 1)
        _draw_matrix(ax1, input_matrix, highlight_positions[step])
        if step == 0:
            ax1.set_title("Input Matrix", fontsize=12, fontweight="bold", pad=10)

        # Multiplication symbol
        ax2 = plt.subplot(5, 5, row * 5 + 2)
        _draw_symbol(ax2, "⊗")

        # Filter
        ax3 = plt.subplot(5, 5, row * 5 + 3)
        _draw_filter(ax3, filter_matrix)
        if step == 0:
            ax3.set_title("Filter", fontsize=12, fontweight="bold", pad=10)

        # Arrow
        ax4 = plt.subplot(5, 5, row * 5 + 4)
        _draw_symbol(ax4, "→")

        # Output
        ax5 = plt.subplot(5, 5, row * 5 + 5)
        filled = {output_positions[i]: output_values[i] for i in range(step + 1)}
        _draw_output(ax5, step, filled)
        if step == 0:
            ax5.set_title("Output", fontsize=12, fontweight="bold", pad=10)

    # Add step labels after tight_layout so positions are calculated correctly
    plt.tight_layout(rect=(0.08, 0.0, 1.0, 1.0))

    # Add step labels vertically centered with each row using actual subplot positions
    for step in range(5):
        # Get the position of the first subplot in this row
        ax = plt.subplot(5, 5, step * 5 + 1)
        pos = ax.get_position()
        # Use the center y-position of the subplot
        y_pos = (pos.y0 + pos.y1) / 2
        fig.text(0.015, y_pos, f"Step-{step + 1}", fontsize=12, fontweight="bold", va="center")

    plt.show()

    print("\nCalculations for each step:")
    for i, (_pos, val) in enumerate(zip(output_positions, output_values, strict=True)):
        h_pos = highlight_positions[i]
        values = [input_matrix[p] for p in h_pos]
        calc = f"({values[0]}*0) + ({values[1]}*1) + ({values[2]}*-1) + ({values[3]}*2) = {val}"
        print(f"Step {i + 1}: {calc}")


def visualize_padding() -> None:
    """Visualize the effect of padding on convolution operations.

    Shows how zero-padding is added around the input matrix to maintain
    the same output dimensions as the input.
    """
    # Define input matrix and filter
    input_matrix_pad = np.array([[1, 0, -2, 1], [-1, 0, 1, 2], [0, 2, 1, 0], [1, 0, 0, 1]])
    filter_matrix_pad = np.array([[0, 1], [-1, 2]])

    # Add zero padding
    padded_input = np.pad(input_matrix_pad, pad_width=1, mode="constant", constant_values=0)

    # Create figure
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    # Highlight top-left position (0,0) in padded matrix - all 4 cells of the filter area
    highlight_pos = [(0, 0), (0, 1), (1, 0), (1, 1)]

    # Draw padded input
    _draw_padded_matrix(axes[0], padded_input, highlight_pos, "Padded Input")

    # Draw symbol
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis("off")
    axes[1].text(0.5, 0.5, "⊗", ha="center", va="center", fontsize=24, fontweight="bold")

    # Draw filter
    _draw_filter(axes[2], filter_matrix_pad, "Filter")

    # Draw arrow
    axes[3].set_xlim(0, 1)
    axes[3].set_ylim(0, 1)
    axes[3].axis("off")
    axes[3].text(0.5, 0.5, "→", ha="center", va="center", fontsize=24, fontweight="bold")

    # Calculate output value
    output_val = (
        padded_input[0, 0] * 0
        + padded_input[0, 1] * 1
        + padded_input[1, 0] * -1
        + padded_input[1, 1] * 2
    )
    filled_output = {(0, 0): output_val}

    # Draw output (4x4 to match input dimensions)
    rows, cols = 4, 4
    axes[4].set_xlim(0, cols)
    axes[4].set_ylim(0, rows)
    axes[4].set_aspect("equal")
    axes[4].axis("off")
    axes[4].set_title("Output (Same Size)", fontsize=12, fontweight="bold", pad=10)

    for i in range(rows):
        for j in range(cols):
            is_filled = (i, j) in filled_output
            color = "#4472C4" if is_filled else "white"

            rect = mpatches.Rectangle(
                (j, rows - 1 - i), 1, 1, linewidth=2, edgecolor="black", facecolor=color
            )
            axes[4].add_patch(rect)

            if is_filled:
                axes[4].text(
                    j + 0.5,
                    rows - 1 - i + 0.5,
                    str(filled_output[(i, j)]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                )

    plt.tight_layout()
    plt.show()

    print(
        f"\nCalculation: ({padded_input[0, 0]}*0) + ({padded_input[0, 1]}*1) + "
        f"({padded_input[1, 0]}*-1) + ({padded_input[1, 1]}*2) = {output_val}"
    )


def visualize_stride_comparison() -> None:
    """Compare convolution operations with different stride values.

    Shows side-by-side comparison of stride=1 and stride=2 operations
    on the same input matrix.
    """
    # Create figure to compare stride 1 vs stride 2
    fig, axes = plt.subplots(2, 5, figsize=(15, 8))

    # Define input and filter
    stride_input = np.array([[1, 0, -2, 1], [-1, 0, 1, 2], [0, 2, 1, 0], [1, 0, 0, 1]])
    stride_filter = np.array([[0, 1], [-1, 2]])

    # Stride 1: Show step 2
    stride1_highlight = [(0, 1), (0, 2), (1, 1), (1, 2)]
    stride1_output_vals = {(0, 0): 1, (0, 1): 0}

    # Draw stride 1 example
    _draw_matrix(axes[0, 0], stride_input, stride1_highlight, "Input Matrix (Stride=1)")
    _draw_symbol(axes[0, 1], "⊗")
    _draw_filter(axes[0, 2], stride_filter, "Filter")
    _draw_symbol(axes[0, 3], "→")

    # Output for stride 1
    rows_out, cols_out = 3, 3
    axes[0, 4].set_xlim(0, cols_out)
    axes[0, 4].set_ylim(0, rows_out)
    axes[0, 4].set_aspect("equal")
    axes[0, 4].axis("off")
    axes[0, 4].set_title("Output (Stride=1)", fontsize=12, fontweight="bold", pad=10)

    for i in range(rows_out):
        for j in range(cols_out):
            is_filled = (i, j) in stride1_output_vals
            color = "#4472C4" if is_filled else "white"
            rect = mpatches.Rectangle(
                (j, rows_out - 1 - i), 1, 1, linewidth=2, edgecolor="black", facecolor=color
            )
            axes[0, 4].add_patch(rect)
            if is_filled:
                axes[0, 4].text(
                    j + 0.5,
                    rows_out - 1 - i + 0.5,
                    str(stride1_output_vals[(i, j)]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                )

    # Stride 2: Show step 2
    stride2_highlight = [(0, 2), (0, 3), (1, 2), (1, 3)]
    stride2_output_vals = {(0, 0): 1, (0, 1): 4}

    # Draw stride 2 example
    _draw_matrix(axes[1, 0], stride_input, stride2_highlight, "Input Matrix (Stride=2)")
    _draw_symbol(axes[1, 1], "⊗")
    _draw_filter(axes[1, 2], stride_filter, "Filter")
    _draw_symbol(axes[1, 3], "→")

    # Output for stride 2
    rows_out2, cols_out2 = 2, 2
    axes[1, 4].set_xlim(0, cols_out2)
    axes[1, 4].set_ylim(0, rows_out2)
    axes[1, 4].set_aspect("equal")
    axes[1, 4].axis("off")
    axes[1, 4].set_title("Output (Stride=2)", fontsize=12, fontweight="bold", pad=10)

    for i in range(rows_out2):
        for j in range(cols_out2):
            is_filled = (i, j) in stride2_output_vals
            color = "#4472C4" if is_filled else "white"
            rect = mpatches.Rectangle(
                (j, rows_out2 - 1 - i), 1, 1, linewidth=2, edgecolor="black", facecolor=color
            )
            axes[1, 4].add_patch(rect)
            if is_filled:
                axes[1, 4].text(
                    j + 0.5,
                    rows_out2 - 1 - i + 0.5,
                    str(stride2_output_vals[(i, j)]),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color="white",
                )

    plt.tight_layout()
    plt.show()

    print("\nStride=1, Step 2:")
    print(
        f"({stride_input[0, 1]}*0) + ({stride_input[0, 2]}*1) + "
        f"({stride_input[1, 1]}*-1) + ({stride_input[1, 2]}*2) = {stride1_output_vals[(0, 1)]}"
    )

    print("\nStride=2, Step 2:")
    print(
        f"({stride_input[0, 2]}*0) + ({stride_input[0, 3]}*1) + "
        f"({stride_input[1, 2]}*-1) + ({stride_input[1, 3]}*2) = {stride2_output_vals[(0, 1)]}"
    )


def visualize_activation_functions() -> None:
    """Visualize ReLU and GELU activation functions.

    Plots both ReLU and GELU activation functions on the same graph
    for comparison.
    """
    # Define the x range
    x = np.linspace(-3, 3, 1000)

    # ReLU function: max(0, x)
    relu = np.maximum(0, x)

    # GELU function: x * Φ(x) where Φ is the cumulative distribution function
    # of the standard normal distribution
    gelu = 0.5 * x * (1 + erf(x / np.sqrt(2)))

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot ReLU
    ax.plot(x, relu, label="ReLU", color="blue", linewidth=2)

    # Plot GELU
    ax.plot(x, gelu, label="GELU", color="red", linewidth=2)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add axes through origin
    ax.axhline(y=0, color="k", linewidth=0.5)
    ax.axvline(x=0, color="k", linewidth=0.5)

    # Labels and title
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("f(x)", fontsize=12)
    ax.set_title("ReLU and GELU Activation Functions", fontsize=14, fontweight="bold")

    # Legend
    ax.legend(fontsize=12, loc="upper left")

    # Set axis limits
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 3)

    plt.tight_layout()
    plt.show()


def visualize_max_pooling() -> None:
    """Visualize max pooling operation.

    Shows how a 2x2 max pooling with stride=2 reduces a 4x4 input
    to a 2x2 output by taking the maximum value in each region.
    """
    # Define the input matrix
    pool_input = np.array([[1, 3, 2, 4], [5, 6, 1, 3], [1, 2, 4, 2], [3, 1, 3, 5]])

    # Calculate max pooling manually (2x2 pooling with stride 2)
    pool_output = np.zeros((2, 2))
    pool_output[0, 0] = np.max(pool_input[0:2, 0:2])  # Top-left
    pool_output[0, 1] = np.max(pool_input[0:2, 2:4])  # Top-right
    pool_output[1, 0] = np.max(pool_input[2:4, 0:2])  # Bottom-left
    pool_output[1, 1] = np.max(pool_input[2:4, 2:4])  # Bottom-right

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Define regions for highlighting (2x2 pools)
    pool_regions = [
        [(0, 0), (0, 1), (1, 0), (1, 1)],  # Top-left pool
        [(0, 2), (0, 3), (1, 2), (1, 3)],  # Top-right pool
        [(2, 0), (2, 1), (3, 0), (3, 1)],  # Bottom-left pool
        [(2, 2), (2, 3), (3, 2), (3, 3)],  # Bottom-right pool
    ]

    # Colors for different regions
    pool_colors = ["#FFE699", "#C6E0B4", "#B4C7E7", "#F4B084"]

    # Find max positions in each region
    max_positions = set()
    for region in pool_regions:
        region_values = [(pos, pool_input[pos]) for pos in region]
        max_pos = max(region_values, key=lambda x: x[1])[0]
        max_positions.add(max_pos)

    # Create pooling regions container
    pooling_data = PoolingRegions(
        regions=pool_regions, colors=pool_colors, max_positions=max_positions
    )

    # Draw input matrix
    _draw_pooling_input(axes[0], pool_input, pooling_data, "Input (4x4)")

    # Draw arrow
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis("off")
    axes[1].text(
        0.5,
        0.6,
        "Max Pool\n2x2, stride=2",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    axes[1].annotate(
        "",
        xy=(0.85, 0.5),
        xytext=(0.15, 0.5),
        arrowprops={"arrowstyle": "->", "lw": 2, "color": "black"},
    )

    # Draw output matrix
    _draw_pooling_output(axes[2], pool_output, pool_colors, "Output (2x2)")

    plt.tight_layout()
    plt.show()

    # Print the pooling calculations
    print("Max Pooling Calculations (2x2 with stride 2):\n")
    print(f"Top-left region: max({pool_input[0:2, 0:2].flatten()}) = {int(pool_output[0, 0])}")
    print(f"Top-right region: max({pool_input[0:2, 2:4].flatten()}) = {int(pool_output[0, 1])}")
    print(f"Bottom-left region: max({pool_input[2:4, 0:2].flatten()}) = {int(pool_output[1, 0])}")
    print(f"Bottom-right region: max({pool_input[2:4, 2:4].flatten()}) = {int(pool_output[1, 1])}")


# Helper functions for drawing


def _draw_matrix(
    ax: Axes,
    matrix: NDArray[np.int_],
    highlight_cells: list[tuple[int, int]] | None = None,
    title: str = "",
) -> None:
    """Draw a matrix with optional highlighting of specific cells.

    Args:
        ax: Matplotlib axes to draw on.
        matrix: 2D numpy array to visualize.
        highlight_cells: List of (row, col) tuples to highlight.
        title: Optional title for the matrix.
    """
    rows, cols = matrix.shape
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Draw cells
    for i in range(rows):
        for j in range(cols):
            # Check if this cell should be highlighted
            is_highlighted = highlight_cells and (i, j) in highlight_cells

            # Draw rectangle
            color = "#FFD966" if is_highlighted else "white"
            rect = mpatches.Rectangle(
                (j, rows - 1 - i), 1, 1, linewidth=2, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)

            # Add text
            ax.text(
                j + 0.5,
                rows - 1 - i + 0.5,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )


def _draw_filter(ax: Axes, matrix: NDArray[np.int_], title: str = "") -> None:
    """Draw a convolutional filter matrix.

    Args:
        ax: Matplotlib axes to draw on.
        matrix: 2D numpy array representing the filter.
        title: Optional title for the filter.
    """
    rows, cols = matrix.shape
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Draw cells
    for i in range(rows):
        for j in range(cols):
            rect = mpatches.Rectangle(
                (j, rows - 1 - i), 1, 1, linewidth=2, edgecolor="black", facecolor="#9DC3E6"
            )
            ax.add_patch(rect)

            ax.text(
                j + 0.5,
                rows - 1 - i + 0.5,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )


def _draw_output(
    ax: Axes, step: int, filled_positions: dict[tuple[int, int], int], title: str = ""
) -> None:
    """Draw output matrix with filled positions.

    Args:
        ax: Matplotlib axes to draw on.
        step: Current step number (unused but kept for API consistency).
        filled_positions: Dictionary mapping (row, col) to values.
        title: Optional title for the output.
    """
    rows, cols = 3, 3
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # Draw cells
    for i in range(rows):
        for j in range(cols):
            # Check if this position should be filled
            is_filled = (i, j) in filled_positions
            color = "#4472C4" if is_filled else "white"

            rect = mpatches.Rectangle(
                (j, rows - 1 - i), 1, 1, linewidth=2, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)

            # Add text if filled
            if is_filled:
                value = filled_positions[(i, j)]
                text_color = "white" if is_filled else "black"
                ax.text(
                    j + 0.5,
                    rows - 1 - i + 0.5,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=14,
                    fontweight="bold",
                    color=text_color,
                )


def _draw_symbol(ax: Axes, symbol: str) -> None:
    """Draw a mathematical symbol in the center of an axes.

    Args:
        ax: Matplotlib axes to draw on.
        symbol: Symbol string to display.
    """
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.5, 0.5, symbol, ha="center", va="center", fontsize=24, fontweight="bold")


def _draw_padded_matrix(
    ax: Axes,
    matrix: NDArray[np.int_],
    highlight_cells: list[tuple[int, int]] | None = None,
    title: str = "",
) -> None:
    """Draw a matrix with padding cells shown in gray.

    Args:
        ax: Matplotlib axes to draw on.
        matrix: 2D numpy array to visualize (including padding).
        highlight_cells: List of (row, col) tuples to highlight.
        title: Optional title for the matrix.
    """
    rows, cols = matrix.shape
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    for i in range(rows):
        for j in range(cols):
            is_highlighted = highlight_cells and (i, j) in highlight_cells

            # Padding cells are gray
            if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                # Padding cells can still be highlighted
                color = "#FFD966" if is_highlighted else "#E0E0E0"
            else:
                color = "#FFD966" if is_highlighted else "white"

            rect = mpatches.Rectangle(
                (j, rows - 1 - i), 1, 1, linewidth=2, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)

            ax.text(
                j + 0.5,
                rows - 1 - i + 0.5,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=14,
                fontweight="bold",
            )


def _draw_pooling_input(
    ax: Axes,
    matrix: NDArray[np.int_],
    pooling_data: PoolingRegions,
    title: str = "",
) -> None:
    """Draw input matrix for pooling visualization with colored regions.

    Args:
        ax: Matplotlib axes to draw on.
        matrix: 2D numpy array to visualize.
        pooling_data: Container with regions, colors, and max positions.
        title: Optional title for the matrix.
    """
    rows, cols = matrix.shape
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Draw all cells
    for i in range(rows):
        for j in range(cols):
            # Determine which region this cell belongs to
            color = "white"
            for region_idx, region in enumerate(pooling_data.regions):
                if (i, j) in region:
                    color = pooling_data.colors[region_idx]
                    break

            rect = mpatches.Rectangle(
                (j, rows - 1 - i), 1, 1, linewidth=2, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)

            # Add text - use red color for max values
            text_color = "red" if (i, j) in pooling_data.max_positions else "black"
            ax.text(
                j + 0.5,
                rows - 1 - i + 0.5,
                str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=text_color,
            )


def _draw_pooling_output(
    ax: Axes,
    matrix: NDArray[np.floating],
    pool_colors: list[str],
    title: str = "",
) -> None:
    """Draw output matrix for pooling visualization.

    Args:
        ax: Matplotlib axes to draw on.
        matrix: 2D numpy array to visualize.
        pool_colors: List of colors for each output cell.
        title: Optional title for the matrix.
    """
    rows, cols = matrix.shape
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Draw all cells with corresponding colors
    for i in range(rows):
        for j in range(cols):
            # Map output position to color
            region_idx = i * 2 + j
            color = pool_colors[region_idx]

            rect = mpatches.Rectangle(
                (j, rows - 1 - i), 1, 1, linewidth=2, edgecolor="black", facecolor=color
            )
            ax.add_patch(rect)

            # Add text
            ax.text(
                j + 0.5,
                rows - 1 - i + 0.5,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
            )
