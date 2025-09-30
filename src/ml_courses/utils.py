"""Utility functions."""

import io

import cairosvg
import dtreeviz
import matplotlib.pyplot as plt
from PIL import Image


def display_dtreeviz(render_obj: dtreeviz.utils.DTreeVizRender) -> None:
    """Display a dtreeviz object in a Jupyter notebook.

    Args:
        render_obj (dtreeviz.utils.DTreeVizRender): The dtreeviz render object to display.
    """
    png_data = cairosvg.svg2png(render_obj.svg().encode(), dpi=150)
    # Open the PNG data with PIL
    img: Image.Image = Image.open(io.BytesIO(png_data))

    # Resize the image to make it larger
    width, height = img.size
    img = img.resize((int(width * 1.5), int(height * 1.5)), Image.Resampling.LANCZOS)

    # Display with matplotlib - set figure size for larger display
    plt.figure(figsize=(20, 12))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
