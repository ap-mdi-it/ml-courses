"""Utility functions."""

import dtreeviz
from IPython.display import SVG, display


def display_dtreeviz(render_obj: dtreeviz.utils.DTreeVizRender) -> None:
    """Display a dtreeviz object in a Jupyter notebook.

    Args:
        render_obj (dtreeviz.utils.DTreeVizRender): The dtreeviz render object to display.
    """
    svg = SVG(render_obj.svg())
    display(svg)
