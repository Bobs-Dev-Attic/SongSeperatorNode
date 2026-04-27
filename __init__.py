"""
SongSeparatorNode – ComfyUI custom node package
================================================
Registers the SongSeparator node so that ComfyUI discovers it automatically
when this directory is placed inside ``ComfyUI/custom_nodes/``.
"""

from .guitar_tab import (
    NODE_CLASS_MAPPINGS as GUITAR_TAB_NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS as GUITAR_TAB_NODE_DISPLAY_NAME_MAPPINGS,
)
from .song_separator import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS,
    **GUITAR_TAB_NODE_CLASS_MAPPINGS,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_NAME_MAPPINGS,
    **GUITAR_TAB_NODE_DISPLAY_NAME_MAPPINGS,
}

WEB_DIRECTORY = "./web/js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
