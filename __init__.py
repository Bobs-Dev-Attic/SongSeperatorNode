"""
SongSeparatorNode – ComfyUI custom node package
================================================
Registers the SongSeparator node so that ComfyUI discovers it automatically
when this directory is placed inside ``ComfyUI/custom_nodes/``.
"""

from .song_separator import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
