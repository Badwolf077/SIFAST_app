"""
Icon management for PyPulse application.
"""

from pathlib import Path

from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QStyle


class IconManager:
    """Manages icons for the application."""

    def __init__(self):
        self.icon_path = Path(__file__).parent.parent / "resources" / "icons"
        self.style = QApplication.style() if QApplication.instance() else None

    def get_icon(self, name: str) -> QIcon:
        """
        Get icon by name. First tries to load from resources folder,
        then falls back to standard Qt icons.
        """
        # Try to load from resources
        icon_file = self.icon_path / f"{name}.png"
        if icon_file.exists():
            return QIcon(str(icon_file))

        # Fall back to Qt standard icons
        if self.style:
            icon_map = {
                "calibration": QStyle.SP_ComputerIcon,
                "folder_open": QStyle.SP_DirOpenIcon,
                "acquire": QStyle.SP_MediaPlay,
                "save": QStyle.SP_DialogSaveButton,
                "scan": QStyle.SP_FileDialogDetailedView,
                "folder_scan": QStyle.SP_DirIcon,
                "save_results": QStyle.SP_DialogSaveButton,
            }

            if name in icon_map:
                return self.style.standardIcon(icon_map[name])

        # Return empty icon if nothing found
        return QIcon()
