"""Shared symbols and sizing for recurring StimTrace actions."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPainter, QPalette, QPen, QPixmap
from PySide6.QtWidgets import QPushButton, QStyle


ACTION_ICONS = {
    "open": QStyle.SP_DialogOpenButton,
    "open_folder": QStyle.SP_DirOpenIcon,
    "save": QStyle.SP_DialogSaveButton,
    "delete": QStyle.SP_TrashIcon,
    "clear": QStyle.SP_DialogResetButton,
    "help": QStyle.SP_DialogHelpButton,
    "settings": QStyle.SP_FileDialogDetailedView,
    "run": QStyle.SP_MediaPlay,
    "apply": QStyle.SP_DialogApplyButton,
    "cancel": QStyle.SP_DialogCancelButton,
    "refresh": QStyle.SP_BrowserReload,
    "previous": QStyle.SP_ArrowBack,
    "next": QStyle.SP_ArrowForward,
}


def set_action_icon(button: QPushButton, action: str) -> QPushButton:
    """Apply the application-wide icon assigned to an action type."""
    button.setIcon(button.style().standardIcon(ACTION_ICONS[action]))
    return button


def set_polarity_icon(button: QPushButton) -> QPushButton:
    """Draw a compact, font-independent plus/minus polarity symbol."""
    pixmap = QPixmap(16, 16)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    pen = QPen(button.palette().color(QPalette.ButtonText), 1.8)
    pen.setCapStyle(Qt.RoundCap)
    painter.setPen(pen)
    painter.drawLine(3, 5, 13, 5)
    painter.drawLine(8, 1, 8, 9)
    painter.drawLine(3, 13, 13, 13)
    painter.end()
    button.setText("")
    button.setIcon(QIcon(pixmap))
    return button


def configure_row_delete(button: QPushButton, tooltip: str) -> QPushButton:
    """Configure the compact destructive control used inside tables and lists."""
    button.setText("X")
    button.setFixedWidth(30)
    button.setProperty("role", "danger")
    button.setToolTip(tooltip)
    return button
