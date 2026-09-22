"""Small reusable Qt widgets shared by StimTrace screens."""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTableView,
    QTableWidget,
    QVBoxLayout,
)


class FrozenFirstColumnTable(QTableWidget):
    """A table whose first data column remains visible during horizontal scrolling."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.frozen_view = QTableView(self)
        self.frozen_view.setModel(self.model())
        self.frozen_view.setSelectionModel(self.selectionModel())
        self.frozen_view.setFocusPolicy(Qt.NoFocus)
        self.frozen_view.setFrameShape(QFrame.NoFrame)
        self.frozen_view.verticalHeader().hide()
        self.frozen_view.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.frozen_view.horizontalHeader().setSectionsClickable(True)
        self.frozen_view.horizontalHeader().setSortIndicatorShown(True)
        self.frozen_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frozen_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frozen_view.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.frozen_view.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.frozen_view.setSelectionBehavior(self.selectionBehavior())
        self.frozen_view.setSelectionMode(self.selectionMode())
        self.frozen_view.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.frozen_view.setSortingEnabled(False)
        self.frozen_view.setStyleSheet("QTableView { border: none; }")
        self.viewport().stackUnder(self.frozen_view)

        self.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.horizontalHeader().sectionResized.connect(self._sync_column_width)
        # Do not connect the Qt signal directly to the overloaded C++ method.
        # PySide otherwise tries to generate a dynamic meta-method for
        # QHeaderView.setSortIndicator and prints a startup warning.
        self.horizontalHeader().sortIndicatorChanged.connect(
            self._sync_sort_indicator
        )
        self.frozen_view.horizontalHeader().sectionClicked.connect(
            self._sort_from_frozen_header
        )
        self.verticalHeader().sectionResized.connect(self._sync_row_height)
        self.verticalScrollBar().valueChanged.connect(
            self.frozen_view.verticalScrollBar().setValue
        )
        self.frozen_view.verticalScrollBar().valueChanged.connect(
            self.verticalScrollBar().setValue
        )
        self.frozen_view.hide()

    def refresh_frozen_column(self) -> None:
        """Refresh visibility and dimensions after the table model is repopulated."""
        if self.columnCount() == 0:
            self.frozen_view.hide()
            return
        for column in range(self.columnCount()):
            self.frozen_view.setColumnHidden(column, column != 0)
        self.frozen_view.setColumnWidth(0, self.columnWidth(0))
        self.frozen_view.horizontalHeader().setFixedHeight(
            self.horizontalHeader().height()
        )
        self.frozen_view.horizontalHeader().setSortIndicator(
            self.horizontalHeader().sortIndicatorSection(),
            self.horizontalHeader().sortIndicatorOrder(),
        )
        for row in range(self.rowCount()):
            self.frozen_view.setRowHeight(row, self.rowHeight(row))
        self._update_frozen_geometry()
        self.frozen_view.show()
        self.frozen_view.raise_()

    def _sync_column_width(
        self,
        logical_index: int,
        _old_size: int,
        new_size: int,
    ) -> None:
        if logical_index == 0:
            self.frozen_view.setColumnWidth(0, new_size)
            self._update_frozen_geometry()

    def _sync_sort_indicator(self, section: int, order: Qt.SortOrder) -> None:
        self.frozen_view.horizontalHeader().setSortIndicator(section, order)

    def _sync_row_height(
        self,
        logical_index: int,
        _old_size: int,
        new_size: int,
    ) -> None:
        self.frozen_view.setRowHeight(logical_index, new_size)

    def _sort_from_frozen_header(self, logical_index: int) -> None:
        if not self.isSortingEnabled():
            return
        header = self.horizontalHeader()
        if header.sortIndicatorSection() != logical_index:
            order = Qt.AscendingOrder
        else:
            order = (
                Qt.DescendingOrder
                if header.sortIndicatorOrder() == Qt.AscendingOrder
                else Qt.AscendingOrder
            )
        self.sortItems(logical_index, order)

    def _update_frozen_geometry(self) -> None:
        if self.columnCount() == 0:
            return
        self.frozen_view.setGeometry(
            self.verticalHeader().width() + self.frameWidth(),
            self.frameWidth(),
            self.columnWidth(0),
            self.viewport().height() + self.horizontalHeader().height(),
        )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_frozen_geometry()


class WorkflowStep(QFrame):
    """Status card that can optionally act as a keyboard-accessible action."""

    clicked = Signal()

    def __init__(self, number: int, title: str, detail: str, parent=None):
        super().__init__(parent)
        self.setObjectName("workflowStep")
        self.setProperty("state", "pending")
        self.setProperty("clickable", False)
        self.setFocusPolicy(Qt.NoFocus)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(9)
        number_label = QLabel(str(number))
        number_label.setProperty("role", "stepNumber")
        layout.addWidget(number_label)
        text = QVBoxLayout()
        text.setSpacing(1)
        self.title = QLabel(title)
        self.title.setStyleSheet("font-weight: 600;")
        self.detail = QLabel(detail)
        self.detail.setProperty("role", "muted")
        self.detail.setWordWrap(True)
        self.detail.setMinimumWidth(0)
        self.detail.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        text.addWidget(self.title)
        text.addWidget(self.detail)
        layout.addLayout(text, 1)
        for label in (number_label, self.title, self.detail):
            label.setAttribute(Qt.WA_TransparentForMouseEvents)

    def set_state(self, state: str, detail: str | None = None) -> None:
        self.setProperty("state", state)
        if detail is not None:
            self.detail.setText(detail)
        self.style().unpolish(self)
        self.style().polish(self)

    def set_clickable(self, clickable: bool, tooltip: str = "") -> None:
        self.setProperty("clickable", bool(clickable))
        self.setCursor(Qt.PointingHandCursor if clickable else Qt.ArrowCursor)
        self.setFocusPolicy(Qt.StrongFocus if clickable else Qt.NoFocus)
        self.setToolTip(tooltip)
        self.style().unpolish(self)
        self.style().polish(self)

    def mouseReleaseEvent(self, event) -> None:
        if self.property("clickable") and event.button() == Qt.LeftButton:
            self.clicked.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:
        if self.property("clickable") and event.key() in (
            Qt.Key_Return,
            Qt.Key_Enter,
            Qt.Key_Space,
        ):
            self.clicked.emit()
            event.accept()
            return
        super().keyPressEvent(event)
