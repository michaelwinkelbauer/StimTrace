from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QTableWidgetItem
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
import numpy as np
import pandas as pd

from signal_analysis import SignalAnalysisPage
from point_tracking import PointTrackingPage, VideoSelection
from ui_components import FrozenFirstColumnTable


class FrozenFirstColumnTableTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance() or QApplication([])

    def test_first_column_stays_fixed_and_vertical_scrolling_is_synchronized(self):
        table = FrozenFirstColumnTable()
        self.addCleanup(table.close)
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Trace", "Metric 1", "Metric 2", "Metric 3"])
        table.setRowCount(40)
        for row in range(table.rowCount()):
            table.setItem(row, 0, QTableWidgetItem(f"Trace {row + 1}"))
            for column in range(1, table.columnCount()):
                table.setItem(row, column, QTableWidgetItem(str(row * column)))
        table.setColumnWidth(0, 180)
        for column in range(1, table.columnCount()):
            table.setColumnWidth(column, 220)
        table.resize(440, 280)
        table.refresh_frozen_column()
        table.setSortingEnabled(True)
        table.show()
        self.app.processEvents()

        frozen_x = table.frozen_view.geometry().x()
        self.assertFalse(table.frozen_view.isColumnHidden(0))
        self.assertTrue(table.frozen_view.isColumnHidden(1))
        self.assertGreater(table.horizontalScrollBar().maximum(), 0)

        table.horizontalScrollBar().setValue(table.horizontalScrollBar().maximum())
        table.verticalScrollBar().setValue(table.verticalScrollBar().maximum())
        self.app.processEvents()

        self.assertEqual(table.frozen_view.geometry().x(), frozen_x)
        self.assertEqual(
            table.frozen_view.verticalScrollBar().value(),
            table.verticalScrollBar().value(),
        )
        previous_order = table.horizontalHeader().sortIndicatorOrder()
        table._sort_from_frozen_header(0)
        self.assertNotEqual(
            table.horizontalHeader().sortIndicatorOrder(),
            previous_order,
        )

    def test_signal_trace_rows_support_non_adjacent_ctrl_selection(self):
        page = SignalAnalysisPage()
        self.addCleanup(page.close)
        page.data = pd.DataFrame(
            {
                "time_s": np.arange(10) / 10,
                "A": np.arange(10),
                "B": np.arange(10),
                "C": np.arange(10),
            }
        )
        page.time_column = "time_s"
        page.populate_traces()
        rows = [
            page.trace_list.itemWidget(page.trace_list.item(index))
            for index in range(3)
        ]

        QTest.mouseClick(rows[0], Qt.LeftButton)
        QTest.mouseClick(rows[2], Qt.LeftButton, Qt.ControlModifier)
        self.assertEqual(page.selected_columns(), ["A", "C"])
        self.assertTrue(rows[0].property("selected"))
        self.assertFalse(rows[1].property("selected"))
        self.assertTrue(rows[2].property("selected"))
        self.assertIn("background: #653040", rows[0].styleSheet())
        self.assertEqual(
            rows[0].findChild(QLabel, "traceNameLabel").objectName(),
            "traceNameLabel",
        )

        QTest.mouseClick(rows[0], Qt.LeftButton, Qt.ControlModifier)
        self.assertEqual(page.selected_columns(), ["C"])
        self.assertFalse(rows[0].property("selected"))

    def test_signal_trace_drag_selects_from_top_to_bottom(self):
        page = SignalAnalysisPage()
        self.addCleanup(page.close)
        page.data = pd.DataFrame(
            {
                "time_s": np.arange(10) / 10,
                "A": np.arange(10),
                "B": np.arange(10),
                "C": np.arange(10),
                "D": np.arange(10),
            }
        )
        page.time_column = "time_s"
        page.populate_traces()
        first = page.trace_list.item(0)
        last = page.trace_list.item(3)

        page.select_trace_row(first, Qt.NoModifier)
        page.extend_trace_selection_to(last)

        self.assertEqual(page.selected_columns(), ["A", "B", "C", "D"])

    def test_ctrl_drag_adds_a_non_adjacent_trace_range(self):
        page = SignalAnalysisPage()
        self.addCleanup(page.close)
        page.data = pd.DataFrame(
            {
                "time_s": np.arange(10) / 10,
                "A": np.arange(10),
                "B": np.arange(10),
                "C": np.arange(10),
                "D": np.arange(10),
            }
        )
        page.time_column = "time_s"
        page.populate_traces()

        page.select_trace_row(page.trace_list.item(0), Qt.NoModifier)
        page.select_trace_row(page.trace_list.item(2), Qt.ControlModifier)
        page.extend_trace_selection_to(page.trace_list.item(3))

        self.assertEqual(page.selected_columns(), ["A", "C", "D"])

    def test_ctrl_mouse_drag_adds_multiple_trace_rows(self):
        page = SignalAnalysisPage()
        self.addCleanup(page.close)
        page.data = pd.DataFrame(
            {
                "time_s": np.arange(10) / 10,
                "A": np.arange(10),
                "B": np.arange(10),
                "C": np.arange(10),
                "D": np.arange(10),
            }
        )
        page.time_column = "time_s"
        page.populate_traces()
        page.resize(1100, 700)
        page.show()
        self.app.processEvents()
        rows = [
            page.trace_list.itemWidget(page.trace_list.item(index))
            for index in range(4)
        ]

        QTest.mouseClick(rows[0], Qt.LeftButton)
        target = rows[2].mapFromGlobal(rows[3].mapToGlobal(rows[3].rect().center()))
        QTest.mousePress(rows[2], Qt.LeftButton, Qt.ControlModifier)
        QTest.mouseMove(rows[2], target, 10)
        QTest.mouseRelease(rows[2], Qt.LeftButton, Qt.ControlModifier, target)

        self.assertEqual(page.selected_columns(), ["A", "C", "D"])

    def test_ctrl_drag_from_selected_trace_removes_that_range(self):
        page = SignalAnalysisPage()
        self.addCleanup(page.close)
        page.data = pd.DataFrame(
            {
                "time_s": np.arange(10) / 10,
                "A": np.arange(10),
                "B": np.arange(10),
                "C": np.arange(10),
                "D": np.arange(10),
            }
        )
        page.time_column = "time_s"
        page.populate_traces()
        for row in range(page.trace_list.count()):
            page.trace_list.item(row).setSelected(True)

        page.select_trace_row(page.trace_list.item(1), Qt.ControlModifier)
        page.extend_trace_selection_to(page.trace_list.item(2))

        self.assertEqual(page.selected_columns(), ["A", "D"])

    def test_inline_frame_arrow_shifts_its_own_trace_not_latest_selection(self):
        page = SignalAnalysisPage()
        self.addCleanup(page.close)
        page.data = pd.DataFrame(
            {
                "time_s": np.arange(10) / 10,
                "A": np.arange(10),
                "B": np.arange(10),
            }
        )
        page.time_column = "time_s"
        page.populate_traces()
        row_a = page.trace_list.itemWidget(page.trace_list.item(0))
        row_b = page.trace_list.itemWidget(page.trace_list.item(1))
        QTest.mouseClick(row_b, Qt.LeftButton)

        earlier = row_a.findChild(QPushButton, "traceShiftEarlierButton")
        self.assertIsNotNone(earlier)
        QTest.mouseClick(earlier, Qt.LeftButton)

        self.assertEqual(page.selected_columns(), ["B"])
        self.assertEqual(page.trace_processing["A"].time_shift_frames, -1)
        self.assertEqual(page.trace_processing.get("B").time_shift_frames if "B" in page.trace_processing else 0, 0)

    def test_signal_legend_controls_apply_to_current_trace_selection(self):
        page = SignalAnalysisPage()
        self.addCleanup(page.close)
        page.data = pd.DataFrame(
            {"time_s": np.arange(10) / 10, "A": np.arange(10), "B": np.arange(10)}
        )
        page.time_column = "time_s"
        page.populate_traces()
        page.resize(1100, 700)
        page.show()
        self.app.processEvents()
        self.assertGreater(
            page.legend_polarity_button.geometry().top(),
            page.mouse_controls_label.geometry().bottom(),
        )
        page.select_trace_row(page.trace_list.item(0), Qt.NoModifier)
        page.select_trace_row(page.trace_list.item(1), Qt.ControlModifier)

        page.legend_polarity_button.click()
        page.legend_shift_earlier_button.click()

        self.assertTrue(page.trace_processing["A"].invert)
        self.assertTrue(page.trace_processing["B"].invert)
        self.assertEqual(page.trace_processing["A"].time_shift_frames, -1)
        self.assertEqual(page.trace_processing["B"].time_shift_frames, -1)
        row = page.trace_list.itemWidget(page.trace_list.item(0))
        polarity_button = row.findChild(QPushButton, "traceInvertButton")
        self.assertEqual(polarity_button.text(), "")
        self.assertFalse(polarity_button.icon().isNull())
        for object_name in (
            "traceInvertButton",
            "traceShiftEarlierButton",
            "traceShiftLaterButton",
        ):
            button = row.findChild(QPushButton, object_name)
            expected_size = max(16, page.trace_list.fontMetrics().height() + 2)
            self.assertEqual(
                (button.width(), button.height()), (expected_size, expected_size)
            )
            self.assertIn("padding: 0", button.styleSheet())

    def test_smoothing_control_explains_savgol_window_and_order(self):
        page = SignalAnalysisPage()
        self.addCleanup(page.close)

        tooltip = page.smoothing.toolTip()
        self.assertIn("Savitzky–Golay", tooltip)
        self.assertIn("third-order", tooltip)
        self.assertIn("0", tooltip)

    def test_signal_processing_changes_preserve_plot_view(self):
        page = SignalAnalysisPage()
        self.addCleanup(page.close)
        time_values = np.linspace(0.0, 10.0, 501)
        page.data = pd.DataFrame(
            {"time_s": time_values, "Trace": np.sin(2 * np.pi * time_values)}
        )
        page.time_column = "time_s"
        page.populate_traces()
        page.trace_list.setCurrentRow(0)
        page.plot_timer.stop()
        page.plot_selected()
        page.axes.set_xlim(2.0, 4.0)
        page.axes.set_ylim(-0.5, 0.5)
        expected_view = (page.axes.get_xlim(), page.axes.get_ylim())

        page.smoothing.setValue(0.2)
        page.plot_timer.stop()
        page.plot_selected()
        self.assertTrue(np.allclose(page.axes.get_xlim(), expected_view[0]))
        self.assertTrue(np.allclose(page.axes.get_ylim(), expected_view[1]))

        page.baseline_method.setCurrentIndex(
            page.baseline_method.findData("constant_percentile")
        )
        page.plot_timer.stop()
        page.plot_selected()
        self.assertTrue(np.allclose(page.axes.get_xlim(), expected_view[0]))
        self.assertTrue(np.allclose(page.axes.get_ylim(), expected_view[1]))

    def test_track_points_clear_button_removes_all_points_from_current_video(self):
        page = PointTrackingPage()
        self.addCleanup(page.close)
        selection = VideoSelection(
            path=Path("recording.avi"), fps=20.0, frame_count=10, width=100, height=100,
            points=[(10.0, 20.0), (30.0, 40.0)],
        )
        page.current_selection = lambda: selection
        page.clear_current_points()

        self.assertEqual(selection.points, [])
        self.assertFalse(page.clear_points_button.isHidden())

    def test_track_points_only_shows_controls_used_by_active_mode(self):
        page = PointTrackingPage()
        self.addCleanup(page.close)

        self.assertEqual(page.controls_panel.minimumWidth(), 360)
        self.assertEqual(page.controls_panel.maximumWidth(), 520)

        self.assertFalse(page.baseline.isHidden())
        self.assertTrue(page.reference_frame.isHidden())
        self.assertTrue(page.threshold.isHidden())
        self.assertTrue(page.fps_override.isHidden())

        page.measurement.setCurrentIndex(page.measurement.findData("relative"))
        self.assertTrue(page.baseline.isHidden())

        page.measurement.setCurrentIndex(page.measurement.findData("absolute"))
        page.baseline.setCurrentIndex(page.baseline.findData("selected_frame"))
        self.assertFalse(page.reference_frame.isHidden())

        page.strategy.setCurrentIndex(page.strategy.findData("automatic"))
        self.assertFalse(page.threshold.isHidden())
        self.assertTrue(page.clear_points_button.isHidden())

        page.strategy.setCurrentIndex(page.strategy.findData("hybrid"))
        self.assertFalse(page.threshold.isHidden())
        self.assertFalse(page.clear_points_button.isHidden())
        self.assertFalse(page.clear_roi_button.isHidden())
        self.assertTrue(page.canvas.allow_points_with_roi)
        self.assertIn("top 20%", page.threshold.toolTip())


if __name__ == "__main__":
    unittest.main()
