"""Simple PyQt viewer for GSSI SIR-30 DZT files.

Features:
- File > Open menu for .dzt files
- DZT header parsing (nsamp, bits, channel count)
- 3 orthogonal slice views (XY, YZ, XZ)
- Axis mapping: X=channel, Y=scan(trace), Z=data index(depth)
"""

from __future__ import annotations

import os
import struct
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

try:
	from PyQt6.QtCore import Qt  # type: ignore[import-not-found]
	from PyQt6.QtGui import QAction, QColor, QImage, QPalette, QPixmap  # type: ignore[import-not-found]
	from PyQt6.QtWidgets import (  # type: ignore[import-not-found]
		QApplication,
		QFileDialog,
		QGridLayout,
		QGroupBox,
		QHBoxLayout,
		QLabel,
		QMainWindow,
		QMessageBox,
		QScrollArea,
		QSlider,
		QVBoxLayout,
		QWidget,
	)

	PYQT6 = True
except ImportError:
	from PyQt5.QtCore import Qt
	from PyQt5.QtGui import QColor, QImage, QPalette, QPixmap
	from PyQt5.QtWidgets import (
		QAction,
		QApplication,
		QFileDialog,
		QGridLayout,
		QGroupBox,
		QHBoxLayout,
		QLabel,
		QMainWindow,
		QMessageBox,
		QScrollArea,
		QSlider,
		QVBoxLayout,
		QWidget,
	)

	PYQT6 = False


@dataclass
class DztMeta:
	nsamp: int
	bits: int
	nchan: int
	traces: int
	data_offset: int


def _read_dzt_header(file_path: str) -> DztMeta:
	with open(file_path, "rb") as f:
		# Positions from public DZT references.
		rh_tag = struct.unpack("<h", f.read(2))[0]
		rh_data = struct.unpack("<h", f.read(2))[0]
		rh_nsamp = struct.unpack("<h", f.read(2))[0]
		rh_bits = struct.unpack("<h", f.read(2))[0]
		f.seek(52)
		rh_nchan = struct.unpack("<h", f.read(2))[0]

	if rh_tag <= 0 or rh_nsamp <= 0:
		raise ValueError("Invalid DZT header (tag/nsamp).")
	if rh_bits not in (8, 16, 32):
		raise ValueError(f"Unsupported sample size: {rh_bits} bits")
	if rh_nchan <= 0:
		rh_nchan = 1

	min_header = 1024
	if rh_data < min_header:
		data_offset = min_header * rh_data
	else:
		data_offset = min_header * rh_nchan

	file_size = os.path.getsize(file_path)
	bytes_per_sample = rh_bits // 8
	payload_bytes = file_size - data_offset
	if payload_bytes <= 0:
		raise ValueError("No payload data found in DZT file.")

	samples_per_trace_all_channels = rh_nsamp * rh_nchan
	trace_bytes = samples_per_trace_all_channels * bytes_per_sample
	if trace_bytes <= 0 or payload_bytes < trace_bytes:
		raise ValueError("Payload is too small for declared header values.")

	traces = payload_bytes // trace_bytes
	if traces <= 0:
		raise ValueError("Could not compute trace count from payload.")

	return DztMeta(
		nsamp=rh_nsamp,
		bits=rh_bits,
		nchan=rh_nchan,
		traces=traces,
		data_offset=data_offset,
	)


def _load_dzt_volume(file_path: str) -> tuple[np.ndarray, DztMeta]:
	meta = _read_dzt_header(file_path)

	dtype: np.dtype
	if meta.bits == 8:
		dtype = np.uint8
	elif meta.bits == 16:
		dtype = np.uint16
	else:
		dtype = np.int32

	raw = np.fromfile(file_path, dtype=dtype, offset=meta.data_offset)
	items_per_trace = meta.nsamp * meta.nchan
	total_items = meta.traces * items_per_trace
	raw = raw[:total_items]
	if raw.size != total_items:
		raise ValueError("Unexpected file truncation while reading payload.")

	traces_channels_samples = raw.reshape(meta.traces, meta.nchan, meta.nsamp)
	# Keep volume axis order as (Z, Y, X) while mapping axes as:
	# Z=sample(depth), Y=trace(scan), X=channel.
	volume = np.transpose(traces_channels_samples, (2, 0, 1))

	if meta.bits in (8, 16):
		volume = volume.astype(np.float32) - (2 ** meta.bits) / 2.0
	else:
		volume = volume.astype(np.float32)

	return volume, meta


def _normalize_to_u8(arr2d: np.ndarray) -> np.ndarray:
	arr = np.asarray(arr2d, dtype=np.float32)
	lo = float(np.percentile(arr, 1))
	hi = float(np.percentile(arr, 99))
	if hi <= lo:
		lo = float(np.min(arr))
		hi = float(np.max(arr))
	if hi <= lo:
		return np.zeros(arr.shape, dtype=np.uint8)
	scaled = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
	return (scaled * 255.0).astype(np.uint8)


def _array_to_qpixmap(arr2d: np.ndarray) -> QPixmap:
	img_u8 = _normalize_to_u8(arr2d)
	h, w = img_u8.shape
	bytes_per_line = w
	# Ensure array is C-contiguous and convert to bytes
	img_u8_c = np.ascontiguousarray(img_u8)
	qimg = QImage(img_u8_c.tobytes(), w, h, bytes_per_line, QImage.Format.Format_Grayscale8 if PYQT6 else QImage.Format_Grayscale8)
	qimg = qimg.copy()
	return QPixmap.fromImage(qimg)


def _map_index_to_display(index: int, src_len: int, dst_len: int) -> int:
	if src_len <= 1 or dst_len <= 1:
		return 0
	pos = int(round(((index + 0.5) * dst_len / float(src_len)) - 0.5))
	return int(np.clip(pos, 0, dst_len - 1))


def _array_to_qpixmap_with_crosshair(
	arr2d: np.ndarray,
	*,
	x_pos: int | None = None,
	y_pos: int | None = None,
	color: tuple[int, int, int] = (0, 255, 0),
) -> QPixmap:
	img_u8 = _normalize_to_u8(arr2d)
	rgb = np.stack((img_u8, img_u8, img_u8), axis=-1)
	h, w = img_u8.shape

	if x_pos is not None:
		x = int(np.clip(x_pos, 0, w - 1))
		rgb[:, x, :] = color
	if y_pos is not None:
		y = int(np.clip(y_pos, 0, h - 1))
		rgb[y, :, :] = color

	rgb_c = np.ascontiguousarray(rgb)
	bytes_per_line = w * 3
	fmt = QImage.Format.Format_RGB888 if PYQT6 else QImage.Format_RGB888
	qimg = QImage(rgb_c.tobytes(), w, h, bytes_per_line, fmt)
	qimg = qimg.copy()
	return QPixmap.fromImage(qimg)


def _expand_axis_for_display(arr2d: np.ndarray, axis: int, min_pixels: int = 120) -> np.ndarray:
	"""Expand very small dimensions (e.g. 7 channels) so slices remain visible."""
	size = arr2d.shape[axis]
	if size >= min_pixels or size <= 0:
		return arr2d
	repeat = int(np.ceil(min_pixels / float(size)))
	return np.repeat(arr2d, repeat, axis=axis)


def _resample_1d(arr1d: np.ndarray, target_len: int = 512) -> np.ndarray:
	arr = np.asarray(arr1d, dtype=np.float32).reshape(-1)
	if arr.size == 0:
		return np.zeros(target_len, dtype=np.float32)
	if arr.size == target_len:
		return arr
	if arr.size == 1:
		return np.repeat(arr, target_len)
	x_old = np.linspace(0.0, 1.0, arr.size, endpoint=True)
	x_new = np.linspace(0.0, 1.0, target_len, endpoint=True)
	return np.interp(x_new, x_old, arr)


def _ascan_to_qpixmap(arr1d: np.ndarray, width: int = 256, height: int = 512) -> QPixmap:
	wave = _resample_1d(arr1d, target_len=height)
	lo = float(np.percentile(wave, 1))
	hi = float(np.percentile(wave, 99))
	if hi <= lo:
		lo = float(np.min(wave))
		hi = float(np.max(wave))
	if hi <= lo:
		hi = lo + 1.0
	center_x = (width - 1) / 2.0
	span_x = max(1.0, (width - 8) / 2.0)
	norm = np.clip((wave - lo) / (hi - lo), 0.0, 1.0)
	xs = np.rint(center_x + (norm - 0.5) * 2.0 * span_x).astype(int)
	ys = np.arange(height, dtype=int)

	canvas = np.zeros((height, width), dtype=np.uint8)
	canvas[:, width // 2] = 48
	for idx in range(height - 1):
		x0 = int(np.clip(xs[idx], 0, width - 1))
		x1 = int(np.clip(xs[idx + 1], 0, width - 1))
		y0 = int(ys[idx])
		y1 = int(ys[idx + 1])
		steps = max(abs(x1 - x0), abs(y1 - y0)) + 1
		line_x = np.rint(np.linspace(x0, x1, steps)).astype(int)
		line_y = np.rint(np.linspace(y0, y1, steps)).astype(int)
		canvas[line_y, line_x] = 255
		if x0 + 1 < width:
			canvas[line_y, np.clip(line_x + 1, 0, width - 1)] = np.maximum(
				canvas[line_y, np.clip(line_x + 1, 0, width - 1)],
				200,
			)
		if x0 - 1 >= 0:
			canvas[line_y, np.clip(line_x - 1, 0, width - 1)] = np.maximum(
				canvas[line_y, np.clip(line_x - 1, 0, width - 1)],
				200,
			)

	return _array_to_qpixmap(canvas)


def _apply_designer_palette(app: QApplication, ui_path: str) -> None:
	"""Apply MainWindow palette from a Qt Designer .ui file, if available."""
	if not os.path.exists(ui_path):
		return

	try:
		root = ET.parse(ui_path).getroot()
	except ET.ParseError:
		return

	palette_elem = root.find(".//widget[@name='MainWindow']/property[@name='palette']/palette")
	if palette_elem is None:
		return

	base_palette = app.palette()

	if PYQT6:
		group_map = {
			"active": QPalette.ColorGroup.Active,
			"inactive": QPalette.ColorGroup.Inactive,
			"disabled": QPalette.ColorGroup.Disabled,
		}
	else:
		group_map = {
			"active": QPalette.Active,
			"inactive": QPalette.Inactive,
			"disabled": QPalette.Disabled,
		}

	for group_name, group_enum in group_map.items():
		group_elem = palette_elem.find(group_name)
		if group_elem is None:
			continue

		for colorrole in group_elem.findall("colorrole"):
			role_name = colorrole.get("role", "")
			color_elem = colorrole.find("./brush/color")
			if color_elem is None:
				continue

			role_enum = getattr(QPalette, role_name, None)
			if role_enum is None and PYQT6:
				role_enum = getattr(QPalette.ColorRole, role_name, None)
			if role_enum is None:
				continue

			try:
				r = int(color_elem.findtext("red", default="0"))
				g = int(color_elem.findtext("green", default="0"))
				b = int(color_elem.findtext("blue", default="0"))
				a = int(color_elem.get("alpha", "255"))
			except ValueError:
				continue

			base_palette.setColor(group_enum, role_enum, QColor(r, g, b, a))

	app.setPalette(base_palette)


class SlicePanel(QGroupBox):
	def __init__(self, title: str) -> None:
		super().__init__(title)
		self.keep_aspect = True
		self.image_label = QLabel("No data")
		self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter if PYQT6 else Qt.AlignCenter)
		self.image_label.setMinimumSize(260, 180)
		self.image_label.setStyleSheet("QLabel { background: #111; color: #ddd; border: 1px solid #555; }")
		self.scroll_area = QScrollArea()
		self.scroll_area.setWidgetResizable(False)
		self.scroll_area.setWidget(self.image_label)
		if PYQT6:
			self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
			self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
		else:
			self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
			self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

		self.info_label = QLabel("-")
		self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft if PYQT6 else Qt.AlignLeft)

		layout = QVBoxLayout()
		layout.addWidget(self.scroll_area, stretch=1)
		layout.addWidget(self.info_label)
		self.setLayout(layout)

	def set_view_size(self, width: int, height: int) -> None:
		self.image_label.setFixedSize(width, height)

	def set_scaling_mode(self, keep_aspect: bool) -> None:
		self.keep_aspect = keep_aspect

	def set_scroll_axes(self, scroll_x: bool, scroll_y: bool) -> None:
		if PYQT6:
			h_policy = Qt.ScrollBarPolicy.ScrollBarAsNeeded if scroll_x else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
			v_policy = Qt.ScrollBarPolicy.ScrollBarAsNeeded if scroll_y else Qt.ScrollBarPolicy.ScrollBarAlwaysOff
		else:
			h_policy = Qt.ScrollBarAsNeeded if scroll_x else Qt.ScrollBarAlwaysOff
			v_policy = Qt.ScrollBarAsNeeded if scroll_y else Qt.ScrollBarAlwaysOff
		self.scroll_area.setHorizontalScrollBarPolicy(h_policy)
		self.scroll_area.setVerticalScrollBarPolicy(v_policy)

	def set_image(self, arr2d: np.ndarray, text: str) -> None:
		pixmap = _array_to_qpixmap(arr2d)
		self.set_pixmap(pixmap, text)

	def set_pixmap(self, pixmap: QPixmap, text: str) -> None:
		aspect_mode = (
			Qt.AspectRatioMode.KeepAspectRatio if self.keep_aspect else Qt.AspectRatioMode.IgnoreAspectRatio
		) if PYQT6 else (
			Qt.KeepAspectRatio if self.keep_aspect else Qt.IgnoreAspectRatio
		)
		scaled = pixmap.scaled(
			self.image_label.size(),
			aspect_mode,
			Qt.TransformationMode.SmoothTransformation if PYQT6 else Qt.SmoothTransformation,
		)
		self.image_label.setPixmap(scaled)
		self.info_label.setText(text)

	def resizeEvent(self, event):  # type: ignore[override]
		super().resizeEvent(event)
		# Re-apply current pixmap to keep scaling responsive.
		current = self.image_label.pixmap()
		if current is not None:
			aspect_mode = (
				Qt.AspectRatioMode.KeepAspectRatio if self.keep_aspect else Qt.AspectRatioMode.IgnoreAspectRatio
			) if PYQT6 else (
				Qt.KeepAspectRatio if self.keep_aspect else Qt.IgnoreAspectRatio
			)
			scaled = current.scaled(
				self.image_label.size(),
				aspect_mode,
				Qt.TransformationMode.SmoothTransformation if PYQT6 else Qt.SmoothTransformation,
			)
			self.image_label.setPixmap(scaled)


class MainWindow(QMainWindow):
	def __init__(self, auto_load_file: str | None = None) -> None:
		super().__init__()
		self.setWindowTitle("GSSI SIR-30 DZT Viewer (X=Channel, Y=Trace, Z=Depth)")
		self.resize(1920, 1080)

		self.volume: np.ndarray | None = None  # shape: (Z, Y, X)
		self.meta: DztMeta | None = None

		self.x_idx = 0
		self.y_idx = 0
		self.z_idx = 0
		self.selected_channel_no = 0
		self.selected_scan_no = 0
		self.selected_data_index = 0
		
		self.auto_load_file = auto_load_file

		self.xy_panel = SlicePanel("XY (trace vs channel @ Z)")
		self.yz_panel = SlicePanel("YZ (depth vs trace @ X)")
		self.xz_panel = SlicePanel("XZ (depth vs channel @ Y)")
		self.ascan_panel = SlicePanel("A-Scan")
		self.xy_panel.set_view_size(1408, 504)
		self.yz_panel.set_view_size(1408, 504)
		self.xz_panel.set_view_size(256, 512)
		self.ascan_panel.set_view_size(256, 512)
		self.xz_panel.setMinimumSize(266, 562)
		self.xz_panel.setMaximumSize(266, 562)
		self.ascan_panel.setMinimumSize(266, 562)
		self.ascan_panel.setMaximumSize(266, 562)
		self.xy_panel.set_scaling_mode(keep_aspect=False)
		self.yz_panel.set_scaling_mode(keep_aspect=False)
		self.xz_panel.set_scaling_mode(keep_aspect=False)
		self.ascan_panel.set_scaling_mode(keep_aspect=False)
		self.xy_panel.set_scroll_axes(scroll_x=True, scroll_y=False)
		self.yz_panel.set_scroll_axes(scroll_x=True, scroll_y=False)
		self.xz_panel.set_scroll_axes(scroll_x=False, scroll_y=False)
		self.ascan_panel.set_scroll_axes(scroll_x=False, scroll_y=False)

		self.y_slider = QSlider(Qt.Orientation.Horizontal if PYQT6 else Qt.Horizontal)

		self.y_slider.valueChanged.connect(self._on_slider_change)
		self.xy_panel.image_label.mousePressEvent = self._on_xy_click  # type: ignore[assignment]
		self.yz_panel.image_label.mousePressEvent = self._on_yz_click  # type: ignore[assignment]
		self.xz_panel.image_label.mousePressEvent = self._on_xz_click  # type: ignore[assignment]
		self.xy_panel.scroll_area.horizontalScrollBar().valueChanged.connect(self._on_xy_scroll_changed)
		self.yz_panel.scroll_area.horizontalScrollBar().valueChanged.connect(self._on_yz_scroll_changed)


		self._build_ui()
		self._build_menu()
		self._set_sliders_enabled(False)
		
		# Auto-load file if provided
		if self.auto_load_file and os.path.exists(self.auto_load_file):
			self.load_file(self.auto_load_file)

	def _build_ui(self) -> None:
		central = QWidget()
		self.setCentralWidget(central)

		main_layout = QVBoxLayout()
		central.setLayout(main_layout)

		grid = QGridLayout()
		grid.setHorizontalSpacing(12)
		grid.setVerticalSpacing(12)
		grid.addWidget(self.xy_panel, 0, 0)
		grid.addWidget(self.yz_panel, 1, 0)
		grid.addWidget(self.xz_panel, 1, 1)
		grid.addWidget(self.ascan_panel, 1, 2)
		main_layout.addLayout(grid, stretch=1)

		# slider_layout = QVBoxLayout()

		# slider_layout.addLayout(slider_layout, self.y_slider)
		# main_layout.addLayout(slider_layout)
		

	def _slider_row(self, title: str, slider: QSlider) -> QHBoxLayout:
		row = QHBoxLayout()
		label = QLabel(title)
		label.setMinimumWidth(160)
		row.addWidget(label)
		row.addWidget(slider)
		return row

	def _build_menu(self) -> None:
		if PYQT6:
			open_action = QAction("Open...", self)
			open_action.setShortcut("Ctrl+O")
			open_action.triggered.connect(self.open_file)

			exit_action = QAction("Exit", self)
			exit_action.setShortcut("Ctrl+Q")
			exit_action.triggered.connect(self.close)

			file_menu = self.menuBar().addMenu("File")
			file_menu.addAction(open_action)
			file_menu.addSeparator()
			file_menu.addAction(exit_action)
		else:
			open_action = QAction("Open...", self)
			open_action.setShortcut("Ctrl+O")
			open_action.triggered.connect(self.open_file)

			exit_action = QAction("Exit", self)
			exit_action.setShortcut("Ctrl+Q")
			exit_action.triggered.connect(self.close)

			file_menu = self.menuBar().addMenu("File")
			file_menu.addAction(open_action)
			file_menu.addSeparator()
			file_menu.addAction(exit_action)

	def _set_sliders_enabled(self, enabled: bool) -> None:
		self.y_slider.setEnabled(enabled)


	def open_file(self) -> None:
		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Open GSSI DZT File",
			"",
			"GSSI files (*.dzt *.DZT);;All files (*.*)",
		)
		if not file_path:
			return
		
		self.load_file(file_path)
	
	def load_file(self, file_path: str) -> None:
		"""Load a DZT file and render the 3D volume."""
		try:
			volume, meta = _load_dzt_volume(file_path)
		except Exception as exc:  # pragma: no cover - UI path
			QMessageBox.critical(self, "Open Error", f"Could not load file:\n{exc}")
			return

		self.volume = volume
		self.meta = meta

		z_dim, y_dim, x_dim = self.volume.shape
		self.y_slider.setRange(0, max(0, y_dim - 1))

		# XY keeps scan width while matching channel height to the visible XY view.
		# xy_h = max(1, self.xy_panel.scroll_area.viewport().height()-20)
		self.xy_panel.set_view_size(y_dim, 320)
		# yz_h = max(1, self.yz_panel.scroll_area.viewport().height()-20)
		self.yz_panel.set_view_size(y_dim, 484)


		self.x_idx = x_dim // 2
		self.y_idx = y_dim // 2
		self.z_idx = z_dim // 2
		self.selected_channel_no = self.x_idx
		self.selected_scan_no = self.y_idx
		self.selected_data_index = self.z_idx

		self.y_slider.setValue(self.y_idx)

		self._set_sliders_enabled(True)
		self._render_views()

		self.statusBar().showMessage(
			f"Loaded: {os.path.basename(file_path)} | nsamp={meta.nsamp}, nchan={meta.nchan}, traces={meta.traces}",
			10000,
		)

	def resizeEvent(self, event):  # type: ignore[override]
		super().resizeEvent(event)
		if self.volume is None:
			return
		_, y_dim, _ = self.volume.shape
		xy_h = max(1, self.xy_panel.scroll_area.viewport().height())
		if self.xy_panel.image_label.height() != xy_h:
			self.xy_panel.set_view_size(y_dim, xy_h)
			self._render_views()

	def _on_slider_change(self) -> None:
		if self.volume is None:
			return
		self.y_idx = self.y_slider.value()
		self.selected_scan_no = self.y_idx

		self._render_views()

	def _on_xy_scroll_changed(self, value: int) -> None:
		yz_bar = self.yz_panel.scroll_area.horizontalScrollBar()
		target = int(np.clip(value, yz_bar.minimum(), yz_bar.maximum()))
		if yz_bar.value() != target:
			yz_bar.setValue(target)

	def _on_yz_scroll_changed(self, value: int) -> None:
		xy_bar = self.xy_panel.scroll_area.horizontalScrollBar()
		target = int(np.clip(value, xy_bar.minimum(), xy_bar.maximum()))
		if xy_bar.value() != target:
			xy_bar.setValue(target)

	def _on_xy_click(self, event) -> None:  # type: ignore[override]
		if self.volume is None:
			return

		if PYQT6:
			if event.button() != Qt.MouseButton.LeftButton:
				return
			pos = event.position()
			click_x = int(pos.x())
			click_y = int(pos.y())
		else:
			if event.button() != Qt.LeftButton:
				return
			pos = event.pos()
			click_x = int(pos.x())
			click_y = int(pos.y())

		label_w = max(1, self.xy_panel.image_label.width())
		label_h = max(1, self.xy_panel.image_label.height())
		click_x = int(np.clip(click_x, 0, label_w - 1))
		click_y = int(np.clip(click_y, 0, label_h - 1))

		_, y_dim, x_dim = self.volume.shape
		scan_no = int(np.clip(np.floor(click_x * y_dim / label_w), 0, y_dim - 1))
		channel_no = int(np.clip(np.floor(click_y * x_dim / label_h), 0, x_dim - 1))

		self.selected_scan_no = scan_no
		self.selected_channel_no = channel_no
		self.y_idx = scan_no
		self.x_idx = channel_no

		if self.y_slider.value() != scan_no:
			self.y_slider.setValue(scan_no)
		else:
			self._render_views()

	def _on_yz_click(self, event) -> None:  # type: ignore[override]
		if self.volume is None:
			return

		if PYQT6:
			if event.button() != Qt.MouseButton.LeftButton:
				return
			pos = event.position()
			click_x = int(pos.x())
			click_y = int(pos.y())
		else:
			if event.button() != Qt.LeftButton:
				return
			pos = event.pos()
			click_x = int(pos.x())
			click_y = int(pos.y())

		label_w = max(1, self.yz_panel.image_label.width())
		label_h = max(1, self.yz_panel.image_label.height())
		click_x = int(np.clip(click_x, 0, label_w - 1))
		click_y = int(np.clip(click_y, 0, label_h - 1))

		z_dim, y_dim, _ = self.volume.shape
		scan_no = int(np.clip(np.floor(click_x * y_dim / label_w), 0, y_dim - 1))
		data_idx = int(np.clip(np.floor(click_y * z_dim / label_h), 0, z_dim - 1))

		self.selected_scan_no = scan_no
		self.selected_data_index = data_idx
		self.y_idx = scan_no
		self.z_idx = data_idx

		if self.y_slider.value() != scan_no:
			self.y_slider.setValue(scan_no)
		else:
			self._render_views()

	def _on_xz_click(self, event) -> None:  # type: ignore[override]
		if self.volume is None:
			return

		if PYQT6:
			if event.button() != Qt.MouseButton.LeftButton:
				return
			pos = event.position()
			click_x = int(pos.x())
			click_y = int(pos.y())
		else:
			if event.button() != Qt.LeftButton:
				return
			pos = event.pos()
			click_x = int(pos.x())
			click_y = int(pos.y())

		label_w = max(1, self.xz_panel.image_label.width())
		label_h = max(1, self.xz_panel.image_label.height())
		click_x = int(np.clip(click_x, 0, label_w - 1))
		click_y = int(np.clip(click_y, 0, label_h - 1))

		z_dim, _, x_dim = self.volume.shape
		channel_no = int(np.clip(np.floor(click_x * x_dim / label_w), 0, x_dim - 1))
		data_idx = int(np.clip(np.floor(click_y * z_dim / label_h), 0, z_dim - 1))

		self.selected_channel_no = channel_no
		self.selected_data_index = data_idx
		self.x_idx = channel_no
		self.z_idx = data_idx

		self._render_views()

	def _render_views(self) -> None:
		if self.volume is None:
			return

		z_dim, y_dim, x_dim = self.volume.shape
		current_data_index = int(np.clip(self.selected_data_index, 0, z_dim - 1))
		self.z_idx = current_data_index

		# XY at selected Z: display as (channel, scan) so
		# Y-axis=channel(X), X-axis=scan number(Y).
		xy = self.volume[current_data_index, :, :].T
		yz = self.volume[:, :, self.x_idx]
		xz = self.volume[:, self.y_idx, :]
		ascan = self.volume[:, self.y_idx, self.x_idx]

		xy_min_pixels = max(1, self.xy_panel.scroll_area.viewport().height())
		xy_display = _expand_axis_for_display(xy, axis=0, min_pixels=xy_min_pixels)
		# YZ uses (Z, Y); expand Z (axis 0) when needed, not Y.
		yz_display = _expand_axis_for_display(yz, axis=0, min_pixels=120)
		xz_display = _expand_axis_for_display(xz, axis=1, min_pixels=120)

		xy_x = _map_index_to_display(self.selected_scan_no, y_dim, xy_display.shape[1])
		xy_y = _map_index_to_display(self.selected_channel_no, x_dim, xy_display.shape[0])
		yz_x = _map_index_to_display(self.selected_scan_no, y_dim, yz_display.shape[1])
		yz_y = _map_index_to_display(current_data_index, z_dim, yz_display.shape[0])
		xz_x = _map_index_to_display(self.selected_channel_no, x_dim, xz_display.shape[1])
		xz_y = _map_index_to_display(current_data_index, z_dim, xz_display.shape[0])

		self.xy_panel.set_pixmap(
			_array_to_qpixmap_with_crosshair(xy_display, x_pos=xy_x, y_pos=xy_y),
			f"XY: data_index={current_data_index}/{z_dim - 1}, shape={xy.shape}",
		)
		self.yz_panel.set_pixmap(
			_array_to_qpixmap_with_crosshair(yz_display, x_pos=yz_x, y_pos=yz_y),
			f"YZ: X={self.x_idx}/{x_dim - 1}, shape={yz.shape}",
		)
		self.xz_panel.set_pixmap(
			_array_to_qpixmap_with_crosshair(xz_display, x_pos=xz_x, y_pos=xz_y),
			f"XZ: Y={self.y_idx}/{y_dim - 1}, shape={xz.shape}",
		)
		self.ascan_panel.set_pixmap(
			_ascan_to_qpixmap(ascan),
			f"A-Scan: X={self.x_idx}/{x_dim - 1}, Y={self.y_idx}/{y_dim - 1}, samples={ascan.size}",
		)


def main() -> int:
	app = QApplication(sys.argv)
	_apply_designer_palette(app, os.path.join(os.path.dirname(__file__), "gssiviewer.ui"))
	
	# Get DZT file from command line if provided
	auto_load = None
	if len(sys.argv) > 1:
		auto_load = sys.argv[1]
	
	w = MainWindow(auto_load_file=auto_load)
	w.show()
	return app.exec()


if __name__ == "__main__":
	raise SystemExit(main())
