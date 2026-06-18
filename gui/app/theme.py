"""Modern dark theme for the application.

A single QSS stylesheet plus a tuned palette. Kept in one place so the look of
the whole app can be changed without touching widget code.
"""
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication

# Palette ---------------------------------------------------------------------
BG = "#1b1d29"          # window background
SURFACE = "#252838"     # cards / group boxes
SURFACE_2 = "#2e3145"   # inputs
BORDER = "#3a3e57"
TEXT = "#e7e9f3"
TEXT_DIM = "#9aa0bd"
ACCENT = "#6c8cff"
ACCENT_HOVER = "#809bff"
ACCENT_PRESSED = "#5878f0"
DANGER = "#ff6b81"
OK = "#3ddc97"

_STYLESHEET = f"""
* {{
    font-size: 13px;
    color: {TEXT};
}}

QMainWindow, QWidget {{
    background-color: {BG};
}}

QToolTip {{
    background-color: {SURFACE_2};
    color: {TEXT};
    border: 1px solid {BORDER};
    padding: 5px 7px;
    border-radius: 6px;
}}

/* Tabs ------------------------------------------------------------------ */
QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    top: -1px;
    background: {BG};
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    padding: 9px 18px;
    margin-right: 4px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    font-weight: 600;
}}
QTabBar::tab:selected {{
    background: {SURFACE};
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:hover:!selected {{
    color: {TEXT};
}}

/* Group boxes (cards) --------------------------------------------------- */
QGroupBox {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 12px;
    margin-top: 16px;
    padding: 14px 14px 12px 14px;
    font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    padding: 2px 8px;
    color: {ACCENT};
    background: transparent;
}}
QGroupBox::indicator {{
    width: 18px; height: 18px;
}}

/* Inputs ---------------------------------------------------------------- */
QLineEdit, QComboBox, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 6px 9px;
    selection-background-color: {ACCENT};
    selection-color: #0d0f18;
}}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {{
    border: 1px solid {ACCENT};
}}
QLineEdit:disabled, QComboBox:disabled {{
    color: {TEXT_DIM};
    background-color: #22243200;
}}
QLineEdit[invalid="true"] {{
    border: 1px solid {DANGER};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox QAbstractItemView {{
    background-color: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    selection-background-color: {ACCENT};
    selection-color: #0d0f18;
    outline: none;
}}

/* Buttons --------------------------------------------------------------- */
QPushButton {{
    background-color: {SURFACE_2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    padding: 7px 16px;
    font-weight: 600;
}}
QPushButton:hover {{ border: 1px solid {ACCENT}; }}
QPushButton:pressed {{ background-color: {SURFACE}; }}
QPushButton:disabled {{ color: {TEXT_DIM}; border-color: {BORDER}; }}

QPushButton[accent="true"] {{
    background-color: {ACCENT};
    color: #0d0f18;
    border: none;
}}
QPushButton[accent="true"]:hover {{ background-color: {ACCENT_HOVER}; }}
QPushButton[accent="true"]:pressed {{ background-color: {ACCENT_PRESSED}; }}

QPushButton[danger="true"] {{
    background-color: transparent;
    color: {DANGER};
    border: 1px solid {DANGER};
}}
QPushButton[danger="true"]:hover {{ background-color: rgba(255,107,129,0.12); }}

/* Tables ---------------------------------------------------------------- */
QTableWidget {{
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 10px;
    gridline-color: {BORDER};
    selection-background-color: rgba(108,140,255,0.25);
    selection-color: {TEXT};
}}
QTableWidget::item {{ padding: 4px 6px; }}
QHeaderView::section {{
    background-color: {SURFACE_2};
    color: {TEXT_DIM};
    padding: 7px 8px;
    border: none;
    border-right: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    font-weight: 700;
}}
QTableCornerButton::section {{
    background-color: {SURFACE_2};
    border: none;
}}

/* Checkboxes ------------------------------------------------------------ */
QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{
    width: 18px; height: 18px;
    border: 1px solid {BORDER};
    border-radius: 5px;
    background: {SURFACE_2};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border: 1px solid {ACCENT};
}}

/* Scrollbars ------------------------------------------------------------ */
QScrollBar:vertical {{
    background: transparent; width: 12px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER}; border-radius: 6px; min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {TEXT_DIM}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 12px; margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: {BORDER}; border-radius: 6px; min-width: 30px;
}}

/* Toolbar / status / menu ---------------------------------------------- */
QToolBar {{
    background: {SURFACE};
    border: none;
    border-bottom: 1px solid {BORDER};
    padding: 6px 8px;
    spacing: 6px;
}}
QStatusBar {{
    background: {SURFACE};
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
}}
QLabel[hint="true"] {{ color: {TEXT_DIM}; }}
QLabel[h1="true"] {{ font-size: 16px; font-weight: 800; }}
"""


def apply_theme(app: QApplication):
    app.setStyle("Fusion")

    pal = QPalette()
    pal.setColor(QPalette.Window, QColor(BG))
    pal.setColor(QPalette.WindowText, QColor(TEXT))
    pal.setColor(QPalette.Base, QColor(SURFACE_2))
    pal.setColor(QPalette.AlternateBase, QColor(SURFACE))
    pal.setColor(QPalette.Text, QColor(TEXT))
    pal.setColor(QPalette.Button, QColor(SURFACE_2))
    pal.setColor(QPalette.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.HighlightedText, QColor("#0d0f18"))
    pal.setColor(QPalette.ToolTipBase, QColor(SURFACE_2))
    pal.setColor(QPalette.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.PlaceholderText, QColor(TEXT_DIM))
    app.setPalette(pal)

    font = QFont("Segoe UI", 10)
    app.setFont(font)

    app.setStyleSheet(_STYLESHEET)
