import customtkinter as ctk

try:
    from .editor_window import ConfigEditor
except ImportError:
    from editor_window import ConfigEditor


class ModernConfigEditor(ConfigEditor):
    def __init__(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        super().__init__(False)


if __name__ == "__main__":
    editor = ModernConfigEditor()
    editor.run()
