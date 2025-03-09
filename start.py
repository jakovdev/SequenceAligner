import sys
import platform


def main():
    try:
        import tkinter as tk
    except ImportError:
        os_name = platform.system().lower()
        print("Error: tkinter is not installed.")
        if os_name == "linux":
            print("Ubuntu/Debian: sudo apt-get install python3-tk")
            print("Fedora: sudo dnf install python3-tkinter")
            print("Arch: sudo pacman -S tk")
            print("WSL2: install tkinter and use an xserver for display")
        sys.exit(1)

    try:
        import importlib.util

        if importlib.util.find_spec("customtkinter"):
            only_tk = False
        else:
            only_tk = True
            print("Note: customtkinter not found, using basic interface")
            print("To use modern interface: pip install customtkinter")
            input("Press Enter to continue...")

        if sys.executable.endswith("python.exe"):
            import subprocess

            subprocess.Popen(
                [sys.executable.replace("python.exe", "pythonw.exe")] + sys.argv
            )
            sys.exit(0)

        if only_tk:
            from scripts.editor_window import ConfigEditor

            editor = ConfigEditor()
        else:
            from scripts.modern import ModernConfigEditor

            editor = ModernConfigEditor()

        editor.run()
    except ImportError as e:
        print(f"Error loading interface: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
