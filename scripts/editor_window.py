import platform, sys
from pathlib import Path
from tkinter import ttk, filedialog, messagebox
import tkinter as tk
import logging
from datetime import datetime
import csv
import os
import subprocess

try:
    from .build_system import DISPLAY_BINARY_PROFILE, build_env
    from .config_schema import (
        DISPLAY_NAMES,
        TOOLTIPS,
        DEFAULT_VALUES,
        DEFAULT_CHECKBOXES,
        validate_config,
        save_config,
    )
except ImportError:
    from build_system import DISPLAY_BINARY_PROFILE, build_env
    from config_schema import (
        DISPLAY_NAMES,
        TOOLTIPS,
        DEFAULT_VALUES,
        DEFAULT_CHECKBOXES,
        validate_config,
        save_config,
    )

try:
    import customtkinter as ctk
except ImportError:
    ctk = None


class ConfigEditor:
    def __init__(self, only_tk=True):
        self.only_tk = only_tk
        self.widgets = {
            k: (getattr(ttk, v[0]) if only_tk else getattr(ctk, v[1]))
            for k, v in {
                "frame": ("Frame", "CTkFrame"),
                "label": ("Label", "CTkLabel"),
                "entry": ("Entry", "CTkEntry"),
                "button": ("Button", "CTkButton"),
                "checkbox": ("Checkbutton", "CTkCheckBox"),
            }.items()
        }
        self.similarity_fields = [
            "WRITE_CSV_MATCHES_POS",
            "WRITE_CSV_MISMATCHES_POS",
            "WRITE_CSV_GAPS_POS",
            "WRITE_CSV_SIMILARITY_POS",
        ]
        self.CURRENT_MAPPINGS = {}
        field_to_key = {
            "WRITE_CSV_SEQ1_POS": "First Seq",
            "WRITE_CSV_SCORE_POS": "Score",
            "WRITE_CSV_ALIGN_POS": "Alignment",
            "WRITE_CSV_MATCHES_POS": "Matches",
            "WRITE_CSV_MISMATCHES_POS": "Mismatches",
            "WRITE_CSV_GAPS_POS": "Gaps",
            "WRITE_CSV_SIMILARITY_POS": "Similarity",
        }

        for field, key in field_to_key.items():
            value = int(DEFAULT_VALUES[field])
            if value >= 0:
                self.CURRENT_MAPPINGS[key] = value
                if key == "First Seq":  # Add Second Seq automatically
                    self.CURRENT_MAPPINGS["Second Seq"] = value + 1

        self.widgets["text_output"] = (
            tk.Text if only_tk else lambda *a, **k: ctk.CTkTextbox(*a, wrap="word", **k)
        )
        self.fields = {}
        self.checkboxes = {}
        self.input_csv_frame = None
        self.output_csv_frame = None
        self.root = (tk.Tk if only_tk else ctk.CTk)()
        self.root.title("Sequence Aligner Configuration")
        self.show_backend = tk.BooleanVar(value=False)
        self._init_logging()
        self._init_ui()
        self._update_csv_info(DEFAULT_VALUES["INPUT_FILE"], initial_load=True)
        build_env.init(self.update_output)

    def _create_scrollable(self, parent):
        if not self.only_tk:
            frame = ctk.CTkScrollableFrame(parent, fg_color="transparent")
            frame.pack(fill="both", expand=True, padx=10, pady=10)
            return frame

        canvas = tk.Canvas(container := ttk.Frame(parent))
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        frame = ttk.Frame(canvas)
        frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind_all(
            (
                "<MouseWheel>"
                if platform.system().lower() == "windows"
                else "<Button-4><Button-5>"
            ),
            lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"),
        )
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        container.pack(fill="both", expand=True)
        return frame

    def _read_csv_preview(self, file_path):
        try:
            with open(file_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                header = next(reader)
                first_row = next(reader)
                return header, first_row
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV file: {e}")
            return [], []

    def _create_csv_preview(self, parent, header, first_row, is_input=True):
        if is_input:
            if not self.input_csv_frame:
                self.input_csv_frame = self.widgets["frame"](parent)
                self.input_csv_frame.pack(fill="x", pady=5)
            else:
                for widget in self.input_csv_frame.winfo_children():
                    widget.destroy()
            frame = self.input_csv_frame

            for col_num, col_name in enumerate(header):
                label_args = {"text": col_name}
                if self.only_tk:
                    label_args.update(
                        {"relief": "raised", "borderwidth": 1, "cursor": "hand2"}
                    )
                    label = self.widgets["label"](self.input_csv_frame, **label_args)
                    label.grid(row=0, column=col_num, sticky="nsew")
                else:
                    label_args.update(
                        {"fg_color": "gray25", "cursor": "hand2", "padx": 5, "pady": 5}
                    )
                    label = self.widgets["label"](self.input_csv_frame, **label_args)
                    label.grid(row=0, column=col_num, sticky="nsew")
                label.bind(
                    "<Button-1>",
                    lambda e, col=col_num: self._set_csv_column(
                        col, "READ_CSV_SEQ_POS"
                    ),
                )

            for col_num, cell in enumerate(first_row):
                label_args = {"text": cell}
                if self.only_tk:
                    label_args.update({"relief": "sunken", "borderwidth": 1})
                    label = self.widgets["label"](self.input_csv_frame, **label_args)
                    label.grid(row=1, column=col_num, sticky="nsew")
                else:
                    label = self.widgets["label"](self.input_csv_frame, **label_args)
                    label.grid(row=1, column=col_num, sticky="nsew")

            for col_num in range(len(header)):
                self.input_csv_frame.grid_columnconfigure(col_num, weight=1)

        else:
            if not self.output_csv_frame:
                self.output_csv_frame = self.widgets["frame"](parent)
                self.output_csv_frame.pack(fill="x", pady=5)
            else:
                for widget in self.output_csv_frame.winfo_children():
                    widget.destroy()
            frame = self.output_csv_frame
            for col_num, col_name in enumerate(header):
                var = tk.StringVar()
                var.set(col_name)
                var.trace_add("write", lambda *args: self._update_write_header())
                entry = self.widgets["entry"](self.output_csv_frame, textvariable=var)
                entry.grid(row=0, column=col_num, sticky="nsew")
                text = next(
                    (k for k, v in self.CURRENT_MAPPINGS.items() if v == col_num), "Set"
                )
                text = (
                    "Second Seq"
                    if col_num == self.CURRENT_MAPPINGS["First Seq"] + 1
                    else text
                )
                btn = self.widgets["button"](
                    self.output_csv_frame,
                    text=text,
                    command=lambda c=col_num: self._choose_special_position(c),
                )
                btn.grid(row=1, column=col_num, sticky="nsew")

            for col_num in range(len(header)):
                self.output_csv_frame.grid_columnconfigure(col_num, weight=1)

    def _update_write_header(self):
        header = ",".join(
            [
                entry.get()
                for entry in self.output_csv_frame.winfo_children()
                if isinstance(entry, self.widgets["entry"])
            ]
        )
        self.fields["WRITE_CSV_HEADER"].delete(0, "end")
        self.fields["WRITE_CSV_HEADER"].insert(0, header)

    def _update_csv_info(self, file_path, is_input=True, initial_load=True):
        if is_input:
            if not initial_load:
                self.CURRENT_MAPPINGS.clear()
                for key in [
                    "WRITE_CSV_SEQ1_POS",
                    "WRITE_CSV_SCORE_POS",
                    "WRITE_CSV_ALIGN_POS",
                    "WRITE_CSV_MATCHES_POS",
                    "WRITE_CSV_MISMATCHES_POS",
                    "WRITE_CSV_GAPS_POS",
                    "WRITE_CSV_SIMILARITY_POS",
                ]:
                    self.fields[key].delete(0, "end")
                    self.fields[key].insert(0, "-1")

            header, first_row = self._read_csv_preview(file_path)
            if header:
                self.fields["READ_CSV_COLS"].delete(0, "end")
                self.fields["READ_CSV_COLS"].insert(0, str(len(header)))
                self.fields["READ_CSV_HEADER"].delete(0, "end")
                self.fields["READ_CSV_HEADER"].insert(0, ",".join(header))
                self._create_csv_preview(self.input_csv_frame, header, first_row)
                self._update_seq_col_info(header)

                num_cols = 2 * len(header) + 2
                if self.checkboxes["SIMILARITY_ANALYSIS"].get():
                    num_cols += 4

                new_header = [""] * num_cols

                if hasattr(self, "output_csv_frame"):
                    old_header = [
                        entry.get()
                        for entry in self.output_csv_frame.winfo_children()
                        if isinstance(entry, self.widgets["entry"])
                    ]
                    for i in range(min(len(new_header), len(old_header))):
                        new_header[i] = old_header[i]

                self._create_csv_preview(
                    self.output_csv_frame, new_header, [], is_input=False
                )
                self._update_write_header()

    def _choose_special_position(self, col_num):
        popup = tk.Toplevel(self.root)
        popup.title("Choose Special Position")
        popup.geometry(
            "300x400+{}+{}".format(self.root.winfo_x() + 50, self.root.winfo_y() + 50)
        )
        popup.resizable(False, False)
        frame = self.widgets["frame"](popup)
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        special_keys = [
            "WRITE_CSV_SEQ1_POS",
            "WRITE_CSV_SCORE_POS",
            "WRITE_CSV_ALIGN_POS",
        ]
        if self.checkboxes["SIMILARITY_ANALYSIS"].get():
            special_keys.extend(self.similarity_fields)
        special_keys.extend(["UNSET", "NONE"])

        for special_key in special_keys:
            if special_key == "NONE":
                btn = self.widgets["button"](
                    frame, text="Cancel", command=popup.destroy
                )
                btn.pack(fill="x", pady=5)
            elif special_key == "UNSET":
                btn = self.widgets["button"](
                    frame,
                    text="Unset",
                    command=lambda c=col_num: self._unset_csv_column(c, popup),
                )
                btn.pack(fill="x", pady=5)
            else:
                btn = self.widgets["button"](
                    frame,
                    text=DISPLAY_NAMES[special_key],
                    command=lambda c=col_num, k=special_key: self._set_csv_column(
                        c, k, popup
                    ),
                )
                btn.pack(fill="x", pady=5)

    def _unset_csv_column(self, col_num, parent=None):
        for k, v in list(self.CURRENT_MAPPINGS.items()):
            if v == col_num:
                self.CURRENT_MAPPINGS.pop(k)
                for field_key, map_key in [
                    ("WRITE_CSV_SEQ1_POS", "First Seq"),
                    ("WRITE_CSV_SCORE_POS", "Score"),
                    ("WRITE_CSV_ALIGN_POS", "Alignment"),
                    ("WRITE_CSV_MATCHES_POS", "Matches"),
                    ("WRITE_CSV_MISMATCHES_POS", "Mismatches"),
                    ("WRITE_CSV_GAPS_POS", "Gaps"),
                    ("WRITE_CSV_SIMILARITY_POS", "Similarity"),
                ]:
                    if map_key == k:
                        self.fields[field_key].delete(0, "end")
                        self.fields[field_key].insert(0, "-1")
                        break

        header = [
            entry.get()
            for entry in self.output_csv_frame.winfo_children()
            if isinstance(entry, self.widgets["entry"])
        ]
        self._create_csv_preview(self.output_csv_frame, header, [], is_input=False)

        if parent:
            parent.destroy()

    def _set_csv_column(self, col_num, key, parent=None):
        mapping_key = {
            "WRITE_CSV_SEQ1_POS": "First Seq",
            "WRITE_CSV_SCORE_POS": "Score",
            "WRITE_CSV_ALIGN_POS": "Alignment",
            "WRITE_CSV_MATCHES_POS": "Matches",
            "WRITE_CSV_MISMATCHES_POS": "Mismatches",
            "WRITE_CSV_GAPS_POS": "Gaps",
            "WRITE_CSV_SIMILARITY_POS": "Similarity",
        }.get(key)

        if mapping_key:
            for k, v in list(self.CURRENT_MAPPINGS.items()):
                if v == col_num or v == col_num + 1:
                    self.CURRENT_MAPPINGS.pop(k)
                    for field_key, map_key in [
                        ("WRITE_CSV_SEQ1_POS", "First Seq"),
                        ("WRITE_CSV_SCORE_POS", "Score"),
                        ("WRITE_CSV_ALIGN_POS", "Alignment"),
                        ("WRITE_CSV_MATCHES_POS", "Matches"),
                        ("WRITE_CSV_MISMATCHES_POS", "Mismatches"),
                        ("WRITE_CSV_GAPS_POS", "Gaps"),
                        ("WRITE_CSV_SIMILARITY_POS", "Similarity"),
                    ]:
                        if map_key == k:
                            self.fields[field_key].delete(0, "end")
                            self.fields[field_key].insert(0, "-1")

            if mapping_key in self.CURRENT_MAPPINGS:
                old_pos = self.CURRENT_MAPPINGS[mapping_key]
                self.CURRENT_MAPPINGS.pop(mapping_key)
                if mapping_key == "First Seq":
                    if (
                        "Second Seq" in self.CURRENT_MAPPINGS
                        and self.CURRENT_MAPPINGS["Second Seq"] == old_pos + 1
                    ):
                        self.CURRENT_MAPPINGS.pop("Second Seq")

            self.fields[key].delete(0, "end")
            self.fields[key].insert(0, str(col_num))
            self.CURRENT_MAPPINGS[mapping_key] = col_num

            if mapping_key == "First Seq":
                self.CURRENT_MAPPINGS["Second Seq"] = col_num + 1

            header = [
                entry.get()
                for entry in self.output_csv_frame.winfo_children()
                if isinstance(entry, self.widgets["entry"])
            ]
            self._create_csv_preview(self.output_csv_frame, header, [], is_input=False)

        if key == "READ_CSV_SEQ_POS":
            self.fields[key].delete(0, "end")
            self.fields[key].insert(0, str(col_num))

            header = []
            for widget in self.input_csv_frame.winfo_children():
                if (
                    isinstance(widget, self.widgets["label"])
                    and str(widget.grid_info().get("row", "")) == "0"
                ):
                    try:
                        header.append(widget.cget("text"))
                    except:
                        continue
            self._update_seq_col_info(header)

        if parent:
            parent.destroy()

    def _init_ui(self):
        self.content_frame = self._create_scrollable(self.root)
        frame = self.widgets["frame"](self.content_frame)
        frame.pack(fill="x", pady=5)

        bf = self.widgets["frame"](frame)
        bf.pack(fill="x", pady=5)
        for display, binary, profile in DISPLAY_BINARY_PROFILE:
            btn = self.widgets["button"](
                bf,
                text=display,
                command=lambda b=binary, p=profile: build_env.build_and_run(
                    self.update_output, b, p
                ),
            )
            btn.pack(side="left", padx=5)
            build_env.register_button_callback(
                lambda en, b=btn: b.configure(state="normal" if en else "disabled")
            )

        view_results_btn = self.widgets["button"](
            bf, text="View Results", command=self._open_output_file
        )
        view_results_btn.pack(side="left", padx=5)

        self.output_text = self.widgets["text_output"](
            frame, height=10 if self.only_tk else 200
        )
        self.output_text.pack(fill="both", expand=True, pady=5)

        for title, fields in {
            "Size Limits": [
                k for k in ("MAX_CSV_LINE", "MAX_SEQ_LEN", "GAP_PENALTY", "BATCH_SIZE")
            ],
            "CSV Format": [
                k
                for k in DEFAULT_VALUES
                if "CSV" in k and not k.endswith("_FILE") and k != "MAX_CSV_LINE"
            ],
            "File Paths": [k for k in DEFAULT_VALUES if k.endswith("_FILE")],
        }.items():
            self._create_section(frame, title, fields)

        options = self.widgets["frame"](self.content_frame)
        options.pack(fill="x", pady=5)
        self.widgets["label"](options, text="Options").pack()

        for key, default in DEFAULT_CHECKBOXES.items():
            var = tk.BooleanVar(value=default)
            cb = self.widgets["checkbox"](
                options, text=DISPLAY_NAMES[key], variable=var
            )
            cb.pack(fill="x", pady=2)
            self.checkboxes[key], tooltip = var, TOOLTIPS[key]
            cb.bind("<Enter>", lambda e, w=cb, t=tooltip: self._show_tooltip(w, t))
            cb.bind("<Leave>", lambda e, w=cb: self._hide_tooltip(w))

            if key == "SIMILARITY_ANALYSIS":
                var.trace_add("write", lambda *args: self._toggle_similarity_fields())

        cb = self.widgets["checkbox"](
            options, text="Show Backend Configuration", variable=self.show_backend
        )
        cb.pack(fill="x", pady=2)
        self.show_backend.trace_add(
            "write", lambda *args: self._toggle_backend_fields()
        )

        self.root.after(100, self._toggle_backend_fields)

        actions = self.widgets["frame"](self.content_frame)
        actions.pack(fill="x", pady=5)
        self.save_button = self.widgets["button"](
            actions, text="Save", command=self.save
        )
        self.save_button.pack(side="left", padx=5)
        build_env.register_button_callback(
            lambda en: self.save_button.configure(state="normal" if en else "disabled")
        )

        for text, cmd in [("Reset", self.reset_defaults), ("Exit", self.root.quit)]:
            self.widgets["button"](actions, text=text, command=cmd).pack(
                side="left", padx=5
            )

        self.root.minsize(1000, 800)
        self.root.geometry(
            f"800x600+{(self.root.winfo_screenwidth()-800)//2}+{(self.root.winfo_screenheight()-600)//2}"
        )

        ok, error = save_config(self.fields, self.checkboxes)
        if not ok:
            messagebox.showerror("Error", error)
            return

    def _create_alignment_preview(self, parent):
        preview_frame = self.widgets["frame"](parent)
        preview_frame.pack(fill="x", pady=5)

        label = self.widgets["label"](
            preview_frame,
            text="Alignment Format Preview | Edit the format to see how it will appear in the output",
        )
        label.pack(fill="x", pady=5)

        example_seq = "ATGCGT"
        format_frame = self.widgets["frame"](preview_frame)
        format_frame.pack(fill="x")

        self.format_parts = []
        for i in range(3):
            entry = self.widgets["entry"](format_frame, width=10)
            entry.grid(row=0, column=i * 2, sticky="ew", padx=2)
            self.format_parts.append(entry)

            if i < 2:
                seq_label = self.widgets["label"](format_frame, text=example_seq)
                seq_label.grid(row=0, column=i * 2 + 1, padx=5)

        for i in range(5):
            format_frame.grid_columnconfigure(i, weight=1)

        self.preview_label = self.widgets["label"](preview_frame, text="")
        self.preview_label.pack(fill="x", pady=(10, 0))

        current_format = DEFAULT_VALUES["WRITE_CSV_ALIGN_FMT"]
        parts = current_format.split("%s")

        for i, part in enumerate(parts):
            self.format_parts[i].insert(0, part)

        for entry in self.format_parts:
            entry.bind("<KeyRelease>", self._update_alignment_format)

        self._update_preview()

    def _update_alignment_format(self, event=None):
        parts = [entry.get() for entry in self.format_parts]
        self._update_preview()
        format_str = f"{parts[0]}%s{parts[1]}%s{parts[2]}"

        if "WRITE_CSV_ALIGN_FMT" in self.fields:
            self.fields["WRITE_CSV_ALIGN_FMT"].delete(0, "end")
            self.fields["WRITE_CSV_ALIGN_FMT"].insert(0, format_str)

    def _update_preview(self):
        example_seq = "ATGCGT"
        parts = [entry.get() for entry in self.format_parts]
        preview = f"{parts[0]}{example_seq}{parts[1]}{example_seq}{parts[2]}"
        self.preview_label.configure(text=f"Preview in CSV: {preview}")

    def _update_seq_col_info(self, header):
        if hasattr(self, "seq_col_info"):
            try:
                seq_pos = int(self.fields["READ_CSV_SEQ_POS"].get())
                if 0 <= seq_pos < len(header):
                    self.seq_col_info.configure(
                        text=f"Sequence column name: {header[seq_pos]}, column number: {seq_pos + 1}"
                    )
                else:
                    self.seq_col_info.configure(text="Invalid sequence column position")
            except (ValueError, AttributeError):
                self.seq_col_info.configure(text="Invalid sequence column position")

    def _create_section(self, parent, title, fields):
        section = (
            ttk.LabelFrame(parent, padding=10, text=title)
            if self.only_tk
            else (
                lambda f: (
                    f.pack(fill="x", padx=10, pady=5),
                    self.widgets["label"](f, text=title).pack(
                        anchor="w", padx=5, pady=(5, 0)
                    ),
                    self.widgets["frame"](f),
                )[-1]
            )(self.widgets["frame"](parent, fg_color="transparent"))
        )
        section.pack(fill="x", pady=5)

        if title == "CSV Format":
            self.backend_section = section

            label = self.widgets["label"](
                section,
                text="Input CSV Preview | Click on a column to set it as the sequence column",
            )
            label.pack(fill="x", pady=5)
            header, first_row = self._read_csv_preview(DEFAULT_VALUES["INPUT_FILE"])
            self._create_csv_preview(section, header, first_row)

            self.seq_col_info = self.widgets["label"](section, text="")
            self.seq_col_info.pack(fill="x", pady=5)

            label = self.widgets["label"](
                section,
                text="Output CSV Format | Edit header and set special positions",
            )
            label.pack(fill="x", pady=5)
            output_header = DEFAULT_VALUES["WRITE_CSV_HEADER"].split(",")
            self._create_csv_preview(section, output_header, [], is_input=False)

            self._create_alignment_preview(section)

            self.backend_label = self.widgets["label"](
                section,
                text="Back-end Configuration | See the results of the above preview here",
            )
            self.backend_container = self.widgets["frame"](section)
            self.backend_label.pack(fill="x", pady=5)
            self.backend_container.pack(fill="x", pady=5)

        container = self.backend_container if title == "CSV Format" else section

        for key in fields:
            row = self.widgets["frame"](container)
            row.pack(fill="x", pady=2)
            label = self.widgets["label"](
                row, text=f"{DISPLAY_NAMES[key]}:", width=25 if self.only_tk else 200
            )
            label.pack(side="left")
            entry = self.widgets["entry"](row)
            entry.insert(0, DEFAULT_VALUES[key])
            entry.pack(side="left", fill="x", expand=True, padx=5)

            if key in self.similarity_fields:
                row.similarity_row = True

            if key.endswith("_FILE"):
                self.widgets["button"](
                    row,
                    text="Browse",
                    width=None if self.only_tk else 100,
                    command=lambda e=entry, i=key.startswith(
                        "INPUT"
                    ): self._browse_file(e, i),
                ).pack(side="left", padx=5)

            self.fields[key] = entry
            if key in TOOLTIPS:
                t = TOOLTIPS[key]
                for w in (label, entry):
                    w.bind("<Enter>", lambda e, w=w, t=t: self._show_tooltip(w, t))
                    w.bind("<Leave>", lambda e, w=w: self._hide_tooltip(w))

    def _toggle_backend_fields(self):
        show = self.show_backend.get()
        if hasattr(self, "backend_container"):
            if show:
                self.backend_label.pack(fill="x", pady=5)
                self.backend_container.pack(fill="x", pady=5)
            else:
                self.backend_label.pack_forget()
                self.backend_container.pack_forget()

    def _toggle_similarity_fields(self):
        enabled = self.checkboxes["SIMILARITY_ANALYSIS"].get()

        for widget in self.content_frame.winfo_children():
            if isinstance(widget, (ttk.Frame, self.widgets["frame"])):
                for child in widget.winfo_children():
                    if hasattr(child, "similarity_row"):
                        if enabled:
                            child.pack()
                        else:
                            child.pack_forget()

        if not enabled:
            for key in self.similarity_fields:
                if key in self.fields:
                    self.fields[key].delete(0, "end")
                    self.fields[key].insert(0, "-2")

            similarity_keys = ["Matches", "Mismatches", "Gaps", "Similarity"]
            for key in similarity_keys:
                if key in self.CURRENT_MAPPINGS:
                    self.CURRENT_MAPPINGS.pop(key)
        else:
            for key in self.similarity_fields:
                if key in self.fields:
                    self.fields[key].delete(0, "end")
                    self.fields[key].insert(0, "-1")

        if hasattr(self, "output_csv_frame"):
            read_cols = len(self.fields["READ_CSV_HEADER"].get().split(","))
            num_cols = 2 * read_cols + 2
            if enabled:
                num_cols += 4

            header = [""] * num_cols
            old_header = [
                entry.get()
                for entry in self.output_csv_frame.winfo_children()
                if isinstance(entry, self.widgets["entry"])
            ]
            for i in range(min(len(header), len(old_header))):
                header[i] = old_header[i]

            self._create_csv_preview(self.output_csv_frame, header, [], is_input=False)
            self._update_write_header()

    def _show_tooltip(self, widget, text):
        widget.tooltip = tk.Toplevel()
        if self.only_tk:
            ttk.Label(
                widget.tooltip,
                text=text,
                background="#ffffe0",
                relief="solid",
                borderwidth=1,
            ).pack(padx=2, pady=2)
            o_x = 0
            o_y = 7 if text.startswith("Path") else 3
        else:
            widget.tooltip.configure(bg="black")
            if build_env.os_name == "windows":
                widget.tooltip.attributes("-transparentcolor", "black")
            else:
                widget.tooltip.configure(bg="#2b2b2b")

            ctk.CTkLabel(
                widget.tooltip,
                text=text,
                fg_color=("gray75", "gray25"),
                corner_radius=4,
                justify="left",
                anchor="w",
            ).pack(padx=4, pady=4)
            o_x = -5
            o_y = 0

        widget.tooltip.wm_overrideredirect(True)
        o_x += widget.winfo_rootx()
        o_y += widget.winfo_rooty() + widget.winfo_height()
        widget.tooltip.wm_geometry(f"+{o_x}+{o_y}")

    def _hide_tooltip(self, widget):
        if hasattr(widget, "tooltip"):
            widget.tooltip.destroy()
            del widget.tooltip

    def _browse_file(self, entry, is_input):
        current = Path(entry.get())

        if filename := (
            filedialog.askopenfilename if is_input else filedialog.asksaveasfilename
        )(
            initialdir=current.parent,
            initialfile=current.name,
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
        ):
            entry.delete(0, "end")
            entry.insert(0, str(Path(filename).absolute().as_posix()))
            if is_input:
                self._update_csv_info(filename)

    def _init_logging(self):
        log_dir = Path(__file__).parent.parent / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = (
            log_dir / f"sequence_aligner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        logging.basicConfig(
            filename=str(log_file),
            level=logging.INFO,
            format="%(asctime)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def update_output(self, text):
        if hasattr(self, "output_text"):

            def update():
                should_log = True
                self.output_text.configure(state="normal")
                if text.startswith("\f"):
                    self.output_text.delete("1.0", "end")
                elif text.startswith("\r"):
                    self.output_text.delete("end-2c linestart", "end-1c")
                    should_log = False

                clean_text = text.lstrip("\f\r")
                self.output_text.insert("end", clean_text)
                self.output_text.see("end")
                self.output_text.configure(state="disabled")
                if should_log:
                    logging.info(clean_text.rstrip())

            self.root.after(0, update)

    def save(self):
        ok, error = validate_config(self.fields, self.checkboxes)
        if not ok:
            messagebox.showerror("Error", error)
            return

        ok, error = save_config(self.fields, self.checkboxes)
        if not ok:
            messagebox.showerror("Error", error)
            return

        messagebox.showinfo("Success", "Configuration saved successfully!")
        build_env.reset_build_status()
        build_env.run_make(
            self.update_output,
            "all",
            on_complete=lambda s: (
                self.update_output(
                    "\fBuild complete! You can now press the Run button or modify the settings below.\n"
                )
                if s
                else None
            ),
        )

    def reset_defaults(self):
        for key, value in DEFAULT_CHECKBOXES.items():
            if key in self.checkboxes:
                self.checkboxes[key].set(value)

        for key, value in DEFAULT_VALUES.items():
            if key in self.fields:
                self.fields[key].delete(0, "end")
                self.fields[key].insert(0, value)

        self.CURRENT_MAPPINGS.clear()
        field_to_key = {
            "WRITE_CSV_SEQ1_POS": "First Seq",
            "WRITE_CSV_SCORE_POS": "Score",
            "WRITE_CSV_ALIGN_POS": "Alignment",
            "WRITE_CSV_MATCHES_POS": "Matches",
            "WRITE_CSV_MISMATCHES_POS": "Mismatches",
            "WRITE_CSV_GAPS_POS": "Gaps",
            "WRITE_CSV_SIMILARITY_POS": "Similarity",
        }

        for field, key in field_to_key.items():
            value = int(DEFAULT_VALUES[field])
            if value >= 0:
                self.CURRENT_MAPPINGS[key] = value
                if key == "First Seq":
                    self.CURRENT_MAPPINGS["Second Seq"] = value + 1

        if hasattr(self, "input_csv_frame"):
            header, first_row = self._read_csv_preview(DEFAULT_VALUES["INPUT_FILE"])
            if header:
                self._create_csv_preview(self.input_csv_frame, header, first_row)
                self._update_seq_col_info(header)

        if hasattr(self, "output_csv_frame"):
            output_header = DEFAULT_VALUES["WRITE_CSV_HEADER"].split(",")
            self._create_csv_preview(
                self.output_csv_frame, output_header, [], is_input=False
            )

        default_similarity_enabled = DEFAULT_CHECKBOXES["SIMILARITY_ANALYSIS"]
        for widget in self.content_frame.winfo_children():
            if isinstance(widget, (ttk.Frame, self.widgets["frame"])):
                for child in widget.winfo_children():
                    if hasattr(child, "similarity_row"):
                        if default_similarity_enabled:
                            child.pack()

        if hasattr(self, "format_parts"):
            default_parts = DEFAULT_VALUES["WRITE_CSV_ALIGN_FMT"].split("%s")
            for entry, part in zip(self.format_parts, default_parts):
                entry.delete(0, "end")
                entry.insert(0, part)
            self._update_preview()
            self._update_alignment_format()

    def _open_output_file(self):
        output_path = self.fields["OUTPUT_FILE"].get()
        if Path(output_path).exists():
            if sys.platform == "win32":
                os.startfile(output_path)
            elif sys.platform == "darwin":
                subprocess.run(["open", output_path])
            else:
                spreadsheet_apps = ["libreoffice", "gnumeric", "sheets"]
                for app in spreadsheet_apps:
                    try:
                        subprocess.run([app, output_path], stderr=subprocess.DEVNULL)
                        return
                    except FileNotFoundError:
                        continue

                text_editors = [
                    "gedit",
                    "gnome-text-editor",
                    "kate",
                    "mousepad",
                    "leafpad",
                    "xed",
                ]
                for editor in text_editors:
                    try:
                        subprocess.run([editor, output_path], stderr=subprocess.DEVNULL)
                        return
                    except FileNotFoundError:
                        continue

                try:
                    subprocess.run(
                        ["xdg-open", output_path], stderr=subprocess.PIPE, check=True
                    )
                except subprocess.CalledProcessError as e:
                    messagebox.showwarning(
                        "Warning",
                        "Could not open file with default application.\n\n"
                        f"File location: {output_path}\n\n"
                        "You can:\n"
                        "1. Open it manually with a spreadsheet application\n"
                        "2. Install LibreOffice ('sudo apt install libreoffice')\n"
                        "3. Install a text editor ('sudo apt install gedit')",
                    )
        else:
            messagebox.showerror("Error", f"Output file does not exist: {output_path}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    editor = ConfigEditor()
    editor.run()
