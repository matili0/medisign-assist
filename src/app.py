"""Desktop application for isolated medical signs in LSFB."""

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from artifacts import ROOT
from camera import Camera
from model import RecognitionEngine


ENGLISH = {
    "APPELER": "call",
    "BOIRE": "drink",
    "CHAUD": "hot / warm",
    "COMPRENDRE": "understand",
    "DORMIR": "sleep",
    "ENCEINTE": "pregnant",
    "FAIBLE": "weak",
    "FORT": "strong",
    "FROID": "cold",
    "HOPITAL": "hospital",
    "MALADE.VENTRE": "stomach discomfort",
    "MANGER": "eat",
    "MEDECIN": "doctor",
    "NON": "no",
    "OUI": "yes",
    "PAS.SOUFFLER": "not blow (literal)",
    "RESPIRER": "breathe",
    "SOUFFRIR": "suffer / be in pain",
    "TOMBER": "fall",
    "VITE": "quickly"
}


@dataclass
class CaptureState:
    """Frames and timing for the current capture."""

    recording: bool = False
    frames: list = field(default_factory=list)
    started: float | None = None

    def reset(self):
        self.recording = False
        self.frames.clear()
        self.started = None

    def start(self):
        self.reset()
        self.recording = True
        self.started = time.perf_counter()

    def finish(self):
        self.recording = False
        return np.asarray(self.frames, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    local_run = ROOT / "models/spoter_legacy"
    default_run = local_run if local_run.is_dir() else ROOT / "runs/spoter_aug"
    parser.add_argument("--run", type=Path, default=default_run)
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.70,
        help="Initial SPOTER threshold; choose the final value on validation",
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--capture-frames", type=int, default=50,
                        help="Stop webcam capture after this many frames (default: 50, capped by the model)")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads (default: 1)")
    parser.add_argument(
        "--input",
        type=Path,
        help="Raw NPY sequence: return JSON without opening the window",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.capture_frames < 1:
        parser.error("--capture-frames must be positive")
    import torch
    torch.set_num_threads(args.threads)
    if args.input:
        engine = RecognitionEngine(args.run, args.threshold)
        print(
            json.dumps(
                engine.predict(np.load(args.input, allow_pickle=False)), indent=2
            )
        )
        return
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("MediSign Assist — LSFB medical signs")
    root.geometry(f"{min(1020, root.winfo_screenwidth() - 48)}x"
                  f"{min(680, root.winfo_screenheight() - 90)}")
    root.minsize(780, 500)
    state = CaptureState()
    engine = None
    camera = None
    timer = None
    protected = []
    path = tk.StringVar(value=str(args.run))
    show_body = tk.BooleanVar(value=False)
    status = tk.StringVar(value="Click Load to start.")
    output = tk.StringVar(value="Ready for a sign")
    detail = tk.StringVar(value="")

    def sync_controls():
        for widget in protected:
            widget.configure(state="disabled" if state.recording else "normal")
        record.configure(text="Finish" if state.recording else "Capture")

    def clear_result():
        output.set("Ready for a sign")
        detail.set("")
        ranking.delete(*ranking.get_children())
        ranking.grid_remove()

    def guarded(fn):
        def call():
            try:
                fn()
            except Exception as exc:
                state.recording = False
                clear_result()
                sync_controls()
                status.set(str(exc))
                messagebox.showerror("MediSign", str(exc))

        return call

    def choose():
        folder = filedialog.askdirectory()
        if folder:
            path.set(folder)

    def load():
        nonlocal engine
        clear_result()
        engine = None
        engine = RecognitionEngine(Path(path.get()), args.threshold)
        status.set("Model loaded. Open the camera or import poses.")

    def recognize(sequence):
        clear_result()
        if engine is None:
            raise ValueError("Load a model first")
        result = engine.predict(sequence)
        sign = result["label"]
        if sign:
            output.set(ENGLISH.get(sign, sign))
        else:
            output.set("Sign not recognized")
        for candidate, score in result.get("top", []):
            ranking.insert("", "end", values=(candidate, ENGLISH.get(candidate, "—"), f"{score:.3f}"))
        if ranking.get_children():
            ranking.grid()
        elif sign:
            detail.set(sign)
        if "processing_ms" in result:
            timing = f"Processing: {result['processing_ms']:.0f} ms"
            detail.set("\n".join(filter(None, [detail.get(), timing])))
        status.set(result["message"])

    def open_file():
        filename = filedialog.askopenfilename(filetypes=[("Raw poses", "*.npy")])
        if filename:
            recognize(np.load(filename, allow_pickle=False))

    def close_camera():
        nonlocal camera, timer
        if timer is not None:
            root.after_cancel(timer)
            timer = None
        if camera is not None:
            camera.close()
            camera = None
        state.reset()
        clear_result()
        sync_controls()
        preview.configure(image="", text="Camera closed")
        preview.image = None
        status.set("")

    def toggle_record():
        if camera is None or engine is None:
            raise ValueError("Load a model and open the camera")
        if state.recording:
            sequence = state.finish()
            sync_controls()
            capture_seconds = time.perf_counter() - state.started
            recognize(sequence)
            detail.set(
                detail.get() + f"\nCapture and extraction: {capture_seconds:.1f} s"
            )
        else:
            state.start()
            clear_result()
            output.set("Recording…")
            sync_controls()

    def tick():
        nonlocal timer
        timer = None
        if camera is None:
            return
        try:
            from PIL import Image, ImageTk

            frame_start = time.perf_counter()
            rgb, points = camera.read(
                extract=state.recording, show_body=show_body.get()
            )
            if state.recording:
                state.frames.append(points)
                guidance = "click Finish after the sign"
                if not points[:99].any():
                    guidance = "keep your upper body in the frame"
                elif not points[99:].any():
                    guidance = "no hand detected — keep the signing hand in view"
                limit = min(args.capture_frames, engine.max_len)
                status.set(f"Recording: {len(state.frames)}/{limit} frames — {guidance}")
                if len(state.frames) >= limit:
                    toggle_record()
            frame = Image.fromarray(rgb)
            frame.thumbnail((max(1, preview.winfo_width()), max(1, preview.winfo_height())))
            photo = ImageTk.PhotoImage(frame)
            preview.configure(image=photo, text="")
            preview.image = photo
            elapsed_ms = int((time.perf_counter() - frame_start) * 1000)
            timer = root.after(max(1, 20 - elapsed_ms), tick)
        except Exception as exc:
            close_camera()
            status.set(f"Camera stopped: {exc}")

    def open_camera():
        nonlocal camera
        if engine is None:
            raise ValueError("Load a model first")
        if camera is not None:
            return
        camera = Camera(args.camera).open()
        status.set("Face the camera. Click Capture before signing.")
        tick()

    paper, white = "#f1f5f4", "#ffffff"
    ink, muted, teal = "#17342f", "#4b655f", "#09675b"
    root.configure(bg=paper)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", font=("Segoe UI", 10), foreground=ink, background=white)
    style.configure("TEntry", padding=7, fieldbackground=white, bordercolor="#b6c9c3")
    style.configure("TButton", padding=(12, 7), width=0, background="#e8efec",
                    borderwidth=0, font=("Segoe UI", 10, "bold"))
    style.map("TButton", background=[("disabled", "#eef2f0"), ("pressed", "#c5d8d2"),
                                      ("active", "#d5e5df")])
    style.configure("Accent.TButton", background=teal, foreground=white)
    style.map("Accent.TButton", foreground=[("disabled", muted)],
              background=[("disabled", "#eef2f0"), ("pressed", "#06483e"),
                          ("active", "#09584d")])
    style.map("TCheckbutton", background=[("active", white)])
    style.configure("Treeview", rowheight=26, fieldbackground=white)

    def label(parent, *, size=10, bold=False, color=ink, **options):
        return tk.Label(parent, bg=parent.cget("background"), fg=color,
                        font=("Segoe UI", size, "bold" if bold else "normal"),
                        anchor="w", justify="left", bd=0, **options)

    def button(parent, caption, command, *, accent=False):
        return ttk.Button(parent, text=caption, command=guarded(command),
                          style="Accent.TButton" if accent else "TButton")

    header = tk.Frame(root, bg=ink, padx=22, pady=12)
    header.pack(fill="x")
    label(header, text="MediSign Assist", size=22, bold=True, color=white).pack(anchor="w")

    model_bar = tk.Frame(root, bg=white, padx=14, pady=10)
    model_bar.pack(fill="x", padx=16, pady=(14, 0))
    model_bar.columnconfigure(1, weight=1)
    label(model_bar, text="Model", bold=True).grid(row=0, column=0, padx=(0, 12))
    entry = ttk.Entry(model_bar, textvariable=path, width=1)
    entry.grid(row=0, column=1, sticky="ew")
    entry.xview_moveto(1)
    folder = button(model_bar, "Model folder", choose)
    folder.grid(row=0, column=2, padx=8)
    load_button = button(model_bar, "Load", load, accent=True)
    load_button.grid(row=0, column=3)
    protected.extend([entry, folder, load_button])

    workspace = tk.Frame(root, bg=paper)
    workspace.pack(fill="both", expand=True, padx=16, pady=12)
    workspace.columnconfigure(0, weight=1)
    workspace.rowconfigure(0, weight=1)
    capture = tk.Frame(workspace, bg=white, padx=14, pady=12)
    capture.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
    capture.columnconfigure(0, weight=1)
    capture.rowconfigure(1, weight=1)

    camera_bar = tk.Frame(capture, bg=white)
    camera_bar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
    label(camera_bar, text="Camera preview", size=13, bold=True).pack(side="left")
    ttk.Checkbutton(camera_bar, text="Show body points", variable=show_body).pack(side="right")
    viewport = tk.Frame(capture, bg="#182e30")
    viewport.grid(row=1, column=0, sticky="nsew")
    preview = tk.Label(viewport, bg="#182e30", fg="#cbddda", bd=0,
                       font=("Segoe UI", 11), text="Camera is off")
    preview.place(relwidth=1, relheight=1)

    controls = tk.Frame(capture, bg=white)
    controls.grid(row=2, column=0, sticky="ew", pady=(12, 0))
    open_button = button(controls, "Open camera", open_camera)
    open_button.pack(side="left")
    record = button(controls, "Capture", toggle_record, accent=True)
    record.pack(side="left", padx=6)
    button(controls, "Close camera", close_camera).pack(side="left")
    import_button = button(camera_bar, "Import poses", open_file)
    import_button.pack(side="right", padx=8)
    protected.extend([open_button, import_button])

    result = tk.Frame(workspace, bg=white, padx=16, pady=14)
    result.grid(row=0, column=1, sticky="nsew")
    result.columnconfigure(0, minsize=314)
    result.rowconfigure(4, weight=1)
    label(result, text="Recognition result", size=13, bold=True).grid(sticky="w")
    label(result, textvariable=output, size=22, bold=True, color=teal,
          wraplength=310).grid(row=1, column=0, sticky="w", pady=(12, 6))
    ranking = ttk.Treeview(result, columns=("sign", "english", "score"),
                           show="headings", height=5, selectmode="none")
    for column, heading, width in (("sign", "Top 5 · LSFB", 112), ("english", "English", 145),
                                   ("score", "Score", 54)):
        anchor = "e" if column == "score" else "w"
        ranking.heading(column, text=heading, anchor=anchor)
        ranking.column(column, width=width, anchor=anchor)
    ranking.grid(row=2, column=0, sticky="ew", pady=(0, 8))
    ranking.grid_remove()
    label(result, textvariable=detail, color=muted, wraplength=310).grid(
        row=3, column=0, sticky="w")
    label(result, textvariable=status, wraplength=310).grid(row=4, column=0, sticky="nw", pady=(8, 0))

    def quit_app():
        close_camera()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", quit_app)
    root.mainloop()


if __name__ == "__main__":
    main()
