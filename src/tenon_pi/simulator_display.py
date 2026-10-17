"""Optional main-thread live simulator camera window; never policy inference."""

from __future__ import annotations

import os
import sys
from typing import Any, Literal

import numpy as np
from PIL import Image

DisplayMode = Literal["auto", "window", "headless"]


class SimulatorDisplay:
    def __init__(self, requested: DisplayMode, *, title: str = "Tenon Pi — Live LIBERO"):
        if requested not in {"auto", "window", "headless"}:
            raise ValueError("display must be auto, window or headless")
        self.requested = requested
        self.mode = "headless"
        self.reason: str | None = None
        self.frames_presented = 0
        self.closed = False
        self._root: Any = None
        self._label: Any = None
        self._image: Any = None
        if requested == "headless":
            self.reason = "explicit_headless"
            return
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            self._unavailable("no_x11_display")
            return
        try:
            import tkinter as tk

            from PIL import ImageTk

            self._tk = tk
            self._photo = ImageTk.PhotoImage
            self._root = tk.Tk()
            self._root.title(title)
            self._root.protocol("WM_DELETE_WINDOW", self.close)
            self._label = tk.Label(self._root, text="Waiting for simulator camera frames")
            self._label.pack()
            self._root.update_idletasks()
            self._root.update()
            self.mode = "window"
        except (ImportError, RuntimeError, OSError) as exc:
            self._cleanup_failed_initialization()
            self._unavailable(f"gui_initialization_{type(exc).__name__}")
        except Exception as exc:
            # TclError is only available if the optional import succeeded.
            self._cleanup_failed_initialization()
            if "tk" in locals() and isinstance(exc, tk.TclError):
                self._unavailable("gui_initialization_TclError")
            else:
                raise

    def _cleanup_failed_initialization(self) -> None:
        if self._root is not None:
            self._root.destroy()
            self._root = None

    def _unavailable(self, reason: str) -> None:
        self.reason = reason
        if self.requested == "window":
            raise RuntimeError(
                f"live simulator window unavailable ({reason}); use a GUI-capable Python/Tk "
                "on a desktop (Linux requires DISPLAY), or explicitly choose --display headless"
            )

    def show(self, observation: dict[str, Any]) -> bool:
        """Present actual environment camera pixels; False requests local rollout stop."""
        if self.closed:
            return False
        if self.mode == "headless":
            return True
        views = []
        for name in ("agentview_image", "robot0_eye_in_hand_image"):
            pixels = np.asarray(observation[name])
            if pixels.ndim != 3 or pixels.shape[2] != 3 or pixels.dtype != np.uint8:
                raise ValueError("simulator viewer requires RGB uint8 camera frames")
            # Same camera orientation used by the LIBERO observation adapter.
            views.append(Image.fromarray(np.ascontiguousarray(pixels[::-1, ::-1])))
        canvas = Image.new("RGB", (sum(v.width for v in views), max(v.height for v in views)))
        offset = 0
        for view in views:
            canvas.paste(view, (offset, 0))
            offset += view.width
        canvas.thumbnail((1024, 768))
        self._image = self._photo(canvas, master=self._root)
        self._label.configure(image=self._image, text="")
        try:
            self._root.update_idletasks()
            self._root.update()
        except self._tk.TclError:
            if self.closed:
                return False
            raise
        if self.closed:
            return False
        self.frames_presented += 1
        return True

    def summary(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "mode": self.mode,
            "reason": self.reason,
            "frames_presented": self.frames_presented,
            "window_closed": self.closed,
        }

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            if self._root is not None:
                self._root.destroy()
                self._root = None
