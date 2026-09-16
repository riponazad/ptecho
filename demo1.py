"""Interactive point-tracking demo — single window.

Plays an echocardiography video in a single matplotlib window. Pause to
click query points on the current frame, pick one or more models from the
on-screen checkboxes, then run tracking and watch the results play back
side by side, each with an approximate GLS (global longitudinal strain)
curve computed from the tracked points. The comparison video is also saved
to ``outputs/output.mp4``.

Usage: python demo1.py [--video_path data/input1.mp4]

@author: Azad Md Abulkalam
@location: ISB, NTNU
"""
import os
import threading
from typing import Tuple

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, Slider
from matplotlib.animation import FuncAnimation
import mediapy as media
from fire import Fire

import configs
from models.net import ECHOTRACKER, PIPS2, COTRACKER3, SPECKNET, LOCOTRACK, ECHOTRACKER2
from utils.utils_ import concatenate_videos_in_grid
from utils import viz_utils

AVAILABLE_MODELS = ("EchoTracker2", "LocoTrack", "CoTracker3", "EchoTracker", "PIPs++", "SpeckNet")
GRAYSCALE_MODELS = {"EchoTracker", "SpeckNet"}  # models that expect single-channel input

DISPLAY_SIZE = 256      # per-panel width/height used to render the input video
MAX_QUERY_POINTS = 40   # upper bound on how many points a user may click


def _with_query_frame(query_points: torch.Tensor, q_frame: int) -> torch.Tensor:
    """Prepends a time column to (B, N, 2) query points, returning (B, N, 3): (t, x, y)."""
    time_dim = torch.full((*query_points.shape[:2], 1), float(q_frame))
    return torch.cat((time_dim, query_points), dim=-1)


def _compute_gls_curve(points_norm: np.ndarray, width: int, height: int, ref_frame: int = 0) -> np.ndarray:
    """Approximates a global longitudinal strain (GLS) curve from tracked points.

    `points_norm` is (N, T, 2), normalized to [0, 1] as returned by `predict()`. Treats the
    N query points, in click order, as a polyline along the myocardial wall and returns the
    percent change in its total length at every frame relative to `ref_frame`:
    ``(L(t) - L(ref)) / L(ref) * 100``. `ref_frame` defaults to the video's first frame so
    the curve is comparable across runs regardless of which frame the query points were
    actually selected on.

    This is only anatomically meaningful if the points were clicked in order along the wall
    (e.g. apex to base). It's a simple 2D polyline-length approximation, not a substitute for
    vendor speckle-tracking GLS.
    """
    pts_px = points_norm * np.array([width, height], dtype=np.float32)
    seg_lengths = np.linalg.norm(pts_px[1:] - pts_px[:-1], axis=-1)  # (N-1, T)
    total_length = seg_lengths.sum(axis=0)  # (T,)
    ref_length = total_length[ref_frame]
    if ref_length == 0:
        return np.zeros_like(total_length)
    return (total_length - ref_length) / ref_length * 100.0


def predict(model_name: str, frames: torch.Tensor, query_points: torch.Tensor, q_frame: int) -> torch.Tensor:
    """Runs a single point-tracking model and returns predicted trajectories, shape [1, T, N, 2]."""
    if model_name == "EchoTracker":
        model = ECHOTRACKER(device_ids=[0])
        model.load(path=configs.model_weights_path["echotracker"])
        return model.infer(frames, query_points, resize=(256, 256))

    if model_name == "PIPs++":
        model = PIPS2()
        model.load(path=configs.model_weights_path["pips++"])
        return model.infer(frames, query_points, resize=(256, 256))

    if model_name == "CoTracker3":
        query_points = _with_query_frame(query_points, q_frame)
        model = COTRACKER3(ft_model=True)
        model.load(path=configs.model_weights_path["cotracker3"])
        trajs_e, _ = model.infer(frames, query_points, resize=(256, 256))
        return trajs_e.cpu()

    if model_name == "SpeckNet":
        # NOTE: unlike the other models, SpeckNet always queries from frame 0, not `q_frame`.
        query_points = _with_query_frame(query_points, 0)
        model = SPECKNET(device_ids=[0], multi_scale_level=3)
        model.load(configs.model_weights_path["specknet"])
        trajs_e, _ = model.infer(frames, query_points, resize=(512, 512))
        return trajs_e.cpu()

    if model_name == "LocoTrack":
        query_points = _with_query_frame(query_points, q_frame)
        model = LOCOTRACK(model_size="base", ft_model=True, device_ids=[0])
        model.load(path=configs.model_weights_path["locotrack"])
        trajs_e, _ = model.infer(frames, query_points, resize=(256, 256))
        return trajs_e.cpu()

    if model_name == "EchoTracker2":
        query_points = _with_query_frame(query_points, q_frame)
        model = ECHOTRACKER2(model_size="base", ft_model=True, device_ids=[0],
                              sp_attn=True, resnet_temp_module=True, patch_size=9)
        model.load(path=configs.model_weights_path["echotracker2"])
        trajs_e = model.infer(frames, query_points, resize=(256, 256))
        del model
        return trajs_e.cpu()

    raise ValueError(f"Unknown model: '{model_name}'. Available models: {AVAILABLE_MODELS}")


class PointTrackingDemo:
    """Single-window interactive demo.

    State machine: playing -> selecting -> tracking -> results (-> selecting ...).
    Model inference runs on a background thread so the video keeps playing
    smoothly while the UI waits.
    """

    def __init__(self, video_path: str, default_models: Tuple[str, ...] = (AVAILABLE_MODELS[0],)):
        self.video_path = video_path
        self.display_w = DISPLAY_SIZE * 2 + 80
        self.display_h = DISPLAY_SIZE * 2
        self.colormap = viz_utils.get_colors(MAX_QUERY_POINTS)

        self.state = "playing"
        self.select_points = []
        self.query_frame_idx = 0
        self.play_idx = 0
        self.result_idx = 0
        self.result_titles = []
        self.result_frames = []
        self.result_gls = []
        self.result_show_gls = True
        self._tracking_done = False
        self._syncing_slider = False
        self.grid_axes = []
        self.grid_ims = []
        self.grid_gls_axes = []
        self.grid_gls_markers = []

        self.frames_gray, self.frames_rgb = self._load_video()
        self.T = len(self.frames_gray)

        self._build_ui(default_models)

    # ------------------------------------------------------------------
    # Video loading
    # ------------------------------------------------------------------
    def _load_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")
        gray, rgb = [], []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (self.display_w, self.display_h))
            gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if not gray:
            raise ValueError("Video contains no frames.")
        return np.array(gray), np.array(rgb)

    # ------------------------------------------------------------------
    # UI setup
    # ------------------------------------------------------------------
    def _build_ui(self, default_models):
        self.fig = plt.figure(figsize=(13, 8))
        self.fig.patch.set_facecolor("#1a1a2e")
        try:
            self.fig.canvas.manager.set_window_title("Point Tracking in Echocardiography")
        except Exception:
            pass

        self.ax = self.fig.add_axes([0.01, 0.16, 0.76, 0.82])
        self.ax.set_facecolor("black")
        self.ax.axis("off")
        self.im = self.ax.imshow(self.frames_rgb[0], aspect="auto")

        slider_ax = self.fig.add_axes([0.06, 0.112, 0.66, 0.025])
        slider_ax.set_facecolor("#2a2a4a")
        self.frame_slider = Slider(
            slider_ax, "Frame", 0, self.T - 1, valinit=0, valstep=1,
            color="#6a6aff", initcolor="none",
        )
        self.frame_slider.label.set_color("white")
        self.frame_slider.valtext.set_color("white")
        self.frame_slider.on_changed(self._on_slider_change)

        self.status_txt = self.fig.text(
            0.40, 0.075,
            'Video playing.  Click "Pause & Select Points" to choose tracking targets.',
            ha="center", va="center", fontsize=10, color="#ccccff",
        )
        self.pts_txt = self.fig.text(
            0.40, 0.048, "", ha="center", va="center", fontsize=9, color="#ffdd88",
        )

        btn_kw = dict(color="#2a2a4a", hovercolor="#44446a")
        self.btn_select = Button(
            self.fig.add_axes([0.03, 0.005, 0.24, 0.038]), "Pause & Select Points", **btn_kw)
        self.btn_clear = Button(
            self.fig.add_axes([0.29, 0.005, 0.22, 0.038]), "Clear Points", **btn_kw)
        self.btn_track = Button(
            self.fig.add_axes([0.53, 0.005, 0.24, 0.038]), "Run Tracking", **btn_kw)
        for b in (self.btn_select, self.btn_clear, self.btn_track):
            b.label.set_color("white")
        self.btn_select.on_clicked(self._on_pause_select)
        self.btn_clear.on_clicked(self._on_clear)
        self.btn_track.on_clicked(self._on_run_tracking)

        # Model selection sidebar — pick any subset of AVAILABLE_MODELS to run.
        self.fig.text(0.80, 0.93, "Models to run", color="white", fontsize=10, fontweight="bold")
        check_ax = self.fig.add_axes([0.80, 0.58, 0.19, 0.34])
        check_ax.set_facecolor("#222244")
        actives = [name in default_models for name in AVAILABLE_MODELS]
        self.model_checks = CheckButtons(
            check_ax, AVAILABLE_MODELS, actives,
            label_props={"color": ["white"], "fontsize": [11]},
            frame_props={"facecolor": "white", "edgecolor": "white", "s": 110},
            check_props={"facecolor": "#39d353", "s": 110},
        )
        self.model_checks.on_clicked(self._on_option_changed)

        # GLS is optional — uncheck to skip strain computation and get tracking only.
        self.fig.text(0.80, 0.53, "Options", color="white", fontsize=10, fontweight="bold")
        gls_toggle_ax = self.fig.add_axes([0.80, 0.45, 0.19, 0.06])
        gls_toggle_ax.set_facecolor("#222244")
        self.gls_toggle = CheckButtons(
            gls_toggle_ax, ["Compute GLS"], [True],
            label_props={"color": ["white"], "fontsize": [11]},
            frame_props={"facecolor": "white", "edgecolor": "white", "s": 110},
            check_props={"facecolor": "#39d353", "s": 110},
        )
        self.gls_toggle.on_clicked(self._on_option_changed)

        self.ani = FuncAnimation(self.fig, self._tick, interval=50, blit=False, cache_frame_data=False)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)

    # ------------------------------------------------------------------
    # Animation tick — main thread, called ~20 fps
    # ------------------------------------------------------------------
    def _tick(self, _):
        if self._tracking_done:
            self._tracking_done = False
            self.state = "results"
            self.result_idx = 0
            self._switch_to_grid(self.result_frames, self.result_titles, self.result_gls, self.result_show_gls)
            self.status_txt.set_text('Tracking complete!  Click "Pause & Select Points" to try again.')
            self.pts_txt.set_text("")

        if self.state == "playing":
            self.play_idx = (self.play_idx + 1) % self.T
            self.im.set_data(self.frames_rgb[self.play_idx])
        elif self.state == "results" and self.result_frames:
            self.result_idx = (self.result_idx + 1) % self.result_frames[0].shape[0]
            for im, frames in zip(self.grid_ims, self.result_frames):
                im.set_data(frames[self.result_idx])
            for marker in self.grid_gls_markers:
                if marker is not None:
                    marker.set_xdata([self.result_idx, self.result_idx])
        # 'selecting', 'tracking' — no frame update needed

        return []

    # ------------------------------------------------------------------
    # Button callbacks — main thread
    # ------------------------------------------------------------------
    def _on_pause_select(self, _):
        if self.state == "tracking":
            return  # can't interrupt an ongoing run

        self.state = "selecting"
        self.select_points = []
        self.query_frame_idx = self.play_idx  # query points belong to whichever frame we paused on
        self._sync_slider(self.query_frame_idx)

        self._switch_to_single(
            self.frames_rgb[self.query_frame_idx],
            title=f"Click to place tracking points (frame {self.query_frame_idx})",
        )
        self.status_txt.set_text('Left-click to add points.  Then click "Run Tracking".')
        self.pts_txt.set_text("Points selected: 0")

    def _on_option_changed(self, _label):
        """Model / GLS checkboxes were touched after results are already showing.

        Changing what to run no longer matches what's on screen, so start over from the
        playing view rather than leaving a stale results grid up.
        """
        if self.state == "results":
            self._reset_to_playing()

    def _reset_to_playing(self):
        """Return to the initial full-video playing view, as at startup."""
        for ax in self.grid_axes:
            ax.remove()
        for ax in self.grid_gls_axes:
            ax.remove()
        self.grid_axes, self.grid_ims = [], []
        self.grid_gls_axes, self.grid_gls_markers = [], []

        self.state = "playing"
        self.select_points = []
        self.result_frames = []
        self.result_titles = []
        self.result_gls = []
        self.play_idx = 0
        self._sync_slider(0)

        self.ax.set_visible(True)
        self.ax.clear()
        self.ax.set_facecolor("black")
        self.ax.axis("off")
        self.im = self.ax.imshow(self.frames_rgb[0], aspect="auto")

        self.status_txt.set_text('Video playing.  Click "Pause & Select Points" to choose tracking targets.')
        self.pts_txt.set_text("")
        self.fig.canvas.draw_idle()

    def _sync_slider(self, idx):
        """Move the slider to `idx` without re-triggering `_on_slider_change`."""
        self._syncing_slider = True
        try:
            self.frame_slider.set_val(idx)
        finally:
            self._syncing_slider = False

    def _on_slider_change(self, val):
        if self._syncing_slider:
            return  # value was set programmatically, not dragged by the user
        if self.state == "tracking":
            self._sync_slider(self.query_frame_idx)  # revert — can't interrupt an ongoing run
            return

        self.state = "selecting"
        self.select_points = []
        self.query_frame_idx = int(val)

        self._switch_to_single(
            self.frames_rgb[self.query_frame_idx],
            title=f"Click to place tracking points (frame {self.query_frame_idx})",
        )
        self.status_txt.set_text('Left-click to add points.  Then click "Run Tracking".')
        self.pts_txt.set_text("Points selected: 0")

    def _on_clear(self, _):
        if self.state != "selecting":
            return
        self.select_points = []
        self._switch_to_single(
            self.frames_rgb[self.query_frame_idx],
            title=f"Click to place tracking points (frame {self.query_frame_idx})",
        )
        self.pts_txt.set_text("Points selected: 0")
        self.status_txt.set_text("Points cleared.  Click to select new points.")

    def _on_run_tracking(self, _):
        if self.state != "selecting":
            return
        if not self.select_points:
            self.status_txt.set_text("Select at least one point before running tracking.")
            self.fig.canvas.draw_idle()
            return

        model_names = [name for name, active in zip(AVAILABLE_MODELS, self.model_checks.get_status()) if active]
        if not model_names:
            self.status_txt.set_text("Select at least one model before running tracking.")
            self.fig.canvas.draw_idle()
            return

        compute_gls = self.gls_toggle.get_status()[0]

        self.state = "tracking"
        self.pts_txt.set_text("")
        gls_note = "with GLS" if compute_gls else "without GLS"
        self.status_txt.set_text(
            f"Running {', '.join(model_names)} {gls_note} on {len(self.select_points)} point(s) — please wait…"
        )
        self.fig.canvas.draw_idle()
        threading.Thread(
            target=self._run_tracking_thread, args=(model_names, compute_gls), daemon=True,
        ).start()

    # ------------------------------------------------------------------
    # Point selection click — main thread
    # ------------------------------------------------------------------
    def _on_click(self, event):
        if self.state != "selecting":
            return
        if event.inaxes is not self.ax or event.button != 1:
            return
        if len(self.select_points) >= MAX_QUERY_POINTS:
            self.status_txt.set_text(f"Maximum of {MAX_QUERY_POINTS} points reached.")
            self.fig.canvas.draw_idle()
            return

        x = int(np.clip(np.round(event.xdata), 0, self.display_w - 1))
        y = int(np.clip(np.round(event.ydata), 0, self.display_h - 1))
        self.select_points.append(np.array([x, y]))
        color = tuple(np.array(self.colormap[len(self.select_points) - 1]) / 255.0)
        self.ax.plot(x, y, "o", color=color, markersize=9,
                     markeredgecolor="white", markeredgewidth=1.0)
        n = len(self.select_points)
        self.pts_txt.set_text(f"Points selected: {n}")
        self.status_txt.set_text(f'{n} point(s) placed.  Add more or click "Run Tracking".')
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------
    # Inference (background thread)
    # ------------------------------------------------------------------
    def _run_tracking_thread(self, model_names, compute_gls):
        q_frame = self.query_frame_idx

        gframes = torch.from_numpy(self.frames_gray[..., np.newaxis]).unsqueeze(0)  # [1, T, H, W, 1]
        rgbs = torch.from_numpy(self.frames_rgb).unsqueeze(0)                       # [1, T, H, W, 3]

        query_points = torch.tensor(np.array(self.select_points), dtype=torch.float).unsqueeze(0)
        query_points[..., 0] /= self.display_w
        query_points[..., 1] /= self.display_h

        titles, frames_out, gls_curves = [], [], []
        for name in model_names:
            frames = gframes if name in GRAYSCALE_MODELS else rgbs
            trajs_e = predict(name, frames, query_points, q_frame)
            # squeeze(0) drops only the batch dim — plain squeeze() would also drop the
            # point dim when exactly one query point is selected (N=1), corrupting the shape.
            pts = trajs_e.squeeze(0).numpy()  # (N, T, 2), normalized to [0, 1]
            seq = viz_utils.visualize_tracking(
                frames=gframes.squeeze(0).squeeze(-1).numpy(),
                points=pts,
                gray=True,
                vis_color="random",
                thickness=2,
                track_length=30,
            )
            titles.append(name)
            frames_out.append(seq)
            gls_curves.append(
                _compute_gls_curve(pts, self.display_w, self.display_h, ref_frame=0)
                if compute_gls and pts.shape[0] >= 2 else None
            )

        self.result_titles = titles
        self.result_frames = frames_out
        self.result_gls = gls_curves
        self.result_show_gls = compute_gls

        os.makedirs("outputs", exist_ok=True)
        out_path = "outputs/output.mp4"
        media.write_video(out_path, concatenate_videos_in_grid(frames_out), fps=25)
        print(f'Results saved to "{out_path}"')

        self._tracking_done = True  # picked up by _tick on the main thread

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    def _switch_to_single(self, img, title=""):
        """Full-width single axes — used for playing and selecting."""
        for ax in self.grid_axes:
            ax.set_visible(False)
        for ax in self.grid_gls_axes:
            ax.set_visible(False)
        self.ax.set_visible(True)
        self.ax.clear()
        self.ax.set_facecolor("black")
        self.ax.axis("off")
        self.im = self.ax.imshow(img, aspect="equal")
        self.ax.set_title(title, color="white", pad=5, fontsize=10)
        self.fig.canvas.draw_idle()

    def _switch_to_grid(self, imgs, titles, gls_curves=None, show_gls=True):
        """One video (+ optional GLS-strain panel) per selected model, in an auto-sized grid."""
        self.ax.set_visible(False)
        for ax in self.grid_axes:
            ax.remove()
        for ax in self.grid_gls_axes:
            ax.remove()
        self.grid_axes, self.grid_ims = [], []
        self.grid_gls_axes, self.grid_gls_markers = [], []

        if gls_curves is None:
            gls_curves = [None] * len(imgs)

        n = len(imgs)
        rows = int(np.sqrt(n))
        cols = int(np.ceil(n / rows))
        left, bottom, width, height = 0.01, 0.16, 0.76, 0.82
        cell_w, cell_h = width / cols, height / rows
        # Fraction of each cell's height given to the video vs. the GLS plot below it. With GLS
        # off, the video simply takes the full cell — no empty panel reserved for it.
        video_frac = 0.72 if show_gls else 1.0

        for i in range(n):
            r, c = divmod(i, cols)
            cell_left = left + c * cell_w
            cell_w_use = cell_w * 0.97  # shared by the video and its GLS plot, so widths match exactly
            cell_bottom = bottom + (rows - 1 - r) * cell_h
            video_h = cell_h * video_frac

            video_ax = self.fig.add_axes([
                cell_left, cell_bottom + cell_h - video_h, cell_w_use, video_h * 0.99,
            ])
            video_ax.set_facecolor("black")
            video_ax.axis("off")
            # aspect="equal" preserves the video's true aspect ratio (no stretching); the GLS
            # plot below shares this axes box's width, even though the letterboxed video
            # content itself may render slightly narrower.
            im = video_ax.imshow(imgs[i][0], aspect="equal")
            # Model name is drawn inside the frame (not as an axes title) so it doesn't eat
            # extra vertical space or risk overlapping the row above it in multi-row grids.
            video_ax.text(
                0.02, 0.96, titles[i], transform=video_ax.transAxes,
                ha="left", va="top", fontsize=10, color="#88ff88", fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", boxstyle="round,pad=0.25"),
            )
            self.grid_axes.append(video_ax)
            self.grid_ims.append(im)

            if not show_gls:
                continue

            gls_h = cell_h * (1 - video_frac) * 0.8
            gls_ax = self.fig.add_axes([cell_left, cell_bottom, cell_w_use, gls_h])
            gls_ax.set_facecolor("#12122a")
            curve = gls_curves[i]
            if curve is not None:
                peak = curve.min()  # most negative = peak myocardial shortening
                gls_ax.plot(curve, color="#39d353", linewidth=1.2)
                gls_ax.axhline(0, color="#666688", linewidth=0.6)
                marker = gls_ax.axvline(self.result_idx, color="white", linewidth=1.0)
                gls_ax.set_title(f"GLS peak: {peak:.1f}%", color="#ffdd88", fontsize=8, pad=2)
            else:
                marker = None
                gls_ax.text(0.5, 0.5, "GLS needs ≥ 2 points", ha="center", va="center",
                            color="#888899", fontsize=7, transform=gls_ax.transAxes)
            gls_ax.tick_params(colors="#888899", labelsize=6)
            for spine in gls_ax.spines.values():
                spine.set_color("#444466")
            self.grid_gls_axes.append(gls_ax)
            self.grid_gls_markers.append(marker)

        self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


def main(video_path: str = "data/input1.mp4", default_models: Tuple[str, ...] = (AVAILABLE_MODELS[0],)):
    """Interactive single-window tracking demo.

    Plays `video_path`; click "Pause & Select Points" to freeze on the
    current frame and place query points, pick one or more models from the
    checkboxes, then "Run Tracking". Results play back side by side and are
    also saved to `outputs/output.mp4`.

    Args:
        video_path: Path to the input video.
        default_models: Models checked by default when the demo opens.
    """
    PointTrackingDemo(video_path, default_models=default_models).show()


if __name__ == "__main__":
    Fire(main)
