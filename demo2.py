"""Quantitative evaluation demo.

Loads ground-truth annotated point trajectories, runs each requested model
on the annotated query points, and saves a side-by-side comparison video
with per-model accuracy metrics (points within 1/2/4 px, median trajectory
error) overlaid, to ``outputs/demo2_output.mp4``.

See the "Demo 2" section of the README for usage examples.

@author: Azad Md Abulkalam
@location: ISB, NTNU
"""
import os
import pickle
from typing import Tuple

import numpy as np
import torch
import mediapy as media
from fire import Fire

import configs
from models.net import ECHOTRACKER, PIPS2, COTRACKER3, SPECKNET, LOCOTRACK, ECHOTRACKER2
from utils.utils_ import add_text_to_frames, concatenate_videos_in_grid, get_resized_frames
from utils import viz_utils, evaluate

AVAILABLE_MODELS = ("EchoTracker2", "LocoTrack", "CoTracker3", "EchoTracker", "PIPs++", "SpeckNet")
GRAYSCALE_MODELS = {"EchoTracker", "SpeckNet"}  # models that expect single-channel input

CANVAS_SIZE = 800  # width/height frames are resized to for visualization


def _with_query_frame(query_points: torch.Tensor, q_frame: int) -> torch.Tensor:
    """Prepends a time column to (B, N, 2) query points, returning (B, N, 3): (t, x, y)."""
    time_dim = torch.full((*query_points.shape[:2], 1), float(q_frame))
    return torch.cat((time_dim, query_points), dim=-1)


def predict(model_name: str, frames: torch.Tensor, query_points: torch.Tensor, q_frame: int) -> torch.Tensor:
    """Runs a single point-tracking model and returns predicted trajectories, shape [1, N, T, 2]."""
    if model_name == "EchoTracker":
        model = ECHOTRACKER(device_ids=[0])
        model.load(path=configs.model_weights_path["echotracker"])
        trajs_e = model.infer(frames, query_points, resize=(256, 256))
        del model
        return trajs_e

    if model_name == "PIPs++":
        model = PIPS2()
        model.load(path=configs.model_weights_path["pips++"])
        trajs_e = model.infer(frames, query_points, resize=(256, 256))
        del model
        return trajs_e

    if model_name == "CoTracker3":
        query_points = _with_query_frame(query_points, q_frame)
        model = COTRACKER3(ft_model=True)
        model.load(path=configs.model_weights_path["cotracker3"])
        trajs_e, _ = model.infer(frames, query_points, resize=(256, 256))
        del model
        return trajs_e.cpu()

    if model_name == "SpeckNet":
        # NOTE: unlike the other models, SpeckNet always queries from frame 0, not `q_frame`.
        query_points = _with_query_frame(query_points, 0)
        model = SPECKNET(device_ids=[0], multi_scale_level=3)
        model.load(configs.model_weights_path["specknet"])
        trajs_e, _ = model.infer(frames, query_points, resize=(512, 512))
        del model
        return trajs_e.cpu()

    if model_name == "LocoTrack":
        query_points = _with_query_frame(query_points, q_frame)
        model = LOCOTRACK(model_size="base", ft_model=True, device_ids=[0])
        model.load(path=configs.model_weights_path["locotrack"])
        trajs_e, _ = model.infer(frames, query_points, resize=(256, 256))
        del model
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


def main(
    model_names: Tuple[str, ...] = AVAILABLE_MODELS,
    data_path: str = "data/lv_sample.pkl",
    q_frame: int = -1,
    add_text: bool = True,
    overlay_label: str = None,
):
    """Quantitative evaluation demo.

    Loads annotated ground-truth trajectories from `data_path`, runs each
    model in `model_names` on the ground-truth query points, and writes a
    side-by-side comparison video with per-model accuracy metrics overlaid
    to `outputs/demo2_output.mp4`.

    Args:
        model_names: Models to run. Any of AVAILABLE_MODELS.
        data_path: Path to a pickle file with 'frames', 'trajs', and
            'visibility' arrays (see the "Training / Fine-tuning" section of
            the README for the expected tensor shapes). `data/lv_sample.pkl`,
            `data/rv_sample.pkl`, and `data/camus_sample.pkl` ship with the repo.
        q_frame: Frame index the ground-truth query points are taken from.
            -1 selects a frame 72% into the sequence.
        add_text: Whether to overlay the model name and metrics on each panel.
        overlay_label: Optional text to overlay on the final concatenated video. If None, no text is added.
    """

    with open(data_path, "rb") as f:
        ds_list = pickle.load(f)

    frames, trajs_g, visibs_g = ds_list["frames"][0], ds_list["trajs"][0], ds_list["visibility"][0]
    if q_frame == -1:
        q_frame = int(frames.shape[0] * 0.72)
    query_points = trajs_g[:, :, q_frame]  # (B, N, 2) ground-truth points at the query frame

    paint_frames = get_resized_frames(frames.squeeze(0).numpy(), CANVAS_SIZE, CANVAS_SIZE)
    gt_frames = viz_utils.visualize_tracking(
        frames=paint_frames, points=trajs_g.squeeze(0).numpy(), gray=True,
        vis_color="green", thickness=2, track_length=0,
    )

    panels = []
    for model_name in model_names:
        model_frames = frames if model_name in GRAYSCALE_MODELS else frames.repeat(1, 1, 1, 1, 3)
        trajs_e = predict(model_name, model_frames, query_points, q_frame)

        seq = viz_utils.visualize_tracking(
            frames=gt_frames, points=trajs_e.squeeze(0).numpy(),
            vis_color="red", thickness=2, track_length=25,
        )

        query_points_txy = _with_query_frame(query_points, q_frame)
        metrics = evaluate.compute_metrics(
            query_points_txy.numpy(), trajs_g.numpy(), visibs_g.numpy(), trajs_e.numpy(), visibs_g.numpy(),
            first=False,
        )

        if add_text:
            seq = add_text_to_frames(seq, model_name, (5, 100), color=(255, 255, 0), thickness=3, font_scale=1.8)
            seq = add_text_to_frames(seq, f"pts_w_1_px: {metrics['pts_within_1'] * 100:0.0f}%",
                                      (5, 160), color=(229, 229, 219), thickness=2, font_scale=1.0)
            seq = add_text_to_frames(seq, f"pts_w_2_px: {metrics['pts_within_2'] * 100:0.0f}%",
                                      (5, 210), color=(229, 229, 219), thickness=2, font_scale=1.0)
            seq = add_text_to_frames(seq, f"pts_w_4_px: {metrics['pts_within_4'] * 100:0.0f}%",
                                      (5, 260), color=(229, 229, 219), thickness=2, font_scale=1.0)
            seq = add_text_to_frames(seq, f"MTE: {metrics['median_traj_error']:0.2f} px",
                                      (5, 310), color=(255, 80, 10), thickness=2, font_scale=1.0)
        panels.append(seq)

    pd_frames = concatenate_videos_in_grid(panels)
    if add_text and overlay_label:
        pd_frames = add_text_to_frames(pd_frames, overlay_label, (800, 800),
                                        color=(0, 255, 10), thickness=3, font_scale=2.5)
    pd_frames = np.concatenate([pd_frames] * 3, axis=0)  # loop 3x for easier viewing

    os.makedirs("outputs", exist_ok=True)
    out_path = "outputs/demo2_output.mp4"
    media.write_video(out_path, pd_frames, fps=25)
    print(f'Saved comparison video to "{out_path}"')


if __name__ == "__main__":
    Fire(main)
