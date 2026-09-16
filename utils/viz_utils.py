import colorsys
import random
from typing import List, Tuple, Optional
import numpy as np
import cv2 
from skimage.color import gray2rgb


def visualize_tracking(
    frames: np.ndarray, 
    points: np.ndarray, 
    tracking_quality: np.ndarray = None,
    vis_color='random',
    color_map: np.ndarray = None,
    gray: bool = False,
    alpha: float = 1.0,
    track_length: int = 0,
    thickness: int = 2,
) -> np.ndarray:

    num_points, num_frames = points.shape[:2]
    height, width = frames.shape[1:3]

    if gray and frames.shape[-1] != 3:
        frames = gray2rgb(frames.squeeze())

    radius = max(6, int(0.006 * min(height, width)))

    quality_colors = {
        0: np.array([255, 0, 0]),
        1: np.array([255, 255, 0]),
        2: np.array([0, 255, 0]),
    }

    video = frames.copy()

    # Stable random colors
    if vis_color == 'random' and tracking_quality is None and color_map is None:
        rand_colors = np.random.randint(0, 256, size=(num_points, 3))

    for t in range(num_frames):
        overlay = np.zeros_like(video[t], dtype=np.uint8)
        t_start = max(1, t - track_length)

        for i in range(num_points):

            # -------------------------------------------------
            # Resolve color ONCE (fixes UnboundLocalError)
            # -------------------------------------------------
            if tracking_quality is not None:
                color = quality_colors.get(
                    int(tracking_quality[i, t]),
                    np.array([255, 255, 255])
                )

            elif color_map is not None:
                color = np.asarray(color_map[i])

            elif isinstance(vis_color, (list, tuple, np.ndarray)):
                color = np.asarray(vis_color)

            else:
                if vis_color == 'random':
                    color = rand_colors[i]
                elif vis_color == 'red':
                    color = quality_colors[0]
                elif vis_color == 'yellow':
                    color = quality_colors[1]
                elif vis_color == 'green':
                    color = quality_colors[2]
                else:
                    raise ValueError(f"Unknown vis_color: {vis_color}")

            color = color.astype(np.uint8)

            # -------------------------------------------------
            # Draw track lines
            # -------------------------------------------------
            for tt in range(t_start, t):
                fade = (tt - t_start + 1) / max(1, (t - t_start))

                x0n, y0n = points[i, tt - 1]
                x1n, y1n = points[i, tt]

                x0 = int(np.clip(x0n * width, 0, width - 1))
                y0 = int(np.clip(y0n * height, 0, height - 1))
                x1 = int(np.clip(x1n * width, 0, width - 1))
                y1 = int(np.clip(y1n * height, 0, height - 1))

                faded_color = (color * fade).astype(np.uint8)

                cv2.line(
                    overlay,
                    (x0, y0),
                    (x1, y1),
                    faded_color.tolist(),
                    thickness=thickness,
                    lineType=cv2.LINE_AA
                )

            # -------------------------------------------------
            # Draw dot (current position)
            # -------------------------------------------------
            xc = int(points[i, t, 0] * width)
            yc = int(points[i, t, 1] * height)

            cv2.circle(
                overlay,
                (xc, yc),
                radius=radius,
                color=color.tolist(),
                thickness=-1
            )

        video[t] = cv2.addWeighted(video[t], 1.0, overlay, alpha, 0)

    return video



# Generate random colormaps for visualizing different points.
def get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
  """Gets colormap for points."""
  colors = []
  for i in np.arange(0.0, 360.0, 360.0 / num_colors):
    hue = i / 360.0
    lightness = (50 + np.random.rand() * 10) / 100.0
    saturation = (90 + np.random.rand() * 10) / 100.0
    color = colorsys.hls_to_rgb(hue, lightness, saturation)
    colors.append(
        (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
    )
  random.shuffle(colors)
  return colors


def get_dynamic_colors(num_points, num_frames):
    """Generates dynamic colors for each trajectory across frames.
    
    Args:
        num_points: The number of trajectories/points.
        num_frames: The number of frames.

    Returns:
        A color map of shape [num_points, num_frames, 3] where each
        point has a unique RGB color that changes across frames.
    """
    np.random.seed(42)  # Fixed seed for repeatability of color patterns

    # Generate base colors for each trajectory
    base_colors = np.random.randint(0, 255, size=(num_points, 3))

    # Create a dynamic color variation over frames
    frame_offsets = np.linspace(0, 1, num_frames)
    dynamic_colors = np.zeros((num_points, num_frames, 3), dtype=np.uint8)

    for i in range(num_points):
        for j in range(num_frames):
            # Add a small variation in color for each frame
            dynamic_colors[i, j] = (base_colors[i] + (frame_offsets[j] * 100) % 255).astype(np.uint8)

    return dynamic_colors


def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibles: np.ndarray,
    colormap: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Converts a sequence of points to color-coded video with dynamic colors.

    Args:
        frames: [num_frames, height, width, 3], np.uint8, [0, 255]
        point_tracks: [num_points, num_frames, 2], np.float32, [0, width / height]
        visibles: [num_points, num_frames], bool
        dynamic_colormap: colormap for points, each point has a different dynamic RGB color.

    Returns:
        video: [num_frames, height, width, 3], np.uint8, [0, 255]
    """
    num_points, num_frames = point_tracks.shape[0:2]
    
    # Assign dynamic color to each trajectory across frames
    if colormap is None:
        colormap = get_dynamic_colors(num_points, num_frames)
    
    height, width = frames.shape[1:3]
    dot_size_as_fraction_of_min_edge = 0.015
    radius = int(round(min(height, width) * dot_size_as_fraction_of_min_edge))
    diam = radius * 2 + 1
    quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
    quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
    icon = (quadratic_y + quadratic_x) - (radius**2) / 2.0
    sharpness = 0.15
    icon = np.clip(icon / (radius * 2 * sharpness), 0, 1)
    icon = 1 - icon[:, :, np.newaxis]
    icon1 = np.pad(icon, [(0, 1), (0, 1), (0, 0)])
    icon2 = np.pad(icon, [(1, 0), (0, 1), (0, 0)])
    icon3 = np.pad(icon, [(0, 1), (1, 0), (0, 0)])
    icon4 = np.pad(icon, [(1, 0), (1, 0), (0, 0)])

    video = frames.copy()
    for t in range(num_frames):
        # Pad so that points that extend outside the image frame don't crash us
        image = np.pad(
            video[t],
            [
                (radius + 1, radius + 1),
                (radius + 1, radius + 1),
                (0, 0),
            ],
        )
        for i in range(num_points):
            # The icon is centered at the center of a pixel, but the input coordinates
            # are raster coordinates.  Therefore, to render a point at (1,1) (which
            # lies on the corner between four pixels), we need 1/4 of the icon placed
            # centered on the 0'th row, 0'th column, etc.  We need to subtract
            # 0.5 to make the fractional position come out right.
            x, y = point_tracks[i, t, :] + 0.5
            x = min(max(x, 0.0), width)
            y = min(max(y, 0.0), height)

            if visibles[i, t]:
                x1, y1 = np.floor(x).astype(np.int32), np.floor(y).astype(np.int32)
                x2, y2 = x1 + 1, y1 + 1

                # bilinear interpolation
                patch = (
                    icon1 * (x2 - x) * (y2 - y)
                    + icon2 * (x2 - x) * (y - y1)
                    + icon3 * (x - x1) * (y2 - y)
                    + icon4 * (x - x1) * (y - y1)
                )
                x_ub = x1 + 2 * radius + 2
                y_ub = y1 + 2 * radius + 2
                # Apply the dynamic color for each trajectory for each frame
                image[y1:y_ub, x1:x_ub, :] = (1 - patch) * image[
                    y1:y_ub, x1:x_ub, :
                ] + patch * np.array(colormap[i, t])[np.newaxis, np.newaxis, :]

        # Remove the pad
        video[t] = image[
            radius + 1 : -radius - 1, radius + 1 : -radius - 1
        ].astype(np.uint8)
    return video