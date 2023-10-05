# Adapted from:
#   - https://github.com/deepmind/tapnet/blob/b8e0641e3c6a3483060e49df1def87fef16c8d1a/data/visualize.py
#   - https://github.com/deepmind/tapnet/blob/b8e0641e3c6a3483060e49df1def87fef16c8d1a/data/viz_utils.py
#
# The original license is stated below.
#
# ==============================================================================
# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Visualize frames of a random video of the given dataset."""

import colorsys
import io
import os
import pickle
import pathlib
import random
from typing import List, Tuple
from typing import Sequence

import mediapy as media
import numpy as np
from PIL import Image
from absl import app
from absl import flags
from absl import logging


FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_path', None, 'Path to the pickle file.', required=True
)
flags.DEFINE_string(
    'output_path', None, 'Path to the output mp4 video.', required=True
)
flags.DEFINE_integer(
    'video_index', None, 'Index of video in dataset to visualize.', required=False
)

def ensure_dir(dirname):
    """
    Ensure that a directory exists. If it doesn't, create it.
    Parameters
    ----------
    dirname : str or pathlib.Path
        The directory, the existence of which will be ensured
    Returns
    -------
    None
    """
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

# Generate random colormaps for visualizing different points.
def _get_colors(num_colors: int) -> List[Tuple[int, int, int]]:
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


def paint_point_track(
        frames: np.ndarray,
        point_tracks: np.ndarray,
        visibles: np.ndarray,
) -> np.ndarray:
    """Converts a sequence of points to color code video.
    Args:
      frames: [num_frames, height, width, 3], np.uint8, [0, 255]
      point_tracks: [num_points, num_frames, 2], np.float32, [0, width / height]
      visibles: [num_points, num_frames], bool
    Returns:
      video: [num_frames, height, width, 3], np.uint8, [0, 255]
    """
    num_points, num_frames = point_tracks.shape[0:2]
    colormap = _get_colors(num_colors=num_points)
    height, width = frames.shape[1:3]
    dot_size_as_fraction_of_min_edge = 0.015
    radius = int(round(min(height, width) * dot_size_as_fraction_of_min_edge))
    diam = radius * 2 + 1
    quadratic_y = np.square(np.arange(diam)[:, np.newaxis] - radius - 1)
    quadratic_x = np.square(np.arange(diam)[np.newaxis, :] - radius - 1)
    icon = (quadratic_y + quadratic_x) - (radius ** 2) / 2.0
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
                image[y1:y_ub, x1:x_ub, :] = (1 - patch) * image[
                                                           y1:y_ub, x1:x_ub, :
                                                           ] + patch * np.array(colormap[i])[np.newaxis, np.newaxis, :]

            # Remove the pad
            video[t] = image[radius + 1: -radius - 1, radius + 1: -radius - 1].astype(np.uint8)
    return video


def main(argv: Sequence[str]) -> None:
    del argv

    logging.info('Loading data from %s. This takes time.', FLAGS.input_path)
    with open(FLAGS.input_path, 'rb') as f:
        data = pickle.load(f)
        if isinstance(data, dict):
            data = list(data.values())
    ensure_dir(os.path.dirname(FLAGS.output_path))

    idx = FLAGS.video_index
    if idx is None:
        idx = random.randint(0, len(data) - 1)
    video = data[idx]
    print(f"Randomly selected video with index {idx} out of {len(data)} videos.")

    frames = video['video']

    if isinstance(frames[0], bytes):
        # Tapnet is stored and JPEG bytes rather than `np.ndarray`s.
        def decode(frame):
            byteio = io.BytesIO(frame)
            img = Image.open(byteio)
            return np.array(img)

        frames = np.array([decode(frame) for frame in frames])

    if frames.shape[1] > 360:
        frames = media.resize_video(frames, (360, 640))

    scale_factor = np.array(frames.shape[2:0:-1])[np.newaxis, np.newaxis, :]
    painted_frames = paint_point_track(
        frames,
        video['points'] * scale_factor,
        ~video['occluded'],
    )

    media.write_video(FLAGS.output_path, painted_frames, fps=25)
    logging.info('Examplar point visualization saved to %s', FLAGS.output_path)


if __name__ == '__main__':
    app.run(main)