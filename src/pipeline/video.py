"""Optional annotated-MP4 exporter for the ``--make_video`` flag.

Consolidates the video machinery that used to be scattered across the per-frame
loop (writer state, frame grab, overlay, pause scheduling, write, finalize).
A no-op when --make_video is off or imageio is unavailable.
"""
import os

import numpy as np

try:
    import imageio.v2 as iio   # v2 API: get_writer/append_data (v3 lacks them)
except Exception:
    iio = None
try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None


class VideoRecorder:
    def __init__(self, args, repo_root, sequence_path):
        self.args = args
        self.repo_root = repo_root
        self.sequence_path = sequence_path
        self.enabled = bool(args.make_video) and iio is not None
        self.writer = None
        self.frame_size = None
        self.pause_frames = 0

    def grab(self, image_with_dt):
        """Return an RGB uint8 frame for the current image, or None."""
        if not self.enabled or not image_with_dt.is_valid():
            return None
        try:
            frame = image_with_dt.data().to_numpy_array()
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            if self.writer is None:
                self.frame_size = (frame.shape[1], frame.shape[0])
            return frame
        except Exception:
            return None

    @staticmethod
    def _overlay(frame_np, lines):
        if Image is None:
            return frame_np
        img = Image.fromarray(frame_np)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("Arial.ttf", 22)
        except Exception:
            font = ImageFont.load_default()
        padding, line_height = 8, 24
        box_width = max([draw.textlength(l, font=font) for l in lines] + [0]) + 2 * padding
        box_height = line_height * len(lines) + 2 * padding
        draw.rectangle([0, 0, box_width, box_height], fill=(0, 0, 0, 180))
        for idx, text in enumerate(lines):
            draw.text((padding, padding + idx * line_height), text, fill=(255, 255, 255), font=font)
        return np.array(img)

    def mark_activation(self, frame_np, lines):
        """Draw the prediction overlay and schedule a pause; returns the new frame."""
        if not self.enabled or frame_np is None:
            return frame_np
        frame_np = self._overlay(frame_np, lines)
        self.pause_frames = int(max(0.0, self.args.pause_duration) * max(1, self.args.fps))
        return frame_np

    def write(self, frame_np):
        if not self.enabled or frame_np is None:
            return
        try:
            if self.writer is None:
                if self.args.video_out:
                    video_path = os.path.expanduser(self.args.video_out)
                else:
                    tmp_dir = os.path.join(self.repo_root, 'results', 'predictions',
                                           os.path.basename(os.path.normpath(self.sequence_path)), 'tmp')
                    os.makedirs(tmp_dir, exist_ok=True)
                    video_path = os.path.join(tmp_dir, 'preview.mp4')
                self.writer = iio.get_writer(video_path, fps=max(1, self.args.fps), codec='libx264')
            self.writer.append_data(frame_np)
            if self.pause_frames > 0:
                for _ in range(self.pause_frames):
                    self.writer.append_data(frame_np)
                self.pause_frames = 0
        except Exception:
            pass

    def finalize(self, predictions_folder, parameter_folder_name):
        try:
            if self.writer is not None:
                self.writer.close()
                if not self.args.video_out and self.frame_size is not None:
                    src_tmp = os.path.join(self.repo_root, 'results', 'predictions',
                                           os.path.basename(os.path.normpath(self.sequence_path)), 'tmp', 'preview.mp4')
                    dst_path = os.path.join(predictions_folder, f"preview_{parameter_folder_name}.mp4")
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    if os.path.exists(src_tmp):
                        try:
                            os.replace(src_tmp, dst_path)
                        except Exception:
                            pass
        except Exception:
            pass
