"""Render a video explaining how the object-anticipation algorithm works.

Overlays, on the egocentric RGB stream:
  - 2D object detections (yellow) and the eye-gaze point (red)
  - the object currently in focus (green)
  - the LLM's live top-3 prediction + inferred goal (from a predictions folder)
  - the ground-truth object the user actually moves (from objects_that_moved.json)

Uses existing prediction JSONs — no API key needed.

Usage:
  python tools/make_algorithm_video.py \
      --sequence /path/to/seq \
      --predictions /path/to/seq/time_2_highdot_0.9_highdotcount_60_dist_3_distcount_30 \
      --out docs/algorithm_demo.mp4
"""
import argparse, json, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

from projectaria_tools.projects.adt import (
    AriaDigitalTwinDataPathsProvider, AriaDigitalTwinDataProvider,
)
from projectaria_tools.core.stream_id import StreamId
from projectaria_tools.core.mps.utils import get_gaze_vector_reprojection


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", required=True)
    p.add_argument("--predictions", required=True, help="folder with large_language_model_*.json")
    p.add_argument("--out", default="docs/algorithm_demo.mp4")
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--stride", type=int, default=4, help="use every Nth frame")
    p.add_argument("--factor", type=int, default=3, help="image downsample factor")
    return p.parse_args()


def load_json(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur); cur = w
    if cur:
        lines.append(cur)
    return lines


def main():
    args = parse_args()
    gtp = AriaDigitalTwinDataPathsProvider(args.sequence)
    gt = AriaDigitalTwinDataProvider(gtp.get_datapaths_by_device_num(0))
    rgb = StreamId("214-1")
    t0, t1 = gt.get_start_time_ns(), gt.get_end_time_ns()
    ts_all = [t for t in gt.get_aria_device_capture_timestamps_ns(rgb) if t0 <= t <= t1]
    ts_sel = ts_all[:: args.stride]
    start_s = ts_all[0] / 1e9

    cam = gt.get_aria_camera_calibration(rgb)
    new_res = (cam.get_image_size() / args.factor).astype(int)
    cam_rs = cam.rescale(new_res, 1.0 / args.factor)
    dev_calib = gt.raw_data_provider_ptr().get_device_calibration()

    # algorithm outputs
    preds = load_json(os.path.join(args.predictions, "large_language_model_prediction.json"), {})
    goals = load_json(os.path.join(args.predictions, "large_language_model_goals.json"), {})
    # objects_that_moved.json lives under results/gt/<seq> (or the sequence folder).
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _name = os.path.basename(os.path.normpath(args.sequence))
    _moved_path = next((p for p in [
        os.path.join(_repo, "results", "gt", _name, "objects_that_moved.json"),
        os.path.join(args.sequence, "objects_that_moved.json"),
    ] if os.path.exists(p)), os.path.join(args.sequence, "objects_that_moved.json"))
    moved = load_json(_moved_path, {})
    pred_times = sorted(float(k) for k in preds)
    moved_times = sorted((float(k), v) for k, v in moved.items())

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 13)
        fontb = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 15)
    except Exception:
        font = fontb = ImageFont.load_default()

    W = int(new_res[1])          # width after 90deg rotation = original height
    PANEL = 96
    writer = imageio.get_writer(os.path.expanduser(args.out), fps=args.fps, codec="libx264",
                                quality=5)

    def latest(times, t):
        prev = None
        for x in times:
            xt = x[0] if isinstance(x, tuple) else x
            if xt <= t:
                prev = x
            else:
                break
        return prev

    n = 0
    for ts in ts_sel:
        img_dt = gt.get_aria_image_by_timestamp_ns(ts, rgb)
        if not img_dt.is_valid():
            continue
        cur_s = ts / 1e9 - start_s
        im = Image.fromarray(img_dt.data().to_numpy_array()).resize((int(new_res[0]), int(new_res[1])))
        dr = ImageDraw.Draw(im)

        # gaze
        gpx = None
        try:
            eg = gt.get_eyegaze_by_timestamp_ns(ts)
            if eg.is_valid():
                g = get_gaze_vector_reprojection(eg.data(), cam.get_label(), dev_calib, cam_rs, eg.data().depth, False)
                if g is not None:
                    gpx = (float(g[0]), float(g[1]))
        except Exception:
            pass

        # 2D boxes + focus highlight
        looked = None
        farea = float(new_res[0] * new_res[1])
        try:
            bb = gt.get_object_2d_boundingboxes_by_timestamp_ns(ts, rgb)
            if bb.is_valid():
                for iid, box in bb.data().items():
                    br = np.array(box.box_range) / args.factor
                    if getattr(box, "visibility_ratio", 1.0) < 0.2:
                        continue
                    x0, x1, y0, y1 = br
                    if (x1 - x0) * (y1 - y0) > 0.5 * farea:
                        continue
                    hit = gpx is not None and x0 <= gpx[0] <= x1 and y0 <= gpx[1] <= y1
                    if hit and (looked is None or (x1 - x0) * (y1 - y0) < looked[1]):
                        looked = (iid, (x1 - x0) * (y1 - y0), (x0, y0, x1, y1))
                    dr.rectangle([x0, y0, x1, y1], outline=(255, 215, 0), width=1)
        except Exception:
            pass
        if looked:
            x0, y0, x1, y1 = looked[2]
            dr.rectangle([x0, y0, x1, y1], outline=(0, 230, 0), width=3)
        if gpx:
            r = 6
            dr.ellipse([gpx[0]-r, gpx[1]-r, gpx[0]+r, gpx[1]+r], outline=(255, 0, 0), width=3)

        im = im.transpose(Image.ROTATE_270)

        # bottom info panel
        H = im.size[1]
        canvas = Image.new("RGB", (im.size[0], H + PANEL), (12, 12, 14))
        canvas.paste(im, (0, 0))
        d2 = ImageDraw.Draw(canvas)

        lp = latest(pred_times, cur_s)
        top3 = preds.get(f"{lp}".rstrip("0").rstrip(".") if False else (("%g" % lp) if lp is not None else ""), None)
        if top3 is None and lp is not None:
            # keys may be like "4.866"; match by formatting
            for k in preds:
                if abs(float(k) - lp) < 1e-6:
                    top3 = preds[k]; break
        goal = ""
        if lp is not None:
            for k in goals:
                if abs(float(k) - lp) < 1e-6:
                    goal = goals[k]; break

        # recent GT movement (within 1.5s window)
        gt_evt = None
        for mt, name in moved_times:
            if mt <= cur_s <= mt + 1.5:
                gt_evt = name; break

        y = H + 5
        d2.text((6, y), f"t = {cur_s:5.1f}s", fill=(255, 255, 255), font=fontb); y += 17
        ptxt = "  ".join(f"{i+1}. {o}" for i, o in enumerate(top3)) if top3 else "(LLM not yet activated)"
        d2.text((6, y), "Prediction (top-3):", fill=(120, 200, 255), font=font)
        d2.text((140, y), ptxt, fill=(255, 255, 255), font=font); y += 16
        if goal:
            gl = wrap("Goal: " + goal, 78)[:2]
            for ln in gl:
                d2.text((6, y), ln, fill=(180, 180, 180), font=font); y += 14
        if gt_evt:
            d2.text((6, H + PANEL - 16), f"→ User actually moves: {gt_evt}", fill=(0, 230, 0), font=fontb)

        writer.append_data(np.array(canvas))
        n += 1

    writer.close()
    print(f"WROTE {args.out}  ({n} frames @ {args.fps}fps, {os.path.getsize(os.path.expanduser(args.out))/1024:.0f} KB)")


if __name__ == "__main__":
    main()
