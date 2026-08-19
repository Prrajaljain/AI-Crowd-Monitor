"""
AI Crowd Monitor — real-time headcount and overcrowding alerts.

Detects people in a video stream with YOLOv8, maintains a live count, and raises a
visible alert when the count crosses a configured threshold. Writes an annotated
video file so output can be used as a demo.

Detection runs at imgsz=1280 rather than the 640 default: people at the back of a
crowd occupy very few pixels, and downscaling loses them entirely.

Counting is a two-pass operation per frame — count every person first, then draw.
Colouring boxes during a single pass tints them by detection order rather than by
the actual crowd size, which looks arbitrary on screen.

Usage
-----
    # webcam
    python crowd_monitor.py --limit 20

    # recorded footage, writing an annotated video
    python crowd_monitor.py --source footage.mp4 --output demo.mp4 --limit 30

    # headless
    python crowd_monitor.py --source footage.mp4 --output out.mp4 --no-preview
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import deque
from pathlib import Path

import cv2
import cvzone
from ultralytics import YOLO

log = logging.getLogger("crowd_monitor")

PERSON_CLASS = "person"
DEFAULT_MODEL = "yolov8l.pt"  # large weights: better recall on small, distant people
INFER_SIZE = 1280             # keeps distant figures above the detection floor
SMOOTHING_WINDOW = 5          # frames to average the count over

SAFE = (0, 200, 90)
ALERT = (0, 0, 235)
OTHER = (235, 130, 0)
PANEL = (40, 40, 40)


class CountSmoother:
    """Rolling mean of the raw per-frame count.

    Raw counts jitter by several people as figures are briefly occluded. Without
    smoothing the display flickers and the alert trips on noise.
    """

    def __init__(self, window: int = SMOOTHING_WINDOW) -> None:
        self._recent: deque[int] = deque(maxlen=window)

    def push(self, count: int) -> int:
        self._recent.append(count)
        return round(sum(self._recent) / len(self._recent))


class AlertState:
    """Tracks threshold breaches, logging once per crossing rather than per frame."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._armed = True

    def update(self, count: int) -> bool:
        """Returns True while over the limit. Logs only on the initial crossing."""
        over = count > self.limit
        if over and self._armed:
            log.warning("overcrowding — %d people (limit %d)", count, self.limit)
            self._armed = False
        elif not over:
            self._armed = True
        return over


class FpsMeter:
    def __init__(self, window: int = 10) -> None:
        self._window = window
        self._count = 0
        self._start = time.monotonic()
        self.value = 0.0

    def tick(self) -> None:
        self._count += 1
        if self._count % self._window == 0:
            now = time.monotonic()
            elapsed = now - self._start
            if elapsed > 0:
                self.value = self._window / elapsed
            self._start = now


def resolve_source(source: str) -> int | str:
    """Camera indices arrive as strings; OpenCV needs an int for those."""
    return int(source) if source.isdigit() else source


def draw_overlay(
    img,
    people: int,
    others: list[tuple[int, int, int, int, str, float]],
    person_boxes: list[tuple[int, int, int, int, float]],
    alerting: bool,
    fps: float,
    limit: int,
    peak: int,
) -> None:
    colour = ALERT if alerting else SAFE

    for x1, y1, x2, y2, label, conf in others:
        cv2.rectangle(img, (x1, y1), (x2, y2), OTHER, 2)
        cvzone.putTextRect(
            img, f"{label} {conf:.2f}", (max(0, x1), max(35, y1)),
            scale=1, thickness=1, colorR=OTHER,
        )

    for x1, y1, x2, y2, conf in person_boxes:
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3)
        cvzone.putTextRect(
            img, f"person {conf:.2f}", (max(0, x1), max(35, y1)),
            scale=1, thickness=1, colorR=colour,
        )

    cvzone.putTextRect(
        img, f"People: {people}", (40, 50),
        scale=2, thickness=2, offset=10, colorR=colour,
    )
    cvzone.putTextRect(
        img, f"limit {limit}   peak {peak}   {fps:.1f} FPS", (40, 105),
        scale=1, thickness=1, offset=8, colorR=PANEL,
    )

    if alerting:
        cvzone.putTextRect(
            img, "OVERCROWDING DETECTED", (40, 175),
            scale=3, thickness=3, colorR=ALERT, offset=12,
        )


def run(
    source: str,
    output: Path | None,
    model_path: str,
    confidence: float,
    limit: int,
    show_others: bool,
    preview: bool,
) -> None:
    model = YOLO(model_path)
    names = model.names  # class-index -> label, straight from the weights

    capture = cv2.VideoCapture(resolve_source(source))
    if not capture.isOpened():
        raise SystemExit(f"could not open source: {source}")

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    src_fps = capture.get(cv2.CAP_PROP_FPS) or 25.0

    writer = None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), src_fps, (1280, 720)
        )
        log.info("writing annotated video to %s", output)

    smoother = CountSmoother()
    alerts = AlertState(limit)
    fps_meter = FpsMeter()
    peak = 0
    frames = 0

    log.info("model=%s imgsz=%d conf=%.2f limit=%d", model_path, INFER_SIZE, confidence, limit)

    try:
        while True:
            ok, img = capture.read()
            if not ok:
                break

            img = cv2.resize(img, (1280, 720))
            results = model(img, imgsz=INFER_SIZE, conf=confidence, verbose=False)

            # Pass 1 — collect. Count before drawing so box colour reflects the
            # final crowd size, not the order detections happened to arrive in.
            person_boxes: list[tuple[int, int, int, int, float]] = []
            others: list[tuple[int, int, int, int, str, float]] = []

            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                    conf = float(box.conf[0])
                    label = names[int(box.cls[0])]

                    if label == PERSON_CLASS:
                        person_boxes.append((x1, y1, x2, y2, conf))
                    elif show_others:
                        others.append((x1, y1, x2, y2, label, conf))

            count = smoother.push(len(person_boxes))
            peak = max(peak, count)
            alerting = alerts.update(count)

            fps_meter.tick()
            frames += 1

            # Pass 2 — draw
            draw_overlay(
                img, count, others, person_boxes,
                alerting, fps_meter.value, limit, peak,
            )

            if writer:
                writer.write(img)

            if preview:
                cv2.imshow("Smart Crowd Monitor", img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        log.info("interrupted")
    finally:
        capture.release()
        if writer:
            writer.release()
        if preview:
            cv2.destroyAllWindows()
        log.info("processed %d frames — peak count %d", frames, peak)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-time crowd detection and overcrowding alerts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", default="0", help="video file path, or camera index")
    parser.add_argument("--output", type=Path, default=None,
                        help="write an annotated MP4 (for demos)")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="YOLO weights; yolov8n.pt is much faster on CPU")
    parser.add_argument("--conf", type=float, default=0.3, help="confidence floor")
    parser.add_argument("--limit", type=int, default=20, help="overcrowding threshold")
    parser.add_argument("--show-others", action="store_true",
                        help="also box non-person classes")
    parser.add_argument("--no-preview", dest="preview", action="store_false",
                        help="run headless")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    run(
        source=args.source,
        output=args.output,
        model_path=args.model,
        confidence=args.conf,
        limit=args.limit,
        show_others=args.show_others,
        preview=args.preview,
    )


if __name__ == "__main__":
    main()
