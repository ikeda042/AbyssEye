from __future__ import annotations

from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from scipy.stats import percentileofscore
from skimage.feature import peak_local_max
from sqlalchemy import Column, Float, Integer, LargeBinary, String, create_engine, inspect, text
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.types import JSON as SAJSON

Base = declarative_base()


class RoiRecord(Base):
    __tablename__ = "roi_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_stem = Column(String, nullable=False)
    scale = Column(Float, nullable=False)
    num_rois = Column(Integer, nullable=False)
    roi_id = Column(Integer, nullable=False)
    roi_start_x = Column(Integer, nullable=False)
    roi_start_y = Column(Integer, nullable=False)
    roi_end_x = Column(Integer, nullable=False)
    roi_end_y = Column(Integer, nullable=False)
    roi_center_x = Column(Integer, nullable=False)
    roi_center_y = Column(Integer, nullable=False)
    roi_meta = Column(SAJSON, nullable=False)
    image_width_px = Column(Integer, nullable=False)
    image_height_px = Column(Integer, nullable=False)
    png_blob = Column(LargeBinary, nullable=False)
    manual_label = Column(String, nullable=True)
    ai_label = Column(String, nullable=True)
    ai_model_name = Column(String, nullable=True)


class ROIExtractor:
    """Utility helpers for ROI detection and persistence."""

    HEIGHT = 48
    WIDTH = 48
    GREEN_RATE = 0.07
    MIN_DISTANCE = 0

    @staticmethod
    def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = float((ix2 - ix1) * (iy2 - iy1))
        area_a = float(max(0, ax2 - ax1) * max(0, ay2 - ay1))
        area_b = float(max(0, bx2 - bx1) * max(0, by2 - by1))
        denom = area_a + area_b - inter
        if denom <= 0.0:
            return 0.0
        return inter / denom

    @classmethod
    def _percentile_threshold(cls, green: np.ndarray, green_rate: float | None = None) -> int:
        rate = cls.GREEN_RATE if green_rate is None else green_rate
        h, w = green.shape[:2]
        num_pixels = h * w
        hist = np.histogram(green, bins=256, range=(0, 256))
        cumulative = np.cumsum(hist[0])
        percentile = percentileofscore(cumulative, (1 - rate) * num_pixels, kind="strict")
        return int(percentile * 255 * 0.01)

    @classmethod
    def detect_rois(
        cls,
        img_rgb: np.ndarray,
        *,
        roi_width: int | None = None,
        roi_height: int | None = None,
        green_rate: float | None = None,
        min_distance: int | None = None,
        min_green: int = 30,
        ratio_primary: float = 1.0,
        ratio_secondary: float = 1.5,
        kernel_size: int = 5,
        dilate_iterations: int = 2,
        disallow_overlap: bool = True,
        nms_iou_threshold: float = 0.15,
        iterative_passes: int = 1,
    ) -> list[dict[str, Iterable[int]]]:
        height, width = img_rgb.shape[:2]
        patch_w = max(8, int(cls.WIDTH if roi_width is None else roi_width))
        patch_h = max(8, int(cls.HEIGHT if roi_height is None else roi_height))
        min_dist = max(0, int(cls.MIN_DISTANCE if min_distance is None else min_distance))
        iou_threshold = float(max(0.0, min(0.95, nms_iou_threshold)))
        num_passes = max(1, int(iterative_passes))

        red = img_rgb[:, :, 0].astype(np.float32)
        green = img_rgb[:, :, 1].astype(np.float32)

        thresh = cls._percentile_threshold(green, green_rate=green_rate)

        mask1 = (green > thresh) & (green > float(min_green)) & ((green / (red + 1e-6)) > float(ratio_primary))
        mask2 = (green < thresh) & (green > float(min_green)) & ((green / (red + 1e-6)) >= float(ratio_secondary))
        mask = (mask1 | mask2).astype(np.uint8) * 255

        kernel_edge = max(1, int(kernel_size))
        if kernel_edge % 2 == 0:
            kernel_edge += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_edge, kernel_edge))
        working = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=max(0, int(dilate_iterations)))

        rois: list[dict[str, Iterable[int]]] = []
        kept_boxes: list[tuple[int, int, int, int]] = []
        next_id = 1

        for _ in range(num_passes):
            peaks = peak_local_max(working, min_distance=min_dist)
            if peaks.size == 0:
                break

            tmp = np.zeros_like(mask, dtype=np.uint8)
            for y, x in peaks:
                tmp[y, x] = 1

            nlabels, _, _, centers = cv2.connectedComponentsWithStats(tmp)
            if nlabels <= 1:
                break

            candidates: list[tuple[float, tuple[int, int, int, int]]] = []
            for i in range(1, nlabels):
                xc, yc = int(centers[i][0]), int(centers[i][1])
                ys, xs = yc - patch_h // 2, xc - patch_w // 2
                ye, xe = ys + patch_h, xs + patch_w

                if ys < 0:
                    ys, ye = 0, patch_h
                if xs < 0:
                    xs, xe = 0, patch_w
                if ye > height:
                    ys, ye = max(0, height - patch_h), height
                if xe > width:
                    xs, xe = max(0, width - patch_w), width

                score = float(green[yc, xc]) if 0 <= yc < height and 0 <= xc < width else 0.0
                candidates.append((score, (int(xs), int(ys), int(xe), int(ye))))

            candidates.sort(key=lambda item: item[0], reverse=True)
            accepted_any = False

            for _, box in candidates:
                if disallow_overlap and any(cls._bbox_iou(box, kept) > iou_threshold for kept in kept_boxes):
                    continue
                xs, ys, xe, ye = box
                kept_boxes.append(box)
                rois.append(
                    {
                        "ID": next_id,
                        "ST": [xs, ys],
                        "EN": [xe, ye],
                        "CE": [int((xs + xe) / 2), int((ys + ye) / 2)],
                    }
                )
                next_id += 1
                accepted_any = True

                # Optional iterative mode: clear accepted ROI area and try another pass.
                if num_passes > 1:
                    working[ys:ye, xs:xe] = 0

            if num_passes == 1 or not accepted_any:
                break

        return rois

    @staticmethod
    def _ensure_path(db_path: str | Path) -> Path:
        # 重要: ~ 展開と絶対パス化。親ディレクトリの自動作成。
        path = Path(db_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            path.unlink()
        return path

    @staticmethod
    def _normalize_image_dims(
        img_rgb: np.ndarray,
        image_width_px: int | None,
        image_height_px: int | None,
    ) -> tuple[int, int]:
        if image_width_px is not None and image_height_px is not None:
            return int(image_width_px), int(image_height_px)
        processed_h, processed_w = img_rgb.shape[:2]
        return int(processed_w), int(processed_h)

    @classmethod
    def save_rois_to_db(
        cls,
        img_rgb: np.ndarray,
        rois: list[dict[str, Iterable[int]]],
        db_path: str | Path,
        stem: str,
        *,
        scale: float = 0.5,
        image_width_px: int,
        image_height_px: int,
    ) -> None:
        path = cls._ensure_path(db_path)

        # echo=True でSQLを確認できる。不要なら False に戻す。
        engine = create_engine(f"sqlite:///{path}", echo=False)

        # 新規DBでも念のため「存在すれば」削除→作成
        Base.metadata.drop_all(engine, checkfirst=True)
        Base.metadata.create_all(engine)

        # ここで実際のカラムを検査して保証
        insp = inspect(engine)
        cols = {c["name"] for c in insp.get_columns("roi_records")}
        required = {"image_width_px", "image_height_px"}
        missing = required - cols
        if missing:
            raise RuntimeError(f"roi_records に必須カラムが見つかりません: {sorted(missing)}  @ {path}")

        SessionLocal = sessionmaker(bind=engine)
        session = SessionLocal()

        resolved_width, resolved_height = cls._normalize_image_dims(
            img_rgb, image_width_px, image_height_px
        )
        num_rois = len(rois)

        try:
            for k, roi in enumerate(rois, 1):
                xs, ys = roi["ST"]
                xe, ye = roi["EN"]
                patch_rgb = img_rgb[ys:ye, xs:xe, :]

                ok, buf = cv2.imencode(".png", cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR))
                if not ok:
                    continue
                png_bytes = buf.tobytes()

                filename = f"{stem}_roi_{k:04d}.png"
                roi_meta = {"image": stem, "scale": scale, "filename": filename, **roi}

                record = RoiRecord(
                    image_stem=stem,
                    scale=scale,
                    num_rois=num_rois,
                    roi_id=roi["ID"],
                    roi_start_x=int(xs),
                    roi_start_y=int(ys),
                    roi_end_x=int(xe),
                    roi_end_y=int(ye),
                    roi_center_x=int(roi["CE"][0]),
                    roi_center_y=int(roi["CE"][1]),
                    roi_meta=roi_meta,
                    png_blob=png_bytes,
                    image_width_px=int(resolved_width),
                    image_height_px=int(resolved_height),
                    manual_label=None,
                    ai_label=None,
                    ai_model_name=None,
                )
                session.add(record)

            session.commit()

            # 動作確認用: 1件だけ読み出して幅高さを見る
            any_row = session.execute(text("SELECT image_width_px, image_height_px FROM roi_records LIMIT 1")).fetchone()
            if any_row is None:
                # ROIが0件でもスキーマはできている。ここでは何もしない。
                pass
        finally:
            session.close()
            engine.dispose()
