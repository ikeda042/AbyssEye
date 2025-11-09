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


class ROIExtractor:
    """Utility helpers for ROI detection and persistence."""

    HEIGHT = 48
    WIDTH = 48
    GREEN_RATE = 0.07
    MIN_DISTANCE = 0

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
    def detect_rois(cls, img_rgb: np.ndarray) -> list[dict[str, Iterable[int]]]:
        height, width = img_rgb.shape[:2]
        red = img_rgb[:, :, 0].astype(np.float32)
        green = img_rgb[:, :, 1].astype(np.float32)

        thresh = cls._percentile_threshold(green)

        mask1 = (green > thresh) & (green > 30) & ((green / (red + 1e-6)) > 1.0)
        mask2 = (green < thresh) & (green > 30) & ((green / (red + 1e-6)) >= 1.5)
        mask = (mask1 | mask2).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
        peaks = peak_local_max(dilated, min_distance=cls.MIN_DISTANCE)

        tmp = np.zeros_like(mask, dtype=np.uint8)
        for y, x in peaks:
            tmp[y, x] = 1

        nlabels, _, _, centers = cv2.connectedComponentsWithStats(tmp)

        rois: list[dict[str, Iterable[int]]] = []
        for i in range(1, nlabels):
            xc, yc = int(centers[i][0]), int(centers[i][1])
            ys, xs = yc - cls.HEIGHT // 2, xc - cls.WIDTH // 2
            ye, xe = yc + cls.HEIGHT // 2, xc + cls.WIDTH // 2

            if ys < 0:
                ys, ye = 0, cls.HEIGHT
            if xs < 0:
                xs, xe = 0, cls.WIDTH
            if ye > height:
                ys, ye = height - cls.HEIGHT, height
            if xe > width:
                xs, xe = width - cls.WIDTH, width

            rois.append(
                {
                    "ID": i,
                    "ST": [int(xs), int(ys)],
                    "EN": [int(xe), int(ye)],
                    "CE": [int((xs + xe) / 2), int((ys + ye) / 2)],
                }
            )
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
