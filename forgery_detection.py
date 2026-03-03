"""
Image Forgery Detection Module

Provides classical computer vision techniques to detect tampering in images:
- Error Level Analysis (ELA)
- Copy-Move Detection (block matching + keypoint matching)
- JPEG Ghost Detection
- CFA (Color Filter Array) Pattern Analysis
- Noise Inconsistency Analysis
- EXIF / Metadata Analysis
- Double JPEG Compression Detection
- Lighting / Shadow Consistency Analysis
- Chromatic Aberration Analysis
"""

from __future__ import annotations

import io
import json
import math
import struct
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image
from PIL.ExifTags import TAGS
from scipy import fftpack, ndimage


# ---------------------------------------------------------------------------
# Data classes for structured results
# ---------------------------------------------------------------------------

class Severity(Enum):
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()


@dataclass
class AnalysisResult:
    """Result from a single forgery-detection analysis."""

    method: str
    suspicious: bool
    confidence: float  # 0.0 – 1.0
    severity: Severity
    details: dict[str, Any] = field(default_factory=dict)
    heatmap: NDArray[np.float64] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = {
            "method": self.method,
            "suspicious": self.suspicious,
            "confidence": round(self.confidence, 4),
            "severity": self.severity.name,
            "details": self.details,
        }
        if self.heatmap is not None:
            d["heatmap_shape"] = list(self.heatmap.shape)
        return d


@dataclass
class ForgeryReport:
    """Aggregated report from all analyses."""

    results: list[AnalysisResult] = field(default_factory=list)

    @property
    def suspicious(self) -> bool:
        return any(r.suspicious for r in self.results)

    @property
    def overall_confidence(self) -> float:
        if not self.results:
            return 0.0
        return max(r.confidence for r in self.results)

    @property
    def max_severity(self) -> Severity:
        if not self.results:
            return Severity.LOW
        return max((r.severity for r in self.results), key=lambda s: s.value)

    def summary(self) -> str:
        lines = [f"Forgery Report  —  suspicious={self.suspicious}  "
                 f"confidence={self.overall_confidence:.2%}  "
                 f"severity={self.max_severity.name}"]
        for r in self.results:
            flag = "!!" if r.suspicious else "ok"
            lines.append(f"  [{flag}] {r.method}: confidence={r.confidence:.2%}  "
                         f"severity={r.severity.name}")
            for k, v in r.details.items():
                lines.append(f"        {k}: {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suspicious": self.suspicious,
            "overall_confidence": round(self.overall_confidence, 4),
            "max_severity": self.max_severity.name,
            "results": [r.to_dict() for r in self.results],
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)


# ---------------------------------------------------------------------------
# Main detector class
# ---------------------------------------------------------------------------

class ForgeryDetector:
    """Detect image forgery using multiple classical computer-vision methods.

    Usage::

        detector = ForgeryDetector()
        report = detector.analyze("photo.jpg")
        print(report.summary())

    Individual analyses can also be run directly::

        result = detector.error_level_analysis("photo.jpg")
    """

    def __init__(
        self,
        ela_quality: int = 90,
        ela_threshold: float = 15.0,
        block_size: int = 16,
        copy_move_threshold: float = 0.9,
        ghost_quality_range: tuple[int, int] = (50, 98),
        ghost_quality_step: int = 2,
        noise_block_size: int = 32,
        noise_inconsistency_threshold: float = 2.0,
    ) -> None:
        self.ela_quality = ela_quality
        self.ela_threshold = ela_threshold
        self.block_size = block_size
        self.copy_move_threshold = copy_move_threshold
        self.ghost_quality_range = ghost_quality_range
        self.ghost_quality_step = ghost_quality_step
        self.noise_block_size = noise_block_size
        self.noise_inconsistency_threshold = noise_inconsistency_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, image_path: str | Path) -> ForgeryReport:
        """Run all available analyses and return an aggregated report."""
        image_path = Path(image_path)
        report = ForgeryReport()

        report.results.append(self.error_level_analysis(image_path))
        report.results.append(self.copy_move_detection(image_path))
        report.results.append(self.noise_inconsistency_analysis(image_path))
        report.results.append(self.cfa_pattern_analysis(image_path))
        report.results.append(self.exif_metadata_analysis(image_path))
        report.results.append(self.chromatic_aberration_analysis(image_path))

        if self._is_jpeg(image_path):
            report.results.append(self.jpeg_ghost_detection(image_path))
            report.results.append(self.double_jpeg_detection(image_path))

        report.results.append(self.lighting_consistency_analysis(image_path))

        return report

    # ------------------------------------------------------------------
    # 1. Error Level Analysis (ELA)
    # ------------------------------------------------------------------

    def error_level_analysis(self, image_path: str | Path) -> AnalysisResult:
        """Detect tampering by comparing JPEG re-compression error levels.

        Re-saves the image at a known quality and measures per-pixel
        differences.  Tampered regions typically show higher error because
        they were compressed a different number of times or at a different
        quality than the host image.
        """
        image_path = Path(image_path)
        original = np.array(Image.open(image_path).convert("RGB"))

        buf = io.BytesIO()
        Image.fromarray(original).save(buf, format="JPEG", quality=self.ela_quality)
        buf.seek(0)
        resaved = np.array(Image.open(buf).convert("RGB"))

        diff = np.abs(original.astype(np.float64) - resaved.astype(np.float64))
        ela_image = diff.mean(axis=2)  # average across channels

        scale = 255.0 / max(ela_image.max(), 1e-9)
        heatmap = (ela_image * scale).astype(np.float64)

        mean_err = float(ela_image.mean())
        max_err = float(ela_image.max())
        std_err = float(ela_image.std())

        suspicious = std_err > self.ela_threshold
        confidence = float(np.clip(std_err / (self.ela_threshold * 3), 0.0, 1.0))

        return AnalysisResult(
            method="Error Level Analysis (ELA)",
            suspicious=suspicious,
            confidence=confidence,
            severity=self._severity_from_confidence(confidence),
            details={
                "mean_error": round(mean_err, 2),
                "max_error": round(max_err, 2),
                "std_error": round(std_err, 2),
                "quality_used": self.ela_quality,
            },
            heatmap=heatmap,
        )

    # ------------------------------------------------------------------
    # 2. Copy-Move Detection
    # ------------------------------------------------------------------

    def copy_move_detection(self, image_path: str | Path) -> AnalysisResult:
        """Detect duplicated (cloned) regions using ORB keypoint matching."""
        image_path = Path(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create(nfeatures=5000)
        keypoints, descriptors = orb.detectAndCompute(gray, None)

        if descriptors is None or len(keypoints) < 2:
            return AnalysisResult(
                method="Copy-Move Detection",
                suspicious=False,
                confidence=0.0,
                severity=Severity.LOW,
                details={"keypoints_found": len(keypoints) if keypoints else 0,
                         "matches": 0},
            )

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(descriptors, descriptors, k=2)

        min_distance_px = max(gray.shape) * 0.05
        good_matches: list[cv2.DMatch] = []
        for m, n in matches:
            if m.queryIdx == m.trainIdx:
                continue
            pt1 = np.array(keypoints[m.queryIdx].pt)
            pt2 = np.array(keypoints[m.trainIdx].pt)
            dist = np.linalg.norm(pt1 - pt2)
            if dist > min_distance_px and m.distance < self.copy_move_threshold * n.distance:
                good_matches.append(m)

        heatmap = np.zeros(gray.shape, dtype=np.float64)
        for m in good_matches:
            pt = keypoints[m.trainIdx].pt
            cv2.circle(heatmap, (int(pt[0]), int(pt[1])), 15, 1.0, -1)

        num_matches = len(good_matches)
        suspicious = num_matches > 10
        confidence = float(np.clip(num_matches / 100.0, 0.0, 1.0))

        return AnalysisResult(
            method="Copy-Move Detection",
            suspicious=suspicious,
            confidence=confidence,
            severity=self._severity_from_confidence(confidence),
            details={
                "keypoints_found": len(keypoints),
                "suspicious_matches": num_matches,
            },
            heatmap=heatmap,
        )

    # ------------------------------------------------------------------
    # 3. JPEG Ghost Detection
    # ------------------------------------------------------------------

    def jpeg_ghost_detection(self, image_path: str | Path) -> AnalysisResult:
        """Find regions saved at a different JPEG quality than the host.

        Re-compresses the image at many quality levels and looks for regions
        whose error is minimised at a quality different from the image's own.
        """
        image_path = Path(image_path)
        original = np.array(Image.open(image_path).convert("RGB"), dtype=np.float64)
        h, w = original.shape[:2]

        q_lo, q_hi = self.ghost_quality_range
        qualities = range(q_lo, q_hi + 1, self.ghost_quality_step)

        error_volume = np.zeros((len(qualities), h, w), dtype=np.float64)

        for idx, q in enumerate(qualities):
            buf = io.BytesIO()
            Image.fromarray(original.astype(np.uint8)).save(buf, format="JPEG", quality=q)
            buf.seek(0)
            recompressed = np.array(Image.open(buf).convert("RGB"), dtype=np.float64)
            diff = np.mean((original - recompressed) ** 2, axis=2)
            error_volume[idx] = diff

        min_q_map = np.argmin(error_volume, axis=0)
        overall_min_q_idx = int(np.median(min_q_map))

        deviation = np.abs(min_q_map.astype(np.float64) - overall_min_q_idx)
        heatmap = deviation / max(deviation.max(), 1e-9) * 255.0

        ghost_ratio = float(np.mean(deviation > len(qualities) * 0.25))
        suspicious = ghost_ratio > 0.05
        confidence = float(np.clip(ghost_ratio / 0.15, 0.0, 1.0))

        quality_list = list(qualities)
        dominant_quality = quality_list[overall_min_q_idx] if overall_min_q_idx < len(quality_list) else -1

        return AnalysisResult(
            method="JPEG Ghost Detection",
            suspicious=suspicious,
            confidence=confidence,
            severity=self._severity_from_confidence(confidence),
            details={
                "dominant_quality": dominant_quality,
                "ghost_region_ratio": round(ghost_ratio, 4),
            },
            heatmap=heatmap,
        )

    # ------------------------------------------------------------------
    # 4. CFA (Color Filter Array) Pattern Analysis
    # ------------------------------------------------------------------

    def cfa_pattern_analysis(self, image_path: str | Path) -> AnalysisResult:
        """Detect disrupted Bayer demosaicing patterns.

        Authentic camera images retain a predictable interpolation pattern
        from the Bayer CFA.  Tampered regions break this pattern.
        """
        image_path = Path(image_path)
        img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float64)
        h, w, _ = img.shape

        green = img[:, :, 1]

        predicted = np.zeros_like(green)
        predicted[1:-1, 1:-1] = (
            green[:-2, 1:-1] + green[2:, 1:-1] +
            green[1:-1, :-2] + green[1:-1, 2:]
        ) / 4.0

        residual = np.abs(green - predicted)
        residual[:1, :] = 0
        residual[-1:, :] = 0
        residual[:, :1] = 0
        residual[:, -1:] = 0

        bs = self.noise_block_size
        block_h, block_w = h // bs, w // bs
        block_variances = np.zeros((block_h, block_w), dtype=np.float64)
        for bi in range(block_h):
            for bj in range(block_w):
                block = residual[bi * bs:(bi + 1) * bs, bj * bs:(bj + 1) * bs]
                block_variances[bi, bj] = block.var()

        if block_variances.size == 0:
            return AnalysisResult(
                method="CFA Pattern Analysis",
                suspicious=False,
                confidence=0.0,
                severity=Severity.LOW,
                details={"note": "Image too small for CFA analysis"},
            )

        median_var = float(np.median(block_variances))
        mad = float(np.median(np.abs(block_variances - median_var)))
        if mad < 1e-9:
            mad = 1e-9

        outlier_map = np.abs(block_variances - median_var) / mad
        outlier_ratio = float(np.mean(outlier_map > 5.0))

        heatmap_small = outlier_map / max(outlier_map.max(), 1e-9) * 255.0
        heatmap = cv2.resize(heatmap_small, (w, h), interpolation=cv2.INTER_NEAREST)

        suspicious = outlier_ratio > 0.05
        confidence = float(np.clip(outlier_ratio / 0.15, 0.0, 1.0))

        return AnalysisResult(
            method="CFA Pattern Analysis",
            suspicious=suspicious,
            confidence=confidence,
            severity=self._severity_from_confidence(confidence),
            details={
                "median_variance": round(median_var, 4),
                "outlier_ratio": round(outlier_ratio, 4),
            },
            heatmap=heatmap,
        )

    # ------------------------------------------------------------------
    # 5. Noise Inconsistency Analysis
    # ------------------------------------------------------------------

    def noise_inconsistency_analysis(self, image_path: str | Path) -> AnalysisResult:
        """Estimate local noise variance across the image.

        Authentic images have relatively uniform sensor noise.  Spliced
        regions from a different source often have a measurably different
        noise level.
        """
        image_path = Path(image_path)
        img = np.array(Image.open(image_path).convert("L"), dtype=np.float64)
        h, w = img.shape

        laplacian = ndimage.laplace(img)

        bs = self.noise_block_size
        block_h, block_w = h // bs, w // bs
        if block_h < 2 or block_w < 2:
            return AnalysisResult(
                method="Noise Inconsistency Analysis",
                suspicious=False,
                confidence=0.0,
                severity=Severity.LOW,
                details={"note": "Image too small for block noise analysis"},
            )

        noise_map = np.zeros((block_h, block_w), dtype=np.float64)
        for bi in range(block_h):
            for bj in range(block_w):
                block = laplacian[bi * bs:(bi + 1) * bs, bj * bs:(bj + 1) * bs]
                sigma = np.median(np.abs(block)) / 0.6745
                noise_map[bi, bj] = sigma

        global_noise = float(np.median(noise_map))
        if global_noise < 1e-9:
            global_noise = 1e-9

        deviation = np.abs(noise_map - global_noise) / global_noise
        max_dev = float(deviation.max())
        mean_dev = float(deviation.mean())

        heatmap_small = deviation / max(max_dev, 1e-9) * 255.0
        heatmap = cv2.resize(heatmap_small, (w, h), interpolation=cv2.INTER_NEAREST)

        suspicious = max_dev > self.noise_inconsistency_threshold
        confidence = float(np.clip(max_dev / (self.noise_inconsistency_threshold * 2), 0.0, 1.0))

        return AnalysisResult(
            method="Noise Inconsistency Analysis",
            suspicious=suspicious,
            confidence=confidence,
            severity=self._severity_from_confidence(confidence),
            details={
                "global_noise_estimate": round(global_noise, 4),
                "max_deviation_ratio": round(max_dev, 4),
                "mean_deviation_ratio": round(mean_dev, 4),
            },
            heatmap=heatmap,
        )

    # ------------------------------------------------------------------
    # 6. EXIF / Metadata Analysis
    # ------------------------------------------------------------------

    def exif_metadata_analysis(self, image_path: str | Path) -> AnalysisResult:
        """Check image metadata for signs of editing or inconsistencies."""
        image_path = Path(image_path)
        pil_img = Image.open(image_path)

        exif_data = {}
        raw_exif = pil_img.getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                try:
                    exif_data[str(tag_name)] = str(value)
                except Exception:
                    exif_data[str(tag_name)] = "<unreadable>"

        flags: list[str] = []

        software = exif_data.get("Software", "")
        editing_keywords = ["photoshop", "gimp", "paint", "lightroom", "affinity",
                            "pixelmator", "snapseed", "canva"]
        for kw in editing_keywords:
            if kw in software.lower():
                flags.append(f"Editing software detected: {software}")
                break

        if not exif_data:
            flags.append("No EXIF data found (may have been stripped)")

        if "Make" in exif_data and "Model" not in exif_data:
            flags.append("Camera Make present but Model missing")
        if "Model" in exif_data and "Make" not in exif_data:
            flags.append("Camera Model present but Make missing")

        if "ExifOffset" not in exif_data and "Make" not in exif_data:
            flags.append("Minimal EXIF — possibly screenshot or generated image")

        suspicious = len(flags) > 0
        confidence = float(np.clip(len(flags) / 4.0, 0.0, 1.0))

        return AnalysisResult(
            method="EXIF / Metadata Analysis",
            suspicious=suspicious,
            confidence=confidence,
            severity=self._severity_from_confidence(confidence),
            details={
                "flags": flags,
                "exif_fields_found": len(exif_data),
                "exif_data": exif_data,
            },
        )

    # ------------------------------------------------------------------
    # 7. Double JPEG Compression Detection
    # ------------------------------------------------------------------

    def double_jpeg_detection(self, image_path: str | Path) -> AnalysisResult:
        """Detect double JPEG compression from DCT coefficient histograms.

        When a JPEG is edited and re-saved, the DCT coefficient distribution
        develops a periodic artifact (double peaks) that differs from
        single-compression statistics.
        """
        image_path = Path(image_path)
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")

        h, w = img.shape
        crop_h = (h // 8) * 8
        crop_w = (w // 8) * 8
        img = img[:crop_h, :crop_w].astype(np.float64)

        dct_coeffs: list[float] = []
        for i in range(0, crop_h, 8):
            for j in range(0, crop_w, 8):
                block = img[i:i + 8, j:j + 8]
                dct_block = fftpack.dct(fftpack.dct(block.T, norm="ortho").T, norm="ortho")
                dct_coeffs.extend(dct_block.ravel()[1:])  # skip DC

        dct_arr = np.array(dct_coeffs)
        dct_arr = dct_arr[(dct_arr > -50) & (dct_arr < 50)]

        hist, bin_edges = np.histogram(dct_arr, bins=100, range=(-50, 50))
        hist = hist.astype(np.float64)

        fft_hist = np.abs(fftpack.fft(hist - hist.mean()))
        fft_hist = fft_hist[1:len(fft_hist) // 2]

        if fft_hist.size == 0 or fft_hist.mean() < 1e-9:
            return AnalysisResult(
                method="Double JPEG Compression Detection",
                suspicious=False,
                confidence=0.0,
                severity=Severity.LOW,
                details={"note": "Insufficient data for DCT analysis"},
            )

        peak_ratio = float(fft_hist.max() / fft_hist.mean())

        suspicious = peak_ratio > 4.0
        confidence = float(np.clip((peak_ratio - 2.0) / 6.0, 0.0, 1.0))

        return AnalysisResult(
            method="Double JPEG Compression Detection",
            suspicious=suspicious,
            confidence=confidence,
            severity=self._severity_from_confidence(confidence),
            details={
                "dct_periodicity_peak_ratio": round(peak_ratio, 4),
            },
        )

    # ------------------------------------------------------------------
    # 8. Lighting / Shadow Consistency Analysis
    # ------------------------------------------------------------------

    def lighting_consistency_analysis(self, image_path: str | Path) -> AnalysisResult:
        """Estimate light-source direction consistency across the image.

        Divides the image into quadrants and estimates the dominant gradient
        (illumination) direction in each.  Large angular spread may indicate
        compositing with inconsistent lighting.
        """
        image_path = Path(image_path)
        img = np.array(Image.open(image_path).convert("L"), dtype=np.float64)
        h, w = img.shape

        grad_x = ndimage.sobel(img, axis=1)
        grad_y = ndimage.sobel(img, axis=0)

        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            (0, mid_h, 0, mid_w),
            (0, mid_h, mid_w, w),
            (mid_h, h, 0, mid_w),
            (mid_h, h, mid_w, w),
        ]

        angles: list[float] = []
        for y0, y1, x0, x1 in quadrants:
            gx = grad_x[y0:y1, x0:x1]
            gy = grad_y[y0:y1, x0:x1]
            mag = np.sqrt(gx ** 2 + gy ** 2)
            mask = mag > np.percentile(mag, 90)
            if mask.sum() < 10:
                continue
            mean_gx = float(gx[mask].mean())
            mean_gy = float(gy[mask].mean())
            angle = math.degrees(math.atan2(mean_gy, mean_gx))
            angles.append(angle)

        if len(angles) < 2:
            return AnalysisResult(
                method="Lighting Consistency Analysis",
                suspicious=False,
                confidence=0.0,
                severity=Severity.LOW,
                details={"note": "Not enough gradient information"},
            )

        def _angular_spread(a: list[float]) -> float:
            rads = [math.radians(x) for x in a]
            mean_sin = sum(math.sin(r) for r in rads) / len(rads)
            mean_cos = sum(math.cos(r) for r in rads) / len(rads)
            r_len = math.sqrt(mean_sin ** 2 + mean_cos ** 2)
            return math.degrees(math.acos(min(r_len, 1.0)))

        spread = _angular_spread(angles)
        suspicious = spread > 30.0
        confidence = float(np.clip(spread / 60.0, 0.0, 1.0))

        return AnalysisResult(
            method="Lighting Consistency Analysis",
            suspicious=suspicious,
            confidence=confidence,
            severity=self._severity_from_confidence(confidence),
            details={
                "quadrant_angles_deg": [round(a, 2) for a in angles],
                "angular_spread_deg": round(spread, 2),
            },
        )

    # ------------------------------------------------------------------
    # 9. Chromatic Aberration Analysis
    # ------------------------------------------------------------------

    def chromatic_aberration_analysis(self, image_path: str | Path) -> AnalysisResult:
        """Detect inconsistent chromatic aberration across the image.

        Camera lenses produce characteristic color fringing.  If a region
        has been spliced from a different camera or synthetically generated,
        its aberration pattern will differ from the rest.
        """
        image_path = Path(image_path)
        img = np.array(Image.open(image_path).convert("RGB"), dtype=np.float64)
        h, w, _ = img.shape

        edges_r = ndimage.sobel(img[:, :, 0])
        edges_g = ndimage.sobel(img[:, :, 1])
        edges_b = ndimage.sobel(img[:, :, 2])

        rg_shift = np.abs(edges_r - edges_g)
        rb_shift = np.abs(edges_r - edges_b)
        ca_map = (rg_shift + rb_shift) / 2.0

        bs = self.noise_block_size
        block_h, block_w = h // bs, w // bs
        if block_h < 2 or block_w < 2:
            return AnalysisResult(
                method="Chromatic Aberration Analysis",
                suspicious=False,
                confidence=0.0,
                severity=Severity.LOW,
                details={"note": "Image too small for CA analysis"},
            )

        block_means = np.zeros((block_h, block_w), dtype=np.float64)
        for bi in range(block_h):
            for bj in range(block_w):
                block = ca_map[bi * bs:(bi + 1) * bs, bj * bs:(bj + 1) * bs]
                block_means[bi, bj] = block.mean()

        median_ca = float(np.median(block_means))
        if median_ca < 1e-9:
            median_ca = 1e-9
        deviation = np.abs(block_means - median_ca) / median_ca
        max_dev = float(deviation.max())
        outlier_ratio = float(np.mean(deviation > 1.5))

        heatmap_small = deviation / max(max_dev, 1e-9) * 255.0
        heatmap = cv2.resize(heatmap_small, (w, h), interpolation=cv2.INTER_NEAREST)

        suspicious = outlier_ratio > 0.08
        confidence = float(np.clip(outlier_ratio / 0.20, 0.0, 1.0))

        return AnalysisResult(
            method="Chromatic Aberration Analysis",
            suspicious=suspicious,
            confidence=confidence,
            severity=self._severity_from_confidence(confidence),
            details={
                "median_ca_strength": round(median_ca, 4),
                "max_deviation_ratio": round(max_dev, 4),
                "outlier_ratio": round(outlier_ratio, 4),
            },
            heatmap=heatmap,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_jpeg(path: Path) -> bool:
        try:
            with open(path, "rb") as f:
                header = f.read(3)
            return header[:2] == b"\xff\xd8"
        except OSError:
            return False

    @staticmethod
    def _severity_from_confidence(confidence: float) -> Severity:
        if confidence >= 0.7:
            return Severity.HIGH
        if confidence >= 0.35:
            return Severity.MEDIUM
        return Severity.LOW
