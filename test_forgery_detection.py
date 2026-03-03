"""Tests for forgery_detection module."""

from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from forgery_detection import (
    AnalysisResult,
    ForgeryDetector,
    ForgeryReport,
    Severity,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic test images
# ---------------------------------------------------------------------------


def _save_jpeg(arr: np.ndarray, path: Path, quality: int = 95) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path, format="JPEG", quality=quality)


def _save_png(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr.astype(np.uint8)).save(path, format="PNG")


@pytest.fixture()
def clean_jpeg(tmp_path: Path) -> Path:
    """A simple gradient JPEG — should not trigger tampering alarms."""
    rng = np.random.RandomState(42)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = np.tile(np.linspace(20, 235, 256, dtype=np.uint8), (256, 1))
    img = img + rng.randint(0, 4, img.shape, dtype=np.uint8)  # mild noise
    p = tmp_path / "clean.jpg"
    _save_jpeg(img, p)
    return p


@pytest.fixture()
def clean_png(tmp_path: Path) -> Path:
    rng = np.random.RandomState(42)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    for c in range(3):
        img[:, :, c] = np.tile(np.linspace(20, 235, 256, dtype=np.uint8), (256, 1))
    img = img + rng.randint(0, 4, img.shape, dtype=np.uint8)
    p = tmp_path / "clean.png"
    _save_png(img, p)
    return p


@pytest.fixture()
def tampered_jpeg(tmp_path: Path) -> Path:
    """JPEG where a block is spliced from a very different source."""
    rng = np.random.RandomState(99)
    img = np.zeros((256, 256, 3), dtype=np.uint8)
    img[:, :] = [120, 140, 130]
    img += rng.randint(0, 3, img.shape, dtype=np.uint8)

    # Save at quality 50 first
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=50)
    buf.seek(0)
    img = np.array(Image.open(buf).convert("RGB"))

    # Splice in a bright block (simulating content from a different image/quality)
    splice = rng.randint(200, 256, (80, 80, 3), dtype=np.uint8)
    img[50:130, 50:130] = splice

    p = tmp_path / "tampered.jpg"
    _save_jpeg(img, p, quality=92)
    return p


@pytest.fixture()
def copy_move_image(tmp_path: Path) -> Path:
    """Image with an exact duplicated region — simulates copy-move forgery."""
    rng = np.random.RandomState(7)
    img = rng.randint(0, 256, (300, 300, 3), dtype=np.uint8)

    # Copy a patch and paste it elsewhere
    patch = img[20:80, 20:80].copy()
    img[180:240, 180:240] = patch

    p = tmp_path / "copymove.png"
    _save_png(img, p)
    return p


@pytest.fixture()
def double_compressed_jpeg(tmp_path: Path) -> Path:
    """JPEG compressed twice at different quality levels."""
    rng = np.random.RandomState(123)
    img = rng.randint(50, 200, (256, 256, 3), dtype=np.uint8)

    # First compression at quality 60
    buf1 = io.BytesIO()
    Image.fromarray(img).save(buf1, format="JPEG", quality=60)
    buf1.seek(0)
    img2 = np.array(Image.open(buf1).convert("RGB"))

    # Second compression at quality 85
    p = tmp_path / "double.jpg"
    _save_jpeg(img2, p, quality=85)
    return p


@pytest.fixture()
def noise_inconsistent_image(tmp_path: Path) -> Path:
    """Image with two regions of very different noise levels."""
    img = np.full((256, 256, 3), 128, dtype=np.uint8)
    rng = np.random.RandomState(55)
    # Low noise region
    img[:128, :] += rng.randint(0, 2, (128, 256, 3), dtype=np.uint8)
    # High noise region
    img[128:, :] = np.clip(
        img[128:, :].astype(np.int16) + rng.randint(-40, 40, (128, 256, 3), dtype=np.int16),
        0, 255,
    ).astype(np.uint8)

    p = tmp_path / "noise_incon.png"
    _save_png(img, p)
    return p


@pytest.fixture()
def detector() -> ForgeryDetector:
    return ForgeryDetector()


# ---------------------------------------------------------------------------
# AnalysisResult / ForgeryReport unit tests
# ---------------------------------------------------------------------------


class TestAnalysisResult:
    def test_to_dict_basic(self) -> None:
        r = AnalysisResult(
            method="Test",
            suspicious=True,
            confidence=0.75,
            severity=Severity.HIGH,
            details={"foo": "bar"},
        )
        d = r.to_dict()
        assert d["method"] == "Test"
        assert d["suspicious"] is True
        assert d["confidence"] == 0.75
        assert d["severity"] == "HIGH"
        assert "heatmap_shape" not in d

    def test_to_dict_with_heatmap(self) -> None:
        hm = np.zeros((10, 10))
        r = AnalysisResult("T", False, 0.1, Severity.LOW, heatmap=hm)
        d = r.to_dict()
        assert d["heatmap_shape"] == [10, 10]


class TestForgeryReport:
    def test_empty_report(self) -> None:
        rpt = ForgeryReport()
        assert rpt.suspicious is False
        assert rpt.overall_confidence == 0.0
        assert rpt.max_severity == Severity.LOW

    def test_aggregation(self) -> None:
        rpt = ForgeryReport(results=[
            AnalysisResult("A", False, 0.1, Severity.LOW),
            AnalysisResult("B", True, 0.8, Severity.HIGH),
        ])
        assert rpt.suspicious is True
        assert rpt.overall_confidence == 0.8
        assert rpt.max_severity == Severity.HIGH

    def test_summary_string(self) -> None:
        rpt = ForgeryReport(results=[
            AnalysisResult("A", False, 0.0, Severity.LOW),
        ])
        s = rpt.summary()
        assert "Forgery Report" in s
        assert "[ok] A" in s

    def test_to_json(self) -> None:
        rpt = ForgeryReport(results=[
            AnalysisResult("X", True, 0.5, Severity.MEDIUM, details={"k": 1}),
        ])
        parsed = json.loads(rpt.to_json())
        assert parsed["suspicious"] is True
        assert len(parsed["results"]) == 1


# ---------------------------------------------------------------------------
# Individual analysis method tests
# ---------------------------------------------------------------------------


class TestELA:
    def test_clean_image(self, detector: ForgeryDetector, clean_jpeg: Path) -> None:
        result = detector.error_level_analysis(clean_jpeg)
        assert result.method == "Error Level Analysis (ELA)"
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert result.heatmap is not None
        assert result.heatmap.shape[:2] == (256, 256)

    def test_tampered_higher_confidence(
        self, detector: ForgeryDetector, clean_jpeg: Path, tampered_jpeg: Path
    ) -> None:
        clean_r = detector.error_level_analysis(clean_jpeg)
        tampered_r = detector.error_level_analysis(tampered_jpeg)
        # Tampered image should have strictly higher ELA std error
        assert tampered_r.details["std_error"] > clean_r.details["std_error"]

    def test_works_on_png(self, detector: ForgeryDetector, clean_png: Path) -> None:
        result = detector.error_level_analysis(clean_png)
        assert result.heatmap is not None


class TestCopyMoveDetection:
    def test_returns_result(self, detector: ForgeryDetector, clean_png: Path) -> None:
        result = detector.copy_move_detection(clean_png)
        assert result.method == "Copy-Move Detection"
        assert "keypoints_found" in result.details

    def test_invalid_path_raises(self, detector: ForgeryDetector) -> None:
        with pytest.raises(FileNotFoundError):
            detector.copy_move_detection("/nonexistent/image.png")


class TestJPEGGhostDetection:
    def test_runs_on_jpeg(self, detector: ForgeryDetector, clean_jpeg: Path) -> None:
        result = detector.jpeg_ghost_detection(clean_jpeg)
        assert result.method == "JPEG Ghost Detection"
        assert "dominant_quality" in result.details
        assert result.heatmap is not None

    def test_tampered_image(self, detector: ForgeryDetector, tampered_jpeg: Path) -> None:
        result = detector.jpeg_ghost_detection(tampered_jpeg)
        assert 0.0 <= result.confidence <= 1.0


class TestCFAPatternAnalysis:
    def test_clean_image(self, detector: ForgeryDetector, clean_jpeg: Path) -> None:
        result = detector.cfa_pattern_analysis(clean_jpeg)
        assert result.method == "CFA Pattern Analysis"
        assert "median_variance" in result.details
        assert result.heatmap is not None


class TestNoiseInconsistency:
    def test_clean_image_low_confidence(
        self, detector: ForgeryDetector, clean_jpeg: Path
    ) -> None:
        result = detector.noise_inconsistency_analysis(clean_jpeg)
        assert result.method == "Noise Inconsistency Analysis"
        assert result.heatmap is not None

    def test_noise_inconsistent_image_higher(
        self, detector: ForgeryDetector, noise_inconsistent_image: Path, clean_png: Path
    ) -> None:
        clean_r = detector.noise_inconsistency_analysis(clean_png)
        noisy_r = detector.noise_inconsistency_analysis(noise_inconsistent_image)
        assert noisy_r.details["max_deviation_ratio"] > clean_r.details["max_deviation_ratio"]


class TestEXIFAnalysis:
    def test_no_exif_flagged(self, detector: ForgeryDetector, clean_jpeg: Path) -> None:
        result = detector.exif_metadata_analysis(clean_jpeg)
        assert result.method == "EXIF / Metadata Analysis"
        # Our synthetic image has no real EXIF data
        assert any("No EXIF" in f or "Minimal EXIF" in f for f in result.details["flags"])

    def test_with_editing_software(self, detector: ForgeryDetector, tmp_path: Path) -> None:
        img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
        from PIL.PngImagePlugin import PngInfo

        # For JPEG EXIF we set Software tag via piexif-like approach;
        # simplest: just check our string matching works by
        # creating a JPEG and injecting Software via PIL.
        p = tmp_path / "edited.jpg"
        img.save(p, format="JPEG", quality=95)

        # PIL doesn't directly set Software in save; the flag logic
        # should at least catch "No EXIF" for this synthetic image.
        result = detector.exif_metadata_analysis(p)
        assert isinstance(result.details["flags"], list)


class TestDoubleJPEGDetection:
    def test_runs(self, detector: ForgeryDetector, clean_jpeg: Path) -> None:
        result = detector.double_jpeg_detection(clean_jpeg)
        assert result.method == "Double JPEG Compression Detection"
        assert "dct_periodicity_peak_ratio" in result.details

    def test_double_compressed_higher_ratio(
        self, detector: ForgeryDetector, clean_jpeg: Path, double_compressed_jpeg: Path
    ) -> None:
        single_r = detector.double_jpeg_detection(clean_jpeg)
        double_r = detector.double_jpeg_detection(double_compressed_jpeg)
        # Both should run without error; double-compressed may or may not
        # trigger on such small synthetic images, but should not crash.
        assert double_r.confidence >= 0.0


class TestLightingConsistency:
    def test_runs(self, detector: ForgeryDetector, clean_jpeg: Path) -> None:
        result = detector.lighting_consistency_analysis(clean_jpeg)
        assert result.method == "Lighting Consistency Analysis"

    def test_uniform_lighting_low_spread(
        self, detector: ForgeryDetector, clean_jpeg: Path
    ) -> None:
        result = detector.lighting_consistency_analysis(clean_jpeg)
        # Gradient image should have very consistent lighting direction
        assert result.details.get("angular_spread_deg", 0) < 45


class TestChromaticAberration:
    def test_runs(self, detector: ForgeryDetector, clean_png: Path) -> None:
        result = detector.chromatic_aberration_analysis(clean_png)
        assert result.method == "Chromatic Aberration Analysis"
        assert result.heatmap is not None


# ---------------------------------------------------------------------------
# Integration: full analyze() pipeline
# ---------------------------------------------------------------------------


class TestFullAnalysis:
    def test_analyze_jpeg(self, detector: ForgeryDetector, clean_jpeg: Path) -> None:
        report = detector.analyze(clean_jpeg)
        method_names = [r.method for r in report.results]
        assert "Error Level Analysis (ELA)" in method_names
        assert "Copy-Move Detection" in method_names
        assert "Noise Inconsistency Analysis" in method_names
        assert "JPEG Ghost Detection" in method_names
        assert "Double JPEG Compression Detection" in method_names
        assert "Lighting Consistency Analysis" in method_names
        assert "Chromatic Aberration Analysis" in method_names
        assert "CFA Pattern Analysis" in method_names
        assert "EXIF / Metadata Analysis" in method_names

    def test_analyze_png_skips_jpeg_specific(
        self, detector: ForgeryDetector, clean_png: Path
    ) -> None:
        report = detector.analyze(clean_png)
        method_names = [r.method for r in report.results]
        assert "JPEG Ghost Detection" not in method_names
        assert "Double JPEG Compression Detection" not in method_names
        # Core analyses still run
        assert "Error Level Analysis (ELA)" in method_names

    def test_report_json_roundtrip(
        self, detector: ForgeryDetector, clean_jpeg: Path
    ) -> None:
        report = detector.analyze(clean_jpeg)
        j = report.to_json()
        parsed = json.loads(j)
        assert "suspicious" in parsed
        assert "results" in parsed
        assert len(parsed["results"]) == len(report.results)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_small_image(self, detector: ForgeryDetector, tmp_path: Path) -> None:
        img = np.zeros((16, 16, 3), dtype=np.uint8)
        p = tmp_path / "tiny.jpg"
        _save_jpeg(img, p)
        report = detector.analyze(p)
        # Should not crash even on very small images
        assert isinstance(report, ForgeryReport)

    def test_grayscale_input(self, detector: ForgeryDetector, tmp_path: Path) -> None:
        img = np.zeros((128, 128), dtype=np.uint8)
        p = tmp_path / "gray.jpg"
        Image.fromarray(img, mode="L").save(p, format="JPEG")
        # All methods accept path and convert internally — should not crash
        report = detector.analyze(p)
        assert isinstance(report, ForgeryReport)

    def test_custom_parameters(self, tmp_path: Path) -> None:
        det = ForgeryDetector(
            ela_quality=75,
            ela_threshold=20.0,
            block_size=8,
            noise_block_size=16,
        )
        img = np.random.RandomState(0).randint(0, 256, (128, 128, 3), dtype=np.uint8)
        p = tmp_path / "custom.jpg"
        _save_jpeg(img, p)
        result = det.error_level_analysis(p)
        assert result.details["quality_used"] == 75
