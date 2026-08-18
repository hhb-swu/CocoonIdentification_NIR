"""Batch generation of recurrence plots from released 1D spectral sequences.

The released spectral sequences are already vector-normalized. This script applies
no additional spectral preprocessing. Each 1D ``.npy`` file is converted to a
recurrence plot (RP) using phase-space reconstruction followed by a Euclidean
distance matrix and min-max scaling to [0, 1].

The input directory structure is preserved exactly in the output directory, and
each RP file retains the same file name as its corresponding 1D spectrum.

Example
-------
Input:
    data/Spectral_Sequences/TestSet/Day1/0/sample_001.npy

Output:
    data/Recurrence_Plots/TestSet/Day1/0/sample_001.npy

Class encoding:
    0 = Female cocoon
    1 = Male cocoon
    2 = Imprinted-dead cocoon

Path configuration
------------------
Default source directory:
    data/Spectral_Sequences/TestSet

Default destination root:
    data/Recurrence_Plots

The script automatically appends the source split name (normally ``TestSet``)
to the destination root. Therefore, the generated RP files are written to:

    data/Recurrence_Plots/TestSet/DayX/{0,1,2}/

For a different repository layout, specify custom paths with:
    --source
    --destination-root
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


EXPECTED_LENGTH = 922
EMBEDDING_DIM = 2
DELAY = 6
RESIZE_SCALE = 0.5
VALID_LABELS = {"0", "1", "2"}


def phase_space_reconstruction(
    spectrum: np.ndarray,
    embedding_dim: int = EMBEDDING_DIM,
    delay: int = DELAY,
) -> np.ndarray:
    """Reconstruct the phase space of a one-dimensional spectral sequence."""
    spectrum = np.asarray(spectrum, dtype=np.float32).reshape(-1)
    n_points = spectrum.size

    reconstructed_length = n_points - (embedding_dim - 1) * delay
    if reconstructed_length <= 0:
        raise ValueError(
            "Phase-space reconstruction is not possible: "
            f"sequence_length={n_points}, embedding_dim={embedding_dim}, delay={delay}."
        )

    phase_space = np.empty(
        (reconstructed_length, embedding_dim),
        dtype=np.float32,
    )

    for index in range(embedding_dim):
        start = index * delay
        end = start + reconstructed_length
        phase_space[:, index] = spectrum[start:end]

    return phase_space


def compute_distance_matrix(phase_space: np.ndarray) -> np.ndarray:
    """Compute the pairwise Euclidean distance matrix in reconstructed phase space."""
    phase_space = np.asarray(phase_space, dtype=np.float32)

    difference = (
        phase_space[:, None, :]
        - phase_space[None, :, :]
    )

    distance_matrix = np.sqrt(
        np.sum(difference ** 2, axis=2)
    )

    return distance_matrix.astype(np.float32, copy=False)


def generate_recurrence_plot(
    spectrum: np.ndarray,
    embedding_dim: int = EMBEDDING_DIM,
    delay: int = DELAY,
) -> np.ndarray:
    """Generate a recurrence plot and min-max scale it to the range [0, 1]."""
    phase_space = phase_space_reconstruction(
        spectrum,
        embedding_dim=embedding_dim,
        delay=delay,
    )

    distance_matrix = compute_distance_matrix(phase_space)

    minimum = float(distance_matrix.min())
    maximum = float(distance_matrix.max())
    denominator = maximum - minimum

    if denominator == 0.0:
        recurrence_plot = np.zeros_like(distance_matrix, dtype=np.float32)
    else:
        recurrence_plot = (
            (distance_matrix - minimum) / denominator
        ).astype(np.float32, copy=False)

    return recurrence_plot


def resize_recurrence_plot(
    recurrence_plot: np.ndarray,
    scale: Optional[float] = RESIZE_SCALE,
) -> np.ndarray:
    """Resize an RP using area interpolation; return the original RP when scale is None."""
    if scale is None:
        return np.asarray(recurrence_plot, dtype=np.float32)

    if scale <= 0:
        raise ValueError(f"Resize scale must be positive, but received {scale}.")

    height, width = recurrence_plot.shape
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = cv2.resize(
        recurrence_plot,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    return np.asarray(resized, dtype=np.float32)


def discover_spectral_files(source_root: Path) -> List[Path]:
    """Find all spectral ``.npy`` files under Day*/{0,1,2}/ directories."""
    files = []

    day_directories = sorted(
        [path for path in source_root.iterdir() if path.is_dir() and path.name.startswith("Day")],
        key=lambda path: int(path.name[3:]) if path.name[3:].isdigit() else float("inf"),
    )

    for day_directory in day_directories:
        for label_directory in sorted(
            [path for path in day_directory.iterdir() if path.is_dir() and path.name in VALID_LABELS],
            key=lambda path: path.name,
        ):
            files.extend(sorted(label_directory.glob("*.npy")))

    return files


def convert_file(
    input_path: Path,
    output_path: Path,
    expected_length: int = EXPECTED_LENGTH,
    embedding_dim: int = EMBEDDING_DIM,
    delay: int = DELAY,
    resize_scale: Optional[float] = RESIZE_SCALE,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    """Convert one released vector-normalized spectrum to an RP file."""
    spectrum = np.load(input_path)
    spectrum = np.asarray(spectrum, dtype=np.float32).reshape(-1)

    if spectrum.size != expected_length:
        raise ValueError(
            f"Unexpected spectral length for {input_path.name}: "
            f"{spectrum.size}; expected {expected_length}."
        )

    if not np.all(np.isfinite(spectrum)):
        raise ValueError(
            f"Non-finite values were found in {input_path}. "
            "No additional preprocessing is applied by this script."
        )

    recurrence_plot = generate_recurrence_plot(
        spectrum,
        embedding_dim=embedding_dim,
        delay=delay,
    )
    original_rp_shape = recurrence_plot.shape

    recurrence_plot = resize_recurrence_plot(
        recurrence_plot,
        scale=resize_scale,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, recurrence_plot.astype(np.float32, copy=False))

    return spectrum.shape, original_rp_shape, recurrence_plot.shape


# =============================================================================
# PATH CONFIGURATION
# =============================================================================
# Default input:
#   <repository>/data/Spectral_Sequences/TestSet
#
# Default output:
#   <repository>/data/Recurrence_Plots/TestSet
#
# These defaults are repository-relative and are recommended for GitHub release.
# Users with a different directory layout can override them with:
#   --source <path>
#   --destination-root <path>
# =============================================================================

def parse_args() -> argparse.Namespace:
    """Parse source, destination, and RP-generation options."""
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description=(
            "Batch-convert released vector-normalized 1D spectra into recurrence plots "
            "while preserving the TestSet/DayX/label directory structure and file names."
        )
    )

    parser.add_argument(
        "--source",
        type=Path,
        default=script_dir / "data" / "Spectral_Sequences" / "TestSet",
        help="Source TestSet directory containing Day1-Day10 and class folders 0, 1, and 2.",
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=script_dir / "data" / "Recurrence_Plots",
        help=(
            "Destination root. The source directory name (normally 'TestSet') is "
            "automatically appended so that the output is Recurrence_Plots/TestSet/..."
        ),
    )
    parser.add_argument(
        "--expected-length",
        type=int,
        default=EXPECTED_LENGTH,
        help="Expected number of spectral variables per sample.",
    )
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=EMBEDDING_DIM,
        help="Phase-space embedding dimension.",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=DELAY,
        help="Phase-space delay.",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=RESIZE_SCALE,
        help=(
            "RP resize factor. With 922 spectral variables, embedding_dim=2, delay=6, "
            "and resize_scale=0.5, the RP changes from 916x916 to 458x458."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    source_root = args.source.resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source directory does not exist: {source_root}")

    destination_root = args.destination_root.resolve() / source_root.name
    spectral_files = discover_spectral_files(source_root)

    if not spectral_files:
        raise RuntimeError(
            f"No .npy spectral files were found under {source_root}. "
            "Expected Day*/{0,1,2}/sample.npy."
        )

    total_count = len(spectral_files)
    success_count = 0
    error_records: List[Tuple[Path, str]] = []
    class_counts = {"0": 0, "1": 0, "2": 0}
    day_class_counts: Dict[str, Dict[str, int]] = {}

    print(f"Source directory:      {source_root}")
    print(f"Destination directory: {destination_root}")
    print("Input spectra are assumed to be already vector-normalized.")
    print("No additional spectral preprocessing is applied.")
    print(f"Files discovered: {total_count}")
    print("-" * 80)

    for index, input_path in enumerate(spectral_files, start=1):
        relative_path = input_path.relative_to(source_root)
        output_path = destination_root / relative_path

        day = relative_path.parts[0]
        label = relative_path.parts[1]

        day_class_counts.setdefault(day, {"0": 0, "1": 0, "2": 0})
        class_counts[label] += 1
        day_class_counts[day][label] += 1

        try:
            spectrum_shape, original_rp_shape, final_rp_shape = convert_file(
                input_path=input_path,
                output_path=output_path,
                expected_length=args.expected_length,
                embedding_dim=args.embedding_dim,
                delay=args.delay,
                resize_scale=args.resize_scale,
            )

            success_count += 1
            print(
                f"[{index:04d}/{total_count:04d}] "
                f"{relative_path.as_posix()} | "
                f"1D={spectrum_shape} | "
                f"RP={original_rp_shape}->{final_rp_shape}"
            )

        except Exception as exc:
            error_records.append((input_path, str(exc)))
            print(
                f"[{index:04d}/{total_count:04d}] ERROR | "
                f"{relative_path.as_posix()} | {exc}"
            )

    print("\n" + "=" * 80)
    print("Conversion summary")
    print("=" * 80)

    for day in sorted(
        day_class_counts,
        key=lambda name: int(name[3:]) if name[3:].isdigit() else float("inf"),
    ):
        counts = day_class_counts[day]
        print(
            f"{day}: Female={counts['0']}, Male={counts['1']}, "
            f"Imprinted-dead={counts['2']}"
        )

    print("-" * 80)
    print(
        "Overall class counts: "
        f"Female={class_counts['0']}, "
        f"Male={class_counts['1']}, "
        f"Imprinted-dead={class_counts['2']}"
    )
    print(f"Total files:     {total_count}")
    print(f"Converted:       {success_count}")
    print(f"Failed:          {len(error_records)}")

    if error_records:
        print("\nFailed files:")
        for path, message in error_records:
            print(f"- {path}: {message}")
        raise RuntimeError(
            f"RP generation completed with {len(error_records)} failed file(s)."
        )

    print("\nAll spectral files were successfully converted to recurrence plots.")


if __name__ == "__main__":
    main()
