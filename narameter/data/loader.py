"""
Data loader for Narameter pipeline.

Supports loading text data from CSV, JSON, and directories of files.
Standardizes output to a DataFrame with columns: ["text", "timestamp", "source"].
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads and standardizes text data from various file formats."""

    REQUIRED_COLUMNS = ["text", "timestamp"]

    def load_csv(
        self,
        path: str | Path,
        text_col: str = "text",
        date_col: str = "date",
        date_format: Optional[str] = None,
        source_col: Optional[str] = None,
        encoding: str = "utf-8",
        separator: str = ",",
    ) -> pd.DataFrame:
        """
        Load text data from a CSV file.

        Args:
            path: Path to the CSV file.
            text_col: Name of the column containing text data.
            date_col: Name of the column containing timestamps.
            date_format: strftime format for parsing dates. Auto-detect if None.
            source_col: Optional column indicating data source.
            encoding: File encoding.
            separator: CSV separator character.

        Returns:
            Standardized DataFrame with columns ["text", "timestamp", "source"].
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        logger.info(f"Loading CSV from {path}")
        df = pd.read_csv(path, encoding=encoding, sep=separator)

        return self._standardize(df, text_col, date_col, date_format, source_col, str(path))

    def load_json(
        self,
        path: str | Path,
        text_key: str = "text",
        date_key: str = "date",
        date_format: Optional[str] = None,
        source_key: Optional[str] = None,
        orient: str = "records",
    ) -> pd.DataFrame:
        """
        Load text data from a JSON file.

        Args:
            path: Path to the JSON file.
            text_key: Key for text data.
            date_key: Key for timestamps.
            date_format: strftime format for parsing dates.
            source_key: Optional key indicating data source.
            orient: JSON orientation ('records', 'columns', etc.).

        Returns:
            Standardized DataFrame.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found: {path}")

        logger.info(f"Loading JSON from {path}")

        if orient == "records":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Handle both list of records and {"data": [...]} format
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            df = pd.DataFrame(data)
        else:
            df = pd.read_json(path, orient=orient)

        return self._standardize(df, text_key, date_key, date_format, source_key, str(path))

    def load_directory(
        self,
        dir_path: str | Path,
        pattern: str = "*.csv",
        text_col: str = "text",
        date_col: str = "date",
        date_format: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Load and concatenate data from all matching files in a directory.

        Args:
            dir_path: Path to directory.
            pattern: Glob pattern for file matching.
            text_col: Column name for text.
            date_col: Column name for dates.
            date_format: Date parsing format.

        Returns:
            Combined standardized DataFrame.
        """
        dir_path = Path(dir_path)
        if not dir_path.is_dir():
            raise NotADirectoryError(f"Directory not found: {dir_path}")

        files = sorted(dir_path.glob(pattern))
        if not files:
            raise FileNotFoundError(f"No files matching '{pattern}' in {dir_path}")

        logger.info(f"Loading {len(files)} files from {dir_path}")
        frames = []

        for file_path in files:
            if file_path.suffix == ".csv":
                df = self.load_csv(file_path, text_col, date_col, date_format)
            elif file_path.suffix == ".json":
                df = self.load_json(file_path, text_col, date_col, date_format)
            else:
                logger.warning(f"Skipping unsupported file: {file_path}")
                continue
            frames.append(df)

        combined = pd.concat(frames, ignore_index=True)
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        logger.info(f"Loaded {len(combined)} total records from {len(files)} files")
        return combined

    def load_auto(
        self,
        path: str | Path,
        text_col: str = "text",
        date_col: str = "date",
        date_format: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Auto-detect file type and load accordingly.

        Args:
            path: Path to file or directory.
            text_col: Column/key name for text.
            date_col: Column/key name for dates.
            date_format: Date parsing format.

        Returns:
            Standardized DataFrame.
        """
        path = Path(path)

        if path.is_dir():
            return self.load_directory(path, "*.csv", text_col, date_col, date_format)
        elif path.suffix == ".csv":
            return self.load_csv(path, text_col, date_col, date_format)
        elif path.suffix == ".json":
            return self.load_json(path, text_col, date_col, date_format)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv or .json")

    def _standardize(
        self,
        df: pd.DataFrame,
        text_col: str,
        date_col: str,
        date_format: Optional[str],
        source_col: Optional[str],
        source_label: str,
    ) -> pd.DataFrame:
        """Standardize DataFrame to common schema."""
        # Validate columns exist
        if text_col not in df.columns:
            raise ValueError(
                f"Text column '{text_col}' not found. Available: {list(df.columns)}"
            )
        # Fallback logic for date_col
        if date_col not in df.columns:
            if "timestamp" in df.columns:
                date_col = "timestamp"
            elif "date" in df.columns:
                date_col = "date"
            elif "created_at" in df.columns:
                date_col = "created_at"
            else:
                raise ValueError(
                    f"Date column '{date_col}' not found. Available: {list(df.columns)}"
                )

        # Build standardized DataFrame
        result = pd.DataFrame()
        result["text"] = df[text_col].astype(str)

        # Parse dates
        if date_format:
            result["timestamp"] = pd.to_datetime(df[date_col], format=date_format)
        else:
            result["timestamp"] = pd.to_datetime(df[date_col], format="mixed")

        # Source column
        if source_col and source_col in df.columns:
            result["source"] = df[source_col]
        else:
            result["source"] = source_label

        # Clean up
        result = result.dropna(subset=["text", "timestamp"])
        result = result[result["text"].str.strip().str.len() > 0]
        result = result.sort_values("timestamp").reset_index(drop=True)

        logger.info(
            f"Standardized {len(result)} records "
            f"({result['timestamp'].min()} to {result['timestamp'].max()})"
        )
        return result
