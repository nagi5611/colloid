"""コロイド溶液の濃度とレーザー照射時の明度データを可視化するスクリプト。

このスクリプトは、CSVファイルから濃度(`concentration`)と明度(`brightness`)
の列を読み込み、濃度と明度の関係を折れ線グラフとして描画します。
明度はレーザー反射や散乱の測定値など、任意の定量指標を想定しています。

使用例:

    python app.py data.csv --output figure.png

"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import sys

import matplotlib.pyplot as plt
from matplotlib import font_manager
import pandas as pd


REQUIRED_COLUMNS: Sequence[str] = ("concentration", "brightness")
JAPANESE_FONT_CANDIDATES: Sequence[str] = (
    "Yu Gothic",
    "YuGothic",
    "Yu Gothic UI",
    "Meiryo",
    "MS Gothic",
    "MS Mincho",
    "Hiragino Sans",
    "Hiragino Kaku Gothic ProN",
    "Noto Sans JP",
    "Noto Sans CJK JP",
    "TakaoGothic",
    "TakaoPGothic",
    "IPAGothic",
    "IPAexGothic",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="濃度ごとのレーザー明度データをプロット"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        nargs="?",
        help="濃度と明度を含むCSVファイル (列名: concentration, brightness)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="生成した図を保存するパス。指定しない場合は画面表示",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="濃度とレーザー明度の関係",
        help="グラフタイトル",
    )
    parser.add_argument(
        "--x-label",
        type=str,
        default="濃度 (任意単位)",
        help="x軸ラベル",
    )
    parser.add_argument(
        "--y-label",
        type=str,
        default="明度 (任意単位)",
        help="y軸ラベル",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="公開研究データが無いため合成したサンプルデータを使用する",
    )
    return parser.parse_args()


def validate_columns(frame: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        missing_columns = ", ".join(missing)
        raise ValueError(
            f"CSVに必要な列が不足しています: {missing_columns}. "
            "列名は concentration と brightness が必要です。"
        )


def configure_japanese_font() -> str:
    for candidate in JAPANESE_FONT_CANDIDATES:
        for font in font_manager.fontManager.ttflist:
            if candidate.lower() in font.name.lower():
                plt.rcParams["font.family"] = font.name
                plt.rcParams["axes.unicode_minus"] = False
                return font.name
    return ""


def plot_data(frame: pd.DataFrame, title: str, x_label: str, y_label: str) -> plt.Figure:
    sorted_frame = frame.sort_values("concentration").reset_index(drop=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(
        sorted_frame["concentration"],
        sorted_frame["brightness"],
        marker="o",
        linestyle="-",
    )
    axis.set_title(title)
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(True, which="major", linestyle="--", linewidth=0.5)

    return figure


def demo_dataset() -> pd.DataFrame:
    """公開データが見つからないため、仮想的に生成したデモ用データを返す。"""

    # 濃度: mg/mL, 明度: カメラで撮影した輝度値 (相対値) を想定
    # → 実測値ではなく、濃度増加に伴い散乱が増えるケースを模擬
    data = {
        "concentration": [0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
        "brightness": [92.0, 80.5, 70.1, 58.6, 46.3, 34.8, 24.5],
    }
    return pd.DataFrame(data)


def main() -> None:
    args = parse_args()

    configured_font = configure_japanese_font()
    if not configured_font:
        print(
            "警告: 日本語フォントが見つからなかったため、グラフ上で文字化けする可能性があります。",
            file=sys.stderr,
        )

    if args.demo:
        data_frame = demo_dataset()
    else:
        if args.csv_path is None:
            raise SystemExit("CSVファイルを指定するか、--demo を使用してください。")

        try:
            data_frame = pd.read_csv(args.csv_path)
        except FileNotFoundError as exc:
            raise SystemExit(f"CSVファイルが見つかりません: {args.csv_path}") from exc
        except pd.errors.EmptyDataError as exc:
            raise SystemExit(f"CSVファイルが空です: {args.csv_path}") from exc

    validate_columns(data_frame)

    figure = plot_data(data_frame, args.title, args.x_label, args.y_label)

    if args.output is not None:
        figure.savefig(args.output, bbox_inches="tight", dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()

