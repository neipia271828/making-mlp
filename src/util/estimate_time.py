import sys
import argparse
from pathlib import Path

# uv run src/util/estimate_time.py --model CNN-v3 --epochs 60
# uv run src/util/estimate_time.py --model CNN-v4 --epochs 120

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from CONSTANTS import CONSTANTS
from lib.factory import load_model_constants


def _format_seconds(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def estimate(model: str, epochs: int) -> None:
    csv_path = Path(f"data/{CONSTANTS.PROJECT}/logs/{CONSTANTS.PROJECT}.csv")
    if not csv_path.exists():
        print(f"[warn] ログが見つかりません: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    past = df[df["MODEL"] == model]

    if past.empty:
        print(f"[warn] モデル '{model}' の過去ランが存在しません。参考値として全モデルの平均を使用します。")
        past = df

    ep_time_mean = past["ep_time"].mean()
    ep_time_min  = past["ep_time"].min()
    ep_time_max  = past["ep_time"].max()
    n_runs       = len(past)

    total_mean = ep_time_mean * epochs
    total_min  = ep_time_min  * epochs
    total_max  = ep_time_max  * epochs

    print(f"モデル       : {model}")
    print(f"エポック数   : {epochs}")
    print(f"参照ラン数   : {n_runs} runs")
    print(f"エポック時間 : avg={ep_time_mean:.1f}s  min={ep_time_min:.1f}s  max={ep_time_max:.1f}s")
    print()
    print(f"推定合計時間")
    print(f"  平均値 : {_format_seconds(total_mean)}  ({total_mean:.0f}s)")
    print(f"  最短   : {_format_seconds(total_min)}")
    print(f"  最長   : {_format_seconds(total_max)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="学習時間の推定")
    parser.add_argument("--model",  default=CONSTANTS.MODEL,  help="モデル名 (default: CONSTANTS.MODEL)")
    parser.add_argument("--epochs", type=int, default=None,   help="エポック数 (default: モデル定数の NUM_EPOCHS)")
    args = parser.parse_args()

    epochs = args.epochs
    if epochs is None:
        try:
            model_constants = load_model_constants(args.model)
            epochs = model_constants.NUM_EPOCHS
        except Exception:
            print(f"[warn] {args.model}/constants.py から NUM_EPOCHS を読めませんでした。--epochs を指定してください。")
            return

    estimate(args.model, epochs)


if __name__ == "__main__":
    main()
