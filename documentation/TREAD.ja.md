# TREAD 学習ドキュメント

> ⚠️ **実験的機能**: SimpleTuner の TREAD サポートは新規実装です。機能はしますが、最適な設定はまだ探索中であり、将来のリリースで挙動が変わる可能性があります。

## 概要

TREAD（Token Routing for Efficient Architecture-agnostic Diffusion Training）は、Transformer レイヤーでトークンを賢くルーティングすることで、拡散モデルの学習を高速化する手法です。特定レイヤーで重要なトークンのみを選んで処理することで、モデル品質を保ちつつ計算コストを大幅に削減できます。

[Krause ら (2025)](https://arxiv.org/abs/2501.04765) の研究に基づき、TREAD は以下の方法で学習を高速化します:
- 各 Transformer レイヤーで処理するトークンを動的に選択する
- スキップ接続を通じてすべてのトークンに勾配を流す
- 重要度に基づくルーティング判断を用いる

速度向上は `selection_ratio` に直接比例します。1.0 に近いほど多くのトークンを落とし、学習が速くなります。

## TREAD の仕組み

### コアコンセプト

学習中、TREAD は次の流れで動作します:
1. **トークンをルーティング** - 指定した Transformer レイヤーで重要度に基づき一部のトークンだけを選択
2. **部分処理** - 選択されたトークンのみが高コストな Attention と MLP を通過
3. **フルシーケンス復元** - 処理後にフルのトークン列へ復元し、全トークンに勾配が流れる

### トークン選択

トークンは L1 ノルム（重要度スコア）に基づいて選択され、探索のためのランダム化も利用できます:
- 重要度が高いトークンほど保持されやすい
- 重要度ベースとランダムの混合で特定パターンへの過学習を防ぐ
- 強制保持マスクにより特定トークン（マスク領域など）を落とさないようにできる

## 設定

### 基本設定

SimpleTuner で TREAD 学習を有効にするには、設定に次を追加します:

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 2,
        "end_layer_idx": 5
      }
    ]
  }
}
```

### ルート設定

各ルートはトークンルーティングが有効になるウィンドウを定義します:
- `selection_ratio`: 落とすトークンの割合（0.5 = トークンの 50% を保持）
- `start_layer_idx`: ルーティング開始レイヤー（0 始まり）
- `end_layer_idx`: ルーティングが有効な最後のレイヤー

負のインデックスも使えます。`-1` は最終レイヤーです。

### 高度な例

選択率が異なる複数のルーティングウィンドウ:

```json
{
  "tread_config": {
    "routes": [
      {
        "selection_ratio": 0.3,
        "start_layer_idx": 1,
        "end_layer_idx": 3
      },
      {
        "selection_ratio": 0.5,
        "start_layer_idx": 4,
        "end_layer_idx": 8
      },
      {
        "selection_ratio": 0.7,
        "start_layer_idx": -4,
        "end_layer_idx": -1
      }
    ]
  }
}
```

## 互換性

### 対応モデル
- **FLUX Dev/Kontext、Wan、AuraFlow、PixArt、SD3** - 現時点でサポートされているモデルファミリー
- 今後、他の拡散 Transformer への対応を予定

### 相性の良い組み合わせ
- **マスク損失学習** - マスク/セグメンテーション条件と組み合わせると、TREAD がマスク領域を自動的に保持します
- **マルチ GPU 学習** - 分散学習構成と互換
- **量子化学習** - int8/int4/NF4 量子化と併用可能

### 制限事項
- 学習時のみ有効（推論では無効）
- 勾配計算が必要（eval モードでは動作しない）
- 現状は FLUX と Wan に特化した実装で、Lumina2 など他のアーキテクチャでは未対応

## 性能上の考慮事項

### 速度面のメリット
- 学習速度は `selection_ratio` に比例（1.0 に近いほどトークンを多く落とせるため高速）
- Attention の O(n²) 依存により、**長い動画入力や高解像度で最も効果が大きい**
- 通常 20〜40% の高速化が見込めますが、設定によって変動
- マスク損失学習ではマスクトークンを落とせないため、高速化が低下

### 品質面のトレードオフ
- **トークンを多く落とすほど、LoRA/LoKr 学習の初期損失が高くなる**
- 損失は比較的早く補正され、選択率が高すぎなければ画像はすぐ安定します
  - これは中間レイヤーでのトークン減少にネットワークが適応している可能性があります
- 保守的な比率（0.1〜0.25）は品質を維持しやすい
- 積極的な比率（>0.35）は収束に確実に影響します

### LoRA 特有の注意点
- 性能はデータ依存の可能性があり、最適なルーティング設定は要検証
- 初期損失の上振れはフル学習より LoRA/LoKr のほうが目立つ

### 推奨設定

速度と品質のバランス:
```json
{
  "routes": [
    {"selection_ratio": 0.5, "start_layer_idx": 2, "end_layer_idx": -2}
  ]
}
```

最大速度（大きな損失上昇を想定）:
```json
{
  "routes": [
    {"selection_ratio": 0.7, "start_layer_idx": 1, "end_layer_idx": -1}
  ]
}
```

高解像度学習（1024px+）:
```json
{
  "routes": [
    {"selection_ratio": 0.6, "start_layer_idx": 2, "end_layer_idx": -3}
  ]
}
```

## 技術詳細

### ルーター実装

TREAD ルーター（`TREADRouter` クラス）は以下を担当します:
- L1 ノルムによるトークン重要度の計算
- 効率的なルーティングのための順列生成
- 勾配を維持するトークン復元

### Attention との統合

TREAD はルーティングされたシーケンスに合わせて RoPE（回転位置埋め込み）を変更します:
- テキストトークンは元の位置を維持
- 画像トークンはシャッフル/スライスされた位置を使用
- ルーティング中の位置整合性を保証
- **注記**: FLUX 向け RoPE 実装は 100% 正確ではない可能性がありますが、実用上は動作しています

### マスク損失との互換性

マスク損失学習を使う場合:
- マスク内トークンは自動的に強制保持
- 重要な学習信号を落とさないようにする
- `conditioning_type` が ["mask", "segmentation"] のときに有効
- **注記**: トークンを保持するため高速化は低下します

## 既知の問題と制限

### 実装状況
- **実験的機能** - 新規実装のため未発見の問題がある可能性があります
- **RoPE の扱い** - トークンルーティングの回転位置埋め込みが完全に正しいとは限りません
- **テスト不足** - 最適なルーティング設定は十分に検証されていません

### 学習挙動
- **初期損失の上振れ** - TREAD を使った LoRA/LoKr 学習では初期損失が高くなり、すぐに補正されます
- **LoRA 性能** - 設定によっては LoRA 学習がわずかに遅くなる場合があります
- **設定感度** - 性能はルーティング設定に大きく依存します

### 既知のバグ（修正済み）
- マスク損失学習は以前のバージョンで壊れていましたが、適切なモデルフレーバー判定（`kontext` ガード）で修正済みです

## トラブルシューティング

### よくある問題

**「TREAD の学習にはルート設定が必要です」**
- `tread_config` に `routes` 配列が含まれていることを確認してください
- 各ルートに `selection_ratio`、`start_layer_idx`、`end_layer_idx` が必要です

**期待より遅い**
- ルートが意味のあるレイヤー範囲をカバーしているか確認
- より積極的な選択率を検討
- 勾配チェックポイントが競合していないか確認
- LoRA 学習ではある程度の低下が想定されるため、別の設定を試してください

**LoRA/LoKr で初期損失が高い**
- これは想定された挙動で、ネットワークがトークン減少に適応する必要があります
- 通常は数百ステップ以内に損失が改善します
- 改善しない場合は `selection_ratio` を下げてトークンを多く残してください

**品質が低下する**
- 選択率を下げてトークンを多く残す
- 早いレイヤー（0〜2）や最終レイヤーでのルーティングを避ける
- 効率向上に見合った十分な学習データを用意する

## 実用例

### 高解像度学習（1024px+）
高解像度で最大の効果を得る設定:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.6, "start_layer_idx": 2, "end_layer_idx": -3}
    ]
  }
}
```

### LoRA ファインチューニング
初期損失の上振れを抑える保守的な設定:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.4, "start_layer_idx": 3, "end_layer_idx": -4}
    ]
  }
}
```

### マスク損失学習
マスク学習ではマスク領域のトークンが保持されます:
```json
{
  "tread_config": {
    "routes": [
      {"selection_ratio": 0.7, "start_layer_idx": 2, "end_layer_idx": -2}
    ]
  }
}
```
注記: 実際の高速化はトークン保持の影響で 0.7 より低くなります。

## 今後の取り組み

SimpleTuner における TREAD サポートは新規実装のため、改善余地が多くあります:

- **設定最適化** - ユースケースごとの最適ルーティング設定を見つけるための追加検証
- **LoRA 性能** - 一部の LoRA 設定で低下が出る理由の調査
- **RoPE 実装** - 回転位置埋め込みの正確性向上
- **モデル拡張** - Flux 以外の拡散 Transformer への対応
- **自動設定** - モデルやデータセット特性から最適ルーティングを推定するツール

コミュニティの貢献や検証結果の共有を歓迎します。

## 参考文献

- [TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training](https://arxiv.org/abs/2501.04765)
- [SimpleTuner Flux Documentation](quickstart/FLUX.md#tread-training)

## 引用

```bibtex
@misc{krause2025treadtokenroutingefficient,
      title={TREAD: Token Routing for Efficient Architecture-agnostic Diffusion Training},
      author={Felix Krause and Timy Phan and Vincent Tao Hu and Björn Ommer},
      year={2025},
      eprint={2501.04765},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.04765},
}
```
