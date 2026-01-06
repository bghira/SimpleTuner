# SimpleTuner における Sparse–Linear Attention（SLA）

Sparse–Linear Attention（SLA）は、疎な FlashAttention と線形 Attention の補償器を単一の CUDA カーネル内で融合します。重要な query/key ブロックは高コストな疎経路を通り、周辺ブロックは軽量な線形 Attention と学習可能な射影を使います。これにより、品質をフル Attention に近づけつつ FLOPs を大幅に削減できます。

SimpleTuner は通常の `--attention_mechanism` フラグで SLA を公開しているため、SLA で微調整したモデルを同じカーネルで推論に使えます。

## 要件

1. 参照実装をインストール:

   ```bash
   git clone https://github.com/thu-ml/SLA.git ~/src/SLA
   pip install -e ~/src/SLA
   ```

2. CUDA ビルドの PyTorch を使用（SLA カーネルは現時点で CUDA 専用）。

## SLA の有効化

- `--attention_mechanism=sla` を指定（または `attention_mechanism: "sla"` を設定）。
- 追加フラグは不要。SimpleTuner が PyTorch の SDPA エントリポイントをラップして SLA を挿入します。
- SLA 設定（top-k 比率、ブロックサイズ、特徴マップ種別、query/key 特徴マップを結合するかどうか）は `--sla_config` / `sla_config` で JSON/Python dict 形式で指定できます。例: `--sla_config '{"topk":0.15,"blkq":32,"tie_feature_map_qk":false}'`。既定は top 20%、ブロックサイズ 64、特徴マップ結合です。

## 学習時の挙動

- SLA は学習可能です。コントローラは線形射影ヘッド（`proj_l`）を `float32` に保つため、SLA 全体が BF16/FP16 で動作していても AMP/GradScaler の安定性を保てます。
- バックボーンは SLA の疎/線形混合挙動を前提に微調整されるため、推論でも SLA を使い続けるべきです。学習後に Diffusers SDPA/XFormers に戻すと品質が低下する可能性が高いです。
- チェックポイント保存時、SimpleTuner は通常の accelerator 状態と並んで `sla_attention.pt` を書き込みます。このファイルには、実体化された各ヘッド次元/ dtype ペアの SLA 射影重みと関連バッファが含まれます。チェックポイントと一緒に保持してください。削除すると次回の再開/推論で射影層が再初期化されます。

## 推論

- 学習再開やバリデーション再実行時も `--attention_mechanism=sla` を有効にし、SLA カーネルを引き続き使ってください。
- ローダはチェックポイント内の `sla_attention.pt` を自動的に再生するため、追加フラグは不要です。
- SLA 学習済み重みを標準 SDPA と比較したい場合、品質低下を想定してください。SLA 論文ではバックボーンを適応させるため数千ステップの調整が必要とされており、SLA なし推論はサポート外として扱うべきです。

## トラブルシューティングと注意

- **`sla_attention.pt` がない:** SLA 状態保存が導入される前に作られたチェックポイントか、ファイルが削除されています。SLA 有効のまま短い学習セッション（1 ステップでも可）を実行して再生成してください。
- **AMP/GradScaler エラー:** SLA モジュールを手動で BF16/FP16 にキャストしないでください。SimpleTuner が射影ヘッドを自動的に FP32 に固定します。追加のキャストは学習を不安定にします。
- **Hub へのアップロード:** Hugging Face Hub などにチェックポイントを送る場合は `sla_attention.pt` を含めてください。ダウンロードした利用者が追加手順なしで学習済み SLA 重みを引き継げます。

SLA の設計とアルゴリズム全体の詳細は、[SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse–Linear Attention](https://www.arxiv.org/abs/2509.24006) を参照してください。
