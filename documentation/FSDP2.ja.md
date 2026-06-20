# FSDP2 シャーディング / マルチ GPU 学習

SimpleTuner は PyTorch Fully Sharded Data Parallel v2（DTensor ベースの FSDP）を一次サポートとして提供します。WebUI はフルモデル実行時に v2 実装を既定として使用し、最重要の accelerate フラグを公開することで、カスタム起動スクリプトなしにマルチ GPU へスケールできます。

> ⚠️ FSDP2 は分散 DTensor スタックが有効な CUDA ビルドの最新 PyTorch 2.x を対象とします。WebUI のコンテキスト並列制御は CUDA ホストのみで表示され、他のバックエンドは実験的扱いです。

## FSDP2 とは？

FSDP2 は PyTorch のシャーディング・データ並列エンジンの次世代版です。FSDP v1 の旧フラットパラメータ処理ではなく、v2 プラグインは DTensor の上で動作します。モデルパラメータ、勾配、オプティマイザをランク間でシャーディングしつつ、各ランクの作業セットを小さく保ちます。従来の ZeRO 方式と比較して、Hugging Face accelerate の起動フローを維持するため、チェックポイント、オプティマイザ、推論パスが SimpleTuner の他機能と互換のままになります。

## FSDP2 を選ぶ場面

FSDP2 は主にメモリをスケールさせるための手段です。通常のデータ並列ではモデル、解像度、動画時間、シーケンス長が収まらない場合や、長い attention シーケンスにコンテキスト並列が必要な場合に向いています。マルチ GPU で常に最速になるわけではありません。

LoRA/PEFT 学習では、各 GPU にモデルが収まるなら DDP を優先してください。8x H100 の LTX-2.3 IC-LoRA テストでは、各 GPU `train_batch_size=1` の通常 DDP が、同じ小さい rank あたり batch 形状の 8x H100 FSDP2 よりおよそ 3-6 倍高速でした。LoRA で FSDP2 を使うのは、解像度を下げる、クリップを短くする、強い offload を使う、または実行できないことが代替になる場合です。

Torch Dynamo は FSDP2 実行で guard の churn が増えやすく、特に動的 shape、tokenwise schedule、学習/検証の shape 変化、regional compile と組み合わせた場合に目立ちます。Dynamo の cache limit に達していなくても再コンパイル guard のログが頻繁に出ることがあります。新しい FSDP2 設定で compile が有利だと決める前に、`TORCH_LOGS=recompiles` を profiling 用に使ってください。

## 機能概要

- WebUI トグル（Hardware → Accelerate）で FullyShardedDataParallelPlugin を既定値付きで生成
- CLI の自動正規化（`--fsdp_version`、`--fsdp_state_dict_type`、`--fsdp_auto_wrap_policy`）で手入力の揺れを許容
- 長いシーケンスモデル向けのコンテキスト並列シャーディング（`--context_parallel_size`、`--context_parallel_comm_strategy`）を FSDP2 の上に追加
- トランスフォーマーブロック検出モーダルがベースモデルを解析し、オートラップ用クラス名を提案
- `~/.simpletuner/fsdp_block_cache.json` に検出メタデータをキャッシュし、WebUI からワンクリック保守
- チェックポイント形式切り替え（シャーデッド/フル）と、ホスト RAM を節約する再開モード

## 既知の制限

- FSDP2 はフルモデル実行と、メモリ制約のある PEFT/LoRA 実行で特に有効です。LoRA が DDP で収まるなら、通常は DDP の方が throughput に優れます。
- DeepSpeed と FSDP は併用できません。`--fsdp_enable` と DeepSpeed 設定を同時に指定すると CLI/WebUI で明示的なエラーになります。
- コンテキスト並列は CUDA のみで、`--context_parallel_size > 1` と `--fsdp_version=2` が必要です。
- `--fsdp_reshard_after_forward=true` を使えば検証パスは動作します。FSDP ラップモデルを直接パイプラインに渡し、all-gather/reshard が透過的に処理されます。
- ブロック検出はベースモデルをローカルで生成します。大きなチェックポイントでは短い待ち時間とホストメモリ増加が発生します。
- FSDP v1 は後方互換のため残っていますが、UI と CLI ログで非推奨扱いです。

## FSDP2 の有効化

### 方法 1: WebUI（推奨）

1. SimpleTuner WebUI を開き、使用する学習設定を読み込みます。
2. **Hardware → Accelerate** に切り替えます。
3. **Enable FSDP v2** をオンにします。バージョンは既定で `2` なので、v1 が必要な場合以外は変更しません。
4. （任意）以下を調整:
   - **Reshard After Forward**（VRAM と通信のトレードオフ）
   - **Checkpoint Format**（`Sharded` / `Full`）
   - **CPU RAM Efficient Loading**（ホストメモリが厳しい再開時）
   - **Auto Wrap Policy** と **Transformer Classes to Wrap**（後述の検出ワークフロー）
   - **Context Parallel Size / Rotation**（シーケンスシャーディングが必要な場合）
5. 設定を保存します。起動時に正しい accelerate プラグインが渡されます。

### 方法 2: CLI

WebUI と同じフラグを `simpletuner-train` で指定します。2 GPU の SDXL フルモデル例:

```bash
simpletuner-train \
  --model_type=full \
  --model_family=sdxl \
  --output_dir=/data/experiments/sdxl-fsdp2 \
  --fsdp_enable \
  --fsdp_version=2 \
  --fsdp_state_dict_type=SHARDED_STATE_DICT \
  --fsdp_auto_wrap_policy=TRANSFORMER_BASED_WRAP \
  --num_processes=2
```

既存の accelerate config を使っている場合はそのまま利用できます。SimpleTuner は FSDP プラグインを起動パラメータにマージし、設定全体を上書きしません。

## コンテキスト並列

コンテキスト並列は CUDA ホストで FSDP2 の上に重ねる任意機能です。`--context_parallel_size`（または WebUI の対応フィールド）にシーケンス次元を分割する GPU 数を指定します。通信方式は次のとおりです:

- `allgather`（既定）– オーバーラップを優先し、まずはここから
- `alltoall` – 非常に大きな注意窓を持つニッチなワークロードで有効な場合があるが、調整コストが増える

トレーナーは、コンテキスト並列が要求された場合に `fsdp_enable` と `fsdp_version=2` を強制します。サイズを `1` に戻すと機能は無効化され、保存設定が一致するようローテーション文字列が正規化されます。

## FSDP ブロック検出ワークフロー

SimpleTuner には、選択したベースモデルを解析して FSDP 自動ラップに最適なモジュールクラスを提示する検出機能が組み込まれています:

1. トレーナーフォームで **Model Family**（必要なら **Model Flavour**）を選択します。
2. カスタム重みから学習する場合はチェックポイントパスを入力します。
3. **Transformer Classes to Wrap** の横にある **Detect Blocks** をクリックします。SimpleTuner がモデルを生成し、モジュールを走査してクラスごとのパラメータ合計を記録します。
4. モーダル解析を確認します:
   - **Select** でラップ対象のクラスを選択（先頭列のチェックボックス）
   - **Total Params** でパラメータ配分の大きいモジュールを把握
   - `_no_split_modules`（存在する場合）はバッジとして表示され、除外リストに追加すべき対象です
5. **Apply Selection** を押して `--fsdp_transformer_layer_cls_to_wrap` に反映します。
6. 以降は **Refresh Detection** を押さない限り、キャッシュ結果が再利用されます。

検出結果は `~/.simpletuner/fsdp_block_cache.json` に、モデルファミリー・チェックポイントパス・フレーバーをキーとして保存されます。異なるチェックポイント間を切り替える場合や重みを更新した後は、**Settings → WebUI Preferences → Cache Maintenance → Clear FSDP Detection Cache** を使ってください。

## チェックポイントの扱い

- **シャーデッド state dict**（`SHARDED_STATE_DICT`）はランクごとのシャードを保存し、大規模モデルでもスケールします。
- **フル state dict**（`FULL_STATE_DICT`）はランク 0 にパラメータを集約し、外部ツール互換の代わりにメモリ負荷が増えます。
- **CPU RAM Efficient Loading** は再開時の all-rank マテリアライズを遅らせ、ホストメモリの急増を抑えます。
- **Reshard After Forward** は forward 間でパラメータシャードを小さく保ちます。検証は FSDP ラップモデルを直接 diffusers パイプラインに渡すことで正しく動作します。

再開頻度と下流ツールに合わせて組み合わせを選んでください。大規模モデルには、シャーデッドチェックポイント + RAM 効率ロードが最も安全です。

## 保守ツール

WebUI の **WebUI Preferences → Cache Maintenance** に保守機能があります:

- **Clear FSDP Detection Cache** はキャッシュされたブロックスキャンをすべて削除します（`FSDP_SERVICE.clear_cache()` のラッパ）。
- **Clear DeepSpeed Offload Cache** は ZeRO ユーザー向けに残っており、FSDP とは独立して動作します。

両方ともトースト通知が表示され、保守ステータスが更新されるためログを追う必要はありません。

## トラブルシューティング

| 症状 | 可能性の高い原因 | 対処 |
|---------|--------------|-----|
| `"FSDP and DeepSpeed cannot be enabled simultaneously."` | 両方のプラグインを指定している（例: DeepSpeed JSON + `--fsdp_enable`）。 | DeepSpeed 設定を外すか FSDP を無効化。 |
| `"Context parallelism requires FSDP2."` | `context_parallel_size > 1` だが FSDP が無効または v1。 | FSDP を有効化し `--fsdp_version=2` を維持するか、サイズを `1` に戻す。 |
| `Unknown model_family` で検出失敗 | フォームに対応ファミリー/フレーバーがない。 | ドロップダウンからモデルを選択。カスタムファミリーは `model_families` に登録が必要。 |
| 検出結果が古い | キャッシュ結果を再利用している。 | **Refresh Detection** を押すかキャッシュをクリア。 |
| 再開時にホスト RAM が枯渇 | ロード時にフル state dict を集約している。 | `SHARDED_STATE_DICT` に切り替え、必要なら RAM 効率ロードを有効化。 |
| compile した FSDP2 実行で Dynamo 再コンパイルログが続く | 動的 shape や学習/検証の shape 変化で guard が無効化されている。 | 一度 compile なしで実行するか、cache limit を上げる前に `TORCH_LOGS=recompiles` で変化する入力を確認する。 |

## CLI フラグ一覧

- `--fsdp_enable` – FullyShardedDataParallelPlugin を有効化
- `--fsdp_version` – `1` または `2` を選択（既定 `2`、v1 は非推奨）
- `--fsdp_reshard_after_forward` – forward 後にパラメータシャードを解放（既定 `true`）
- `--fsdp_state_dict_type` – `SHARDED_STATE_DICT`（既定）または `FULL_STATE_DICT`
- `--fsdp_cpu_ram_efficient_loading` – 再開時のホストメモリ急増を低減
- `--fsdp_auto_wrap_policy` – `TRANSFORMER_BASED_WRAP`、`SIZE_BASED_WRAP`、`NO_WRAP`、またはドット区切りの callable パス
- `--fsdp_transformer_layer_cls_to_wrap` – 検出機能で埋められるカンマ区切りクラス一覧
- `--context_parallel_size` – この数のランクで注意をシャーディング（CUDA + FSDP2 のみ）
- `--context_parallel_comm_strategy` – `allgather`（既定）または `alltoall` のローテーション戦略
- `--num_processes` – config ファイルなしで accelerate に渡す総ランク数

これらは Hardware → Accelerate の WebUI 制御と 1:1 に対応するため、UI から書き出した設定は CLI でそのまま再現できます。
