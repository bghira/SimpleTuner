# SimpleTuner 学習スクリプトのオプション

## 概要

このガイドでは、SimpleTuner の `train.py` スクリプトで利用できるコマンドラインオプションを分かりやすく説明します。これらのオプションにより高度なカスタマイズが可能になり、要件に合わせてモデルを学習できます。

### JSON 設定ファイル形式

期待される JSON ファイル名は `config.json` で、キー名は以下の `--arguments` と同じです。JSON では先頭の `--` は不要ですが、残していても構いません。

すぐに使える例を探している場合は、[simpletuner/examples/README.md](/simpletuner/examples/README.md) のプリセットを参照してください。

### 簡単設定スクリプト（***推奨***）

`simpletuner configure` コマンドを使うと、ほぼ理想的な既定値で `config.json` をセットアップできます。

#### 既存設定の変更

`configure` コマンドは互換性のある `config.json` を 1 つ受け取り、対話的に設定を変更できます:

```bash
simpletuner configure config/foo/config.json
```

`foo` は設定環境名です。設定環境を使っていない場合は `config/config.json` を指定してください。

<img width="1484" height="560" alt="image" src="https://github.com/user-attachments/assets/67dec8d8-3e41-42df-96e6-f95892d2814c" />

> ⚠️ Hugging Face Hub にアクセスしづらい国のユーザーは、`HF_ENDPOINT=https://hf-mirror.com` を `~/.bashrc` または `~/.zshrc`（利用中の `$SHELL` に応じて）に追加してください。

---

## 🌟 コアモデル設定

### `--model_type`

- **内容**: LoRA またはフルファインチューニングを作成するかを選択します。
- **選択肢**: lora, full.
- **既定**: lora
  - lora の場合、`--lora_type` で PEFT か LyCORIS かが決まります。PixArt など一部モデルは LyCORIS アダプタのみ対応です。

### `--model_family`

- **内容**: どのモデルアーキテクチャを学習するかを指定します。
- **選択肢**: pixart_sigma, flux, sd3, sdxl, kolors, legacy

### `--lora_format`

- **内容**: LoRA チェックポイントの load/save 形式を選択します。
- **選択肢**: `diffusers`（既定）, `comfyui`
- **注記**:
  - `diffusers` は標準の PEFT/Diffusers 形式です。
  - `comfyui` は ComfyUI 形式（`diffusion_model.*` と `lora_A/lora_B` + `.alpha`）に変換します。Flux、Flux2、Lumina2、Z-Image は `diffusers` のままでも ComfyUI 入力を自動検出しますが、保存時に ComfyUI 出力を強制したい場合は `comfyui` を指定してください。

### `--fuse_qkv_projections`

- **内容**: モデルのアテンションブロック内 QKV 投影を融合し、ハードウェア効率を高めます。
- **注記**: Flash Attention 3 を手動でインストールした NVIDIA H100 または H200 のみで利用可能です。

### `--offload_during_startup`

- **内容**: VAE キャッシュ処理中にテキストエンコーダの重みを CPU にオフロードします。
- **理由**: HiDream や Wan 2.1 のような大型モデルでは、VAE キャッシュ読み込み時に OOM になることがあります。このオプションは学習品質に影響しませんが、非常に大きなテキストエンコーダや低速 CPU の場合、複数データセットで起動時間が大きく伸びることがあります。そのため既定では無効です。
- **ヒント**: 特にメモリが厳しい環境では、後述のグループオフロードと併用すると効果的です。

### `--offload_during_save`

- **内容**: `save_hooks.py` がチェックポイントを準備する間、パイプライン全体を一時的に CPU に移動し、FP8/量子化重みをデバイス外に書き込みます。
- **理由**: fp8-quanto の保存は VRAM 使用量が急増する場合があります（例: `state_dict()` のシリアライズ）。このオプションは学習中はモデルをアクセラレータに保持し、保存時だけオフロードして CUDA OOM を避けます。
- **ヒント**: 保存で OOM が発生する場合のみ有効にしてください。保存後はローダがモデルを戻すため学習はシームレスに再開します。

### `--delete_model_after_load`

- **内容**: モデルをメモリにロードした後、Hugging Face キャッシュからモデルファイルを削除します。
- **理由**: ディスク使用量を抑えたい構成に有効です。モデルを VRAM/RAM にロードした後は、次回の実行までオンディスクキャッシュは不要です。次回実行時のネットワーク帯域負荷と引き換えにストレージを節約します。
- **注記**:
  - 検証が有効な場合、VAE は検証画像の生成に必要なため削除されません。
  - テキストエンコーダはデータバックエンドの起動完了後（埋め込みキャッシュ後）に削除されます。
  - Transformer/UNet はロード直後に削除されます。
  - マルチノード構成では、各ノードの local-rank 0 のみが削除を実行します。共有ストレージでの競合を避けるため、削除失敗は無視されます。
  - 学習チェックポイントには影響せず、事前学習済みベースモデルのキャッシュのみが対象です。

### `--enable_group_offload`

- **内容**: diffusers の grouped module offloading を有効化し、forward 間でモデルブロックを CPU（またはディスク）に退避します。
- **理由**: 大規模 Transformer（Flux、Wan、Auraflow、LTXVideo、Cosmos2Image）で VRAM ピーク使用量を大幅に削減し、CUDA ストリームと併用すれば性能影響は最小です。
- **注記**:
  - `--enable_model_cpu_offload` と排他です。実行ごとにどちらか一方を選択してください。
  - diffusers **v0.33.0** 以上が必要です。

### `--group_offload_type`

- **選択肢**: `block_level`（既定）, `leaf_level`
- **内容**: レイヤーのグルーピング方法を制御します。`block_level` は VRAM 削減とスループットのバランス、`leaf_level` は最大の削減と引き換えに CPU 転送が増えます。

### `--group_offload_blocks_per_group`

- **内容**: `block_level` 使用時、1 グループに束ねる Transformer ブロック数。
- **既定**: `1`
- **理由**: 値を増やすと転送頻度が減り高速化しますが、アクセラレータ上に保持するパラメータが増えます。

### `--group_offload_use_stream`

- **内容**: 専用 CUDA ストリームでホスト/デバイス転送と計算を重ね合わせます。
- **既定**: `False`
- **注記**:
  - 非 CUDA バックエンド（Apple MPS、ROCm、CPU）では自動的に CPU 風の転送にフォールバックします。
  - NVIDIA GPU でコピーエンジンに余力がある場合に推奨します。

### `--group_offload_to_disk_path`

- **内容**: グループ化されたパラメータを RAM ではなくディスクにスピルするためのディレクトリパス。
- **理由**: CPU RAM が極端に厳しい環境（例: 大容量 NVMe を持つワークステーション）で有効です。
- **ヒント**: 高速なローカル SSD を使ってください。ネットワークファイルシステムでは学習が大幅に遅くなります。

### `--musubi_blocks_to_swap`

- **内容**: LongCat-Video、Wan、LTXVideo、Kandinsky5-Video、Qwen-Image、Flux、Flux.2、Cosmos2Image、HunyuanVideo の Musubi ブロックスワップ。最後の N 個の Transformer ブロックを CPU に置き、forward 中にブロック単位で重みをストリーミングします。
- **既定**: `0`（無効）
- **注記**: Musubi 方式の重みオフロードで、スループット低下と引き換えに VRAM を削減します。勾配が有効な場合はスキップされます。
- **注記**: Musubi 方式の重みオフロードで、スループット低下と引き換えに VRAM を削減します。勾配が有効な場合はスキップされます。

### `--musubi_block_swap_device`

- **内容**: スワップする Transformer ブロックの保存先デバイス文字列（例: `cpu`、`cuda:0`）。
- **既定**: `cpu`
- **注記**: `--musubi_blocks_to_swap` > 0 のときのみ使用されます。

### `--ramtorch`

- **内容**: `nn.Linear` レイヤーを RamTorch の CPU ストリーミング実装に置き換えます。
- **理由**: Linear 重みを CPU メモリで共有し、アクセラレータへストリーミングすることで VRAM 圧力を下げます。
- **注記**:
  - CUDA または ROCm が必要（Apple/MPS では非対応）。
  - `--enable_group_offload` と排他。
  - `--set_grads_to_none` を自動的に有効化します。

### `--ramtorch_target_modules`

- **内容**: RamTorch に変換する Linear モジュールを制限するカンマ区切りの glob パターン。
- **既定**: パターン未指定の場合、すべての Linear レイヤーを変換。
- **注記**: 完全修飾モジュール名またはクラス名にマッチ（ワイルドカード可）。

### `--ramtorch_text_encoder`

- **内容**: すべてのテキストエンコーダ Linear レイヤーに RamTorch 置換を適用。
- **既定**: `False`

### `--ramtorch_vae`

- **内容**: VAE の mid-block Linear レイヤーのみを RamTorch に変換する実験的オプション。
- **既定**: `False`
- **注記**: VAE の up/down 畳み込みブロックは変更されません。

### `--ramtorch_controlnet`

- **内容**: ControlNet を学習する際、ControlNet の Linear レイヤーに RamTorch 置換を適用。
- **既定**: `False`

### `--pretrained_model_name_or_path`

- **内容**: 事前学習済みモデルのパス、または <https://huggingface.co/models> の識別子。
- **理由**: 学習を開始するベースモデルを指定します。`--revision` と `--variant` でリポジトリ内の特定バージョンを指定できます。SDXL、Flux、SD3.x の単一ファイル `.safetensors` パスにも対応しています。

### `--pretrained_t5_model_name_or_path`

- **内容**: 事前学習済み T5 モデルのパス、または <https://huggingface.co/models> の識別子。
- **理由**: PixArt の学習時、T5 重みの取得元を指定することで、ベースモデル切り替え時の重複ダウンロードを避けられます。

### `--pretrained_gemma_model_name_or_path`

- **内容**: 事前学習済み Gemma モデルのパス、または <https://huggingface.co/models> の識別子。
- **理由**: Gemma 系モデル（例: LTX-2、Sana、Lumina2）を学習する際、ベース拡散モデルのパスを変えずに Gemma 重みの参照先を指定できます。

### `--custom_text_encoder_intermediary_layers`

- **内容**: FLUX.2 モデルでテキストエンコーダーから抽出する隠れ状態レイヤーを上書き指定します。
- **形式**: レイヤーインデックスの JSON 配列（例: `[10, 20, 30]`）
- **デフォルト**: 未設定時はモデル固有のデフォルト値を使用:
  - FLUX.2-dev (Mistral-3): `[10, 20, 30]`
  - FLUX.2-klein (Qwen3): `[9, 18, 27]`
- **理由**: カスタムアライメントや研究目的で、異なるテキストエンコーダー隠れ状態の組み合わせを実験できます。
- **注意**: この設定は実験的で FLUX.2 モデルにのみ適用されます。レイヤーインデックスを変更するとキャッシュ済みテキスト埋め込みが無効になり、再生成が必要です。レイヤー数はモデルの期待入力と一致させる必要があります（3 レイヤー）。

### `--gradient_checkpointing`

- **内容**: 学習中に勾配をレイヤー単位で計算・蓄積し、ピーク VRAM を削減します（学習速度は低下）。

### `--gradient_checkpointing_interval`

- **内容**: *n* ブロックごとにチェックポイントを作成します。値は 0 より大きい必要があります。1 は `--gradient_checkpointing` と同等で、2 は隔ブロックでチェックポイントを作成します。
- **注記**: 現在このオプションに対応しているのは SDXL と Flux のみです。SDXL は暫定的な実装です。

### `--refiner_training`

- **内容**: カスタムの Mixture-of-Experts モデル系列の学習を有効化します。詳細は [Mixture-of-Experts](MIXTURE_OF_EXPERTS.md) を参照してください。

## 精度

### `--quantize_via`

- **選択肢**: `cpu`、`accelerator`、`pipeline`
  - `accelerator` では若干高速になる可能性がありますが、Flux のように大きなモデルでは 24G カードで OOM するリスクがあります。
  - `cpu` では量子化に約 30 秒かかります。（**既定**）
  - `pipeline` は `--quantization_config` またはパイプライン対応のプリセット（例: `nf4-bnb`、`int8-torchao`、`fp8-torchao`、`int8-quanto`、`.gguf` チェックポイント）を使って Diffusers に量子化を委譲します。

### `--base_model_precision`

- **内容**: モデル精度を下げ、少ないメモリで学習します。対応する量子化バックエンドは BitsAndBytes（pipeline）、TorchAO（pipeline または手動）、Optimum Quanto（pipeline または手動）の 3 つです。

#### Diffusers のパイプラインプリセット

- `nf4-bnb` は Diffusers 経由で 4-bit NF4 BitsAndBytes 設定で読み込みます（CUDA のみ）。`bitsandbytes` と BnB 対応の diffusers が必要です。
- `int4-torchao`、`int8-torchao`、`fp8-torchao` は Diffusers 経由で TorchAoConfig を使用します（CUDA）。`torchao` と最新の diffusers/transformers が必要です。
- `int8-quanto`、`int4-quanto`、`int2-quanto`、`fp8-quanto`、`fp8uz-quanto` は Diffusers 経由で QuantoConfig を使用します。Diffusers は FP8-NUZ を float8 重みにマップするため、NUZ 変種が必要な場合は手動の quanto 量子化を使ってください。
- `.gguf` チェックポイントは自動検出され、`GGUFQuantizationConfig` で読み込まれます。GGUF 対応には最新の diffusers/transformers をインストールしてください。

#### Optimum Quanto

Hugging Face が提供する optimum-quanto は、すべての対応プラットフォームで堅牢に動作します。

- `int8-quanto` は最も互換性が高く、おそらく最良の結果になります
  - RTX4090 やその他 GPU で最速の学習
  - CUDA デバイスでは int8/int4 のハードウェア加速 matmul を使用
    - int4 は依然として非常に遅い
  - `TRAINING_DYNAMO_BACKEND=inductor`（`torch.compile()`）で動作
- `fp8uz-quanto` は CUDA と ROCm 向けの実験的 fp8 バリアントです。
  - Instinct など AMD の新しいアーキテクチャでより良くサポート
  - 4090 では学習が `int8-quanto` よりわずかに速い場合がありますが、推論は遅い（1 秒遅い）
  - `TRAINING_DYNAMO_BACKEND=inductor`（`torch.compile()`）で動作
- `fp8-quanto` は（現時点では）fp8 matmul を使用せず、Apple では動作しません。
  - CUDA/ROCm でハードウェア fp8 matmul が未整備のため、int8 より遅くなる可能性があります
    - fp8 GEMM に MARLIN カーネルを使用
  - dynamo と互換性がなく、組み合わせると自動的に dynamo を無効化します。

#### TorchAO

PyTorch の新しいライブラリで、Linear と 2D 畳み込み（例: unet 形式）を量子化版に置き換えられます。
<!-- Additionally, it provides an experimental CPU offload optimiser that essentially provides a simpler reimplementation of DeepSpeed. -->

- `int8-torchao` は Quanto の精度レベルと同等にメモリ消費を削減します
  - 記載時点では Apple MPS で Quanto（9s/iter）よりやや遅い（11s/iter）
  - `torch.compile` 未使用時、CUDA では `int8-quanto` と同等の速度とメモリ、ROCm は不明
  - `torch.compile` 使用時は `int8-quanto` より遅い
- `fp8-torchao` は Hopper（H100/H200）またはそれ以降（Blackwell B200）のみ対応

##### オプティマイザ

TorchAO は一般利用可能な 4bit/8bit オプティマイザを提供します: `ao-adamw8bit`、`ao-adamw4bit`

さらに Hopper（H100 以上）向けの `ao-adamfp8` と `ao-adamwfp8` も提供します

#### SDNQ（SD.Next Quantization Engine）

[SDNQ](https://github.com/disty0/sdnq) は学習に最適化された量子化ライブラリで、AMD（ROCm）、Apple（MPS）、NVIDIA（CUDA）すべてで動作します。確率的丸めと量子化オプティマイザ状態により、メモリ効率の高い量子化学習を提供します。

##### 推奨精度レベル

**フルファインチューニング**（モデル重みを更新する場合）:
- `uint8-sdnq` - メモリ削減と学習品質のバランスが最良
- `uint16-sdnq` - 最大品質の高精度（例: Stable Cascade）
- `int16-sdnq` - 符号付き 16-bit の代替
- `fp16-sdnq` - 量子化 FP16、SDNQ の利点を保った最大精度

**LoRA 学習**（ベースモデル重みは固定）:
- `int8-sdnq` - 符号付き 8-bit、汎用的に使いやすい選択
- `int6-sdnq`, `int5-sdnq` - 低精度でメモリ削減
- `uint5-sdnq`, `uint4-sdnq`, `uint3-sdnq`, `uint2-sdnq` - 強い圧縮

**注記:** `int7-sdnq` は利用可能ですが推奨しません（遅く、int8 と大差がありません）。

**重要:** 5-bit 未満では、SDNQ は品質維持のために SVD（特異値分解）を 8 ステップで自動的に有効化します。SVD は量子化に時間がかかり非決定的なため、Disty0 は Hugging Face に事前量子化済み SVD モデルを提供しています。SVD は学習中に計算オーバーヘッドを追加するので、重みを更新するフルファインチューニングでは避けてください。

**主な特長:**
- クロスプラットフォーム: AMD、Apple、NVIDIA で同一動作
- 学習最適化: 確率的丸めで量子化誤差の蓄積を低減
- メモリ効率: 量子化されたオプティマイザ状態バッファに対応
- 乗算の分離: 重み精度と matmul 精度を独立に選べる（INT8/FP8/FP16）

##### SDNQ オプティマイザ

SDNQ は追加のメモリ削減のため、量子化状態バッファを持つオプティマイザを含みます:

- `sdnq-adamw` - 量子化状態バッファ付き AdamW（uint8、group_size=32）
- `sdnq-adamw+no_quant` - 量子化なし AdamW（比較用）
- `sdnq-adafactor` - 量子化状態バッファ付き Adafactor
- `sdnq-came` - 量子化状態バッファ付き CAME
- `sdnq-lion` - 量子化状態バッファ付き Lion
- `sdnq-muon` - 量子化状態バッファ付き Muon
- `sdnq-muon+quantized_matmul` - zeropower 計算で INT8 matmul を使う Muon

すべての SDNQ オプティマイザは既定で確率的丸めを使用し、`--optimizer_config` で `use_quantized_buffers=false` を指定すると状態量子化を無効化できます。

**Muon 固有オプション:**
- `use_quantized_matmul` - zeropower_via_newtonschulz5 で INT8/FP8/FP16 matmul を有効化
- `quantized_matmul_dtype` - matmul 精度: `int8`（コンシューマ GPU）、`fp8`（データセンター）、`fp16`
- `zeropower_dtype` - zeropower 計算の精度（`use_quantized_matmul=True` の場合は無視）
- `muon_` または `adamw_` の接頭辞で Muon と AdamW フォールバックに異なる値を設定

**事前量子化モデル:** Disty0 は事前量子化済みの uint4 SVD モデルを [huggingface.co/collections/Disty0/sdnq](https://huggingface.co/collections/Disty0/sdnq) に提供しています。通常どおり読み込み、SDNQ を import した後に `convert_sdnq_model_to_training()` で変換してください（SDNQ は Diffusers への登録のため、読み込み前に import が必要です）。

**チェックポイントに関する注記:** SDNQ 学習モデルは、学習再開用の PyTorch ネイティブ形式（`.pt`）と推論用の safetensors 形式の両方で保存されます。SDNQ の `SDNQTensor` が独自シリアライズを使うため、学習再開にはネイティブ形式が必要です。

**ディスク節約のヒント:** ディスク容量を節約するには量子化済み重みのみを保持し、必要に応じて SDNQ の [dequantize_sdnq_training.py](https://github.com/Disty0/sdnq/blob/main/scripts/dequantize_sdnq_training.py) で推論用に復元できます。

### `--quantization_config`

- **内容**: `--quantize_via=pipeline` 使用時に Diffusers の `quantization_config` を上書きする JSON オブジェクトまたはファイルパス。
- **方法**: インライン JSON（またはファイル）でコンポーネント別の設定を指定します。キーは `unet`、`transformer`、`text_encoder`、`default` など。
- **例**:

```json
{
  "unet": {"load_in_4bit": true, "bnb_4bit_quant_type": "nf4", "bnb_4bit_compute_dtype": "bfloat16"},
  "text_encoder": {"quant_type": {"group_size": 128}}
}
```

この例では UNet に 4-bit NF4 BnB、テキストエンコーダに TorchAO int4 を適用します。

#### Torch Dynamo

WebUI で `torch.compile()` を有効化するには **Hardware → Accelerate (advanced)** に移動し、**Torch Dynamo Backend** を希望のコンパイラ（例: *inductor*）に設定します。追加のトグルで最適化 **mode**、**dynamic shape**、**regional compilation** を選べます（深い Transformer のコールドスタート短縮）。

同じ設定は `config/config.env` に次のように記述できます:

```bash
TRAINING_DYNAMO_BACKEND=inductor
```

必要に応じて `--dynamo_mode=max-autotune` など UI にある Dynamo フラグを組み合わせて細かく制御できます。

コンパイルはバックグラウンドで行われるため、最初の数ステップは通常より遅くなります。

`config.json` に永続化するには以下を追加します:

```json
{
  "dynamo_backend": "inductor",
  "dynamo_mode": "max-autotune",
  "dynamo_fullgraph": false,
  "dynamo_dynamic": false,
  "dynamo_use_regional_compilation": true
}
```

Accelerate の既定値を使いたい項目は省略してください（例: 自動選択の `dynamo_mode` を使うなら省略）。

### `--attention_mechanism`

代替アテンション機構が利用可能で、互換性やトレードオフが異なります:

- `diffusers` は PyTorch の標準 SDPA カーネルを使い、既定です。
- `xformers` はモデルが `enable_xformers_memory_efficient_attention` を公開している場合に Meta の [xformers](https://github.com/facebookresearch/xformers) アテンションカーネル（学習+推論）を有効化します。
- `flash-attn`、`flash-attn-2`、`flash-attn-3`、`flash-attn-3-varlen` は Diffusers の `attention_backend` ヘルパーに接続し、FlashAttention v1/2/3 カーネルへルーティングします。対応する `flash-attn` / `flash-attn-interface` をインストールし、FA3 は Hopper GPU が必要です。
- `flex` は PyTorch 2.5 の FlexAttention バックエンド（CUDA の FP16/BF16）を選択します。Flex カーネルは別途コンパイル/インストールが必要です。詳細は [documentation/attention/FLEX.md](attention/FLEX.md) を参照。
- `cudnn`、`native-efficient`、`native-flash`、`native-math`、`native-npu`、`native-xla` は `torch.nn.attention.sdpa_kernel` が提供する対応 SDPA バックエンドを選択します。`native-math` による決定性、CuDNN SDPA カーネル、NPU/XLA のネイティブアクセラレータ利用に有用です。
- `sla` は [Sparse–Linear Attention (SLA)](https://github.com/thu-ml/SLA) を有効化し、学習と検証の双方で使える疎/線形ハイブリッドカーネルを提供します。
  - SLA パッケージをインストールしてから（例: `pip install -e ~/src/SLA`）選択してください。
  - SimpleTuner は SLA の学習済み投影重みを各チェックポイント内の `sla_attention.pt` に保存します。再開や推論で SLA 状態を保持するため、このファイルをチェックポイントと一緒に保管してください。
  - SLA の疎/線形混合挙動に合わせてバックボーンが調整されるため、推論でも SLA が必要です。詳細は `documentation/attention/SLA.md` を参照してください。
  - 必要に応じて `--sla_config '{"topk":0.15,"blkq":32,"tie_feature_map_qk":false}'`（JSON または Python dict 形式）で SLA の実行時既定値を上書きできます。
- `sageattention`、`sageattention-int8-fp16-triton`、`sageattention-int8-fp16-cuda`、`sageattention-int8-fp8-cuda` は [SageAttention](https://github.com/thu-ml/SageAttention) の対応カーネルを利用します。これらは推論向けであり、誤って学習で使わないよう `--sageattention_usage` と併用してください。
  - 単純に言うと、SageAttention は推論の計算量を削減します。

> ℹ️ Flash/Flex/PyTorch バックエンド選択は Diffusers の `attention_backend` ディスパッチャに依存するため、現状はその経路を使う Transformer 系モデル（Flux、Wan 2.x、LTXVideo、QwenImage など）で効果があります。従来の SD/SDXL UNet は PyTorch SDPA を直接使用します。

`--sageattention_usage` を使って SageAttention で学習を有効化する場合は注意が必要です。カスタム CUDA 実装の QKV Linear から勾配を追跡・伝播しません。

- その結果、これらのレイヤーは完全に学習されず、モデル崩壊や短い学習でのわずかな改善を引き起こす可能性があります。

---

## 📰 Publishing

### `--push_to_hub`

- **内容**: 指定すると、学習完了後にモデルを [Hugging Face Hub](https://huggingface.co) にアップロードします。`--push_checkpoints_to_hub` を使うと中間チェックポイントもすべてプッシュされます。

### `--push_to_hub_background`

- **内容**: バックグラウンドワーカーから Hugging Face Hub にアップロードし、チェックポイント送信で学習ループが停止しないようにします。
- **理由**: Hub へのアップロード中も学習と検証を継続できます。最終アップロードは終了前に待機するため、失敗は可視化されます。

### `--webhook_config`

- **内容**: リアルタイムの学習イベントを受け取る Webhook（Discord、カスタムエンドポイントなど）の設定。
- **理由**: 外部ツールやダッシュボードで学習を監視し、重要なステージで通知を受け取れます。
- **注記**: Webhook ペイロードの `job_id` は、学習前に `SIMPLETUNER_JOB_ID` 環境変数を設定することで埋められます:
  ```bash
  export SIMPLETUNER_JOB_ID="my-training-run-name"
  python train.py
  ```
複数の学習ランからの Webhook を受ける監視ツールで、どの設定がイベントを送信したか識別するのに有用です。SIMPLETUNER_JOB_ID が未設定の場合、Webhook ペイロードの job_id は null になります。

### `--publishing_config`

- **内容**: Hugging Face 以外の公開先（S3 互換ストレージ、Backblaze B2、Azure Blob Storage、Dropbox）を記述する JSON/dict/ファイルパス。
- **理由**: `--webhook_config` と同様の解析を行い、Hub 以外にも成果物を配布できます。公開は検証後にメインプロセスで `output_dir` を使って実行されます。
- **注記**: 追加の公開先は `--push_to_hub` に加算されます。各プロバイダ SDK（例: `boto3`、`azure-storage-blob`、`dropbox`）を `.venv` にインストールしてください。完全な例は `documentation/publishing/README.md` を参照。

### `--hub_model_id`

- **内容**: Hugging Face Hub のモデル名およびローカル結果ディレクトリ名。
- **理由**: `--output_dir` で指定した場所の配下にこの名前のディレクトリが作成されます。`--push_to_hub` を指定した場合、Hugging Face Hub 上のモデル名になります。

### `--modelspec_comment`

- **内容**: safetensors ファイルのメタデータに `modelspec.comment` として埋め込まれるテキスト
- **デフォルト**: None（無効）
- **注記**:
  - 外部モデルビューア（ComfyUI、モデル情報ツール）で表示可能
  - 文字列または文字列配列（改行で結合）を受け付けます
  - 環境変数置換用の `{env:VAR_NAME}` プレースホルダをサポート
  - 各チェックポイントは保存時の現在の設定値を使用

**例（文字列）**:
```json
"modelspec_comment": "カスタムデータセット v2.1 で学習"
```

**例（配列、複数行）**:
```json
"modelspec_comment": [
  "学習ラン: experiment-42",
  "データセット: custom-portraits-v2",
  "メモ: {env:TRAINING_NOTES}"
]
```

### `--disable_benchmark`

- **内容**: step 0 で行う起動時の検証/ベンチマークを無効化します。これらの出力は学習済みモデルの検証画像の左側に連結されます。

## 📂 データストレージと管理

### `--data_backend_config`

- **内容**: SimpleTuner データセット設定へのパス。
- **理由**: 複数のストレージ媒体にあるデータセットを 1 つの学習セッションにまとめられます。
- **例**: 設定例は [multidatabackend.json.example](/multidatabackend.json.example) を参照し、データローダ設定の詳細は [このドキュメント](DATALOADER.md) を参照してください。

### `--override_dataset_config`

- **内容**: 指定すると、データセット内にキャッシュされた設定と現在の値の差異を無視できます。
- **理由**: SimpleTuner を初めてデータセットに対して実行すると、データセット内の情報を含むキャッシュ文書が作られます。これには `crop` や `resolution` などの設定値が含まれます。これらを不用意に変更すると学習ジョブが不規則にクラッシュする可能性があるため、このパラメータの使用は推奨されません。適用したい差分は別の方法で解決してください。

### `--data_backend_sampling`

- **内容**: 複数のデータバックエンドを使う場合、異なるサンプリング戦略を選べます。
- **オプション**:
  - `uniform` - v0.9.8.1 以前の挙動で、データセット長は考慮せず、手動の確率ウェイトのみを使用します。
  - `auto-weighting` - 既定の挙動。データセット長を使って全データセットを均等にサンプリングし、全体分布を均一に保ちます。
    - サイズが異なるデータセットを均等に学習させたい場合に必須です。
    - ただし Dreambooth 画像と正則化セットを適切にサンプリングするには、`repeats` の手動調整が**必須**です。

### `--vae_cache_scan_behaviour`

- **内容**: 整合性スキャンの挙動を設定します。
- **理由**: 学習中の複数のタイミングで誤った設定が適用される可能性があります（例: データセットの `.json` キャッシュを誤って削除し、データバックエンド設定をアスペクトクロップから正方形に変更してしまう）。その結果データキャッシュが不整合になり、`multidatabackend.json` で `scan_for_errors` を `true` にすることで修正できます。スキャン時に `--vae_cache_scan_behaviour` に従って不整合を解決します。既定の `recreate` は該当キャッシュエントリを削除して再作成し、`sync` はバケットメタデータを実サンプルに合わせて更新します。推奨値: `recreate`。

### `--dataloader_prefetch`

- **内容**: バッチを事前に取得します。
- **理由**: 大きなバッチサイズでは、ディスク（NVMe でも）からの読み込み中に学習が停止し、GPU 利用率が低下します。prefetch を有効にするとバッファにバッチを保持し、即時読み込みできます。

> ⚠️ これは H100 以上かつ低解像度で I/O がボトルネックになる場合にのみ有効です。ほとんどの用途では不要な複雑さです。

### `--dataloader_prefetch_qlen`

- **内容**: メモリに保持するバッチ数を増減します。
- **理由**: dataloader prefetch の既定では GPU/プロセスあたり 10 エントリを保持します。多すぎる/少なすぎる場合に調整できます。

### `--compress_disk_cache`

- **内容**: VAE とテキスト埋め込みキャッシュをディスク上で圧縮します。
- **理由**: DeepFloyd、SD3、PixArt が使う T5 エンコーダは非常に大きなテキスト埋め込みを生成し、短いキャプションや冗長なキャプションでは空領域が多くなります。`--compress_disk_cache` で最大 75%、平均 40% の削減が見込めます。

> ⚠️ 既存キャッシュディレクトリは手動で削除し、圧縮付きで再作成されるようにしてください。

---

## 🌈 画像とテキスト処理

多くの設定は [dataloader 設定](DATALOADER.md) 経由ですが、ここでの設定はグローバルに適用されます。

### `--resolution_type`

- **内容**: `area`（面積）計算を使うか、`pixel`（短辺）計算を使うかを指定します。`pixel_area` は面積計算をピクセル単位で扱えるハイブリッド方式です。
- **オプション**:
  - `resolution_type=pixel_area`
    - `resolution` が 1024 の場合、アスペクトバケット用に正確な面積へ内部変換されます。
    - 例: `1024` の場合のサイズ 1024x1024、1216x832、832x1216
  - `resolution_type=pixel`
    - すべての画像は短辺がこの解像度になるようにリサイズされます。結果の画像サイズが大きくなり、VRAM 使用量が増える可能性があります。
    - 例: `1024` の場合のサイズ 1024x1024、1766x1024、1024x1766
  - `resolution_type=area`
    - **非推奨**。`pixel_area` を使用してください。

### `--resolution`

- **内容**: 入力画像解像度（短辺のピクセル長）
- **既定**: 1024
- **注記**: データセットに解像度が設定されていない場合のグローバル既定値です。

### `--validation_resolution`

- **内容**: 出力画像解像度（ピクセル）、または `1024x1024` のように `widthxheight` 形式で指定します。複数の解像度をカンマ区切りで指定できます。
- **理由**: 検証時に生成される画像解像度を指定します。学習解像度と異なる場合に有用です。

### `--validation_method`

- **内容**: 検証の実行方法を選択します。
- **オプション**: `simpletuner-local`（既定）は内蔵パイプラインを実行、`external-script` はユーザー提供の実行ファイルを使います。
- **理由**: ローカルパイプラインを止めずに、外部システムへ検証を委譲できます。

### `--validation_external_script`

- **内容**: `--validation_method=external-script` のときに実行するコマンド。シェル形式で分割されるため、コマンド文字列は適切にクォートしてください。
- **プレースホルダ**: 学習コンテキストを渡すためのトークン（`.format` 形式）を埋め込めます。記載がない場合は空文字になります（特記がない場合）。
  - `{local_checkpoint_path}` → `output_dir` 配下の最新チェックポイントディレクトリ（少なくとも 1 つのチェックポイントが必要）。
  - `{local_checkpoint_path}` → `output_dir` 配下の最新チェックポイントディレクトリ（少なくとも 1 つのチェックポイントが必要）。
  - `{global_step}` → 現在のグローバルステップ。
  - `{tracker_run_name}` → `--tracker_run_name` の値。
  - `{tracker_project_name}` → `--tracker_project_name` の値。
  - `{model_family}` → `--model_family` の値。
  - `{model_type}` / `{lora_type}` → モデルタイプと LoRA フレーバー。
  - `{huggingface_path}` → `--hub_model_id` の値（設定されている場合）。
  - `{remote_checkpoint_path}` → 最後にアップロードしたリモート URL（検証フックでは空）。
  - 任意の `validation_*` 設定値（例: `validation_num_inference_steps`、`validation_guidance`、`validation_noise_scheduler`）。
- **例**: `--validation_external_script="/opt/tools/validate.sh {local_checkpoint_path} {global_step}"`

### `--validation_external_background`

- **内容**: 有効化すると `--validation_external_script` をバックグラウンドで起動します（fire-and-forget）。
- **理由**: 外部スクリプトを待たずに学習を進めます。このモードでは終了コードを確認しません。

### `--post_upload_script`

- **内容**: 各公開先および Hugging Face Hub へのアップロード完了後に実行される任意の実行ファイル（最終モデルとチェックポイントのアップロード）。学習をブロックしないよう非同期で実行されます。
- **プレースホルダ**: `--validation_external_script` と同じ置換に加えて `{remote_checkpoint_path}`（プロバイダが返す URI）が使えます。公開 URL を下流システムへ渡す用途に便利です。
- **注記**:
  - スクリプトはプロバイダ/アップロードごとに実行されます。エラーはログに記録されますが学習は停止しません。
  - リモートアップロードがない場合でも実行されるため、ローカル自動化（例: 別 GPU で推論）にも使えます。
  - SimpleTuner はスクリプトの結果を取り込みません。メトリクスや画像を記録したい場合はトラッカーに直接ログしてください。
- **例**:
  ```bash
  --post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'
  ```
  `/opt/hooks/notify.sh` がトラッキングシステムに投稿する例:
  ```bash
  #!/usr/bin/env bash
  REMOTE="$1"
  PROJECT="$2"
  RUN="$3"
  curl -X POST "https://tracker.internal/api/runs/${PROJECT}/${RUN}/artifacts" \
       -H "Content-Type: application/json" \
       -d "{\"remote_uri\":\"${REMOTE}\"}"
  ```
- **動作サンプル**:
  - `simpletuner/examples/external-validation/replicate_post_upload.py` は `{remote_checkpoint_path}`、`{model_family}`、`{model_type}`、`{lora_type}`、`{huggingface_path}` を使ってアップロード後に推論を起動する Replicate フック。
  - `simpletuner/examples/external-validation/wavespeed_post_upload.py` は同じプレースホルダに加えて WaveSpeed の非同期ポーリングを使うフック。
  - `simpletuner/examples/external-validation/fal_post_upload.py` は fal.ai Flux LoRA フック（`FAL_KEY` が必要）。
  - `simpletuner/examples/external-validation/use_second_gpu.py` は二次 GPU で Flux LoRA 推論を実行し、リモートアップロードがなくても動作します。

### `--post_checkpoint_script`

- **内容**: 各チェックポイントディレクトリがディスクに書き込まれた直後（アップロード開始前）に実行される実行ファイル。メインプロセスで非同期に動作します。
- **プレースホルダ**: `--validation_external_script` と同じ置換（`{local_checkpoint_path}`、`{global_step}`、`{tracker_run_name}`、`{tracker_project_name}`、`{model_family}`、`{model_type}`、`{lora_type}`、`{huggingface_path}`、および任意の `validation_*` 設定）。`{remote_checkpoint_path}` は空になります。
- **注記**:
  - 予約/手動/ローリングの各チェックポイント保存直後に起動します。
  - アップロード完了を待たずにローカル自動化（別ボリュームへのコピーや評価ジョブの実行）をトリガーできます。
- **例**:
  ```bash
  --post_checkpoint_script='/opt/hooks/run_eval.sh {local_checkpoint_path} {global_step}'
  ```


### `--validation_adapter_path`

- **内容**: 予約検証時に一時的に単一の LoRA アダプタを読み込みます。
- **形式**:
  - Hugging Face リポジトリ: `org/repo` または `org/repo:weight_name.safetensors`（既定は `pytorch_lora_weights.safetensors`）。
  - safetensors アダプタを指すローカルファイル/ディレクトリパス。
- **注記**:
  - `--validation_adapter_config` と排他で、両方指定するとエラーになります。
  - アダプタは検証時のみ適用されます（学習中の重みは変更されません）。

### `--validation_adapter_name`

- **内容**: `--validation_adapter_path` で読み込む一時アダプタに付ける任意の識別子。
- **理由**: ログ/Web UI での表示名を制御し、複数アダプタを順次テストする際に安定した名前を確保します。

### `--validation_adapter_strength`

- **内容**: 一時アダプタを有効化する際の強度倍率（既定 `1.0`）。
- **理由**: 学習状態を変更せずに、検証で LoRA の強弱を試せます。0 より大きい任意の値を受け付けます。

### `--validation_adapter_mode`

- **選択肢**: `adapter_only`, `comparison`, `none`
- **内容**:
  - `adapter_only`: 一時アダプタ付きでのみ検証を実行。
  - `comparison`: ベースモデルとアダプタ有効時の両方を生成し、並べて比較。
  - `none`: アダプタを付けずに検証（CLI フラグを残したまま無効化したい場合）。

### `--validation_adapter_config`

- **内容**: 複数の検証アダプタ組み合わせを順に試すための JSON ファイルまたはインライン JSON。
- **形式**: エントリ配列、または `runs` 配列を持つオブジェクト。各エントリは次を含められます:
  - `label`: ログ/UI に表示される名前。
  - `path`: Hugging Face リポジトリ ID またはローカルパス（`--validation_adapter_path` と同形式）。
  - `adapter_name`: アダプタごとの任意識別子。
  - `strength`: 任意のスカラー上書き。
  - `adapters`/`paths`: 単一実行で複数アダプタを読み込むためのオブジェクト/文字列配列。
- **注記**:
  - 指定すると、単一アダプタ用のオプション（`--validation_adapter_path`、`--validation_adapter_name`、`--validation_adapter_strength`、`--validation_adapter_mode`）は UI で無効化されます。
  - 各 run は 1 つずつ読み込まれ、次の run 前に完全に解除されます。

### `--validation_preview`

- **内容**: Tiny AutoEncoder を使って拡散サンプリング中の中間検証プレビューをストリーミングします
- **既定**: False
- **理由**: 軽量な Tiny AutoEncoder でデコードしたプレビュー画像を Webhook で送信し、生成過程をリアルタイムに確認できます。完了まで待つ必要がありません。
- **注記**:
  - Tiny AutoEncoder に対応するモデルファミリーのみ（例: Flux、SDXL、SD3）
  - プレビュー画像を受け取る Webhook 設定が必要
  - `--validation_preview_steps` でプレビュー頻度を制御

### `--validation_preview_steps`

- **内容**: 検証プレビューのデコード/ストリーミング間隔
- **既定**: 1
- **理由**: 検証サンプリング中に中間潜在をどの頻度でデコードするかを制御します。値を大きくすると（例: 3）、Tiny AutoEncoder のオーバーヘッドを減らせます。
- **例**: `--validation_num_inference_steps=20` と `--validation_preview_steps=5` の場合、生成中の 5、10、15、20 ステップで 4 枚のプレビューが届きます。

### `--evaluation_type`

- **内容**: 検証時に生成画像の CLIP 評価を有効化します。
- **理由**: CLIP スコアは生成画像特徴と検証プロンプトの距離を算出します。プロンプトへの一致が改善しているかの指標になりますが、意味のある評価には多数の検証プロンプトが必要です。
- **選択肢**: "none" または "clip"
- **スケジューリング**: ステップ単位の `--eval_steps_interval` またはエポック単位の `--eval_epoch_interval` を使用します（`0.5` のような小数はエポック内で複数回実行）。両方設定すると警告を出し、両方のスケジュールで実行します。

### `--eval_loss_disable`

- **内容**: 検証中の評価損失計算を無効化します。
- **理由**: 評価用データセットを設定すると損失は自動計算されます。CLIP 評価も有効な場合は両方実行されます。このフラグで CLIP を残したまま評価損失だけ無効化できます。

### `--caption_strategy`

- **内容**: 画像キャプションを導出する戦略。**選択肢**: `textfile`, `filename`, `parquet`, `instanceprompt`
- **理由**: 学習画像のキャプション生成方法を決めます。
  - `textfile` は画像と同名の `.txt` ファイルの内容を使います
  - `filename` はファイル名を整形してキャプションにします
  - `parquet` はデータセット内の parquet ファイルを使用し、`parquet_caption_column` が指定されていなければ `caption` 列を使います。`parquet_fallback_caption_column` がない場合、すべてのキャプションが必要です。
  - `instanceprompt` はデータセット設定の `instance_prompt` を全画像のプロンプトとして使います。

### `--conditioning_multidataset_sampling` {#--conditioning_multidataset_sampling}

- **内容**: 複数の条件データセットからのサンプリング方法。**選択肢**: `combined`, `random`
- **理由**: 複数の条件データセット（例: 複数の参照画像や制御信号）を使う場合、どのように利用するかを決めます:
  - `combined` は条件入力を結合し、学習中に同時に提示します。複数画像の合成タスクに有用です。
  - `random` はサンプルごとに条件データセットをランダムに 1 つ選び、学習中に条件を切り替えます。
- **注記**: `combined` を使う場合、条件データセットに個別の `captions` は定義できません。ソースデータセットのキャプションが使用されます。
- **参照**: 複数条件データセットの設定は [DATALOADER.md](DATALOADER.md#conditioning_data) を参照してください。

---

## 🎛 学習パラメータ

### `--num_train_epochs`

- **内容**: 学習エポック数（全画像が見られる回数）。0 にすると `--max_train_steps` が優先されます。
- **理由**: 画像の繰り返し回数を決め、学習期間に影響します。エポック数が多いほど過学習しやすいですが、学習したい概念を拾うには必要な場合があります。目安は 5〜50。

### `--max_train_steps`

- **内容**: このステップ数で学習を終了します。0 にすると `--num_train_epochs` が優先されます。
- **理由**: 学習時間を短縮したい場合に有用です。

### `--ignore_final_epochs`

- **内容**: 最終エポックのカウントを無視して `--max_train_steps` を優先します。
- **理由**: データローダの長さ変更により想定より早く終了する場合、最終エポックを無視して `--max_train_steps` まで学習を続けます。

### `--learning_rate`

- **内容**: ウォームアップ後の初期学習率。
- **理由**: 学習率は勾配更新の「ステップ幅」です。高すぎると解を飛び越え、低すぎると解に到達できません。`full` チューニングでは `1e-7`〜`1e-6`、`lora` では `1e-5`〜`1e-3` が目安です。高い学習率を使う場合は EMA ネットワークと学習率ウォームアップが有利です（`--use_ema`、`--lr_warmup_steps`、`--lr_scheduler`）。

### `--lr_scheduler`

- **内容**: 学習率の時間変化を指定します。
- **選択肢**: constant, constant_with_warmup, cosine, cosine_with_restarts, **polynomial**（推奨）, linear
- **理由**: モデルは学習率の継続的な調整により損失地形をより探索できます。既定は cosine で、2 つの極値の間を滑らかに遷移します。学習率を一定にすると高すぎて発散、低すぎて局所解に停滞することがあります。polynomial はウォームアップと相性が良く、`learning_rate` に近づいた後は減速し、最後に `--lr_end` へ近づきます。

### `--optimizer`

- **内容**: 学習に使用するオプティマイザ。
- **選択肢**: adamw_bf16, ao-adamw8bit, ao-adamw4bit, ao-adamfp8, ao-adamwfp8, adamw_schedulefree, adamw_schedulefree+aggressive, adamw_schedulefree+no_kahan, optimi-stableadamw, optimi-adamw, optimi-lion, optimi-radam, optimi-ranger, optimi-adan, optimi-adam, optimi-sgd, soap, bnb-adagrad, bnb-adagrad8bit, bnb-adam, bnb-adam8bit, bnb-adamw, bnb-adamw8bit, bnb-adamw-paged, bnb-adamw8bit-paged, bnb-lion, bnb-lion8bit, bnb-lion-paged, bnb-lion8bit-paged, bnb-ademamix, bnb-ademamix8bit, bnb-ademamix-paged, bnb-ademamix8bit-paged, prodigy

> 注記: 一部のオプティマイザは NVIDIA 以外のハードウェアでは利用できない場合があります。

### `--optimizer_config`

- **内容**: オプティマイザ設定を調整します。
- **理由**: オプティマイザの設定が多いため、各項目の CLI 引数は用意されていません。代わりにカンマ区切りで値を指定し、既定値を上書きできます。
- **例**: **prodigy** の `d_coef` を設定する場合: `--optimizer_config=d_coef=0.1`

> 注記: オプティマイザのベータ値は専用パラメータ `--optimizer_beta1`、`--optimizer_beta2` で上書きします。

### `--train_batch_size`

- **内容**: 学習データローダのバッチサイズ。
- **理由**: メモリ消費、収束品質、学習速度に影響します。バッチサイズが大きいほど結果は良くなる傾向がありますが、過学習や学習不安定の原因になることもあり、学習時間も増えます。試行が必要ですが、一般には学習速度を落とさずに VRAM を最大限使うよう調整します。

### `--gradient_accumulation_steps`

- **内容**: backward/update を実行するまでに蓄積する更新ステップ数。複数バッチに分割してメモリを節約しますが、学習時間は増えます。
- **理由**: 大きなモデルやデータセットに有用です。

> 注記: 勾配蓄積を使用する場合、どのオプティマイザでも fused backward パスを有効にしないでください。

### `--allow_dataset_oversubscription` {#--allow_dataset_oversubscription}

- **内容**: データセットが実効バッチサイズより小さい場合に `repeats` を自動調整します。
- **理由**: マルチ GPU 構成で最小要件を満たさないデータセットによる学習失敗を防ぎます。
- **仕組み**:
  - **実効バッチサイズ**を計算: `train_batch_size × num_gpus × gradient_accumulation_steps`
  - いずれかのアスペクトバケットが実効バッチサイズを下回る場合、`repeats` を自動的に増加
  - データセット設定で `repeats` を明示していない場合のみ適用
  - 調整内容と理由を警告ログに出力
- **用途**:
  - 複数 GPU と小規模データセット（< 100 枚）
  - データセット再設定なしでバッチサイズを試す
  - フルデータセット収集前のプロトタイピング
- **例**: 25 枚、8 GPU、`train_batch_size=4` の場合、実効バッチサイズは 32。自動で `repeats=1` にして 50 サンプル（25 × 2）を確保します。
- **注記**: データローダ設定で手動指定された `repeats` は上書きしません。`--disable_bucket_pruning` と同様、意図しない挙動を避けつつ利便性を提供します。

マルチ GPU 学習のデータセットサイズ詳細は [DATALOADER.md](DATALOADER.md#automatic-dataset-oversubscription) を参照してください。

---

## 🛠 高度な最適化

### `--use_ema`

- **内容**: 学習期間全体で指数移動平均（EMA）を保持することは、モデルを定期的に自身へバックマージするようなものです。
- **理由**: 追加のシステムリソースとわずかな学習時間増加の代わりに、学習の安定性が向上します。

### `--ema_device`

- **選択肢**: `cpu`, `accelerator`；既定: `cpu`
- **内容**: EMA 重みを更新間に保持する場所を選択します。
- **理由**: アクセラレータに置くと最速ですが VRAM を消費します。CPU に置くとメモリ圧は下がりますが、`--ema_cpu_only` を使わない場合は転送が発生します。

### `--ema_cpu_only`

- **内容**: `--ema_device=cpu` のとき、EMA 重みを更新のためにアクセラレータへ戻さないようにします。
- **理由**: 大規模 EMA のホスト/デバイス転送時間と VRAM 使用量を削減します。`--ema_device=accelerator` では重みが既にアクセラレータにあるため効果はありません。

### `--ema_foreach_disable`

### `--ema_foreach_disable`

- **内容**: EMA 更新に `torch._foreach_*` カーネルを使わないようにします。
- **理由**: foreach オペレーションに問題があるバックエンド/ハードウェアがあります。無効化するとスカラー実装にフォールバックし、更新がやや遅くなります。

### `--ema_update_interval`

- **内容**: EMA シャドウパラメータの更新頻度を下げます。
- **理由**: 毎ステップの更新は不要な場合があります。例: `--ema_update_interval=100` は 100 ステップに 1 回だけ EMA 更新を行い、`--ema_device=cpu` や `--ema_cpu_only` のオーバーヘッドを減らします。

### `--ema_decay`

- **内容**: EMA 更新時の平滑化係数を制御します。
- **理由**: 高い値（例: `0.999`）は反応が遅い代わりに非常に安定した重みを生成します。低い値（例: `0.99`）は新しい学習信号への追随が速くなります。

### `--snr_gamma`

- **内容**: min-SNR 重み付き損失係数を使用します。
- **理由**: Minimum SNR gamma はタイムステップ位置に応じて損失の重みを調整します。ノイズが多いステップの寄与を減らし、ノイズが少ないステップの寄与を増やします。元論文の推奨値は **5** ですが、**1**〜**20** まで使えます。一般的に **20** が上限で、それ以上は効果が小さくなります。**1** が最も強い設定です。

### `--use_soft_min_snr`

- **内容**: 損失地形に対してより緩やかな重み付けで学習します。
- **理由**: ピクセル拡散モデルの学習では、特定の損失重み付けがないと劣化することがあります。DeepFloyd では soft-min-snr-gamma が良好な結果にほぼ必須でした。潜在拡散モデルでは成功する場合もありますが、少数の実験ではぼやけた結果になる可能性がありました。

### `--diff2flow_enabled`

- **内容**: epsilon または v-prediction モデル向けに Diffusion-to-Flow ブリッジを有効化します。
- **理由**: 標準拡散目的で学習したモデルが、モデル構造を変えずに flow-matching ターゲット（noise - latents）を使えるようにします。
- **注記**: 実験的機能です。

### `--diff2flow_loss`

- **内容**: ネイティブ予測損失ではなく Flow Matching 損失で学習します。
- **理由**: `--diff2flow_enabled` と併用すると、モデルのネイティブターゲット（epsilon または velocity）ではなく flow ターゲット（noise - latents）に対して損失を計算します。
- **注記**: `--diff2flow_enabled` が必要です。

### `--scheduled_sampling_max_step_offset`

- **内容**: 学習中に「ロールアウト」する最大ステップ数。
- **理由**: スケジュールドサンプリング（ロールアウト）を有効化し、学習中に数ステップだけモデル自身の入力を生成させます。自己修正を学ばせ、露出バイアスを軽減します。
- **既定**: 0（無効）。有効化するには正の整数（例: 5 または 10）を指定します。

### `--scheduled_sampling_strategy`

- **内容**: ロールアウトのオフセットを選ぶ戦略。
- **選択肢**: `uniform`, `biased_early`, `biased_late`。
- **既定**: `uniform`。
- **理由**: ロールアウト長の分布を制御します。`uniform` は均等、`biased_early` は短めを優先、`biased_late` は長めを優先します。

### `--scheduled_sampling_probability`

- **内容**: サンプルごとに非ゼロのロールアウトオフセットを適用する確率。
- **既定**: 0.0。
- **理由**: スケジュールドサンプリングの適用頻度を制御します。0.0 なら `max_step_offset` が 0 より大きくても無効。1.0 なら全サンプルに適用します。

### `--scheduled_sampling_prob_start`

- **内容**: ランプ開始時のスケジュールドサンプリング確率。
- **既定**: 0.0。

### `--scheduled_sampling_prob_end`

- **内容**: ランプ終了時のスケジュールドサンプリング確率。
- **既定**: 0.5。

### `--scheduled_sampling_ramp_steps`

- **内容**: `prob_start` から `prob_end` まで確率を増やすステップ数。
- **既定**: 0（ランプなし）。

### `--scheduled_sampling_start_step`

- **内容**: スケジュールドサンプリングのランプ開始グローバルステップ。
- **既定**: 0.0。

### `--scheduled_sampling_ramp_shape`

- **内容**: 確率ランプの形状。
- **選択肢**: `linear`, `cosine`。
- **既定**: `linear`。

### `--scheduled_sampling_sampler`

- **内容**: ロールアウト生成ステップに使用するソルバ。
- **選択肢**: `unipc`, `euler`, `dpm`, `rk4`。
- **既定**: `unipc`。

### `--scheduled_sampling_order`

- **内容**: ロールアウトに使用するソルバの次数。
- **既定**: 2。

### `--scheduled_sampling_reflexflow`

- **内容**: flow-matching モデルのスケジュールドサンプリング中に ReflexFlow 風の強化（アンチドリフト + 周波数補償重み付け）を有効化します。
- **理由**: 方向正則化とバイアス考慮の損失重み付けを加え、flow-matching モデルのロールアウトで露出バイアスを軽減します。
- **既定**: `--scheduled_sampling_max_step_offset` > 0 の場合、flow-matching モデルで自動有効化。`--scheduled_sampling_reflexflow=false` で上書きできます。

### `--scheduled_sampling_reflexflow_alpha`

- **内容**: 露出バイアスから導出される周波数補償重みのスケーリング係数。
- **既定**: 1.0。
- **理由**: 値を大きくすると、flow-matching モデルのロールアウトで露出バイアスが大きい領域の重み付けが強くなります。

### `--scheduled_sampling_reflexflow_beta1`

- **内容**: ReflexFlow のアンチドリフト（方向）正則化の重み。
- **既定**: 10.0。
- **理由**: flow-matching モデルでスケジュールドサンプリングを使う際、予測方向をターゲットのクリーンサンプルに揃える強さを制御します。

### `--scheduled_sampling_reflexflow_beta2`

- **内容**: ReflexFlow の周波数補償（損失再重み付け）項の重み。
- **既定**: 1.0。
- **理由**: 再重み付けされた flow-matching 損失をスケールし、ReflexFlow 論文の β₂ に対応します。

---

## 🎯 CREPA（Cross-frame Representation Alignment）

CREPA は動画拡散モデルのファインチューニング向け正則化手法で、隣接フレームの事前学習済み視覚特徴と隠れ状態を整合させることで時間的一貫性を向上させます。論文は ["Cross-Frame Representation Alignment for Fine-Tuning Video Diffusion Models"](https://arxiv.org/abs/2506.09229) を参照してください。

### `--crepa_enabled`

- **内容**: 学習中に CREPA 正則化を有効化します。
- **理由**: 隣接フレームの DINOv2 特徴と DiT 隠れ状態を整合させ、動画フレーム間の意味的一貫性を向上させます。
- **既定**: `false`
- **注記**: 動画モデル（Wan、LTXVideo、SanaVideo、Kandinsky5）のみ対象です。

### `--crepa_block_index`

- **内容**: 整合に使う Transformer ブロックの隠れ状態を指定します。
- **理由**: 論文では CogVideoX は block 8、Hunyuan Video は block 10 を推奨しています。DiT の「エンコーダ」部分に相当するため、早めのブロックが良い傾向があります。
- **必須**: CREPA 有効時は必須です。

### `--crepa_lambda`

- **内容**: CREPA 整合損失の重み（主損失に対する比率）。
- **理由**: 整合正則化が学習に与える強さを制御します。論文では CogVideoX に 0.5、Hunyuan Video に 1.0 を使用。
- **既定**: `0.5`

### `--crepa_adjacent_distance`

- **内容**: 近傍フレーム整合の距離 `d`。
- **理由**: 論文の式 6 にある $K = \{f-d, f+d\}$ に従い、どの隣接フレームと整合するかを定義します。`d=1` なら直近の隣接フレームと整合します。
- **既定**: `1`

### `--crepa_adjacent_tau`

- **内容**: 距離に対する指数重みの温度係数。
- **理由**: $e^{-|k-f|/\tau}$ による距離減衰の速さを制御します。小さい値ほど直近の隣接フレームに強く焦点が当たります。
- **既定**: `1.0`

### `--crepa_cumulative_neighbors`

- **内容**: 隣接モードではなく累積モードを使用します。
- **理由**:
  - **隣接モード（既定）**: 距離 `d` のフレームのみ整合（論文の $K = \{f-d, f+d\}$）。
  - **累積モード**: 距離 1 から `d` まで全フレームと整合し、勾配が滑らかになります。
- **既定**: `false`

### `--crepa_normalize_by_frames`

- **内容**: 整合損失をフレーム数で正規化します。
- **理由**: 動画長に依存せず損失スケールを一定に保ちます。無効化すると長い動画ほど強い整合信号になります。
- **既定**: `true`

### `--crepa_spatial_align`

- **内容**: DiT とエンコーダでトークン数が異なる場合に空間補間を使用します。
- **理由**: DiT 隠れ状態と DINOv2 特徴の空間解像度が異なる場合があります。有効化するとバイリニア補間で整合し、無効化するとグローバルプーリングにフォールバックします。
- **既定**: `true`

### `--crepa_model`

- **内容**: 特徴抽出に使う事前学習済みエンコーダ。
- **理由**: 論文では DINOv2-g（ViT-Giant）を使用。`dinov2_vitb14` など小型はメモリ消費が少なくなります。
- **既定**: `dinov2_vitg14`
- **選択肢**: `dinov2_vitg14`, `dinov2_vitb14`, `dinov2_vits14`

### `--crepa_encoder_frames_batch_size`

- **内容**: 外部特徴エンコーダが並列処理するフレーム数。0 以下でバッチ全フレームを同時処理します。割り切れない場合、残りは小さなバッチとして処理されます。
- **理由**: DINO 系エンコーダは画像モデルのため、分割バッチ処理で VRAM を抑えられます（速度は低下）。
- **既定**: `-1`

### `--crepa_use_backbone_features`

- **内容**: 外部エンコーダを使わず、拡散モデル内部の教師ブロックに学生ブロックを整合させます。
- **理由**: バックボーン内に強い意味層がある場合、DINOv2 のロードを避けられます。
- **既定**: `false`

### `--crepa_teacher_block_index`

- **内容**: バックボーン特徴使用時の教師ブロックインデックス。
- **理由**: 外部エンコーダなしで、早い学生ブロックを後段の教師ブロックに整合できます。未設定なら学生ブロックにフォールバックします。
- **既定**: 指定がなければ `crepa_block_index` を使用。

### `--crepa_encoder_image_size`

- **内容**: エンコーダの入力解像度。
- **理由**: DINOv2 は学習時の解像度で最も良く動作します。巨大モデルは 518x518 を使用します。
- **既定**: `518`

### `--crepa_scheduler`

- **内容**: 学習中の CREPA 係数減衰スケジュール。
- **理由**: 学習が進むにつれて CREPA 正則化の強度を下げることで、深層エンコーダ特徴への過学習を防ぎます。
- **選択肢**: `constant`、`linear`、`cosine`、`polynomial`
- **既定**: `constant`

### `--crepa_warmup_steps`

- **内容**: CREPA 重みを 0 から `crepa_lambda` まで線形に上昇させるステップ数。
- **理由**: 段階的なウォームアップにより、CREPA 正則化が有効になる前の初期学習を安定させます。
- **既定**: `0`

### `--crepa_decay_steps`

- **内容**: 減衰の総ステップ数（ウォームアップ後）。0 に設定すると学習全体で減衰します。
- **理由**: 減衰フェーズの期間を制御します。減衰はウォームアップ完了後に開始されます。
- **既定**: `0`（`max_train_steps` を使用）

### `--crepa_lambda_end`

- **内容**: 減衰完了後の最終 CREPA 重み。
- **理由**: 0 に設定すると学習終了時に CREPA を実質的に無効化できます。text2video で CREPA がアーティファクトを引き起こす場合に有用です。
- **既定**: `0.0`

### `--crepa_power`

- **内容**: 多項式減衰のべき乗係数。1.0 = 線形、2.0 = 二次など。
- **理由**: 値が大きいほど初期の減衰が速く、終盤に向けて緩やかになります。
- **既定**: `1.0`

### `--crepa_cutoff_step`

- **内容**: CREPA を無効化するハードカットオフステップ。
- **理由**: モデルが時間的整合に収束した後に CREPA を無効化するのに有用です。
- **既定**: `0`（ステップベースのカットオフなし）

### `--crepa_similarity_threshold`

- **内容**: CREPA カットオフをトリガーする類似度 EMA 閾値。
- **理由**: 類似度の指数移動平均がこの値に達すると、深層エンコーダ特徴への過学習を防ぐために CREPA が無効化されます。text2video 学習に特に有用です。
- **既定**: なし（無効）

### `--crepa_similarity_ema_decay`

- **内容**: 類似度追跡の指数移動平均減衰係数。
- **理由**: 値が大きいほど滑らかな追跡（0.99 ≈ 100 ステップウィンドウ）、値が小さいほど変化に素早く反応します。
- **既定**: `0.99`

### `--crepa_threshold_mode`

- **内容**: 類似度閾値に達した際の動作。
- **選択肢**: `permanent`（閾値に達すると CREPA はオフのまま）、`recoverable`（類似度が下がると CREPA が再有効化）
- **既定**: `permanent`

### 設定例

```toml
# 動画ファインチューニング用 CREPA を有効化
crepa_enabled = true
crepa_block_index = 8          # モデルに応じて調整
crepa_lambda = 0.5
crepa_adjacent_distance = 1
crepa_adjacent_tau = 1.0
crepa_cumulative_neighbors = false
crepa_normalize_by_frames = true
crepa_spatial_align = true
crepa_model = "dinov2_vitg14"
crepa_encoder_frames_batch_size = -1
crepa_use_backbone_features = false
# crepa_teacher_block_index = 16
crepa_encoder_image_size = 518

# CREPA スケジューリング（オプション）
# crepa_scheduler = "cosine"           # 減衰タイプ: constant, linear, cosine, polynomial
# crepa_warmup_steps = 100             # CREPA 有効化前のウォームアップ
# crepa_decay_steps = 1000             # 減衰ステップ数（0 = 学習全体）
# crepa_lambda_end = 0.0               # 減衰後の最終重み
# crepa_cutoff_step = 5000             # ハードカットオフステップ（0 = 無効）
# crepa_similarity_threshold = 0.9    # 類似度ベースのカットオフ
# crepa_threshold_mode = "permanent"   # permanent または recoverable
```

---

## 🔄 チェックポイントと再開

### `--checkpoint_step_interval`（別名: `--checkpointing_steps`）

- **内容**: 学習状態のチェックポイント保存間隔（ステップ数）。
- **理由**: 学習再開や推論に有用です。*n* イテレーションごとに Diffusers のファイルレイアウトで `.safetensors` の部分チェックポイントが保存されます。

---

## 🔁 LayerSync（隠れ状態の自己整合）

LayerSync は同一 Transformer 内の「学生」レイヤーを、より強い「教師」レイヤーに合わせることで、隠れトークンのコサイン類似度を用いて整合させます。

### `--layersync_enabled`

- **内容**: 同一モデル内の 2 つの Transformer ブロック間で LayerSync の隠れ状態整合を有効化します。
- **注意**: 隠れ状態バッファを確保します。必要なフラグが不足している場合、起動時にエラーになります。
- **既定**: `false`

### `--layersync_student_block`

- **内容**: 学生アンカーとして扱う Transformer ブロックのインデックス。
- **インデックス**: LayerSync 論文の 1 始まり深度、または 0 始まりレイヤー ID を受け付けます。実装は `idx-1` を先に試し、次に `idx` を試します。
- **必須**: LayerSync 有効時は必須です。

### `--layersync_teacher_block`

- **内容**: 教師ターゲットとして扱う Transformer ブロックのインデックス（学生より深くても可）。
- **インデックス**: 学生ブロックと同じく、1 始まり優先で次に 0 始まりにフォールバックします。
- **既定**: 未指定時は学生ブロックを使用し、損失は自己類似になります。

### `--layersync_lambda`

- **内容**: 学生と教師の隠れ状態間の LayerSync コサイン整合損失（負のコサイン類似度）の重み。
- **効果**: 基本損失に加える補助正則化をスケールします。値が大きいほど学生トークンは教師トークンにより強く整合します。
- **上流名**: 元の LayerSync コードベースでは `--reg-weight`。
- **必須**: LayerSync 有効時は 0 より大きくする必要があります（そうでないと学習が中断されます）。
- **既定**: LayerSync 有効時は `0.2`（参照リポジトリに合わせる）、それ以外は `0.0`。

上流オプション対応（LayerSync → SimpleTuner）:
- `--encoder-depth` → `--layersync_student_block`（上流は 1 始まり深度を受け付けるが、0 始まりレイヤーも可）
- `--gt-encoder-depth` → `--layersync_teacher_block`（1 始まり推奨。未指定時は学生を使用）
- `--reg-weight` → `--layersync_lambda`

> 注記: LayerSync は参照実装と同様、類似度計算前に教師の隠れ状態を常にデタッチします。Transformer の隠れ状態を公開するモデル（SimpleTuner の多くの Transformer バックボーン）に依存し、隠れ状態バッファ分のメモリを各ステップで追加します。VRAM が厳しい場合は無効化してください。

### `--checkpoint_epoch_interval`

- **内容**: 完了したエポック N ごとにチェックポイントを保存します。
- **理由**: マルチデータセットサンプリングでステップ数が変動しても、エポック境界で状態を確実に保存できるよう、ステップベースのチェックポイントを補完します。

### `--resume_from_checkpoint`

- **内容**: 学習再開の有無と再開元を指定します。
- **理由**: 手動指定または最新のチェックポイントから再開できます。チェックポイントは `unet` と任意の `unet_ema` サブフォルダで構成されます。`unet` は Diffusers レイアウトの SDXL モデルにそのまま配置でき、通常のモデルとして利用可能です。

> ℹ️ PixArt、SD3、Hunyuan などの Transformer モデルは `transformer` と `transformer_ema` のサブフォルダ名を使用します。

---

## 📊 ログとモニタリング

### `--logging_dir`

- **内容**: TensorBoard ログの保存ディレクトリ。
- **理由**: 学習の進捗と性能メトリクスを監視できます。

### `--report_to`

- **内容**: 結果とログの報告先プラットフォーム。
- **理由**: TensorBoard、wandb、comet_ml などと連携して監視できます。複数指定はカンマ区切りです。
- **選択肢**: wandb, tensorboard, comet_ml

## 環境設定変数

上記オプションは主に `config.json` に適用されますが、一部の項目は `config.env` 内で設定する必要があります。

- `TRAINING_NUM_PROCESSES` はシステム内の GPU 数に設定します。ほとんどの用途で DistributedDataParallel（DDP）を有効化するには十分です。`config.env` を使いたくない場合は、`config.json` の `num_processes` を使用してください。
- `TRAINING_DYNAMO_BACKEND` は既定で `no` ですが、対応する torch.compile バックエンド（例: `inductor`, `aot_eager`, `cudagraphs`）に設定でき、`--dynamo_mode`、`--dynamo_fullgraph`、`--dynamo_use_regional_compilation` と組み合わせて微調整できます。
- `SIMPLETUNER_LOG_LEVEL` は既定で `INFO` ですが、`DEBUG` にすると `debug.log` に問題報告向けの詳細情報を追加できます。
- `VENV_PATH` は Python 仮想環境の場所を指定できます（標準の `.venv` 以外の場合）。
- `ACCELERATE_EXTRA_ARGS` は未設定でも構いませんが、`--multi_gpu` や FSDP 特有のフラグなど追加引数を含められます。

---

これは基本的な概要で、最初の手がかりとして役立つことを目的としています。完全なオプション一覧と詳細な説明については、次の完全な仕様を参照してください:

```
usage: train.py [-h] --model_family
                {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo}
                [--model_flavour MODEL_FLAVOUR] [--controlnet [CONTROLNET]]
                [--pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH]
                --output_dir OUTPUT_DIR [--logging_dir LOGGING_DIR]
                --model_type {full,lora} [--seed SEED]
                [--resolution RESOLUTION]
                [--resume_from_checkpoint RESUME_FROM_CHECKPOINT]
                [--prediction_type {epsilon,v_prediction,sample,flow_matching}]
                [--pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH]
                [--vae_dtype {default,fp32,fp16,bf16}]
                [--vae_cache_ondemand [VAE_CACHE_ONDEMAND]]
                [--accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL]
                [--aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}]
                [--base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}]
                [--gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL]
                [--offload_during_startup [OFFLOAD_DURING_STARTUP]]
                [--quantize_via {cpu,accelerator,pipeline}]
                [--quantization_config QUANTIZATION_CONFIG]
                [--fuse_qkv_projections [FUSE_QKV_PROJECTIONS]]
                [--control [CONTROL]]
                [--controlnet_custom_config CONTROLNET_CUSTOM_CONFIG]
                [--controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH]
                [--tread_config TREAD_CONFIG]
                [--pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH]
                [--pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER]
                [--pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH]
                [--pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER]
                [--pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH]
                [--pretrained_gemma_model_name_or_path PRETRAINED_GEMMA_MODEL_NAME_OR_PATH]
                [--revision REVISION] [--variant VARIANT]
                [--base_model_default_dtype {bf16,fp32}]
                [--unet_attention_slice [UNET_ATTENTION_SLICE]]
                [--num_train_epochs NUM_TRAIN_EPOCHS]
                [--max_train_steps MAX_TRAIN_STEPS]
                [--train_batch_size TRAIN_BATCH_SIZE]
                [--learning_rate LEARNING_RATE] --optimizer
                {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,prodigy}
                [--optimizer_config OPTIMIZER_CONFIG]
                [--lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}]
                [--gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS]
                [--lr_warmup_steps LR_WARMUP_STEPS]
                [--checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT]
                [--gradient_checkpointing [GRADIENT_CHECKPOINTING]]
                [--train_text_encoder [TRAIN_TEXT_ENCODER]]
                [--text_encoder_lr TEXT_ENCODER_LR]
                [--lr_num_cycles LR_NUM_CYCLES] [--lr_power LR_POWER]
                [--use_soft_min_snr [USE_SOFT_MIN_SNR]] [--use_ema [USE_EMA]]
                [--ema_device {accelerator,cpu}]
                [--ema_cpu_only [EMA_CPU_ONLY]]
                [--ema_update_interval EMA_UPDATE_INTERVAL]
                [--ema_foreach_disable [EMA_FOREACH_DISABLE]]
                [--ema_decay EMA_DECAY] [--lora_rank LORA_RANK]
                [--lora_alpha LORA_ALPHA] [--lora_type {standard,lycoris}]
                [--lora_dropout LORA_DROPOUT]
                [--lora_init_type {default,gaussian,loftq,olora,pissa}]
                [--peft_lora_mode {standard,singlora}]
                [--peft_lora_target_modules PEFT_LORA_TARGET_MODULES]
                [--singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS]
                [--init_lora INIT_LORA] [--lycoris_config LYCORIS_CONFIG]
                [--init_lokr_norm INIT_LOKR_NORM]
                [--flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,controlnet,all+ffs+embedder,all+ffs+embedder+controlnet}]
                [--use_dora [USE_DORA]]
                [--resolution_type {pixel,area,pixel_area}]
                --data_backend_config DATA_BACKEND_CONFIG
                [--caption_strategy {filename,textfile,instance_prompt,parquet}]
                [--conditioning_multidataset_sampling {combined,random}]
                [--instance_prompt INSTANCE_PROMPT]
                [--parquet_caption_column PARQUET_CAPTION_COLUMN]
                [--parquet_filename_column PARQUET_FILENAME_COLUMN]
                [--ignore_missing_files [IGNORE_MISSING_FILES]]
                [--vae_cache_scan_behaviour {recreate,sync}]
                [--vae_enable_slicing [VAE_ENABLE_SLICING]]
                [--vae_enable_tiling [VAE_ENABLE_TILING]]
                [--vae_enable_patch_conv [VAE_ENABLE_PATCH_CONV]]
                [--vae_batch_size VAE_BATCH_SIZE]
                [--caption_dropout_probability CAPTION_DROPOUT_PROBABILITY]
                [--tokenizer_max_length TOKENIZER_MAX_LENGTH]
                [--validation_step_interval VALIDATION_STEP_INTERVAL]
                [--validation_epoch_interval VALIDATION_EPOCH_INTERVAL]
                [--disable_benchmark [DISABLE_BENCHMARK]]
                [--validation_prompt VALIDATION_PROMPT]
                [--num_validation_images NUM_VALIDATION_IMAGES]
                [--num_eval_images NUM_EVAL_IMAGES]
                [--eval_steps_interval EVAL_STEPS_INTERVAL]
                [--eval_epoch_interval EVAL_EPOCH_INTERVAL]
                [--eval_timesteps EVAL_TIMESTEPS]
                [--eval_dataset_pooling [EVAL_DATASET_POOLING]]
                [--evaluation_type {none,clip}]
                [--pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH]
                [--validation_guidance VALIDATION_GUIDANCE]
                [--validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS]
                [--validation_on_startup [VALIDATION_ON_STARTUP]]
                [--validation_using_datasets [VALIDATION_USING_DATASETS]]
                [--validation_torch_compile [VALIDATION_TORCH_COMPILE]]
                [--validation_guidance_real VALIDATION_GUIDANCE_REAL]
                [--validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP]
                [--validation_negative_prompt VALIDATION_NEGATIVE_PROMPT]
                [--validation_randomize [VALIDATION_RANDOMIZE]]
                [--validation_seed VALIDATION_SEED]
                [--validation_disable [VALIDATION_DISABLE]]
                [--validation_prompt_library [VALIDATION_PROMPT_LIBRARY]]
                [--user_prompt_library USER_PROMPT_LIBRARY]
                [--eval_dataset_id EVAL_DATASET_ID]
                [--validation_stitch_input_location {left,right}]
                [--validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE]
                [--validation_disable_unconditional [VALIDATION_DISABLE_UNCONDITIONAL]]
                [--validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS]
                [--validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START]
                [--validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP]
                [--validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE]
                [--validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH]
                [--validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++,perflow}]
                [--validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES]
                [--validation_resolution VALIDATION_RESOLUTION]
                [--validation_seed_source {cpu,gpu}]
                [--i_know_what_i_am_doing [I_KNOW_WHAT_I_AM_DOING]]
                [--flow_sigmoid_scale FLOW_SIGMOID_SCALE]
                [--flux_fast_schedule [FLUX_FAST_SCHEDULE]]
                [--flow_use_uniform_schedule [FLOW_USE_UNIFORM_SCHEDULE]]
                [--flow_use_beta_schedule [FLOW_USE_BETA_SCHEDULE]]
                [--flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA]
                [--flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA]
                [--flow_schedule_shift FLOW_SCHEDULE_SHIFT]
                [--flow_schedule_auto_shift [FLOW_SCHEDULE_AUTO_SHIFT]]
                [--flux_guidance_mode {constant,random-range}]
                [--flux_attention_masked_training [FLUX_ATTENTION_MASKED_TRAINING]]
                [--flux_guidance_value FLUX_GUIDANCE_VALUE]
                [--flux_guidance_min FLUX_GUIDANCE_MIN]
                [--flux_guidance_max FLUX_GUIDANCE_MAX]
                [--t5_padding {zero,unmodified}]
                [--sd3_clip_uncond_behaviour {empty_string,zero}]
                [--sd3_t5_uncond_behaviour {empty_string,zero}]
                [--soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA]
                [--mixed_precision {no,fp16,bf16,fp8}]
                [--attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}]
                [--sageattention_usage {training,inference,training+inference}]
                [--disable_tf32 [DISABLE_TF32]]
                [--set_grads_to_none [SET_GRADS_TO_NONE]]
                [--noise_offset NOISE_OFFSET]
                [--noise_offset_probability NOISE_OFFSET_PROBABILITY]
                [--input_perturbation INPUT_PERTURBATION]
                [--input_perturbation_steps INPUT_PERTURBATION_STEPS]
                [--lr_end LR_END] [--lr_scale [LR_SCALE]]
                [--lr_scale_sqrt [LR_SCALE_SQRT]]
                [--ignore_final_epochs [IGNORE_FINAL_EPOCHS]]
                [--freeze_encoder_before FREEZE_ENCODER_BEFORE]
                [--freeze_encoder_after FREEZE_ENCODER_AFTER]
                [--freeze_encoder_strategy {before,between,after}]
                [--layer_freeze_strategy {none,bitfit}]
                [--fully_unload_text_encoder [FULLY_UNLOAD_TEXT_ENCODER]]
                [--save_text_encoder [SAVE_TEXT_ENCODER]]
                [--text_encoder_limit TEXT_ENCODER_LIMIT]
                [--prepend_instance_prompt [PREPEND_INSTANCE_PROMPT]]
                [--only_instance_prompt [ONLY_INSTANCE_PROMPT]]
                [--data_aesthetic_score DATA_AESTHETIC_SCORE]
                [--delete_unwanted_images [DELETE_UNWANTED_IMAGES]]
                [--delete_problematic_images [DELETE_PROBLEMATIC_IMAGES]]
                [--disable_bucket_pruning [DISABLE_BUCKET_PRUNING]]
                [--disable_segmented_timestep_sampling [DISABLE_SEGMENTED_TIMESTEP_SAMPLING]]
                [--preserve_data_backend_cache [PRESERVE_DATA_BACKEND_CACHE]]
                [--override_dataset_config [OVERRIDE_DATASET_CONFIG]]
                [--cache_dir CACHE_DIR] [--cache_dir_text CACHE_DIR_TEXT]
                [--cache_dir_vae CACHE_DIR_VAE]
                [--compress_disk_cache [COMPRESS_DISK_CACHE]]
                [--aspect_bucket_disable_rebuild [ASPECT_BUCKET_DISABLE_REBUILD]]
                [--keep_vae_loaded [KEEP_VAE_LOADED]]
                [--skip_file_discovery SKIP_FILE_DISCOVERY]
                [--data_backend_sampling {uniform,auto-weighting}]
                [--image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE]
                [--write_batch_size WRITE_BATCH_SIZE]
                [--read_batch_size READ_BATCH_SIZE]
                [--enable_multiprocessing [ENABLE_MULTIPROCESSING]]
                [--max_workers MAX_WORKERS]
                [--aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS]
                [--torch_num_threads TORCH_NUM_THREADS]
                [--dataloader_prefetch [DATALOADER_PREFETCH]]
                [--dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN]
                [--aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT]
                [--aspect_bucket_alignment {8,16,24,32,64}]
                [--minimum_image_size MINIMUM_IMAGE_SIZE]
                [--maximum_image_size MAXIMUM_IMAGE_SIZE]
                [--target_downsample_size TARGET_DOWNSAMPLE_SIZE]
                [--max_upscale_threshold MAX_UPSCALE_THRESHOLD]
                [--metadata_update_interval METADATA_UPDATE_INTERVAL]
                [--debug_aspect_buckets [DEBUG_ASPECT_BUCKETS]]
                [--debug_dataset_loader [DEBUG_DATASET_LOADER]]
                [--print_filenames [PRINT_FILENAMES]]
                [--print_sampler_statistics [PRINT_SAMPLER_STATISTICS]]
                [--timestep_bias_strategy {earlier,later,range,none}]
                [--timestep_bias_begin TIMESTEP_BIAS_BEGIN]
                [--timestep_bias_end TIMESTEP_BIAS_END]
                [--timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER]
                [--timestep_bias_portion TIMESTEP_BIAS_PORTION]
                [--training_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--inference_scheduler_timestep_spacing {leading,linspace,trailing}]
                [--loss_type {l2,huber,smooth_l1}]
                [--huber_schedule {snr,exponential,constant}]
                [--huber_c HUBER_C] [--snr_gamma SNR_GAMMA]
                [--masked_loss_probability MASKED_LOSS_PROBABILITY]
                [--hidream_use_load_balancing_loss [HIDREAM_USE_LOAD_BALANCING_LOSS]]
                [--hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT]
                [--adam_beta1 ADAM_BETA1] [--adam_beta2 ADAM_BETA2]
                [--optimizer_beta1 OPTIMIZER_BETA1]
                [--optimizer_beta2 OPTIMIZER_BETA2]
                [--optimizer_cpu_offload_method {none}]
                [--gradient_precision {unmodified,fp32}]
                [--adam_weight_decay ADAM_WEIGHT_DECAY]
                [--adam_epsilon ADAM_EPSILON] [--prodigy_steps PRODIGY_STEPS]
                [--max_grad_norm MAX_GRAD_NORM]
                [--grad_clip_method {value,norm}]
                [--optimizer_offload_gradients [OPTIMIZER_OFFLOAD_GRADIENTS]]
                [--fuse_optimizer [FUSE_OPTIMIZER]]
                [--optimizer_release_gradients [OPTIMIZER_RELEASE_GRADIENTS]]
                [--push_to_hub [PUSH_TO_HUB]]
                [--push_to_hub_background [PUSH_TO_HUB_BACKGROUND]]
                [--push_checkpoints_to_hub [PUSH_CHECKPOINTS_TO_HUB]]
                [--publishing_config PUBLISHING_CONFIG]
                [--hub_model_id HUB_MODEL_ID]
                [--model_card_private [MODEL_CARD_PRIVATE]]
                [--model_card_safe_for_work [MODEL_CARD_SAFE_FOR_WORK]]
                [--model_card_note MODEL_CARD_NOTE]
                [--modelspec_comment MODELSPEC_COMMENT]
                [--report_to {tensorboard,wandb,comet_ml,all,none}]
                [--checkpoint_step_interval CHECKPOINT_STEP_INTERVAL]
                [--checkpoint_epoch_interval CHECKPOINT_EPOCH_INTERVAL]
                [--checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS]
                [--checkpointing_use_tempdir [CHECKPOINTING_USE_TEMPDIR]]
                [--checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT]
                [--tracker_run_name TRACKER_RUN_NAME]
                [--tracker_project_name TRACKER_PROJECT_NAME]
                [--tracker_image_layout {gallery,table}]
                [--enable_watermark [ENABLE_WATERMARK]]
                [--framerate FRAMERATE]
                [--seed_for_each_device [SEED_FOR_EACH_DEVICE]]
                [--snr_weight SNR_WEIGHT]
                [--rescale_betas_zero_snr [RESCALE_BETAS_ZERO_SNR]]
                [--webhook_config WEBHOOK_CONFIG]
                [--webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL]
                [--distillation_method {lcm,dcm,dmd,perflow}]
                [--distillation_config DISTILLATION_CONFIG]
                [--ema_validation {none,ema_only,comparison}]
                [--local_rank LOCAL_RANK] [--ltx_train_mode {t2v,i2v}]
                [--ltx_i2v_prob LTX_I2V_PROB]
                [--ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION]
                [--ltx_protect_first_frame [LTX_PROTECT_FIRST_FRAME]]
                [--offload_param_path OFFLOAD_PARAM_PATH]
                [--offset_noise [OFFSET_NOISE]]
                [--quantize_activations [QUANTIZE_ACTIVATIONS]]
                [--refiner_training [REFINER_TRAINING]]
                [--refiner_training_invert_schedule [REFINER_TRAINING_INVERT_SCHEDULE]]
                [--refiner_training_strength REFINER_TRAINING_STRENGTH]
                [--sdxl_refiner_uses_full_range [SDXL_REFINER_USES_FULL_RANGE]]
                [--sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION]

The following SimpleTuner command-line options are available:

options:
  -h, --help            show this help message and exit
  --model_family {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo}
                        The base model architecture family to train
  --model_flavour MODEL_FLAVOUR
                        Specific variant of the selected model family
  --controlnet [CONTROLNET]
                        Train ControlNet (full or LoRA) branches alongside the
                        primary network.
  --pretrained_model_name_or_path PRETRAINED_MODEL_NAME_OR_PATH
                        Optional override of the model checkpoint. Leave blank
                        to use the default path for the selected model
                        flavour.
  --output_dir OUTPUT_DIR
                        Directory where model checkpoints and logs will be
                        saved
  --logging_dir LOGGING_DIR
                        Directory for TensorBoard logs
  --model_type {full,lora}
                        Choose between full model training or LoRA adapter
                        training
  --seed SEED           Seed used for deterministic training behaviour
  --resolution RESOLUTION
                        Resolution for training images
  --resume_from_checkpoint RESUME_FROM_CHECKPOINT
                        Select checkpoint to resume training from
  --prediction_type {epsilon,v_prediction,sample,flow_matching}
                        The parameterization type for the diffusion model
  --pretrained_vae_model_name_or_path PRETRAINED_VAE_MODEL_NAME_OR_PATH
                        Path to pretrained VAE model
  --vae_dtype {default,fp32,fp16,bf16}
                        Precision for VAE encoding/decoding. Lower precision
                        saves memory.
  --vae_cache_ondemand [VAE_CACHE_ONDEMAND]
                        Process VAE latents during training instead of
                        precomputing them
  --vae_cache_disable [VAE_CACHE_DISABLE]
                        Implicitly enables on-demand caching and disables
                        writing embeddings to disk.
  --accelerator_cache_clear_interval ACCELERATOR_CACHE_CLEAR_INTERVAL
                        Clear the cache from VRAM every X steps to prevent
                        memory leaks
  --aspect_bucket_rounding {1,2,3,4,5,6,7,8,9}
                        Number of decimal places to round aspect ratios to for
                        bucket creation
  --base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for loading the base model. Lower precision
                        saves memory.
  --text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-torchao}
                        Precision for text encoders. Lower precision saves
                        memory.
  --gradient_checkpointing_interval GRADIENT_CHECKPOINTING_INTERVAL
                        Checkpoint every N transformer blocks
  --offload_during_startup [OFFLOAD_DURING_STARTUP]
                        Offload text encoders to CPU during VAE caching
  --quantize_via {cpu,accelerator,pipeline}
                        Where to perform model quantization
  --quantization_config QUANTIZATION_CONFIG
                        JSON or file path describing Diffusers quantization
                        config for pipeline quantization
  --fuse_qkv_projections [FUSE_QKV_PROJECTIONS]
                        Enables Flash Attention 3 when supported; otherwise
                        falls back to PyTorch SDPA.
  --control [CONTROL]   Enable channel-wise control style training
  --controlnet_custom_config CONTROLNET_CUSTOM_CONFIG
                        Custom configuration for ControlNet models
  --controlnet_model_name_or_path CONTROLNET_MODEL_NAME_OR_PATH
                        Path to ControlNet model weights to preload
  --tread_config TREAD_CONFIG
                        Configuration for TREAD training method
  --pretrained_transformer_model_name_or_path PRETRAINED_TRANSFORMER_MODEL_NAME_OR_PATH
                        Path to pretrained transformer model
  --pretrained_transformer_subfolder PRETRAINED_TRANSFORMER_SUBFOLDER
                        Subfolder containing transformer model weights
  --pretrained_unet_model_name_or_path PRETRAINED_UNET_MODEL_NAME_OR_PATH
                        Path to pretrained UNet model
  --pretrained_unet_subfolder PRETRAINED_UNET_SUBFOLDER
                        Subfolder containing UNet model weights
  --pretrained_t5_model_name_or_path PRETRAINED_T5_MODEL_NAME_OR_PATH
                        Path to pretrained T5 model
  --pretrained_gemma_model_name_or_path PRETRAINED_GEMMA_MODEL_NAME_OR_PATH
                        Path to pretrained Gemma model
  --revision REVISION   Git branch/tag/commit for model version
  --variant VARIANT     Model variant (e.g., fp16, bf16)
  --base_model_default_dtype {bf16,fp32}
                        Default precision for quantized base model weights
  --unet_attention_slice [UNET_ATTENTION_SLICE]
                        Enable attention slicing for SDXL UNet
  --num_train_epochs NUM_TRAIN_EPOCHS
                        Number of times to iterate through the entire dataset
  --max_train_steps MAX_TRAIN_STEPS
                        Maximum number of training steps (0 = use epochs
                        instead)
  --train_batch_size TRAIN_BATCH_SIZE
                        Number of samples processed per forward/backward pass
                        (per device).
  --learning_rate LEARNING_RATE
                        Base learning rate for training
  --optimizer {adamw_bf16,ao-adamw8bit,ao-adamw4bit,ao-adamfp8,ao-adamwfp8,adamw_schedulefree,adamw_schedulefree+aggressive,adamw_schedulefree+no_kahan,optimi-stableadamw,optimi-adamw,optimi-lion,optimi-radam,optimi-ranger,optimi-adan,optimi-adam,optimi-sgd,soap,prodigy}
                        Optimization algorithm for training
  --optimizer_config OPTIMIZER_CONFIG
                        Comma-separated key=value pairs forwarded to the
                        selected optimizer
  --lr_scheduler {linear,sine,cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup}
                        How learning rate changes during training
  --gradient_accumulation_steps GRADIENT_ACCUMULATION_STEPS
                        Number of steps to accumulate gradients
  --lr_warmup_steps LR_WARMUP_STEPS
                        Number of steps to gradually increase LR from 0
  --checkpoints_total_limit CHECKPOINTS_TOTAL_LIMIT
                        Maximum number of checkpoints to keep on disk
  --gradient_checkpointing [GRADIENT_CHECKPOINTING]
                        Trade compute for memory during training
  --train_text_encoder [TRAIN_TEXT_ENCODER]
                        Also train the text encoder (CLIP) model
  --text_encoder_lr TEXT_ENCODER_LR
                        Separate learning rate for text encoder
  --lr_num_cycles LR_NUM_CYCLES
                        Number of cosine annealing cycles
  --lr_power LR_POWER   Power for polynomial decay scheduler
  --use_soft_min_snr [USE_SOFT_MIN_SNR]
                        Use soft clamping instead of hard clamping for Min-SNR
  --use_ema [USE_EMA]   Maintain an exponential moving average copy of the
                        model during training.
  --ema_device {accelerator,cpu}
                        Where to keep the EMA weights in-between updates.
  --ema_cpu_only [EMA_CPU_ONLY]
                        Keep EMA weights exclusively on CPU even when
                        ema_device would normally move them.
  --ema_update_interval EMA_UPDATE_INTERVAL
                        Update EMA weights every N optimizer steps
  --ema_foreach_disable [EMA_FOREACH_DISABLE]
                        Fallback to standard tensor ops instead of
                        torch.foreach updates.
  --ema_decay EMA_DECAY
                        Smoothing factor for EMA updates (closer to 1.0 =
                        slower drift).
  --lora_rank LORA_RANK
                        Dimension of LoRA update matrices
  --lora_alpha LORA_ALPHA
                        Scaling factor for LoRA updates
  --lora_type {standard,lycoris}
                        LoRA implementation type
  --lora_dropout LORA_DROPOUT
                        LoRA dropout randomly ignores neurons during training.
                        This can help prevent overfitting.
  --lora_init_type {default,gaussian,loftq,olora,pissa}
                        The initialization type for the LoRA model
  --peft_lora_mode {standard,singlora}
                        PEFT LoRA training mode
  --peft_lora_target_modules PEFT_LORA_TARGET_MODULES
                        JSON array (or path to a JSON file) listing PEFT
                        LoRA target module names. Overrides preset targets.
  --singlora_ramp_up_steps SINGLORA_RAMP_UP_STEPS
                        Number of ramp-up steps for SingLoRA
  --slider_lora_target [SLIDER_LORA_TARGET]
                        Route LoRA training to slider-friendly targets
                        (self-attn + conv/time embeddings). Only affects
                        standard PEFT LoRA.
  --init_lora INIT_LORA
                        Specify an existing LoRA or LyCORIS safetensors file
                        to initialize the adapter
  --lycoris_config LYCORIS_CONFIG
                        Path to LyCORIS configuration JSON file
  --init_lokr_norm INIT_LOKR_NORM
                        Perturbed normal initialization for LyCORIS LoKr
                        layers
  --flux_lora_target {mmdit,context,context+ffs,all,all+ffs,ai-toolkit,tiny,nano,controlnet,all+ffs+embedder,all+ffs+embedder+controlnet}
                        Which layers to train in Flux models
  --use_dora [USE_DORA]
                        Enable DoRA (Weight-Decomposed LoRA)
  --resolution_type {pixel,area,pixel_area}
                        How to interpret the resolution value
  --data_backend_config DATA_BACKEND_CONFIG
                        Select a saved dataset configuration (managed in
                        Datasets & Environments tabs)
  --caption_strategy {filename,textfile,instance_prompt,parquet}
                        How to load captions for images
  --conditioning_multidataset_sampling {combined,random}
                        How to sample from multiple conditioning datasets
  --instance_prompt INSTANCE_PROMPT
                        Instance prompt for training
  --parquet_caption_column PARQUET_CAPTION_COLUMN
                        Column name containing captions in parquet files
  --parquet_filename_column PARQUET_FILENAME_COLUMN
                        Column name containing image paths in parquet files
  --ignore_missing_files [IGNORE_MISSING_FILES]
                        Continue training even if some files are missing
  --vae_cache_scan_behaviour {recreate,sync}
                        How to scan VAE cache for missing files
  --vae_enable_slicing [VAE_ENABLE_SLICING]
                        Enable VAE attention slicing for memory efficiency
  --vae_enable_tiling [VAE_ENABLE_TILING]
                        Enable VAE tiling for large images
  --vae_enable_patch_conv [VAE_ENABLE_PATCH_CONV]
                        Enable patch-based 3D conv for HunyuanVideo VAE to
                        reduce peak VRAM (slight slowdown)
  --vae_batch_size VAE_BATCH_SIZE
                        Batch size for VAE encoding during caching
  --caption_dropout_probability CAPTION_DROPOUT_PROBABILITY
                        Caption dropout will randomly drop captions and, for
                        SDXL, size conditioning inputs based on this
                        probability
  --tokenizer_max_length TOKENIZER_MAX_LENGTH
                        Override the tokenizer sequence length (advanced).
  --validation_step_interval VALIDATION_STEP_INTERVAL
                        Run validation every N training steps (deprecated alias: --validation_steps)
  --validation_epoch_interval VALIDATION_EPOCH_INTERVAL
                        Run validation every N training epochs
  --disable_benchmark [DISABLE_BENCHMARK]
                        Skip generating baseline comparison images before
                        training starts
  --validation_prompt VALIDATION_PROMPT
                        Prompt to use for validation images
  --num_validation_images NUM_VALIDATION_IMAGES
                        Number of images to generate per validation
  --num_eval_images NUM_EVAL_IMAGES
                        Number of images to generate for evaluation metrics
  --eval_steps_interval EVAL_STEPS_INTERVAL
                        Run evaluation every N training steps
  --eval_epoch_interval EVAL_EPOCH_INTERVAL
                        Run evaluation every N training epochs (decimals run
                        multiple times per epoch)
  --eval_timesteps EVAL_TIMESTEPS
                        Number of timesteps for evaluation
  --eval_dataset_pooling [EVAL_DATASET_POOLING]
                        Combine evaluation metrics from all datasets into a
                        single chart
  --evaluation_type {none,clip}
                        Type of evaluation metrics to compute
  --pretrained_evaluation_model_name_or_path PRETRAINED_EVALUATION_MODEL_NAME_OR_PATH
                        Path to pretrained model for evaluation metrics
  --validation_guidance VALIDATION_GUIDANCE
                        CFG guidance scale for validation images
  --validation_num_inference_steps VALIDATION_NUM_INFERENCE_STEPS
                        Number of diffusion steps for validation renders
  --validation_on_startup [VALIDATION_ON_STARTUP]
                        Run validation on the base model before training
                        starts
  --validation_using_datasets [VALIDATION_USING_DATASETS]
                        Use random images from training datasets for
                        validation
  --validation_torch_compile [VALIDATION_TORCH_COMPILE]
                        Use torch.compile() on validation pipeline for speed
  --validation_guidance_real VALIDATION_GUIDANCE_REAL
                        CFG value for distilled models (e.g., FLUX schnell)
  --validation_no_cfg_until_timestep VALIDATION_NO_CFG_UNTIL_TIMESTEP
                        Skip CFG for initial timesteps (Flux only)
  --validation_negative_prompt VALIDATION_NEGATIVE_PROMPT
                        Negative prompt for validation images
  --validation_randomize [VALIDATION_RANDOMIZE]
                        Use random seeds for each validation
  --validation_seed VALIDATION_SEED
                        Fixed seed for reproducible validation images
  --validation_disable [VALIDATION_DISABLE]
                        Completely disable validation image generation
  --validation_prompt_library [VALIDATION_PROMPT_LIBRARY]
                        Use SimpleTuner's built-in prompt library
  --user_prompt_library USER_PROMPT_LIBRARY
                        Path to custom JSON prompt library
  --eval_dataset_id EVAL_DATASET_ID
                        Specific dataset to use for evaluation metrics
  --validation_stitch_input_location {left,right}
                        Where to place input image in img2img validations
  --validation_guidance_rescale VALIDATION_GUIDANCE_RESCALE
                        CFG rescale value for validation
  --validation_disable_unconditional [VALIDATION_DISABLE_UNCONDITIONAL]
                        Disable unconditional image generation during
                        validation
  --validation_guidance_skip_layers VALIDATION_GUIDANCE_SKIP_LAYERS
                        JSON list of transformer layers to skip during
                        classifier-free guidance
  --validation_guidance_skip_layers_start VALIDATION_GUIDANCE_SKIP_LAYERS_START
                        Starting layer index to skip guidance
  --validation_guidance_skip_layers_stop VALIDATION_GUIDANCE_SKIP_LAYERS_STOP
                        Ending layer index to skip guidance
  --validation_guidance_skip_scale VALIDATION_GUIDANCE_SKIP_SCALE
                        Scale guidance strength when applying layer skipping
  --validation_lycoris_strength VALIDATION_LYCORIS_STRENGTH
                        Strength multiplier for LyCORIS validation
  --validation_noise_scheduler {ddim,ddpm,euler,euler-a,unipc,dpm++,perflow}
                        Noise scheduler for validation
  --validation_num_video_frames VALIDATION_NUM_VIDEO_FRAMES
                        Number of frames for video validation
  --validation_resolution VALIDATION_RESOLUTION
                        Override resolution for validation images (pixels or
                        megapixels)
  --validation_seed_source {cpu,gpu}
                        Source device used to generate validation seeds
  --i_know_what_i_am_doing [I_KNOW_WHAT_I_AM_DOING]
                        Unlock experimental overrides and bypass built-in
                        safety limits.
  --flow_sigmoid_scale FLOW_SIGMOID_SCALE
                        Scale factor for sigmoid timestep sampling for flow-
                        matching models.
  --flux_fast_schedule [FLUX_FAST_SCHEDULE]
                        Use experimental fast schedule for Flux training
  --flow_use_uniform_schedule [FLOW_USE_UNIFORM_SCHEDULE]
                        Use uniform schedule instead of sigmoid for flow-
                        matching
  --flow_use_beta_schedule [FLOW_USE_BETA_SCHEDULE]
                        Use beta schedule instead of sigmoid for flow-matching
  --flow_beta_schedule_alpha FLOW_BETA_SCHEDULE_ALPHA
                        Alpha value for beta schedule (default: 2.0)
  --flow_beta_schedule_beta FLOW_BETA_SCHEDULE_BETA
                        Beta value for beta schedule (default: 2.0)
  --flow_schedule_shift FLOW_SCHEDULE_SHIFT
                        Shift the noise schedule for flow-matching models
  --flow_schedule_auto_shift [FLOW_SCHEDULE_AUTO_SHIFT]
                        Auto-adjust schedule shift based on image resolution
  --flux_guidance_mode {constant,random-range}
                        Guidance mode for Flux training
  --flux_attention_masked_training [FLUX_ATTENTION_MASKED_TRAINING]
                        Enable attention masked training for Flux models
  --flux_guidance_value FLUX_GUIDANCE_VALUE
                        Guidance value for constant mode
  --flux_guidance_min FLUX_GUIDANCE_MIN
                        Minimum guidance value for random-range mode
  --flux_guidance_max FLUX_GUIDANCE_MAX
                        Maximum guidance value for random-range mode
  --t5_padding {zero,unmodified}
                        Padding behavior for T5 text encoder
  --sd3_clip_uncond_behaviour {empty_string,zero}
                        How SD3 handles unconditional prompts
  --sd3_t5_uncond_behaviour {empty_string,zero}
                        How SD3 T5 handles unconditional prompts
  --soft_min_snr_sigma_data SOFT_MIN_SNR_SIGMA_DATA
                        Sigma data for soft min SNR weighting
  --mixed_precision {no,fp16,bf16,fp8}
                        Precision for training computations
  --attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}
                        Attention computation backend
  --sageattention_usage {training,inference,training+inference}
                        When to use SageAttention
  --disable_tf32 [DISABLE_TF32]
                        Force IEEE FP32 precision (disables TF32) using
                        PyTorch's fp32_precision controls when available
  --set_grads_to_none [SET_GRADS_TO_NONE]
                        Set gradients to None instead of zero
  --noise_offset NOISE_OFFSET
                        Add noise offset to training
  --noise_offset_probability NOISE_OFFSET_PROBABILITY
                        Probability of applying noise offset
  --input_perturbation INPUT_PERTURBATION
                        Add additional noise only to the inputs fed to the
                        model during training
  --input_perturbation_steps INPUT_PERTURBATION_STEPS
                        Only apply input perturbation over the first N steps
                        with linear decay
  --lr_end LR_END       A polynomial learning rate will end up at this value
                        after the specified number of warmup steps
  --lr_scale [LR_SCALE]
                        Scale the learning rate by the number of GPUs,
                        gradient accumulation steps, and batch size
  --lr_scale_sqrt [LR_SCALE_SQRT]
                        If using --lr_scale, use the square root of (number of
                        GPUs * gradient accumulation steps * batch size)
  --ignore_final_epochs [IGNORE_FINAL_EPOCHS]
                        When provided, the max epoch counter will not
                        determine the end of the training run
  --freeze_encoder_before FREEZE_ENCODER_BEFORE
                        When using 'before' strategy, we will freeze layers
                        earlier than this
  --freeze_encoder_after FREEZE_ENCODER_AFTER
                        When using 'after' strategy, we will freeze layers
                        later than this
  --freeze_encoder_strategy {before,between,after}
                        When freezing the text encoder, we can use the
                        'before', 'between', or 'after' strategy
  --layer_freeze_strategy {none,bitfit}
                        When freezing parameters, we can use the 'none' or
                        'bitfit' strategy
  --fully_unload_text_encoder [FULLY_UNLOAD_TEXT_ENCODER]
                        If set, will fully unload the text_encoder from memory
                        when not in use
  --save_text_encoder [SAVE_TEXT_ENCODER]
                        If set, will save the text encoder after training
  --text_encoder_limit TEXT_ENCODER_LIMIT
                        When training the text encoder, we want to limit how
                        long it trains for to avoid catastrophic loss
  --prepend_instance_prompt [PREPEND_INSTANCE_PROMPT]
                        When determining the captions from the filename,
                        prepend the instance prompt as an enforced keyword
  --only_instance_prompt [ONLY_INSTANCE_PROMPT]
                        Use the instance prompt instead of the caption from
                        filename
  --data_aesthetic_score DATA_AESTHETIC_SCORE
                        Since currently we do not calculate aesthetic scores
                        for data, we will statically set it to one value. This
                        is only used by the SDXL Refiner
  --delete_unwanted_images [DELETE_UNWANTED_IMAGES]
                        If set, will delete images that are not of a minimum
                        size to save on disk space for large training runs
  --delete_problematic_images [DELETE_PROBLEMATIC_IMAGES]
                        If set, any images that error out during load will be
                        removed from the underlying storage medium
  --disable_bucket_pruning [DISABLE_BUCKET_PRUNING]
                        When training on very small datasets, you might not
                        care that the batch sizes will outpace your image
                        count. Setting this option will prevent SimpleTuner
                        from deleting your bucket lists that do not meet the
                        minimum image count requirements. Use at your own
                        risk, it may end up throwing off your statistics or
                        epoch tracking
  --disable_segmented_timestep_sampling [DISABLE_SEGMENTED_TIMESTEP_SAMPLING]
                        By default, the timestep schedule is divided into
                        roughly `train_batch_size` number of segments, and
                        then each of those are sampled from separately. This
                        improves the selection distribution, but may not be
                        desired in certain training scenarios, eg. when
                        limiting the timestep selection range
  --preserve_data_backend_cache [PRESERVE_DATA_BACKEND_CACHE]
                        For very large cloud storage buckets that will never
                        change, enabling this option will prevent the trainer
                        from scanning it at startup, by preserving the cache
                        files that we generate. Be careful when using this,
                        as, switching datasets can result in the preserved
                        cache being used, which would be problematic.
                        Currently, cache is not stored in the dataset itself
                        but rather, locally. This may change in a future
                        release
  --override_dataset_config [OVERRIDE_DATASET_CONFIG]
                        When provided, the dataset's config will not be
                        checked against the live backend config
  --cache_dir CACHE_DIR
                        The directory where the downloaded models and datasets
                        will be stored
  --cache_dir_text CACHE_DIR_TEXT
                        This is the path to a local directory that will
                        contain your text embed cache
  --cache_dir_vae CACHE_DIR_VAE
                        This is the path to a local directory that will
                        contain your VAE outputs
  --compress_disk_cache [COMPRESS_DISK_CACHE]
                        If set, will gzip-compress the disk cache for Pytorch
                        files. This will save substantial disk space, but may
                        slow down the training process
  --aspect_bucket_disable_rebuild [ASPECT_BUCKET_DISABLE_REBUILD]
                        When using a randomised aspect bucket list, the VAE
                        and aspect cache are rebuilt on each epoch. With a
                        large and diverse enough dataset, rebuilding the
                        aspect list may take a long time, and this may be
                        undesirable. This option will not override
                        vae_cache_clear_each_epoch. If both options are
                        provided, only the VAE cache will be rebuilt
  --keep_vae_loaded [KEEP_VAE_LOADED]
                        If set, will keep the VAE loaded in memory. This can
                        reduce disk churn, but consumes VRAM during the
                        forward pass
  --skip_file_discovery SKIP_FILE_DISCOVERY
                        Comma-separated values of which stages to skip
                        discovery for. Skipping any stage will speed up
                        resumption, but will increase the risk of errors, as
                        missing images or incorrectly bucketed images may not
                        be caught. Valid options: aspect, vae, text, metadata
  --data_backend_sampling {uniform,auto-weighting}
                        When using multiple data backends, the sampling
                        weighting can be set to 'uniform' or 'auto-weighting'
  --image_processing_batch_size IMAGE_PROCESSING_BATCH_SIZE
                        When resizing and cropping images, we do it in
                        parallel using processes or threads. This defines how
                        many images will be read into the queue before they
                        are processed
  --write_batch_size WRITE_BATCH_SIZE
                        When using certain storage backends, it is better to
                        batch smaller writes rather than continuous
                        dispatching. In SimpleTuner, write batching is
                        currently applied during VAE caching, when many small
                        objects are written. This mostly applies to S3, but
                        some shared server filesystems may benefit as well.
                        Default: 64
  --read_batch_size READ_BATCH_SIZE
                        Used by the VAE cache to prefetch image data. This is
                        the number of images to read ahead
  --enable_multiprocessing [ENABLE_MULTIPROCESSING]
                        If set, will use processes instead of threads during
                        metadata caching operations
  --max_workers MAX_WORKERS
                        How many active threads or processes to run during VAE
                        caching
  --aws_max_pool_connections AWS_MAX_POOL_CONNECTIONS
                        When using AWS backends, the maximum number of
                        connections to keep open to the S3 bucket at a single
                        time
  --torch_num_threads TORCH_NUM_THREADS
                        The number of threads to use for PyTorch operations.
                        This is not the same as the number of workers
  --dataloader_prefetch [DATALOADER_PREFETCH]
                        When provided, the dataloader will read-ahead and
                        attempt to retrieve latents, text embeds, and other
                        metadata ahead of the time when the batch is required,
                        so that it can be immediately available
  --dataloader_prefetch_qlen DATALOADER_PREFETCH_QLEN
                        Set the number of prefetched batches
  --aspect_bucket_worker_count ASPECT_BUCKET_WORKER_COUNT
                        The number of workers to use for aspect bucketing.
                        This is a CPU-bound task, so the number of workers
                        should be set to the number of CPU threads available.
                        If you use an I/O bound backend, an even higher value
                        may make sense. Default: 12
  --aspect_bucket_alignment {8,16,24,32,64}
                        When training diffusion models, the image sizes
                        generally must align to a 64 pixel interval
  --minimum_image_size MINIMUM_IMAGE_SIZE
                        The minimum resolution for both sides of input images
  --maximum_image_size MAXIMUM_IMAGE_SIZE
                        When cropping images that are excessively large, the
                        entire scene context may be lost, eg. the crop might
                        just end up being a portion of the background. To
                        avoid this, a maximum image size may be provided,
                        which will result in very-large images being
                        downsampled before cropping them. This value uses
                        --resolution_type to determine whether it is a pixel
                        edge or megapixel value
  --target_downsample_size TARGET_DOWNSAMPLE_SIZE
                        When using --maximum_image_size, very-large images
                        exceeding that value will be downsampled to this
                        target size before cropping
  --max_upscale_threshold MAX_UPSCALE_THRESHOLD
                        Limit upscaling of small images to prevent quality
                        degradation (opt-in). When set, filters out aspect
                        buckets requiring upscaling beyond this threshold.
                        For example, 0.2 allows up to 20% upscaling. Default
                        (None) allows unlimited upscaling. Must be between 0
                        and 1.
  --metadata_update_interval METADATA_UPDATE_INTERVAL
                        When generating the aspect bucket indicies, we want to
                        save it every X seconds
  --debug_aspect_buckets [DEBUG_ASPECT_BUCKETS]
                        If set, will print excessive debugging for aspect
                        bucket operations
  --debug_dataset_loader [DEBUG_DATASET_LOADER]
                        If set, will print excessive debugging for data loader
                        operations
  --print_filenames [PRINT_FILENAMES]
                        If any image files are stopping the process eg. due to
                        corruption or truncation, this will help identify
                        which is at fault
  --print_sampler_statistics [PRINT_SAMPLER_STATISTICS]
                        If provided, will print statistics about the dataset
                        sampler. This is useful for debugging
  --timestep_bias_strategy {earlier,later,range,none}
                        Strategy for biasing timestep sampling
  --timestep_bias_begin TIMESTEP_BIAS_BEGIN
                        Beginning of timestep bias range
  --timestep_bias_end TIMESTEP_BIAS_END
                        End of timestep bias range
  --timestep_bias_multiplier TIMESTEP_BIAS_MULTIPLIER
                        Multiplier for timestep bias probability
  --timestep_bias_portion TIMESTEP_BIAS_PORTION
                        Portion of training steps to apply timestep bias
  --training_scheduler_timestep_spacing {leading,linspace,trailing}
                        Timestep spacing for training scheduler
  --inference_scheduler_timestep_spacing {leading,linspace,trailing}
                        Timestep spacing for inference scheduler
  --loss_type {l2,huber,smooth_l1}
                        Loss function for training
  --huber_schedule {snr,exponential,constant}
                        Schedule for Huber loss transition threshold
  --huber_c HUBER_C     Transition point between L2 and L1 regions for Huber
                        loss
  --snr_gamma SNR_GAMMA
                        SNR weighting gamma value (0 = disabled)
  --masked_loss_probability MASKED_LOSS_PROBABILITY
                        Probability of applying masked loss weighting per
                        batch
  --hidream_use_load_balancing_loss [HIDREAM_USE_LOAD_BALANCING_LOSS]
                        Apply experimental load balancing loss when training
                        HiDream models.
  --hidream_load_balancing_loss_weight HIDREAM_LOAD_BALANCING_LOSS_WEIGHT
                        Strength multiplier for HiDream load balancing loss.
  --adam_beta1 ADAM_BETA1
                        First moment decay rate for Adam optimizers
  --adam_beta2 ADAM_BETA2
                        Second moment decay rate for Adam optimizers
  --optimizer_beta1 OPTIMIZER_BETA1
                        First moment decay rate for optimizers
  --optimizer_beta2 OPTIMIZER_BETA2
                        Second moment decay rate for optimizers
  --optimizer_cpu_offload_method {none}
                        Method for CPU offloading optimizer states
  --gradient_precision {unmodified,fp32}
                        Precision for gradient computation
  --adam_weight_decay ADAM_WEIGHT_DECAY
                        L2 regularisation strength for Adam-family optimizers.
  --adam_epsilon ADAM_EPSILON
                        Small constant added for numerical stability.
  --prodigy_steps PRODIGY_STEPS
                        Number of steps Prodigy should spend adapting its
                        learning rate.
  --max_grad_norm MAX_GRAD_NORM
                        Gradient clipping threshold to prevent exploding
                        gradients.
  --grad_clip_method {value,norm}
                        Strategy for applying max_grad_norm during clipping.
  --optimizer_offload_gradients [OPTIMIZER_OFFLOAD_GRADIENTS]
                        Move optimizer gradients to CPU to save GPU memory.
  --fuse_optimizer [FUSE_OPTIMIZER]
                        Enable fused kernels when offloading to reduce memory
                        overhead.
  --optimizer_release_gradients [OPTIMIZER_RELEASE_GRADIENTS]
                        Free gradient tensors immediately after optimizer step
                        when using Optimi optimizers.
  --push_to_hub [PUSH_TO_HUB]
                        Automatically upload the trained model to your Hugging
                        Face Hub repository.
  --push_to_hub_background [PUSH_TO_HUB_BACKGROUND]
                        Run Hub uploads in a background worker so training is
                        not blocked while pushing.
  --push_checkpoints_to_hub [PUSH_CHECKPOINTS_TO_HUB]
                        Upload intermediate checkpoints to the same Hugging
                        Face repository during training.
  --publishing_config PUBLISHING_CONFIG
                        Optional JSON/file path describing additional
                        publishing targets (S3/Backblaze B2/Azure Blob/Dropbox).
  --hub_model_id HUB_MODEL_ID
                        If left blank, SimpleTuner derives a name from the
                        project settings when pushing to Hub.
  --model_card_private [MODEL_CARD_PRIVATE]
                        Create the Hugging Face repository as private instead
                        of public.
  --model_card_safe_for_work [MODEL_CARD_SAFE_FOR_WORK]
                        Remove the default NSFW warning from the generated
                        model card on Hugging Face Hub.
  --model_card_note MODEL_CARD_NOTE
                        Optional note that appears at the top of the generated
                        model card.
  --modelspec_comment MODELSPEC_COMMENT
                        Text embedded in safetensors file metadata as
                        modelspec.comment, visible in external model viewers.
  --report_to {tensorboard,wandb,comet_ml,all,none}
                        Where to log training metrics
  --checkpoint_step_interval CHECKPOINT_STEP_INTERVAL
                        Save model checkpoint every N steps (deprecated alias: --checkpointing_steps)
  --checkpoint_epoch_interval CHECKPOINT_EPOCH_INTERVAL
                        Save model checkpoint every N epochs
  --checkpointing_rolling_steps CHECKPOINTING_ROLLING_STEPS
                        Rolling checkpoint window size for continuous
                        checkpointing
  --checkpointing_use_tempdir [CHECKPOINTING_USE_TEMPDIR]
                        Use temporary directory for checkpoint files before
                        final save
  --checkpoints_rolling_total_limit CHECKPOINTS_ROLLING_TOTAL_LIMIT
                        Maximum number of rolling checkpoints to keep
  --tracker_run_name TRACKER_RUN_NAME
                        Name for this training run in tracking platforms
  --tracker_project_name TRACKER_PROJECT_NAME
                        Project name in tracking platforms
  --tracker_image_layout {gallery,table}
                        How validation images are displayed in trackers
  --enable_watermark [ENABLE_WATERMARK]
                        Add invisible watermark to generated images
  --framerate FRAMERATE
                        Framerate for video model training
  --seed_for_each_device [SEED_FOR_EACH_DEVICE]
                        Use a unique deterministic seed per GPU instead of
                        sharing one seed across devices.
  --snr_weight SNR_WEIGHT
                        Weight factor for SNR-based loss scaling
  --rescale_betas_zero_snr [RESCALE_BETAS_ZERO_SNR]
                        Rescale betas for zero terminal SNR
  --webhook_config WEBHOOK_CONFIG
                        Path to webhook configuration file
  --webhook_reporting_interval WEBHOOK_REPORTING_INTERVAL
                        Interval for webhook reports (seconds)
  --distillation_method {lcm,dcm,dmd,perflow}
                        Method for model distillation
  --distillation_config DISTILLATION_CONFIG
                        Path to distillation configuration file
  --ema_validation {none,ema_only,comparison}
                        Control how EMA weights are used during validation
                        runs.
  --local_rank LOCAL_RANK
                        Local rank for distributed training
  --ltx_train_mode {t2v,i2v}
                        Training mode for LTX models
  --ltx_i2v_prob LTX_I2V_PROB
                        Probability of using image-to-video training for LTX
  --ltx_partial_noise_fraction LTX_PARTIAL_NOISE_FRACTION
                        Fraction of noise to add for LTX partial training
  --ltx_protect_first_frame [LTX_PROTECT_FIRST_FRAME]
                        Protect the first frame from noise in LTX training
  --offload_param_path OFFLOAD_PARAM_PATH
                        Path to offloaded parameter files
  --offset_noise [OFFSET_NOISE]
                        Enable offset-noise training
  --quantize_activations [QUANTIZE_ACTIVATIONS]
                        Quantize model activations during training
  --refiner_training [REFINER_TRAINING]
                        Enable refiner model training mode
  --refiner_training_invert_schedule [REFINER_TRAINING_INVERT_SCHEDULE]
                        Invert the noise schedule for refiner training
  --refiner_training_strength REFINER_TRAINING_STRENGTH
                        Strength of refiner training
  --sdxl_refiner_uses_full_range [SDXL_REFINER_USES_FULL_RANGE]
                        Use full timestep range for SDXL refiner
  --sana_complex_human_instruction SANA_COMPLEX_HUMAN_INSTRUCTION
                        Complex human instruction for Sana model training
```
