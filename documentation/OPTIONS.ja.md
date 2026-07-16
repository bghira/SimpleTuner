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

### `--text_embed_full_cache`

- **内容**: text embed cache に text encoder の完全な raw 出力を保存します。
- **既定値**: `False`
- **理由**: 既定ではモデル固有の compact cache layout を使えます。たとえば Ideogram 4 は、cache file に書き込む前に 13 層の Qwen hidden-state stack を transformer の凍結された `llm_cond_norm` と `llm_cond_proj` で投影します。これらの layer は LoRA と full transformer training の両方で凍結されます。
- **使用タイミング**: cache 互換性のデバッグ、raw で未投影の text encoder features が必要な場合、または text projection が固定済み pretrained component ではない Ideogram 風 architecture を scratch training に適用する場合に有効化します。

### `--trust_remote_code`

- **内容**: チェックポイントが upstream の独自クラスに依存している場合に、Transformers と tokenizer がモデルリポジトリ内のカスタム Python コードを実行できるようにします。
- **既定値**: `False`
- **理由**: upstream リポジトリに独自の `AutoModel` / tokenizer コードを含む ACE-Step v1.5 チェックポイントで必要です。
- **警告**: 信頼できるモデルリポジトリに対してのみ有効にしてください。

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

- **内容**: LongCat-Video、Wan、LTXVideo、Kandinsky5-Video、Qwen-Image、Flux、Flux.2、zlab i1、Cosmos2Image、HunyuanVideo、Krea 2 の Musubi ブロックスワップ。最後の N 個の Transformer ブロックを CPU に置き、forward 中にブロック単位で重みをストリーミングします。
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
- **注記**:
  - `fnmatch` glob 構文を使用して完全修飾モジュール名またはクラス名にマッチ。
  - ブロック内のレイヤーにマッチさせるには、末尾に `.*` ワイルドカードを含める必要があります。例えば、`transformer_blocks.0.*` はブロック 0 内のすべてのレイヤーにマッチし、`transformer_blocks.*` はすべての transformer ブロックにマッチします。`transformer_blocks.0` のような `.*` なしの名前も動作します（自動的に展開されます）が、明確さのために明示的なワイルドカード形式を推奨します。
  - 例: `"transformer_blocks.*,single_transformer_blocks.0.*,single_transformer_blocks.1.*"`

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

### `--ramtorch_transformer_percent`

- **内容**: RamTorch でオフロードする transformer Linear レイヤーの割合（0-100）。
- **既定**: `100`（対象となるすべてのレイヤー）
- **理由**: 部分的なオフロードにより、VRAM 節約とパフォーマンスのバランスを取ることができます。低い値はより多くのレイヤーを GPU に保持し、メモリ使用量を削減しながら高速な学習を可能にします。
- **注記**: レイヤーはモジュール走査順の先頭から選択されます。`--ramtorch_target_modules` と組み合わせ可能。

### `--ramtorch_text_encoder_percent`

- **内容**: RamTorch でオフロードするテキストエンコーダー Linear レイヤーの割合（0-100）。
- **既定**: `100`（対象となるすべてのレイヤー）
- **理由**: `--ramtorch_text_encoder` 有効時にテキストエンコーダーの部分的なオフロードを可能にします。
- **注記**: `--ramtorch_text_encoder` が有効な場合のみ適用。

### `--ramtorch_disable_sync_hooks`

- **内容**: RamTorch レイヤーの後に追加される CUDA 同期フックを無効にします。
- **既定**: `False`（同期フック有効）
- **理由**: 同期フックは RamTorch のピンポンバッファリングシステムにおける競合状態を修正し、非決定的な出力を防ぎます。無効にするとパフォーマンスが向上する可能性がありますが、不正確な結果のリスクがあります。
- **注記**: 同期フックに問題がある場合やテストする場合にのみ無効にしてください。

### `--ramtorch_disable_extensions`

- **内容**: Linear レイヤーのみに RamTorch を適用し、Embedding/RMSNorm/LayerNorm/Conv をスキップします。
- **既定**: `True`（拡張機能無効）
- **理由**: SimpleTuner は RamTorch を Linear レイヤー以外に拡張し、Embedding、RMSNorm、LayerNorm、Conv レイヤーを含めます。この拡張機能を無効にして Linear レイヤーのみをオフロードするにはこのオプションを使用します。
- **注記**: VRAM 節約が減少する可能性がありますが、拡張レイヤータイプの問題をデバッグするのに役立ちます。

### `--pretrained_model_name_or_path`

- **内容**: 事前学習済みモデルのパス、または <https://huggingface.co/models> の識別子。
- **理由**: 学習を開始するベースモデルを指定します。`--revision` と `--variant` でリポジトリ内の特定バージョンを指定できます。SDXL、Flux、SD3.x の単一ファイル `.safetensors` パスにも対応しています。

### `--pretrained_transformer_model_name_or_path`

- **内容**: 事前学習済み transformer 重みの任意パス、または <https://huggingface.co/models> の識別子。
- **既定**: `None`（この上書きをサポートするローダーでは、transformer の参照元は `--pretrained_model_name_or_path` に従います）
- **理由**: transformer コンポーネントがベースモデルパッケージとは別のリポジトリ、ローカルフォルダー、またはチェックポイントにある場合に使用します。
- **注記**: transformer 重みがそのパス内のサブフォルダーにある場合は、`--pretrained_transformer_subfolder` と組み合わせて使用します。

### `--pretrained_t5_model_name_or_path`

- **内容**: 事前学習済み T5 モデルのパス、または <https://huggingface.co/models> の識別子。
- **理由**: PixArt の学習時、T5 重みの取得元を指定することで、ベースモデル切り替え時の重複ダウンロードを避けられます。

### `--pretrained_gemma_model_name_or_path`

- **内容**: 事前学習済み Gemma モデルのパス、または <https://huggingface.co/models> の識別子。
- **理由**: Gemma 系モデル（例: LTX-2、Sana、Lumina2）を学習する際、ベース拡散モデルのパスを変えずに Gemma 重みの参照先を指定できます。

### `--max_grounding_entities`
- GLIGEN スタイルの空間アノテーション用に、画像あたりのグラウンディングエンティティの最大数を指定します。デフォルト: 0（無効）。一般的な値: 4-16。

### `--pretrained_grounding_model_name_or_path`
- エンティティごとの画像特徴抽出に使用するオプションの事前学習済みモデル。デフォルト: None。

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

### `--gradient_checkpointing_backend`

- **選択肢**: `torch`、`unsloth`
- **内容**: 勾配チェックポイントの実装を選択します。
  - `torch`（デフォルト）: 標準の PyTorch チェックポイント。逆伝播時に活性化を再計算します。約 20% の時間オーバーヘッド。
  - `unsloth`: 再計算の代わりに活性化を非同期で CPU にオフロードします。約 30% のメモリ節約、約 2% のオーバーヘッドのみ。高速な PCIe 帯域が必要です。
- **注記**: `--gradient_checkpointing` が有効な場合のみ機能します。`unsloth` バックエンドは CUDA が必要です。

### `--refiner_training`

- **内容**: カスタムの Mixture-of-Experts モデル系列の学習を有効化します。詳細は [Mixture-of-Experts](MIXTURE_OF_EXPERTS.md) を参照してください。

## 精度

### `--quantize_via`

- **選択肢**: `cpu`、`accelerator`、`pipeline`
  - `accelerator` では若干高速になる可能性がありますが、Flux のように大きなモデルでは 24G カードで OOM するリスクがあります。
  - `cpu` では量子化に約 30 秒かかります。（**既定**）
  - `pipeline` は `--quantization_config` またはパイプライン対応のプリセット（例: `nf4-bnb`、`int8-torchao` `fp8-torchao`、`int8-quanto`、`.gguf` チェックポイント）を使って Diffusers に量子化を委譲します。

### `--base_model_precision`

- **内容**: モデル精度を下げ、少ないメモリで学習します。対応する量子化バックエンドは BitsAndBytes（pipeline）、TorchAO（pipeline または手動）、Optimum Quanto（pipeline または手動）の 3 つです。

#### Diffusers のパイプラインプリセット

- `nf4-bnb` は Diffusers 経由で 4-bit NF4 BitsAndBytes 設定で読み込みます（CUDA のみ）。`bitsandbytes` と BnB 対応の diffusers が必要です。
- `int4-torchao`、`int8-torchao` `fp8-torchao`、`fp8wo-torchao` は Diffusers 経由で TorchAoConfig を使用します（CUDA）。`torchao` と最新の diffusers/transformers が必要です。
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
- `fp8-native` and `fp8-torchao` は FP8 scaled matmul に対応する Ada Lovelace（RTX 40/L40S）、Hopper（H100/H200）またはそれ以降が必要
- `fp8-transformerengine` は対象の Linear 層を TransformerEngine FP8 module に置き換え、model forward を TE FP8 autocast で包みます。`pip install 'simpletuner[transformerengine]'` でインストールします。この preset は Ada Lovelace、Hopper、またはそれ以降の CUDA accelerator 向けです。

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
- `fp8-sdnq` - SDNQ のネイティブ FP8 matmul を使う FP8 重み。H100/H200 クラス向け

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

##### SDNQ Native Matmul オプション

- `--sdnq_weights_dtype` - SDNQ の保存 dtype を上書きします。例: `float8_e4m3fn`, `int8`, `uint4`。
- `--sdnq_quantized_matmul_dtype` - matmul dtype: `auto`, `int8`, `float8_e4m3fn`, `fp8`, `float16`, `fp16`。
- `--sdnq_group_size` - 量子化 group size。`-1` はテンソル全体の static matmul。`fp8-sdnq` は既定で `-1`。
- `--sdnq_use_quantized_matmul` - SDNQ quantized matmul を有効/無効にします。未設定の場合、`fp8-sdnq` は SDNQ compile mode と FP8 matmul support の両方が利用可能なときだけ native FP8 matmul を使い、他の preset は SDNQ compile availability に従います。
- `--sdnq_compile_mode` - `auto`, `compile`, `eager`。SDNQ 内部の `torch.compile` 使用を制御します。現在の SDNQ は quantized matmul に compile mode を要求し、eager mode では dequantized matmul を使います。
- `--sdnq_use_static_quantization`, `--sdnq_use_stochastic_rounding`, `--sdnq_dequantize_fp32` - SDNQ training quantization の既定値を上書きします。
- `--sdnq_use_svd`, `--sdnq_svd_rank`, `--sdnq_svd_steps` - 低 bit SDNQ preset の SVDQuant を設定します。
- `--sdnq_use_hadamard`, `--sdnq_hadamard_group_size` - SDNQ Hadamard rotation を有効化/設定します。
- `--sdnq_modules_to_not_convert`, `--sdnq_modules_to_not_use_matmul` - JSON 配列、ファイル、またはカンマ区切りの module pattern。
- `--sdnq_modules_dtype_dict`, `--sdnq_modules_quant_config` - module ごとの dtype/quantization override 用 JSON object またはファイル。

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
- `metal-flash-attention` は Apple Silicon 上で Universal Metal Flash Attention の PyTorch custom-op バックエンドを利用します。まず UMFA の `examples/pytorch-custom-op-ffi` パッケージをインストールしてください。SimpleTuner は PyTorch の MPS SDPA dispatcher 経由でアテンションをルーティングし、現行の UMFA build はこの dispatcher を登録します。対象の MPS FP32/FP16/BF16 4D SDPA 呼び出し — single-head を含む任意の head 数、任意の sequence length、transposed な FLUX スタイル layout、最大 4D の bool/additive mask、causal 呼び出し（トレーニング含む — causal backward は厳密な勾配パリティを通過） — は呼び出しごとの同期なしで PyTorch の MPS command stream に直接エンコードされ、FP16/BF16 入力は FP32 への昇格なしにネイティブ低精度カーネルで実行されます。dropout または `enable_gqa` を伴う呼び出しは PyTorch SDPA にフォールバックします。古い `PrivateUse1` build は `torch.device("mps")` テンソルにバイパスされます。SimpleTuner は起動時に FP32/FP16/BF16 の forward・autograd パリティチェックと causal forward パリティを実行し、数値が一致しない、または `get_dispatch_stats()` を公開しない UMFA build を拒否します。`metal-flash-attention-int8` と `metal-flash-attention-int4` は `set_quantization_mode(ext.QUANT_INT8, ext.QUANT_BLOCK_WISE)` または `set_quantization_mode(ext.QUANT_INT4, ext.QUANT_BLOCK_WISE)` で UMFA のグローバル blockwise quantization mode を設定し、バックエンド切替時に解除、さらに autograd 出力の接続・有限な multi-head 勾配・dispatcher レベルの bool/additive mask サポート・PyTorch フォールバックなしを確認する追加の起動チェックを要求します。実行中は `metal_sdpa_extension.get_dispatch_stats()` で `fp32_instream`（量子化エイリアスでは `quantized_autograd`）が増加し `pytorch_fallback` が `0` のままであることを確認してください。all-true の bool mask は `mask_all_true_skipped` も増やします。拡張はさらに `rope_scaled_dot_product_attention` を公開しており、FLUX.1/FLUX.2/Krea2/Z-Image の interleaved-pair rotary 規約に対応した融合 RoPE+SDPA エントリポイントです。トレーニングは融合 autograd（backward で dQ/dK に逆回転を適用）を通ります。勾配が必要な mask 付き・GQA の呼び出しは eager 回転へフォールバックします。
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
  - メタデータ書き込み時に `{current_step}`、`{current_epoch}`、`{timestamp}` をサポート
  - `{timestamp}` は UTC の ISO 8601 値です
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
- **注記**:
  - `config.json` と `config.toml` から読み込まれる文字列値は `{env:VAR_NAME}` をサポート
  - 参照先の `multidatabackend.json` 内の文字列値も `{env:VAR_NAME}` をサポート
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

### `--enable_nsfw_check`

- **内容**: VAE キャッシュ前処理中に NSFW 分類器による除外を有効にします。
- **既定値**: `false`。
- **注記**: VAE キャッシュがこれから処理する未キャッシュのサンプルだけをスキャンします。既存の VAE キャッシュと `skip_file_discovery=vae` は信頼され、再スキャンされません。
- **詳細**: プライバシーと責任については [NSFW.ja.md](NSFW.ja.md)、VAE キャッシュの詳細は [DATALOADER.ja.md#nsfw-classifier-checks-during-vae-caching](DATALOADER.ja.md#nsfw-classifier-checks-during-vae-caching) を参照してください。

### `--nsfw_check_models`

- **内容**: Hugging Face Transformers の画像分類モデルを CSV で指定します。モデルごとに `:threshold=0.5` を付けられます。
- **既定値**: `Falconsai/nsfw_image_detection:threshold=0.5,AdamCodd/vit-base-nsfw-detector:threshold=0.5`。
- **注記**: 標準の Transformers モデルだけをサポートします。`trust_remote_code` は有効にせず、`timm` 依存の分類器も読み込みません。

### `--nsfw_check_min_votes`

- **内容**: 1 フレームを拒否するために NSFW 判定を返す必要がある分類器数です。
- **既定値**: `2`。
- **注記**: `1` 以上、かつ設定済み分類器数以下である必要があります。

### `--nsfw_check_backend_types`

- **内容**: スキャン対象にするデータバックエンドの `type` 値を CSV で指定します。
- **既定値**: `all`。
- **例**: `local,huggingface,csv,aws`。

### `--nsfw_check_sample_types`

- **内容**: スキャン対象にするデータセットの `dataset_type` 値を CSV で指定します。
- **既定値**: `image,conditioning`。
- **注記**: 評価データセットはスキャンされません。

### `--delete_nsfw_images`

- **内容**: バックエンドが削除をサポートする場合、NSFW 分類器で拒否された元サンプルを削除します。
- **既定値**: `false`。
- **注記**: 無効時は、拒否サンプルは現在の実行のメタデータから外されますが、元データセットには残ります。

### `--nsfw_check_video_frame_count`

- **内容**: 動画または複数フレーム conditioning 入力から抽出するフレーム数です。
- **既定値**: `3`。

### `--nsfw_check_video_frame_selection`

- **内容**: 動画 NSFW チェックのフレーム選択方式です。
- **既定値**: `uniform`。
- **オプション**: `uniform`, `first`, `middle`。

### `--nsfw_check_video_min_flagged_frames`

- **内容**: サンプル全体を拒否するために拒否される必要があるチェック済み動画フレーム数です。
- **既定値**: `1`。

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

### `--deepfloyd_validation_pipeline_mode`

- **内容**: DeepFloyd 検証時のステージ連結を制御します。
- **選択肢**: `auto`, `trained-stage`, `full-pipeline`
- **既定値**: `auto`
- **理由**: `auto` はプロンプト検証では DeepFloyd stage I → stage II を実行し、データセット画像検証では学習中のステージだけを使います。単一ステージに固定する場合は `trained-stage`、常に前後ステージを読み込む場合は `full-pipeline` を使います。

### `--deepfloyd_validation_stage1_model`

- **内容**: 学習中の stage II をフルパイプラインで検証するときに使う固定 stage I モデル。
- **既定値**: `DeepFloyd/IF-I-XL-v1.0`

### `--deepfloyd_validation_stage2_model`

- **内容**: 学習中の stage I をフルパイプラインで検証するときに使う固定 stage II モデル。
- **既定値**: `DeepFloyd/IF-II-M-v1.0`

### `--deepfloyd_validation_stage3_mode`

- **内容**: DeepFloyd stage II の後に使う任意の最終アップスケーラ。
- **選択肢**: `none`, `sd-x4-upscaler`
- **既定値**: `none`
- **理由**: DeepFloyd の未公開 stage III は実質的に 4x アップスケーラでした。`sd-x4-upscaler` はその役割に `stabilityai/stable-diffusion-x4-upscaler` を使います。

### `--deepfloyd_validation_stage3_model`

- **内容**: `--deepfloyd_validation_stage3_mode=sd-x4-upscaler` のときに使うモデルリポジトリ。
- **既定値**: `stabilityai/stable-diffusion-x4-upscaler`

### `--deepfloyd_validation_stage1_num_inference_steps`

- **内容**: stage I 検証ステップ数の任意上書き。
- **既定値**: `--validation_num_inference_steps` を使い、stage I では最大 30 に制限します。

### `--deepfloyd_validation_stage2_num_inference_steps`

- **内容**: stage II 検証ステップ数の任意上書き。
- **既定値**: `--validation_num_inference_steps`

### `--deepfloyd_validation_stage1_guidance`

- **内容**: stage I 検証 guidance の任意上書き。
- **既定値**: `--validation_guidance`

### `--deepfloyd_validation_stage2_guidance`

- **内容**: stage II 検証 guidance の任意上書き。
- **既定値**: `--validation_guidance`

### `--deepfloyd_validation_stage3_guidance`

- **内容**: SD x4 アップスケーラ guidance の任意上書き。
- **既定値**: `--validation_guidance`

### `--deepfloyd_validation_stage3_noise_level`

- **内容**: SD x4 アップスケーラに渡すノイズレベル。
- **既定値**: `100`

### `--wan_validation_load_other_stage`

- **内容**: 検証時に Wan 2.2 の反対側ステージを読み込みます。
- **既定値**: `false`
- **理由**: Wan 2.2 と AnimeGen などの互換ステージ構成では、各ステージを個別に学習できます。有効にすると固定のペアステージを読み込み、検証で完全なペアステージパイプラインを使って境界で denoiser を切り替えます。

### `--sdxl_validation_pipeline_mode`

- **選択肢**: `trained-stage`, `full-pipeline`
- **既定値**: `trained-stage`
- **内容**: SDXL 検証で学習済みステージだけを実行するか、base/refiner の分割パイプラインを実行するかを選びます。
- **理由**: `full-pipeline` は stage 1 を `1 - refiner_training_strength` まで latent 出力で実行し、同じ schedule 境界から stage 2 を続行します。

### `--sdxl_validation_stage1_model`

- **内容**: 学習済み stage 2 モデルを full-pipeline 検証で refine するときに使う固定 SDXL stage 1/base モデル。
- **既定値**: 選択された SDXL バージョンから推定され、通常は `stabilityai/stable-diffusion-xl-base-1.0`

### `--sdxl_validation_stage2_model`

- **内容**: 学習済み stage 1 モデルを先に実行する full-pipeline 検証で使う固定 SDXL stage 2/refiner モデル。
- **既定値**: 選択された SDXL バージョンから推定され、通常は `stabilityai/stable-diffusion-xl-refiner-1.0`

### `--pixart_validation_pipeline_mode`

- **選択肢**: `trained-stage`, `full-pipeline`
- **既定値**: `trained-stage`
- **内容**: PixArt 検証で学習済みステージだけを実行するか、v0.7 分割パイプラインを実行するかを選びます。
- **理由**: `full-pipeline` は stage 1 を `1 - refiner_training_strength` まで latent 出力で実行し、同じ schedule 境界から stage 2 を続行します。

### `--pixart_validation_stage1_model`

- **内容**: 学習済み stage 2 モデルを full-pipeline 検証で refine するときに使う固定 PixArt stage 1 モデル。
- **既定値**: `terminusresearch/pixart-900m-1024-ft-v0.7-stage1`

### `--pixart_validation_stage2_model`

- **内容**: 学習済み stage 1 モデルを先に実行する full-pipeline 検証で使う固定 PixArt stage 2 モデル。
- **既定値**: `terminusresearch/pixart-900m-1024-ft-v0.7-stage2`

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

### `--validation_using_datasets`

- **内容**: 純粋なテキストから画像生成の代わりに、学習データセットから画像を検証に使用します。
- **理由**: 画像から画像 (img2img) または画像から動画 (i2v) 検証モードを有効化し、モデルが学習画像をコンディショニング入力として使用します。以下の場合に便利です：
  - 入力画像が必要な編集/インペインティングモデルのテスト
  - モデルが画像構造をどの程度保持するかの評価
  - テキストから画像と画像から画像の両方のワークフローをサポートするモデル（例：Flux2、LTXVideo2）
  - **I2V 動画モデル**（HunyuanVideo、WAN、Kandinsky5Video）：画像データセットから画像を動画生成検証の最初のフレームコンディショニング入力として使用
- **注意**:
  - モデルに `IMG2IMG` または `IMG2VIDEO` パイプラインが登録されている必要があります
  - `--eval_dataset_id` と組み合わせて特定のデータセットから画像を取得できます
  - i2v モデルの場合、学習時に使用する複雑なコンディショニングデータセットのペアリング設定なしで、シンプルな画像データセットを検証に使用できます
  - Flux Kontext は検証でこのフラグを使用しません。無効のままにして、`--eval_dataset_id` で編集データセットを選択してください。Kontext は対応する参照データセットを自動で読み込みます
  - デノイズ強度は通常の検証タイムステップ設定で制御されます

### `--eval_dataset_id`

- **内容**: 評価/検証画像ソーシング用の特定のデータセットID。
- **理由**: `--validation_using_datasets` またはコンディショニングベースの検証を使用する場合、どのデータセットが入力画像を提供するかを制御します：
  - このオプションなしでは、すべての学習データセットからランダムに画像が選択されます
  - このオプションありでは、指定されたデータセットのみが検証入力に使用されます
- **注意**:
  - データセットIDはデータローダー設定の設定済みデータセットと一致する必要があります
  - 専用の評価データセットを使用して一貫した評価を維持するのに便利です
  - コンディショニングモデルの場合、データセットのコンディショニングデータ（存在する場合）も使用されます
  - Flux Kontext では、これが正しい検証データセット選択方法です。`--validation_using_datasets` は有効にしないでください

---

## コンディショニングと検証モードの理解

SimpleTunerは、コンディショニング入力（参照画像、制御信号など）を使用するモデル向けに3つの主要なパラダイムをサポートしています：

### 1. コンディショニングを必要とするモデル

一部のモデルはコンディショニング入力なしでは機能しません：

- **Flux Kontext**: 編集スタイルの学習には常に参照画像が必要
- **ControlNet学習**: 制御信号画像が必要

これらのモデルではコンディショニングデータセットが必須です。WebUIはコンディショニングオプションを必須として表示し、なければ学習は失敗します。
Flux Kontext の検証もこのコンディショニングベースの経路を使います。検証に使う編集データセットは `--eval_dataset_id` で選び、`--validation_using_datasets` は無効のままにしてください。

### 2. オプションのコンディショニングをサポートするモデル

一部のモデルはテキストから画像と画像から画像の両方のモードで動作できます：

- **Flux2**: オプションの参照画像でデュアルT2I/I2I学習をサポート
- **LTXVideo2**: オプションの最初のフレームコンディショニングでT2VとI2V（画像から動画）の両方をサポート
- **LongCat-Video**: オプションのフレームコンディショニングをサポート
- **HunyuanVideo i2v**: 最初のフレームコンディショニング付きI2Vをサポート（フレーバー: `i2v-480p`、`i2v-720p` など）
- **WAN i2v**: 最初のフレームコンディショニング付きI2Vをサポート
- **Kandinsky5Video i2v**: 最初のフレームコンディショニング付きI2Vをサポート

これらのモデルでは、コンディショニングデータセットを追加できますが必須ではありません。WebUIはコンディショニングオプションをオプションとして表示します。

**I2V 検証ショートカット**: i2v 動画モデルの場合、`--validation_using_datasets` と画像データセット（`--eval_dataset_id` で指定）を使用して、学習時に使用する完全なコンディショニングデータセットペアリング設定なしで、検証コンディショニング画像を直接取得できます。

### 3. 検証モード

| モード | フラグ | 動作 |
|--------|--------|------|
| **テキストから画像/動画** | (デフォルト) | テキストプロンプトのみから生成 |
| **データセットベース (img2img)** | `--validation_using_datasets` | データセットから画像を部分的にデノイズ |
| **データセットベース (i2v)** | `--validation_using_datasets` | i2v 動画モデルの場合、画像を最初のフレームコンディショニングとして使用 |
| **コンディショニングベース** | (コンディショニング設定時に自動) | 検証中にコンディショニング入力を使用 |

**モードの組み合わせ**: モデルがコンディショニングをサポートし、かつ `--validation_using_datasets` が有効な場合：
- 検証システムはデータセットから画像を取得します
- それらのデータセットにコンディショニングデータがあれば、自動的に使用されます
- `--eval_dataset_id` を使用してどのデータセットが入力を提供するかを制御できます

**I2V モデルと `--validation_using_datasets`**: i2v 動画モデル（HunyuanVideo、WAN、Kandinsky5Video）の場合、このフラグを有効にすると、シンプルな画像データセットを検証に使用できます。画像は検証動画を生成するための最初のフレームコンディショニング入力として使用され、複雑なコンディショニングデータセットペアリング設定は不要です。

**Flux Kontext と `--validation_using_datasets`**: このフラグは有効にしないでください。Kontext は編集専用で、通常のペア画像 + コンディショニングデータセット経路で検証します。代わりに `--eval_dataset_id` で編集データセットを選択してください。

### コンディショニングデータタイプ

異なるモデルは異なるコンディショニングデータを期待します：

| タイプ | モデル | データセット設定 |
|--------|--------|-----------------|
| `conditioning` | ControlNet, Control | データセット設定で `type: conditioning` |
| `image` | Flux Kontext | `type: image` (標準画像データセット) |
| `latents` | Flux, Flux2 | コンディショニングは自動的にVAEエンコードされます |

---

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

### `--krea2_reference_latents` {#--krea2_reference_latents}

- **内容**: Krea 2 の reference dataset training を有効化します。
- **理由**: 有効にすると、Krea 2 は Qwen3VL prompt embeddings をキャッシュするときにペアの conditioning image を使い、training 時にはその conditioning image の clean VAE latents を transformer token sequence に追加します。
- **データセット設定**: main image dataset の `conditioning_data` をペアの conditioning dataset に向けます。target image と reference image のファイル名は一致している必要があります。
- **範囲**: これは Krea 2 の model-side option です。conditioning datasets は生成しません。通常の dataloader reference-dataset 設定を使ってください。

### LTX-2 validation options

- **`--ltx2_validation_pipeline_mode`**: LTX-2 validation で trained model だけを実行するか（`trained-stage`）、2 段の spatial upscaler validation pipeline を実行するか（`spatial-upscale`）を選びます。
- **`--ltx2_validation_spatial_upsampler_model`**: LTX-2 spatial latent upsampler の Hugging Face repo、ローカルディレクトリ、またはローカル `.safetensors` ファイル。既定値は `Lightricks/LTX-2.3` です。
- **`--ltx2_validation_spatial_upsampler_filename`**: model option が repo または directory を指す場合の upsampler filename。既定値は `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` です。
- **spatial-upscale の動作**: Stage 1 は指定 validation resolution の半分で video latents を生成し、spatial upsampler が latents を 2 倍にし、stage 2 が LTX-2 stage-2 sigma schedule で指定 resolution に re-denoise します。
- **制限**: Spatial-upscale validation は video 用です。`--validation_audio_only` では通常の single-stage validation path を使います。

### LTX-2 conditioning options

これは LTX-2 training 用の任意の advanced setting です。下記の名前で JSON/TOML config に書くか、`--ltx2_first_frame_conditioning_probability` のような対応する CLI flag で指定します。

- **Intrinsic target-token conditioning**: 選択された target video token を clean latent からコピーし、その token timestep を `0` にして video loss から除外します。
  - `ltx2_intrinsic_conditioning`: condition object の JSON 配列。例: `[{"type":"first_frame","probability":1.0}]`。対応する `type` は `first_frame`, `prefix`, `suffix`, `spatial_crop`, `mask` です。
  - 短縮キー: `ltx2_first_frame_conditioning_probability`, `ltx2_prefix_conditioning_probability`, `ltx2_prefix_conditioning_frames`, `ltx2_suffix_conditioning_probability`, `ltx2_suffix_conditioning_frames`, `ltx2_mask_conditioning_probability`。
  - `mask` では値 `1` が clean conditioning/no loss、値 `0` が通常の noisy training を意味します。
- **IC-LoRA reference scaling**: `ltx2_reference_spatial_scale_factor` と `ltx2_reference_temporal_scale_factor` は reference token の座標を調整します。未指定時は reference/target latent サイズから spatial scale を推定します。
- **IC-LoRA validation reference**: `validation_ltx2_video_conditioning` は validation 用のローカル reference video の JSON 配列です。例: `[{"path":"data/reference.mp4","strength":1.0}]`。
- **Scope**: これらは model-side の LTX-2 conditioning behavior だけを制御します。Dataset pairing、mask files、reference datasets、WebUI controls、dataset templates は別途設定します。

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
- **選択肢**: `unipc`, `euler`, `dpm`。
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
- **注記**: Transformer 系拡散モデル（DiT スタイル）のみ対象です。UNet モデル（SDXL、SD1.5、Kolors）は U-REPA を使用してください。

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

### `--crepa_normalize_neighbour_sum`

- **内容**: 近傍和の整合をフレームごとの重み合計で正規化します。
- **理由**: `crepa_alignment_score` を [-1, 1] に収め、損失スケールをより直感的にします。論文の式 (6) からの実験的な逸脱です。
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

### `--crepa_feature_source`

- **内容**: CREPA が教師信号をどこから取得するかを選びます。
- **理由**: 古典的な外部エンコーダ経路は `encoder`、内部ブロック間整合は `backbone`、Self-Flow の cleaner EMA 教師ビューは `self_flow` を使います。
- **選択肢**: `encoder`, `backbone`, `self_flow`
- **既定**: `encoder`

### `--crepa_self_flow`

- **内容**: Self-Flow モードを有効にする旧式の真偽値エイリアスです。
- **理由**: 既存設定との互換性のために残っていますが、新しい設定では `crepa_feature_source=self_flow` を推奨します。
- **既定**: `false`
- **注意**: `crepa_use_backbone_features` や、別モードを指す `crepa_feature_source` と競合します。

### `--crepa_self_flow_mask_ratio`

- **内容**: Self-Flow で別 timestep を受けるトークンの割合です。
- **理由**: クリーンなトークンとノイズの強いトークンの情報非対称性を制御します。高すぎる値は自己教師あり信号を強めますが、学習を不安定にすることがあります。
- **既定**: `0.1`
- **範囲**: `0.0` から `0.5`

### `--crepa_teacher_block_index`

- **内容**: バックボーン特徴または Self-Flow 使用時の教師ブロックインデックス。
- **理由**: 外部エンコーダなしで、早い学生ブロックを後段の教師ブロックに整合できます。Self-Flow では EMA 教師がより深い意味層を読むため、明示設定が必要です。
- **既定**: バックボーンモードでは未指定時に `crepa_block_index` を使用。Self-Flow モードでは必須です。

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
- **理由**: 対象は整合スコア（`crepa_alignment_score`）の指数移動平均です。これがこの値に達すると、深層エンコーダ特徴への過学習を防ぐために CREPA が無効化されます。text2video 学習に特に有用です。`crepa_normalize_neighbour_sum` を有効にしない場合、整合スコアは 1.0 を超えることがあります。
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
crepa_normalize_neighbour_sum = false
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

## 🎯 U-REPA（UNet 表現整合）

U-REPA は UNet ベースの拡散モデル（SDXL、SD1.5、Kolors）向けの正則化手法です。UNet の中間ブロック特徴を事前学習ビジョン特徴に整合させ、相対的な類似構造を保つためのマニフォールド損失を追加します。

### `--urepa_enabled`

- **内容**: 学習中に U-REPA 正則化を有効化します。
- **理由**: 凍結したビジョンエンコーダを用いて UNet 中間ブロック特徴を整合させます。
- **既定**: `false`
- **注記**: UNet モデル（SDXL、SD1.5、Kolors）のみ対象です。

### `--urepa_lambda`

- **内容**: U-REPA 整合損失の重み（主損失に対する比率）。
- **理由**: 整合正則化の強さを制御します。
- **既定**: `0.5`

### `--urepa_manifold_weight`

- **内容**: マニフォールド損失の重み（整合損失に対する比率）。
- **理由**: 特徴の相対的な類似構造を重視します（論文の既定は 3.0）。
- **既定**: `3.0`

### `--urepa_model`

- **内容**: 凍結ビジョンエンコーダの torch hub 識別子。
- **理由**: 既定は DINOv2 ViT-G/14。`dinov2_vits14` など小型モデルは高速です。
- **既定**: `dinov2_vitg14`

### `--urepa_encoder_image_size`

- **内容**: ビジョンエンコーダ前処理の入力解像度。
- **理由**: エンコーダのネイティブ解像度を使用（DINOv2 ViT-G/14 は 518、ViT-S/14 は 224）。
- **既定**: `518`

### `--urepa_use_tae`

- **内容**: フル VAE の代わりに Tiny AutoEncoder を使用します。
- **理由**: 高速で VRAM 使用量が少ない一方、復元品質は低下します。
- **既定**: `false`

### `--urepa_scheduler`

- **内容**: 学習中の U-REPA 係数の減衰スケジュール。
- **理由**: 学習の進行に合わせて正則化強度を下げられます。
- **選択肢**: `constant`、`linear`、`cosine`、`polynomial`
- **既定**: `constant`

### `--urepa_warmup_steps`

- **内容**: U-REPA 重みを 0 から `urepa_lambda` まで線形に増やすステップ数。
- **理由**: 初期学習の安定化に有効です。
- **既定**: `0`

### `--urepa_decay_steps`

- **内容**: 減衰に使う総ステップ数（ウォームアップ後）。0 にすると学習全体で減衰します。
- **理由**: 減衰フェーズの長さを制御します。
- **既定**: `0`（`max_train_steps` を使用）

### `--urepa_lambda_end`

- **内容**: 減衰完了後の最終 U-REPA 重み。
- **理由**: 0 にすると学習末期で U-REPA を実質無効化します。
- **既定**: `0.0`

### `--urepa_power`

- **内容**: 多項式減衰の指数。1.0 = 線形、2.0 = 二次など。
- **理由**: 値を上げると初期減衰が速く、後半が緩やかになります。
- **既定**: `1.0`

### `--urepa_cutoff_step`

- **内容**: このステップ以降で U-REPA を無効化するハードカット。
- **理由**: 整合が収束した後にオフにするのに便利です。
- **既定**: `0`（ステップ制のカットなし）

### `--urepa_similarity_threshold`

- **内容**: U-REPA の類似度 EMA 閾値。
- **理由**: 類似度（`urepa_similarity`）の指数移動平均がこの値に達すると U-REPA を無効化します。
- **既定**: None（無効）

### `--urepa_similarity_ema_decay`

- **内容**: 類似度追跡の指数移動平均の減衰係数。
- **理由**: 高い値ほど平滑（0.99 ≈ 100 ステップ）、低い値ほど反応が速い。
- **既定**: `0.99`

### `--urepa_threshold_mode`

- **内容**: 閾値到達時の挙動。
- **選択肢**: `permanent`（一度達したら U-REPA を保持してオフ）、`recoverable`（類似度低下で再有効化）
- **既定**: `permanent`

### 設定例

```toml
# UNet 微調整向けに U-REPA を有効化（SDXL、SD1.5、Kolors）
urepa_enabled = true
urepa_lambda = 0.5
urepa_manifold_weight = 3.0
urepa_model = "dinov2_vitg14"
urepa_encoder_image_size = 518
urepa_use_tae = false

# U-REPA スケジューリング（任意）
# urepa_scheduler = "cosine"           # 減衰タイプ：constant、linear、cosine、polynomial
# urepa_warmup_steps = 100             # U-REPA 有効化前のウォームアップ
# urepa_decay_steps = 1000             # 減衰ステップ数（0 = 学習全体）
# urepa_lambda_end = 0.0               # 減衰後の最終重み
# urepa_cutoff_step = 5000             # ハードカット（0 = 無効）
# urepa_similarity_threshold = 0.9     # 類似度ベースのカットオフ
# urepa_threshold_mode = "permanent"   # permanent または recoverable
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

- **内容**: 学習再開の有無と再開元を指定します。`latest`、ローカルのチェックポイント名/パス、または S3/R2 の URI を受け付けます。
- **理由**: 保存済みの状態から再開できます。手動指定、または最新のチェックポイントを利用します。
- **リモート再開**: 完全な URI (`s3://bucket/jobs/.../checkpoint-100`) またはバケット相対キー (`jobs/.../checkpoint-100`) を指定します。`latest` はローカルの `output_dir` のみで動作します。
- **要件**: リモート再開にはチェックポイントを読み取れる S3 の publishing_config（bucket + credentials）が必要です。
- **注記**: リモートチェックポイントには `checkpoint_manifest.json` が含まれている必要があります（最近の SimpleTuner 実行で生成）。チェックポイントは `unet` と任意の `unet_ema` サブフォルダで構成されます。`unet` は Diffusers レイアウトの SDXL モデルにそのまま配置でき、通常のモデルとして利用可能です。

> ℹ️ PixArt、SD3、Hunyuan などの Transformer モデルは `transformer` と `transformer_ema` のサブフォルダ名を使用します。

### `--delete_invalid_checkpoints`

- **内容**: 再開時に読み込めないローカルチェックポイントを削除します。
- **動作**: `--resume_from_checkpoint=latest` では、SimpleTuner は無効なローカルチェックポイントを削除し、次に新しいチェックポイントを試します。新しいチェックポイントは再開に必要なファイルを保存した後に `.guard` ファイルを書き込むため、より新しいチェックポイントにその guard がなく、古い guarded チェックポイントがある場合は破棄できます。
- **安全性**: 削除対象は `output_dir` 配下のローカルチェックポイントディレクトリだけです。明示的なチェックポイントパスでは、削除後も元の読み込み失敗を送出します。
- **既定**: `false`

### `--disk_low_threshold`

- **内容**: チェックポイント保存前に必要な最小空きディスク容量。
- **理由**: ディスク容量不足を早期に検知して設定されたアクションを実行することで、ディスク満杯エラーによる学習クラッシュを防止します。
- **形式**: `100G`、`50M`、`1T`、`500K` のようなサイズ文字列、またはバイト数。
- **デフォルト**: なし（機能無効）

### `--disk_low_action`

- **内容**: ディスク容量がしきい値を下回った場合のアクション。
- **選択肢**: `stop`、`wait`、`script`
- **デフォルト**: `stop`
- **動作**:
  - `stop`: エラーメッセージを表示して学習を即座に終了します。
  - `wait`: 容量が回復するまで 30 秒ごとにループします。無限に待機する可能性があります。
  - `script`: `--disk_low_script` で指定されたスクリプトを実行して空き容量を確保します。

### `--disk_low_script`

- **内容**: ディスク容量不足時に実行するクリーンアップスクリプトのパス。
- **理由**: ディスク容量不足時に自動クリーンアップ（古いチェックポイントの削除、キャッシュのクリアなど）を実行できます。
- **注意**: `--disk_low_action=script` の場合のみ使用されます。スクリプトは実行可能である必要があります。スクリプトが失敗したり、十分な容量を確保できなかった場合、学習はエラーで停止します。
- **デフォルト**: なし

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
                [--base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,int8dq-torchao,int8dq-int4-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-native,fp8-torchao,fp8wo-torchao,fp8-int4-torchao,fp8-transformerengine}]
                [--text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,int8dq-torchao,int8dq-int4-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-native,fp8-torchao,fp8wo-torchao,fp8-int4-torchao,fp8-transformerengine}]
                [--text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,int8dq-torchao,int8dq-int4-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-native,fp8-torchao,fp8wo-torchao,fp8-int4-torchao,fp8-transformerengine}]
                [--text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,int8dq-torchao,int8dq-int4-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-native,fp8-torchao,fp8wo-torchao,fp8-int4-torchao,fp8-transformerengine}]
                [--text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,int8dq-torchao,int8dq-int4-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-native,fp8-torchao,fp8wo-torchao,fp8-int4-torchao,fp8-transformerengine}]
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
                [--validation_audio_only [VALIDATION_AUDIO_ONLY]]
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
                [--flow_custom_timesteps FLOW_CUSTOM_TIMESTEPS]
                [--flow_timesteps_mode {fixed-list,round-robin}]
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
                [--attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,metal-flash-attention,metal-flash-attention-int8,metal-flash-attention-int4,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}]
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
                [--delete_invalid_checkpoints [DELETE_INVALID_CHECKPOINTS]]
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
                [--distillation_method {lcm,dcm,dmd,perflow,flow_dpo,anyflow}]
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
  --model_family {kolors,auraflow,omnigen,flux,deepfloyd,cosmos2image,sana,qwen_image,pixart_sigma,sdxl,sd1x,sd2x,wan,hidream,sd3,lumina2,ltxvideo,ace_step,heartmula}
                        The base model architecture family to train
  --model_flavour MODEL_FLAVOUR
                        Specific variant of the selected model family.
                        ACE-Step の flavour は `base`、`v15-turbo`、
                        `v15-base`、`v15-sft` です。v1.5 flavour は学習と
                        内蔵バリデーション音声生成をサポートし、upstream
                        リポジトリでは `--trust_remote_code` が必要です。
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
  --base_model_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,int8dq-torchao,int8dq-int4-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-native,fp8-torchao,fp8wo-torchao,fp8-int4-torchao,fp8-transformerengine}
                        Precision for loading the base model. Lower precision
                        saves memory.
  --text_encoder_1_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,int8dq-torchao,int8dq-int4-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-native,fp8-torchao,fp8wo-torchao,fp8-int4-torchao,fp8-transformerengine}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_2_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,int8dq-torchao,int8dq-int4-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-native,fp8-torchao,fp8wo-torchao,fp8-int4-torchao,fp8-transformerengine}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_3_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,int8dq-torchao,int8dq-int4-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-native,fp8-torchao,fp8wo-torchao,fp8-int4-torchao,fp8-transformerengine}
                        Precision for text encoders. Lower precision saves
                        memory.
  --text_encoder_4_precision {no_change,int8-quanto,int4-quanto,int2-quanto,int8-torchao,int8dq-torchao,int8dq-int4-torchao,nf4-bnb,int4-torchao,fp8-quanto,fp8uz-quanto,fp8-native,fp8-torchao,fp8wo-torchao,fp8-int4-torchao,fp8-transformerengine}
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
  --validation_audio_only [VALIDATION_AUDIO_ONLY]
                        Disable video generation during validation and emit
                        audio only
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
  --flow_custom_timesteps FLOW_CUSTOM_TIMESTEPS
                        Override flow-matching timestep sampling with a fixed
                        comma-separated list. The list is interpreted as
                        sigmas only when every value is in [0,1]; otherwise
                        all values are interpreted as timesteps [0,1000].
  --flow_timesteps_mode {fixed-list,round-robin}
                        Select how flow_custom_timesteps values are assigned
                        to samples.
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
  --attention_mechanism {diffusers,xformers,flash-attn,flash-attn-2,flash-attn-3,flash-attn-3-varlen,flex,metal-flash-attention,metal-flash-attention-int8,metal-flash-attention-int4,cudnn,native-efficient,native-flash,native-math,native-npu,native-xla,sla,sageattention,sageattention-int8-fp16-triton,sageattention-int8-fp16-cuda,sageattention-int8-fp8-cuda}
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
  --text_embed_full_cache [TEXT_EMBED_FULL_CACHE]
                        Store full raw text encoder outputs in the text embed
                        cache. This opts out of model-specific cache size
                        optimisations, such as Ideogram 4's frozen text
                        projection cache.
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
  --delete_invalid_checkpoints [DELETE_INVALID_CHECKPOINTS]
                        Delete local checkpoints that cannot be loaded while
                        resuming.
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
  --distillation_method {lcm,dcm,dmd,perflow,flow_dpo,anyflow}
                        Method for model distillation
                        Distillation methods cannot be combined with
                        --train_text_encoder.
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
