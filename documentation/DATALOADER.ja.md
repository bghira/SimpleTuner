# Dataloader 設定ファイル

以下は、`multidatabackend.example.json` としての最も基本的なデータローダ設定ファイルの例です。

```json
[
  {
    "id": "something-special-to-remember-by",
    "type": "local",
    "instance_data_dir": "/path/to/data/tree",
    "crop": true,
    "crop_style": "center",
    "crop_aspect": "square",
    "resolution": 1024,
    "minimum_image_size": 768,
    "maximum_image_size": 2048,
    "minimum_aspect_ratio": 0.50,
    "maximum_aspect_ratio": 3.00,
    "target_downsample_size": 1024,
    "resolution_type": "pixel_area",
    "prepend_instance_prompt": false,
    "instance_prompt": "something to label every image",
    "only_instance_prompt": false,
    "caption_strategy": "textfile",
    "cache_dir_vae": "/path/to/vaecache",
    "repeats": 0
  },
  {
    "id": "an example backend for text embeds.",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "aws",
    "aws_bucket_name": "textembeds-something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir": ""
  }
]
```

## 設定オプション

### `id`

- **説明:** データセットの一意な識別子です。設定後は固定し、状態追跡エントリとの紐付けに使います。

### `disabled`

- **値:** `true` | `false`
- **説明:** `true` にすると、このデータセットは学習中に完全にスキップされます。設定を削除せずに一時的に除外したい場合に便利です。
- **注記:** 綴り `disable` も受け付けます。

### `dataset_type`

- **値:** `image` | `video` | `audio` | `text_embeds` | `image_embeds` | `conditioning_image_embeds` | `conditioning`
- **説明:** `image`、`video`、`audio` は主要な学習サンプルを含むデータセットです。`text_embeds` はテキストエンコーダのキャッシュ出力、`image_embeds` は VAE 潜在（モデルが使用する場合）を保持し、`conditioning_image_embeds` は条件画像埋め込み（例: Wan 2.2 I2V の CLIP ビジョン特徴）をキャッシュします。データセットを `conditioning` に設定した場合、[conditioning_data オプション](#conditioning_data) を介して `image` データセットと関連付けられます。
- **注記:** テキスト埋め込みと画像埋め込みのデータセットは、画像データセットとは異なる定義です。テキスト埋め込みデータセットはテキスト埋め込みオブジェクトのみを保存し、画像データセットは学習データを保存します。
- **注記:** 画像と動画を**同一**データセットに混在させないでください。分けてください。

### `default`

- **`dataset_type=text_embeds` のみに適用**
- `true` にすると、このテキスト埋め込みデータセットが SimpleTuner のテキスト埋め込みキャッシュ（例: 検証用プロンプト埋め込み）の保存先になります。画像データと対にならないため、専用の保存先が必要です。

### `cache_dir`

- **`dataset_type=text_embeds` と `dataset_type=image_embeds` のみに適用**
- **説明:** このデータセットの埋め込みキャッシュファイルの保存先を指定します。`text_embeds` はテキストエンコーダ出力、`image_embeds` は VAE 潜在の保存先です。
- **注記:** 主となる画像/動画データセットで VAE キャッシュの保存先を指定する `cache_dir_vae` とは異なります。

### `write_batch_size`

- **`dataset_type=text_embeds` のみに適用**
- **説明:** 1 回のバッチ操作で書き込むテキスト埋め込み数。値を大きくすると書き込みスループットは向上しますが、メモリ使用量が増えます。
- **既定値:** トレーナーの `--write_batch_size` 引数（通常は 128）にフォールバックします。

### `text_embeds`

- **`dataset_type=image` のみに適用**
- 未設定の場合は `default` の text_embeds データセットが使われます。既存の `text_embeds` データセットの `id` を設定すると、それが使われます。特定のテキスト埋め込みデータセットを画像データセットに関連付けられます。

### `image_embeds`

- **`dataset_type=image` のみに適用**
- 未設定の場合は VAE 出力が画像バックエンドに保存されます。既存の `image_embeds` データセットの `id` を設定すると、VAE 出力はそちらに保存されます。画像埋め込みデータセットを画像データに関連付けられます。

### `conditioning_image_embeds`

- **`dataset_type=image` と `dataset_type=video` に適用**
- モデルが `requires_conditioning_image_embeds` を報告する場合、`conditioning_image_embeds` データセットの `id` を設定して条件画像埋め込みキャッシュ（例: Wan 2.2 I2V の CLIP ビジョン特徴）を保存します。未設定の場合、SimpleTuner は既定で `cache/conditioning_image_embeds/<dataset_id>` にキャッシュを書き込み、VAE キャッシュと衝突しないようにします。
- これらの埋め込みが必要なモデルは、主要パイプライン経由で画像エンコーダを公開する必要があります。モデルがエンコーダを提供できない場合、前処理は空ファイルを黙って生成するのではなく早期に失敗します。

#### `cache_dir_conditioning_image_embeds`

- **条件画像埋め込みキャッシュの保存先を上書きするオプションです。**
- キャッシュを特定のファイルシステムに固定したい場合や、専用のリモートバックエンド（`dataset_type=conditioning_image_embeds`）を使う場合に設定します。省略時は上記の既定パスが自動的に使われます。

#### `conditioning_image_embed_batch_size`

- **条件画像埋め込みを生成する際のバッチサイズを上書きするオプションです。**
- 明示的に指定しない場合、`conditioning_image_embed_batch_size` トレーナー引数、または VAE バッチサイズにフォールバックします。

### 音声データセット設定 (`dataset_type=audio`)

音声バックエンドは専用の `audio` ブロックをサポートし、メタデータとバケット計算が再生時間に合わせて行われます。例:

```json
"audio": {
  "max_duration_seconds": 90,
  "channels": 2,
  "bucket_strategy": "duration",
  "duration_interval": 15,
  "truncation_mode": "beginning"
}
```

- **`bucket_strategy`** – 現在は `duration` が既定で、クリップを等間隔のバケットに切り詰め、GPU ごとのサンプリングがバッチ計算に合うようにします。
- **`duration_interval`** – バケット丸めを秒単位で指定します（未設定時の既定は **3**）。`15` にすると、77 秒のクリップは 75 秒に丸められます。これにより、単一の長いクリップが他のランクを阻害するのを防ぎ、同じ間隔で切り詰められます。
- **`max_duration_seconds`** – これを超える長さのクリップはメタデータ探索時に完全にスキップされるため、極端に長いトラックがバケットを予期せず消費しません。
- **`truncation_mode`** – バケット間隔に揃える際に保持するクリップの部分を決めます。選択肢: `beginning`、`end`、`random`（既定: `beginning`）。
- **`audio_only`** – 音声のみトレーニングモード（LTX-2）: 動画ファイルなしで音声生成のみをトレーニングします。動画潜在変数は自動的にゼロになり、動画損失はマスクされます。
- **`target_resolution`** – 音声のみモードでのターゲット動画解像度（潜在変数の次元計算に使用）。
- 標準の音声設定（チャンネル数、キャッシュディレクトリなど）は `simpletuner.helpers.data_backend.factory` によって作成されるランタイム音声バックエンドに直接マッピングされます。パディングは意図的に回避され、クリップは延長ではなく切り詰められるため、ACE-Step のような拡散トレーナーの挙動と整合します。

### 音声キャプション（Hugging Face）
Hugging Face の音声データセットでは、キャプション（プロンプト）を構成する列と歌詞を含む列を指定できます:
```json
"config": {
    "audio_caption_fields": ["prompt", "tags"],
    "lyrics_column": "lyrics"
}
```
*   `audio_caption_fields`: 複数の列を結合してテキストプロンプトを作成します（テキストエンコーダで使用）。
*   `lyrics_column`: 歌詞の列を指定します（歌詞エンコーダで使用）。

メタデータ探索中にローダは各ファイルの `sample_rate`、`num_samples`、`num_channels`、`duration_seconds` を記録します。CLI のバケットレポートは **images** ではなく **samples** を基準に出力され、空のデータセット診断では有効な `bucket_strategy`/`duration_interval`（および `max_duration_seconds` 制限）が表示されるため、ログを深掘りせずに間隔調整できます。

### `type`

- **値:** `aws` | `local` | `csv` | `huggingface`
- **説明:** このデータセットに使用するストレージバックエンド（ローカル、csv、またはクラウド）を決定します。

### `conditioning_type`

- **値:** `controlnet` | `mask` | `reference_strict` | `reference_loose`
- **説明:** `conditioning` データセットの条件付けの種類を指定します。
  - **controlnet**: コントロール信号学習用の ControlNet 条件入力。
  - **mask**: インペインティング学習用のバイナリマスク。
  - **reference_strict**: 厳密なピクセル整合の参照画像（Qwen Edit などの編集モデル向け）。
  - **reference_loose**: 緩い整合の参照画像。

### `source_dataset_id`

- **`dataset_type=conditioning` のみに適用**（`conditioning_type` が `reference_strict`、`reference_loose`、`mask` の場合）
- **説明:** 条件データセットをソースの画像/動画データセットに紐付け、ピクセル整合を取ります。設定すると、SimpleTuner はソースデータセットのメタデータを複製し、条件画像がターゲットと整合するようにします。
- **注記:** 厳密な整合モードでは必須、緩い整合では任意です。

### `conditioning_data`

- **値:** 条件データセットの `id` 値、または `id` 値の配列
- **説明:** [ControlNet ガイド](CONTROLNET.md) の説明どおり、`image` データセットはこのオプションを介して ControlNet または画像マスクデータに紐付けられます。
- **注記:** 複数の条件データセットがある場合は `id` の配列を指定できます。Flux Kontext の学習時には、条件をランダムに切り替えたり入力を結合したりして、より高度な複数画像合成タスクの学習が可能です。

### `instance_data_dir` / `aws_data_prefix`

- **Local:** ファイルシステム上のデータパス。
- **AWS:** バケット内のデータに対する S3 プレフィックス。

### `caption_strategy`

- **textfile** は、image.png の隣に改行区切りで 1 つ以上のキャプションを含む image.txt があることを前提とします。これらの画像+テキストのペアは**同じディレクトリ**にある必要があります。
- **instanceprompt** は `instance_prompt` の値が必要で、その値のみをセット内のすべての画像のキャプションとして使用します。
- **filename** は、ファイル名を変換・整形した文字列（例: アンダースコアを空白に置換）をキャプションとして使用します。
- **parquet** は画像メタデータを含む parquet テーブルからキャプションを取得します。`parquet` フィールドで設定してください。[Parquet caption strategy](#parquet-caption-strategy-json-lines-datasets) を参照。

`textfile` と `parquet` は複数キャプションに対応します:
- textfile は改行で分割され、各行が別キャプションになります。
- parquet テーブルはフィールドに iterable 型を持てます。

### `disable_multiline_split`

- `true` に設定すると、キャプションテキストファイルが改行で複数のキャプションバリアントに分割されなくなります。
- 意図的な改行を含むキャプションを単一のキャプションとして保持したい場合に便利です。
- デフォルト: `false`（改行でキャプションを分割）

### `metadata_backend`

- **値:** `discovery` | `parquet` | `huggingface`
- **説明:** SimpleTuner がデータセット準備中に画像サイズなどのメタデータをどのように取得するかを制御します。
  - **discovery**（既定）: 実画像ファイルをスキャンして寸法を読み取ります。あらゆるストレージバックエンドで動作しますが、大規模データセットでは遅くなります。
  - **parquet**: parquet/JSONL ファイルの `width_column` と `height_column` から寸法を読み取り、ファイルアクセスを省略します。[Parquet caption strategy](#parquet-caption-strategy-json-lines-datasets) を参照。
  - **huggingface**: Hugging Face データセットのメタデータを使用します。[Hugging Face Datasets Support](#hugging-face-datasets-support) を参照。
- **注記:** `parquet` を使う場合は `width_column` と `height_column` を含む `parquet` ブロックの設定も必要です。これにより大規模データセットの起動が大幅に高速化されます。

### `metadata_update_interval`

- **値:** 整数（秒）
- **説明:** 学習中にデータセットメタデータを再取得する間隔（秒）。長時間の学習でデータセットが変更される可能性がある場合に有用です。
- **既定値:** トレーナーの `--metadata_update_interval` 引数にフォールバックします。

### クロップオプション

- `crop`: 画像クロップを有効/無効にします。
- `crop_style`: クロップ方法を選択します（`random`、`center`、`corner`、`face`）。
- `crop_aspect`: クロップのアスペクト比を選択します（`closest`、`random`、`square`、`preserve`）。
- `crop_aspect_buckets`: `crop_aspect` が `closest` または `random` の場合、このリストからバケットが選択されます。既定ではすべてのバケットが利用可能です（無制限のアップスケールを許可）。必要に応じて `max_upscale_threshold` でアップスケールを制限します。

### `resolution`

- **resolution_type=area:** 最終的な画像サイズはメガピクセル数で決まります。ここで 1.05 を指定すると、1024^2（1024x1024）相当の総ピクセル面積、約 1_050_000 ピクセルのアスペクトバケットになります。
- **resolution_type=pixel_area:** `area` と同様に面積で決まりますが、メガピクセルではなくピクセルで測定します。ここで 1024 を指定すると、1024^2（1024x1024）相当の総ピクセル面積、約 1_050_000 ピクセルのアスペクトバケットになります。
- **resolution_type=pixel:** 最終的な画像サイズは短辺がこの値になるように決定されます。

> **注記**: 画像がアップスケール/ダウンスケール/クロップされるかどうかは、`minimum_image_size`、`maximum_target_size`、`target_downsample_size`、`crop`、`crop_aspect` の値に依存します。

### `minimum_image_size`

- サイズがこの値を下回る画像は学習から**除外**されます。
- `resolution` をメガピクセルで指定する場合（`resolution_type=area`）、ここもメガピクセルで指定します（例: 1024x1024 **面積**未満を除外するなら `1.05`）。
- `resolution` をピクセルで指定する場合は同じ単位を使います（例: 1024px **短辺長**未満を除外するなら `1024`）。
- **推奨:** 品質の低いアップスケール画像の学習を避けたい場合を除き、`minimum_image_size` は `resolution` と同じにしてください。

### `minimum_aspect_ratio`

- **説明:** 画像の最小アスペクト比です。これより小さいアスペクト比の画像は学習から除外されます。
- **注記**: 除外対象が多すぎると、トレーナーがスキャンとバケット作成を試みるため起動時に時間を浪費する可能性があります。

> **注記**: データセットのアスペクトおよびメタデータリストが作成済みであれば、`skip_file_discovery="vae aspect metadata"` を使うことで起動時のスキャンを防ぎ、大幅に時間を節約できます。

### `maximum_aspect_ratio`

- **説明:** 画像の最大アスペクト比です。これより大きいアスペクト比の画像は学習から除外されます。
- **注記**: 除外対象が多すぎると、トレーナーがスキャンとバケット作成を試みるため起動時に時間を浪費する可能性があります。

> **注記**: データセットのアスペクトおよびメタデータリストが作成済みであれば、`skip_file_discovery="vae aspect metadata"` を使うことで起動時のスキャンを防ぎ、大幅に時間を節約できます。

### `conditioning`

- **値:** 条件付け設定オブジェクトの配列
- **説明:** ソース画像から条件データセットを自動生成します。各条件タイプは ControlNet 学習などに使用できる独立したデータセットを作成します。
- **注記:** 指定すると、SimpleTuner は `{source_id}_conditioning_{type}` のような ID を持つ条件データセットを自動的に作成します。

各条件オブジェクトに含められる項目:
- `type`: 生成する条件の種類（必須）
- `params`: 種類固有のパラメータ（任意）
- `captions`: 生成データセットのキャプション戦略（任意）
  - `false` を指定可能（キャプションなし）
  - 単一の文字列（すべての画像に対する instance prompt として使用）
  - 文字列の配列（各画像でランダムに選択）
  - 省略時はソースデータセットのキャプションを使用

#### 利用可能な条件タイプ

##### `superresolution`
超解像学習向けに低品質版の画像を生成します:
```json
{
  "type": "superresolution",
  "blur_radius": 2.5,
  "blur_type": "gaussian",
  "add_noise": true,
  "noise_level": 0.03,
  "jpeg_quality": 85,
  "downscale_factor": 2
}
```

##### `jpeg_artifacts`
アーティファクト除去学習向けに JPEG 圧縮アーティファクトを作成します:
```json
{
  "type": "jpeg_artifacts",
  "quality_mode": "range",
  "quality_range": [10, 30],
  "compression_rounds": 1,
  "enhance_blocks": false
}
```

##### `depth` / `depth_midas`
DPT モデルを使って深度マップを生成します:
```json
{
  "type": "depth_midas",
  "model_type": "DPT"
}
```
**注記:** 深度生成には GPU が必要で、メインプロセスで動作します。CPU ベースのジェネレータより遅くなる場合があります。

##### `random_masks` / `inpainting`
インペインティング学習向けにランダムマスクを作成します:
```json
{
  "type": "random_masks",
  "mask_types": ["rectangle", "circle", "brush", "irregular"],
  "min_coverage": 0.1,
  "max_coverage": 0.5,
  "output_mode": "mask"
}
```

##### `canny` / `edges`
Canny エッジ検出マップを生成します:
```json
{
  "type": "canny",
  "low_threshold": 100,
  "high_threshold": 200
}
```

これらの条件データセットの使い方の詳細は [ControlNet ガイド](CONTROLNET.md) を参照してください。

#### 例

##### 動画データセット

動画データセットは（例: mp4）動画ファイルのフォルダと、通常のキャプション保存方法から構成されます。

```json
[
  {
    "id": "disney-black-and-white",
    "type": "local",
    "dataset_type": "video",
    "crop": false,
    "resolution": 480,
    "minimum_image_size": 480,
    "maximum_image_size": 480,
    "target_downsample_size": 480,
    "resolution_type": "pixel_area",
    "cache_dir_vae": "cache/vae/ltxvideo/disney-black-and-white",
    "instance_data_dir": "datasets/disney-black-and-white",
    "disabled": false,
    "caption_strategy": "textfile",
    "metadata_backend": "discovery",
    "repeats": 0,
    "video": {
        "num_frames": 125,
        "min_frames": 125
    }
  },
  {
    "id": "text-embeds",
    "type": "local",
    "dataset_type": "text_embeds",
    "default": true,
    "cache_dir": "cache/text/ltxvideo",
    "disabled": false,
    "write_batch_size": 128
  }
]
```

- `video` サブセクションでは次のキーを設定できます:
  - `num_frames`（任意、int）は学習に使うフレーム数です。
    - 25 fps の場合、125 フレームは 5 秒で標準的な出力です。これを目標にしてください。
  - `min_frames`（任意、int）は学習対象として考慮する最小動画長を指定します。
    - `num_frames` 以上である必要があります。未設定の場合は同じ値になります。
  - `max_frames`（任意、int）は学習対象として考慮する最大動画長を指定します。
  - `is_i2v`（任意、bool）はデータセットで i2v 学習を行うかどうかを指定します。
    - LTX では既定で True に設定されていますが、無効化できます。
  - `bucket_strategy`（任意、string）は動画のバケット分け方法を指定します:
    - `aspect_ratio`（既定）: 空間アスペクト比のみでバケット化（例: `1.78`、`0.75`）。画像データセットと同じ挙動です。
    - `resolution_frames`: 解像度とフレーム数を `WxH@F` 形式（例: `1920x1080@125`）でバケット化します。解像度や長さが混在するデータセットの学習に有用です。
  - `frame_interval`（任意、int）: `bucket_strategy: "resolution_frames"` の場合、フレーム数はこの値の最も近い倍数に切り下げられます。モデルの必要とするフレーム数の係数に合わせて設定してください（モデルによっては `num_frames - 1` が特定の値で割り切れる必要があります）。

**自動フレーム数調整:** SimpleTuner は、モデル固有の制約を満たすように動画のフレーム数を自動的に調整します。例えば、LTX-2 は `frames % 8 == 1` を満たすフレーム数（例: 49、57、65、73、81 など）を必要とします。動画のフレーム数が異なる場合（例: 119 フレーム）、最も近い有効なフレーム数に自動的にトリミングされます（例: 113 フレーム）。調整後に `min_frames` より短くなる動画は警告メッセージとともにスキップされます。この自動調整は学習エラーを防ぎ、ユーザー側での設定は不要です。

**注記:** `bucket_strategy: "resolution_frames"` を `num_frames` と併用すると、単一のフレームバケットになり、`num_frames` より短い動画は破棄されます。より多くのフレームバケットと破棄の削減が必要なら `num_frames` を未設定にしてください。

混在解像度の動画データセットで `resolution_frames` バケットを使う例:

```json
{
  "id": "mixed-resolution-videos",
  "type": "local",
  "dataset_type": "video",
  "resolution": 720,
  "resolution_type": "pixel_area",
  "instance_data_dir": "datasets/videos",
  "video": {
      "bucket_strategy": "resolution_frames",
      "frame_interval": 25,
      "min_frames": 25,
      "max_frames": 250
  }
}
```

この設定では `1280x720@100`、`1920x1080@125`、`640x480@75` などのバケットが作成されます。動画は学習解像度とフレーム数（最も近い 25 フレームに丸め）でグループ化されます。


##### Configuration
```json
    "minimum_image_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel"
```
##### Outcome
- 短辺が **1024px** 未満の画像はすべて学習から除外されます。
- `768x1024` や `1280x768` は除外されますが、`1760x1024` と `1024x1024` は除外されません。
- `minimum_image_size` が `resolution` と同じため、アップサンプリングは行われません。

##### Configuration
```json
    "minimum_image_size": 1024,
    "resolution": 1024,
    "resolution_type": "pixel_area" # different from the above configuration, which is 'pixel'
```
##### Outcome
- 画像の総面積（幅 * 高さ）が最小面積（1024 * 1024）未満の場合、学習から除外されます。
- `1280x960` のような画像は `(1280 * 960)` が `(1024 * 1024)` を上回るため**除外されません**。
- `minimum_image_size` が `resolution` と同じため、アップサンプリングは行われません。

##### Configuration
```json
    "minimum_image_size": 0, # or completely unset, not present in the config
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": false
```

##### Outcome
- アスペクト比を維持したまま短辺が 1024px になるようにリサイズされます。
- サイズに基づいて除外される画像はありません。
- 小さい画像は `PIL.resize` の単純な方法でアップスケールされ、見た目は良くありません。
  - 学習前に好みのアップスケーラーで手動拡大するのでなければ、アップスケールは避けることを推奨します。

### `maximum_image_size` と `target_downsample_size`

`maximum_image_size` と `target_downsample_size` の両方が設定されている場合のみ、クロップ前に画像がリサイズされます。つまり、`4096x4096` の画像は直接 `1024x1024` へクロップされるため、望ましくない場合があります。

- `maximum_image_size` はリサイズを開始する閾値を指定します。これを超える画像はクロップ前にダウンサンプリングされます。
- `target_downsample_size` はリサンプル後、クロップ前の画像サイズを指定します。

#### 例

##### Configuration
```json
    "resolution_type": "pixel_area",
    "resolution": 1024,
    "maximum_image_size": 1536,
    "target_downsample_size": 1280,
    "crop": true,
    "crop_aspect": "square"
```

##### Outcome
- ピクセル面積が `(1536 * 1536)` を超える画像は、元のアスペクト比を維持したまま `(1280 * 1280)` 程度の面積になるようにリサイズされます。
- 最終画像サイズは `(1024 * 1024)` のピクセル面積にランダムクロップされます。
- 例: 20 メガピクセルのデータセットを学習する際、クロップ前に大きくリサイズすることで、人物がタイル壁や背景のぼやけた一部だけになるような過度のコンテキスト損失を避けられます。

### `max_upscale_threshold`

既定では、SimpleTuner は小さな画像を目標解像度に合わせてアップスケールするため、画質劣化が起こり得ます。`max_upscale_threshold` オプションを使うと、このアップスケール動作を制限できます。

- **既定値**: `null`（無制限のアップスケールを許可）
- **設定時**: 指定した閾値を超えるアップスケールが必要なアスペクトバケットを除外します
- **値の範囲**: 0〜1（例: `0.2` = 最大 20% のアップスケールを許可）
- **適用範囲**: `crop_aspect` が `closest` または `random` のときのアスペクトバケット選択

#### 例

##### Configuration
```json
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": true,
    "crop_aspect": "random",
    "crop_aspect_buckets": [1.0, 0.5, 2.0],
    "max_upscale_threshold": null
```

##### Outcome
- すべてのアスペクトバケットが選択可能
- 256x256 の画像を 1024x1024 にアップスケール可能（4 倍）
- 非常に小さな画像では画質劣化が起こり得ます

##### Configuration
```json
    "resolution": 1024,
    "resolution_type": "pixel",
    "crop": true,
    "crop_aspect": "random",
    "crop_aspect_buckets": [1.0, 0.5, 2.0],
    "max_upscale_threshold": 0.2
```

##### Outcome
- 20% 以内のアップスケールが必要なアスペクトバケットのみ利用可能
- 256x256 の画像を 1024x1024 に拡大しようとすると（4 倍 = 300% のアップスケール）、利用可能なバケットがありません
- 850x850 の画像は 1024/850 ≈ 1.2（20% のアップスケール）のためすべてのバケットを使用可能
- 低品質なアップスケール画像を除外し、学習品質の維持に役立ちます

---

### `prepend_instance_prompt`

- 有効化すると、すべてのキャプションの先頭に `instance_prompt` の値が付加されます。

### `only_instance_prompt`

- `prepend_instance_prompt` に加えて、データセット内のすべてのキャプションを単一のフレーズまたはトリガーワードで置き換えます。

### `repeats`

- エポック中にデータセット内のすべてのサンプルが表示される回数を指定します。小さなデータセットの影響を強めたり、VAE キャッシュオブジェクトの使用率を最大化したりするのに有用です。
- 画像数が 1000 のデータセットと 100 のデータセットがある場合、後者には **9 以上**の repeats を設定して合計 1000 画像相当にするのが一般的です。

> ℹ️ この値は Kohya スクリプトの同名オプションとは挙動が異なり、Kohya では 1 が繰り返しなしですが、**SimpleTuner では 0 が繰り返しなし**です。Kohya の値から 1 を引いた値が SimpleTuner に相当します。そのため `(dataset_length + repeats * dataset_length)` の計算で **9** という値になります。

#### マルチ GPU 学習とデータセットサイズ

複数 GPU で学習する場合、データセットは次の **実効バッチサイズ** を満たすだけの大きさが必要です:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

たとえば、GPU が 4 枚、`train_batch_size=4`、`gradient_accumulation_steps=1` の場合、各アスペクトバケットには（repeats 適用後で）最低 **16 サンプル**が必要です。

**重要:** データセット設定から使用可能なバッチが 0 になる場合、SimpleTuner はエラーを出します。エラーメッセージには以下が表示されます:
- 現在の設定値（バッチサイズ、GPU 数、repeats）
- サンプル数が不足しているアスペクトバケット
- 各バケットに必要な最小 repeats
- 推奨される解決策
- 推奨される解決策

##### データセットの自動オーバーサブスクリプション

データセットが実効バッチサイズより小さい場合に `repeats` を自動調整するには、`--allow_dataset_oversubscription` フラグを使用します（[OPTIONS.md](OPTIONS.md#--allow_dataset_oversubscription) 参照）。

有効化すると、SimpleTuner は次を行います:
- 学習に必要な最小 repeats を計算
- 要件を満たすように `repeats` を自動的に増加
- 調整内容を警告ログで出力
- **手動設定された repeats を尊重** - データセット設定で明示的に `repeats` を設定している場合、自動調整はスキップされます

特に次のような場合に有用です:
- 小さなデータセット（100 枚未満）を学習する場合
- GPU 数が多くデータセットが小さい場合
- データセット設定を変更せずにバッチサイズを試行したい場合

**例:**
- データセット: 25 枚
- 設定: GPU 8 枚、`train_batch_size=4`、`gradient_accumulation_steps=1`
- 実効バッチサイズ: 32 サンプル必要
- オーバーサブスクリプションなし: エラー
- `--allow_dataset_oversubscription` あり: repeats が自動的に 1 へ設定（25 × 2 = 50 サンプル）

### `max_num_samples`

- **説明：** データセットの最大サンプル数を制限します。設定すると、完全なデータセットから指定されたサイズの決定論的なランダムサブセットが選択されます。
- **使用例：** 大規模な正則化データセットで、小さなトレーニングセットを圧倒しないようにデータの一部のみを使用したい場合に便利です。
- **決定論的選択：** ランダム選択はデータセット `id` をシードとして使用し、再現性のためにトレーニングセッション間で同じサブセットが選択されることを保証します。
- **デフォルト：** `null`（制限なし、すべてのサンプルを使用）

#### 例
```json
{
  "id": "regularization-data",
  "max_num_samples": 1000,
  ...
}
```

これにより、データセットから 1000 サンプルが決定論的に選択され、トレーニングを実行するたびに同じ選択が使用されます。

### `start_epoch` / `start_step`

- データセットのサンプリング開始タイミングをスケジュールします。
- `start_epoch`（既定: `1`）はエポック番号でゲートし、`start_step`（既定: `0`）は最適化ステップ（勾配蓄積後）でゲートします。両方の条件が満たされるまでサンプルは取得されません。
- 少なくとも 1 つのデータセットは `start_epoch<=1` **かつ** `start_step<=1` を満たす必要があります。そうでない場合、起動時にデータがないため学習がエラーになります。
- 開始条件を満たさないデータセット（例: `start_epoch` が `--num_train_epochs` を超える）はスキップされ、モデルカードに記載されます。
- 進行中にスケジュールされたデータセットが有効になるとエポック長が増えるため、進捗バーのステップ見積もりは概算になります。

### `is_regularisation_data`

- `is_regularization_data` と綴ることもできます。
- LyCORIS アダプタ向けに親教師学習を有効化し、指定データセットに対してベースモデルの結果を優先する予測ターゲットにします。
  - 標準 LoRA は現在サポートされていません。

### `delete_unwanted_images`

- **値:** `true` | `false`
- **説明:** 有効化すると、サイズやアスペクト比のフィルタ（例: `minimum_image_size` 未満、`minimum_aspect_ratio`/`maximum_aspect_ratio` の範囲外）に失敗した画像がデータセットディレクトリから完全に削除されます。
- **警告:** 破壊的で元に戻せません。注意して使用してください。
- **既定値:** トレーナーの `--delete_unwanted_images` 引数（既定: `false`）にフォールバックします。

### `delete_problematic_images`

- **値:** `true` | `false`
- **説明:** 有効化すると、VAE エンコードで失敗した画像（破損ファイル、非対応形式など）がデータセットディレクトリから完全に削除されます。
- **警告:** 破壊的で元に戻せません。注意して使用してください。
- **既定値:** トレーナーの `--delete_problematic_images` 引数（既定: `false`）にフォールバックします。

### `slider_strength`

- **値:** 任意の浮動小数値（正、負、または 0）
- **説明:** データセットをスライダー LoRA 学習に指定し、対照的な「反対概念」を学習して制御可能なコンセプトアダプタを作成します。
  - **正の値**（例: `0.5`）: 「概念を強める」— 目を明るくする、笑顔を強くする等。
  - **負の値**（例: `-0.5`）: 「概念を弱める」— 目を暗くする、表情を中立にする等。
  - **0 または未設定**: 概念をどちらにも押し出さない中立例。
- **注記:** データセットに `slider_strength` がある場合、SimpleTuner はバッチを固定サイクル（正 → 負 → 中立）で回します。各グループ内では標準のバックエンド確率が引き続き適用されます。
- **参照:** スライダー LoRA 学習の完全なガイドは [SLIDER_LORA.md](SLIDER_LORA.md) を参照してください。

### `vae_cache_clear_each_epoch`

- 有効化すると、各データセットの繰り返しサイクルの終了時に VAE キャッシュオブジェクトがファイルシステムから削除されます。大規模データセットでは負荷が大きいですが、`crop_style=random` や `crop_aspect=random` と併用する場合は、各画像から十分なクロップをサンプリングするために有効化することを推奨します。
- 実際、ランダムバケットやランダムクロップを使う場合は**既定で有効**になります。

### `vae_cache_disable`

- **値:** `true` | `false`
- **説明:** 有効化すると（コマンドライン引数 `--vae_cache_disable` 経由）、オンデマンド VAE キャッシュが暗黙的に有効になり、生成された埋め込みのディスク書き込みは無効になります。ディスク容量が懸念される大規模データセットや書き込みが難しい環境に有用です。
- **注記:** これはトレーナーレベルの引数でありデータセット単位の設定ではありませんが、データローダが VAE キャッシュとやり取りする方法に影響します。

### `skip_file_discovery`

- ほとんどの場合は設定不要で、非常に大規模なデータセットでのみ有用です。
- このパラメータは、例: `vae metadata aspect text` のようにカンマまたはスペース区切りの値を受け取り、ローダ設定の 1 つ以上の段階でファイル探索をスキップします。
- コマンドラインオプション `--skip_file_discovery` と同等です。
- すでに潜在/埋め込みが完全にキャッシュされているなど、起動時にスキャンしたくないデータセットがある場合に有用で、起動と学習再開が速くなります。

### `preserve_data_backend_cache`

- ほとんどの場合は設定不要で、非常に大規模な AWS データセットでのみ有用です。
- `skip_file_discovery` と同様に、不要で長時間かつ高コストなファイルシステムスキャンを防ぐために使えます。
- boolean 値を取り、`true` にすると生成されたファイルシステム一覧キャッシュが起動時に削除されません。
- S3 やローカルの SMR スピニングディスクなど、応答が非常に遅いストレージに対して有用です。
- さらに S3 ではバックエンドのリスト取得がコストになるため、避けるべきです。

> ⚠️ **データが継続的に変更されている場合は設定できません。** トレーナーはプールに追加された新しいデータを認識できず、再度フルスキャンが必要になります。

### `hash_filenames`

- VAE キャッシュエントリのファイル名は常にハッシュ化されます。これはユーザー側で変更できず、非常に長いファイル名でもパス長の問題を回避できます。設定に `hash_filenames` があっても無視されます。

## キャプションのフィルタリング

### `caption_filter_list`

- **テキスト埋め込みデータセットのみ対象です。** JSON リスト、txt ファイルへのパス、または JSON ドキュメントへのパスを指定できます。フィルタ文字列はキャプションから削除する単純語句でも正規表現でも構いません。さらに、sed 形式の `s/search/replace/` エントリを使うと、削除ではなくキャプション内文字列の**置換**ができます。

#### フィルタリストの例

完全な例は [こちら](/config/caption_filter_list.txt.example) にあります。BLIP（一般版）、LLaVA、CogVLM が返しがちな反復的・ネガティブな文字列が含まれています。

以下は短い例で、後で説明します:

```
arafed
this .* has a
^this is the beginning of the string
s/this/will be found and replaced/
```

各行の動作は次のとおりです:

- `arafed `（末尾にスペースあり）は、見つかったキャプションから削除されます。末尾スペースを含めると二重スペースが残らず、見た目が良くなります。必須ではありませんが、見栄えが良くなります。
- `this .* has a` は正規表現で、「this ... has a」を含む文字列を削除します。`.*` は「一致する範囲すべて」を意味し、「has a」に到達した時点で停止します。
- `^this is the beginning of the string` は、キャプションの先頭にある場合に限り「this is the beginning of the string」を削除します。
- `s/this/will be found and replaced/` は、キャプション内の最初の「this」を「will be found and replaced」に置換します。

> ❗正規表現のデバッグとテストには [regex 101](https://regex101.com) を使用してください。

# 高度なテクニック

## 高度な設定例

```json
[
  {
    "id": "something-special-to-remember-by",
    "type": "local",
    "instance_data_dir": "/path/to/data/tree",
    "crop": false,
    "crop_style": "random|center|corner|face",
    "crop_aspect": "square|preserve|closest|random",
    "crop_aspect_buckets": [0.33, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    "resolution": 1.0,
    "resolution_type": "area|pixel",
    "minimum_image_size": 1.0,
    "prepend_instance_prompt": false,
    "instance_prompt": "something to label every image",
    "only_instance_prompt": false,
    "caption_strategy": "filename|instanceprompt|parquet|textfile",
    "disable_multiline_split": false,
    "cache_dir_vae": "/path/to/vaecache",
    "vae_cache_clear_each_epoch": true,
    "probability": 1.0,
    "repeats": 0,
    "start_epoch": 1,
    "start_step": 0,
    "text_embeds": "alt-embed-cache",
    "image_embeds": "vae-embeds-example",
    "conditioning_image_embeds": "conditioning-embeds-example"
  },
  {
    "id": "another-special-name-for-another-backend",
    "type": "aws",
    "aws_bucket_name": "something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir_vae": "s3prefix/for/vaecache",
    "vae_cache_clear_each_epoch": true,
    "repeats": 0
  },
  {
      "id": "vae-embeds-example",
      "type": "local",
      "dataset_type": "image_embeds",
      "disabled": false,
  },
  {
      "id": "conditioning-embeds-example",
      "type": "local",
      "dataset_type": "conditioning_image_embeds",
      "disabled": false
  },
  {
    "id": "an example backend for text embeds.",
    "dataset_type": "text_embeds",
    "default": true,
    "type": "aws",
    "aws_bucket_name": "textembeds-something-yummy",
    "aws_region_name": null,
    "aws_endpoint_url": "https://foo.bar/",
    "aws_access_key_id": "wpz-764e9734523434",
    "aws_secret_access_key": "xyz-sdajkhfhakhfjd",
    "aws_data_prefix": "",
    "cache_dir": ""
  },
  {
    "id": "alt-embed-cache",
    "dataset_type": "text_embeds",
    "default": false,
    "type": "local",
    "cache_dir": "/path/to/textembed_cache"
  }
]
```

## CSV URL リストから直接学習

**注記: CSV には画像のキャプションが含まれている必要があります。**

> ⚠️ これは高度かつ**実験的**な機能であり、問題が発生する可能性があります。その場合は [issue](https://github.com/bghira/simpletuner/issues) を開いてください。

URL リストからデータを手動でダウンロードする代わりに、直接トレーナーへ渡すことができます。

**注記:** 画像データは手動でダウンロードするほうが常に望ましいです。ローカルディスク容量を節約したい場合は、代替策として [クラウドデータセットとローカルエンコーダキャッシュの併用](#local-cache-with-cloud-dataset) を試すこともできます。

### 利点

- データを直接ダウンロードする必要がない
- SimpleTuner のキャプションツールキットを使って URL リストに直接キャプション付けできる
- （該当する場合）画像埋め込みとテキスト埋め込みのみ保存するためディスク容量を節約できる

### 欠点

- 各画像をダウンロードしてメタデータを収集する高コストかつ低速なアスペクトバケットスキャンが必要
- ダウンロード済み画像はオンディスクにキャッシュされ、大きく膨らむ可能性があります。現状のキャッシュ管理は非常に基本的で、書き込みのみ・削除なしのため改善余地があります
- 無効な URL が多い場合、現在は不良サンプルが URL リストから**決して**削除されないため、再開時に時間を浪費する可能性があります
  - **提案:** 事前に URL 検証タスクを実行して不良サンプルを削除してください。

### 設定

必須キー:

- `type: "csv"`
- `csv_caption_column`
- `csv_cache_dir`
- `caption_strategy: "csv"`

```json
[
    {
        "id": "csvtest",
        "type": "csv",
        "csv_caption_column": "caption",
        "csv_file": "/Volumes/ml/dataset/test_list.csv",
        "csv_cache_dir": "/Volumes/ml/cache/csv/test",
        "cache_dir_vae": "/Volumes/ml/cache/vae/sdxl",
        "caption_strategy": "csv",
        "image_embeds": "image-embeds",
        "crop": true,
        "crop_aspect": "square",
        "crop_style": "center",
        "resolution": 1024,
        "maximum_image_size": 1024,
        "target_downsample_size": 1024,
        "resolution_type": "pixel",
        "minimum_image_size": 0,
        "disabled": false,
        "skip_file_discovery": "",
        "preserve_data_backend_cache": false
    },
    {
      "id": "image-embeds",
      "type": "local"
    },
    {
        "id": "text-embeds",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/Volumes/ml/cache/text/sdxl",
        "disabled": false,
        "preserve_data_backend_cache": false,
        "skip_file_discovery": "",
        "write_batch_size": 128
    }
]
```

## Parquet キャプション戦略 / JSON Lines データセット

> ⚠️ これは高度な機能であり、ほとんどのユーザーには不要です。

数十万〜数百万枚規模の非常に大きなデータセットで学習する場合、メタデータを txt ファイルではなく parquet データベースに保存するのが最速です。特に学習データを S3 バケットに保存している場合に有効です。

parquet キャプション戦略を使うと、ファイル名を `id` 値で統一でき、キャプション列を変更する際に多数のテキストファイルを更新したりファイル名を変更したりする必要がありません。

以下は、[photo-concept-bucket](https://huggingface.co/datasets/bghira/photo-concept-bucket) データセットのキャプションとデータを使用するデータローダ設定例です:

```json
{
  "id": "photo-concept-bucket",
  "type": "local",
  "instance_data_dir": "/models/training/datasets/photo-concept-bucket-downloads",
  "caption_strategy": "parquet",
  "metadata_backend": "parquet",
  "parquet": {
    "path": "photo-concept-bucket.parquet",
    "filename_column": "id",
    "caption_column": "cogvlm_caption",
    "fallback_caption_column": "tags",
    "width_column": "width",
    "height_column": "height",
    "identifier_includes_extension": false
  },
  "resolution": 1.0,
  "minimum_image_size": 0.75,
  "maximum_image_size": 2.0,
  "target_downsample_size": 1.5,
  "prepend_instance_prompt": false,
  "instance_prompt": null,
  "only_instance_prompt": false,
  "disable": false,
  "cache_dir_vae": "/models/training/vae_cache/photo-concept-bucket",
  "probability": 1.0,
  "skip_file_discovery": "",
  "preserve_data_backend_cache": false,
  "vae_cache_clear_each_epoch": true,
  "repeats": 1,
  "crop": true,
  "crop_aspect": "closest",
  "crop_style": "random",
  "crop_aspect_buckets": [1.0, 0.75, 1.23],
  "resolution_type": "area"
}
```

この設定では:

- `caption_strategy` は `parquet` に設定されます。
- `metadata_backend` は `parquet` に設定されます。
- 新しいセクション `parquet` を定義する必要があります:
  - `path` は parquet または JSONL ファイルへのパスです。
  - `filename_column` はファイル名を含む列名です。この例では数値の `id` 列を使用しています（推奨）。
  - `caption_column` はキャプションを含む列名です。この例では `cogvlm_caption` 列を使用しています。LAION データセットでは TEXT フィールドに相当します。
  - `width_column` と `height_column` は、実際の画像寸法を示す文字列や int、単一エントリの Series 型などを含む列にできます。これにより実画像にアクセスせずに情報を取得でき、データセット準備時間が大幅に短縮されます。
  - `fallback_caption_column` は、主キャプションが空のときに使うフォールバックキャプション列名（任意）です。この例では `tags` 列を使用しています。
  - `identifier_includes_extension` は、ファイル名列に拡張子が含まれる場合は `true` に設定します。そうでない場合、拡張子は `.png` とみなされます。ファイル名列に拡張子を含めることを推奨します。

> ⚠️ Parquet の対応範囲はキャプションの読み取りに限定されています。画像サンプルは別途データソースに `{id}.png` のファイル名で配置してください。ヒントは [scripts/toolkit/datasets](scripts/toolkit/datasets) ディレクトリのスクリプトを参照してください。

他のデータローダ設定と同様に:

- `prepend_instance_prompt` と `instance_prompt` は通常どおり動作します。
- 学習実行の合間にサンプルのキャプションを更新すると新しい埋め込みがキャッシュされますが、古い（孤立した）ユニットは削除されません。
- データセットに画像が存在しない場合、そのファイル名がキャプションとして使用され、エラーが出力されます。

## クラウドデータセットとローカルキャッシュ

高価なローカル NVMe ストレージの利用を最大化するため、画像ファイル（png、jpg）のみを S3 バケットに置き、ローカルストレージにテキストエンコーダや VAE（該当する場合）の特徴マップをキャッシュする構成が有用です。

この設定例では:

- 画像データは S3 互換バケットに保存
- VAE データは /local/path/to/cache/vae に保存
- テキスト埋め込みは /local/path/to/cache/textencoder に保存

> ⚠️ `resolution` や `crop` など、他のデータセットオプションも忘れずに設定してください。

```json
[
    {
        "id": "data",
        "type": "aws",
        "aws_bucket_name": "text-vae-embeds",
        "aws_endpoint_url": "https://storage.provider.example",
        "aws_access_key_id": "exampleAccessKey",
        "aws_secret_access_key": "exampleSecretKey",
        "aws_region_name": null,
        "cache_dir_vae": "/local/path/to/cache/vae/",
        "caption_strategy": "parquet",
        "metadata_backend": "parquet",
        "parquet": {
            "path": "train.parquet",
            "caption_column": "caption",
            "filename_column": "filename",
            "width_column": "width",
            "height_column": "height",
            "identifier_includes_extension": true
        },
        "preserve_data_backend_cache": false,
        "image_embeds": "vae-embed-storage"
    },
    {
        "id": "vae-embed-storage",
        "type": "local",
        "dataset_type": "image_embeds"
    },
    {
        "id": "text-embed-storage",
        "type": "local",
        "dataset_type": "text_embeds",
        "default": true,
        "cache_dir": "/local/path/to/cache/textencoder/",
        "write_batch_size": 128
    }
]
```

**注記:** `image_embeds` データセットにはデータパスを設定するオプションはありません。これらは画像バックエンドの `cache_dir_vae` で設定します。

### Hugging Face Datasets 対応

SimpleTuner は Hugging Face Hub からデータセットを直接読み込み、ローカルに完全ダウンロードせずに扱えるようになりました。この実験的機能は次の用途に適しています:

- Hugging Face にホストされる大規模データセット
- メタデータや品質評価が組み込まれたデータセット
- ローカルストレージを使わない素早い実験

この機能の詳細なドキュメントは [このドキュメント](HUGGINGFACE_DATASETS.md) を参照してください。

Hugging Face データセットの基本的な使用例としては、データローダ設定で `"type": "huggingface"` を指定します:

```json
{
  "id": "my-hf-dataset",
  "type": "huggingface",
  "dataset_name": "username/dataset-name",
  "caption_strategy": "huggingface",
  "metadata_backend": "huggingface",
  "caption_column": "caption",
  "image_column": "image"
}
```

## アスペクト比と解像度のカスタムマッピング

SimpleTuner の初回起動時、10 進のアスペクト比を目標ピクセルサイズに対応付ける解像度別のマッピング一覧が生成されます。

独自のマッピングを作成すると、トレーナーが自動計算ではなく指定した目標解像度に合わせるようにできます。この機能は誤った設定で重大な影響を与える可能性があるため、自己責任で使用してください。

カスタムマッピングを作成するには:

- 下記の例に従ってファイルを作成する
- ファイル名を `aspect_ratio_map-{resolution}.json` の形式にする
  - `resolution=1.0` / `resolution_type=area` の場合、マッピングファイル名は `aspect_resolution_map-1.0.json` になります
- このファイルを `--output_dir` で指定した場所に配置する
  - チェックポイントや検証画像が保存される場所と同じです
- 追加の設定フラグやオプションは不要です。名前と場所が正しければ自動的に検出されて使用されます。

### マッピング設定例

これは SimpleTuner が生成するアスペクト比マッピングの例です。手動で設定する必要はありませんが、結果の解像度を完全に制御したい場合の出発点として提供されています。

- データセットが 100 万枚以上の画像を含む
- データローダの `resolution` が `1.0` に設定されている
- データローダの `resolution_type` が `area` に設定されている

これは最も一般的な設定で、1 メガピクセルモデル向けに学習可能なアスペクトバケットの一覧です。

```json
{
    "0.07": [320, 4544],    "0.38": [640, 1664],    "0.88": [960, 1088],    "1.92": [1472, 768],    "3.11": [1792, 576],    "5.71": [2560, 448],
    "0.08": [320, 3968],    "0.4": [640, 1600],     "0.89": [1024, 1152],   "2.09": [1472, 704],    "3.22": [1856, 576],    "6.83": [2624, 384],
    "0.1": [320, 3328],     "0.41": [704, 1728],    "0.94": [1024, 1088],   "2.18": [1536, 704],    "3.33": [1920, 576],    "7.0": [2688, 384],
    "0.11": [384, 3520],    "0.42": [704, 1664],    "1.06": [1088, 1024],   "2.27": [1600, 704],    "3.44": [1984, 576],    "8.0": [3072, 384],
    "0.12": [384, 3200],    "0.44": [704, 1600],    "1.12": [1152, 1024],   "2.5": [1600, 640],     "3.88": [1984, 512],
    "0.14": [384, 2688],    "0.46": [704, 1536],    "1.13": [1088, 960],    "2.6": [1664, 640],     "4.0": [2048, 512],
    "0.15": [448, 3008],    "0.48": [704, 1472],    "1.2": [1152, 960],     "2.7": [1728, 640],     "4.12": [2112, 512],
    "0.16": [448, 2816],    "0.5": [768, 1536],     "1.36": [1216, 896],    "2.8": [1792, 640],     "4.25": [2176, 512],
    "0.19": [448, 2304],    "0.52": [768, 1472],    "1.46": [1216, 832],    "3.11": [1792, 576],    "4.38": [2240, 512],
    "0.24": [512, 2112],    "0.55": [768, 1408],    "1.54": [1280, 832],    "3.22": [1856, 576],    "5.0": [2240, 448],
    "0.26": [512, 1984],    "0.59": [832, 1408],    "1.83": [1408, 768],    "3.33": [1920, 576],    "5.14": [2304, 448],
    "0.29": [576, 1984],    "0.62": [832, 1344],    "1.92": [1472, 768],    "3.44": [1984, 576],    "5.71": [2560, 448],
    "0.31": [576, 1856],    "0.65": [832, 1280],    "2.09": [1472, 704],    "3.88": [1984, 512],    "6.83": [2624, 384],
    "0.34": [640, 1856],    "0.68": [832, 1216],    "2.18": [1536, 704],    "4.0": [2048, 512],     "7.0": [2688, 384],
    "0.38": [640, 1664],    "0.74": [896, 1216],    "2.27": [1600, 704],    "4.12": [2112, 512],    "8.0": [3072, 384],
    "0.4": [640, 1600],     "0.83": [960, 1152],    "2.5": [1600, 640],     "4.25": [2176, 512],
    "0.41": [704, 1728],    "0.88": [960, 1088],    "2.6": [1664, 640],     "4.38": [2240, 512],
    "0.42": [704, 1664],    "0.89": [1024, 1152],   "2.7": [1728, 640],     "5.0": [2240, 448],
    "0.44": [704, 1600],    "0.94": [1024, 1088],   "2.8": [1792, 640],     "5.14": [2304, 448]
}
```

Stable Diffusion 1.5 / 2.0-base（512px）モデルには、次のマッピングが使用できます:

```json
{
    "1.3": [832, 640], "1.0": [768, 768], "2.0": [1024, 512],
    "0.64": [576, 896], "0.77": [640, 832], "0.79": [704, 896],
    "0.53": [576, 1088], "1.18": [832, 704], "0.85": [704, 832],
    "0.56": [576, 1024], "0.92": [704, 768], "1.78": [1024, 576],
    "1.56": [896, 576], "0.67": [640, 960], "1.67": [960, 576],
    "0.5": [512, 1024], "1.09": [768, 704], "1.08": [832, 768],
    "0.44": [512, 1152], "0.71": [640, 896], "1.4": [896, 640],
    "0.39": [448, 1152], "2.25": [1152, 512], "2.57": [1152, 448],
    "0.4": [512, 1280], "3.5": [1344, 384], "2.12": [1088, 512],
    "0.3": [448, 1472], "2.71": [1216, 448], "8.25": [2112, 256],
    "0.29": [384, 1344], "2.86": [1280, 448], "6.2": [1984, 320],
    "0.6": [576, 960]
}
```
