# セットアップ

Docker または他のコンテナオーケストレーションプラットフォームを使用したいユーザーは、まず[このドキュメント](DOCKER.md)を参照してください。

## インストール

Windows 10 以降で操作しているユーザー向けに、Docker と WSL をベースにしたインストールガイドが[このドキュメント](DOCKER.md)で利用可能です。

### Pip インストール方法

SimpleTuner は pip を使用して簡単にインストールできます。これはほとんどのユーザーに推奨されます:

```bash
# CUDA 用
pip install 'simpletuner[cuda]'
# CUDA 13 / Blackwell 用 (NVIDIA Bシリーズ GPU)
pip install 'simpletuner[cuda13]'
# ROCm 用
pip install 'simpletuner[rocm]'
# Apple Silicon 用
pip install 'simpletuner[apple]'
# CPU のみ(非推奨)
pip install 'simpletuner[cpu]'
# JPEG XL サポート用(オプション)
pip install 'simpletuner[jxl]'

# 開発要件(オプション、PR の提出やテストの実行にのみ必要)
pip install 'simpletuner[dev]'
```

### Git リポジトリ方式

ローカル開発やテスト用に、SimpleTuner リポジトリをクローンして Python venv をセットアップできます:

```bash
git clone --branch=release https://github.com/bghira/SimpleTuner.git

cd SimpleTuner

# python --version が 3.11 または 3.12 を示す場合は 3.13 にアップグレードすることをお勧めします。
python3.13 -m venv .venv

source .venv/bin/activate
```

> ℹ️ `config/config.env` ファイルで `export VENV_PATH=/path/to/.venv` を設定することで、独自のカスタム venv パスを使用できます。

**注意:** ここでは現在 `release` ブランチをインストールしています。`main` ブランチには、より良い結果や低メモリ使用量を持つ可能性のある実験的機能が含まれている場合があります。

自動プラットフォーム検出で SimpleTuner をインストール:

```bash
# 基本インストール(CUDA/ROCm/Apple を自動検出)
pip install -e .

# JPEG XL サポート付き
pip install -e .[jxl]
```

**注意:** setup.py は自動的にプラットフォーム(CUDA/ROCm/Apple)を検出し、適切な依存関係をインストールします。

#### NVIDIA Hopper / Blackwell フォローアップ手順

オプションとして、Hopper(または新しい)機器は、`torch.compile` を使用する際の推論およびトレーニングパフォーマンス向上のために FlashAttention3 を利用できます。

venv をアクティブにした状態で、SimpleTuner ディレクトリから次のコマンドシーケンスを実行する必要があります:

```bash
git clone https://github.com/Dao-AILab/flash-attention
pushd flash-attention
  pushd hopper
    python setup.py install
  popd
popd
```

> ⚠️ SimpleTuner での flash_attn ビルドの管理は、現在十分にサポートされていません。これはアップデート時に壊れる可能性があり、このビルド手順を時々手動で再実行する必要があります。

#### AMD ROCm フォローアップ手順

AMD MI300X を使用可能にするには、以下を実行する必要があります:

```bash
apt install amd-smi-lib
pushd /opt/rocm/share/amd_smi
  python3 -m pip install --upgrade pip
  python3 -m pip install .
popd
```

> ℹ️ **ROCm アクセラレーションのデフォルト**: SimpleTuner が HIP 対応の PyTorch ビルドを検出すると、自動的に `PYTORCH_TUNABLEOP_ENABLED=1` をエクスポート(既に設定していない場合)し、TunableOp カーネルが利用可能になります。MI300/gfx94x デバイスでは、デフォルトで `HIPBLASLT_ALLOW_TF32=1` も設定され、手動での環境設定なしに hipBLASLt の TF32 パスが有効になります。

### 全プラットフォーム共通

- 2a. **オプション1(推奨)**: `simpletuner configure` を実行
- 2b. **オプション2**: `config/config.json.example` を `config/config.json` にコピーして詳細を入力

> ⚠️ Hugging Face Hub に容易にアクセスできない国にいるユーザーは、使用している `$SHELL` に応じて、`~/.bashrc` または `~/.zshrc` に `HF_ENDPOINT=https://hf-mirror.com` を追加する必要があります。

#### マルチ GPU トレーニング {#multiple-gpu-training}

SimpleTuner には、WebUI を通じた**自動 GPU 検出と設定**が含まれるようになりました。初回ロード時に、GPU を検出して Accelerate を自動的に設定するオンボーディング手順がガイドされます。

##### WebUI 自動検出(推奨)

WebUI を初めて起動するか、`simpletuner configure` を使用すると、「Accelerate GPU Defaults」オンボーディング手順が表示されます:

1. **自動検出**: システム上の利用可能な全 GPU を検出
2. **GPU 詳細の表示**: 名前、メモリ、デバイス ID を含む
3. **最適設定の推奨**: マルチ GPU トレーニング用の最適設定を推奨
4. **3つの設定モードを提供:**

   - **オートモード**(推奨): 検出された全 GPU を最適なプロセス数で使用
   - **マニュアルモード**: 特定の GPU を選択するか、カスタムプロセス数を設定
   - **無効モード**: シングル GPU トレーニングのみ

**仕組み:**
- システムは CUDA/ROCm 経由で GPU ハードウェアを検出
- 利用可能なデバイスに基づいて最適な `--num_processes` を計算
- 特定の GPU が選択されている場合、自動的に `CUDA_VISIBLE_DEVICES` を設定
- 将来のトレーニング実行のために設定を保存

##### 手動設定

WebUI を使用しない場合、`config.json` で GPU の可視性を直接制御できます:

```json
{
  "accelerate_visible_devices": [0, 1, 2],
  "num_processes": 3
}
```

これにより、トレーニングは GPU 0、1、2 に制限され、3 つのプロセスが起動されます。

3. `--report_to='wandb'`(デフォルト)を使用している場合、以下は統計情報のレポートに役立ちます:

```bash
wandb login
```

API キーを見つけて設定するために、表示される指示に従ってください。

完了すると、すべてのトレーニングセッションと検証データが Weights & Biases で利用可能になります。

> ℹ️ Weights & Biases または Tensorboard のレポートを完全に無効にしたい場合は、`--report-to=none` を使用してください


4. simpletuner でトレーニングを開始します。ログは `debug.log` に書き込まれます

```bash
simpletuner train
```

> ⚠️ この時点で、`simpletuner configure` を使用した場合は完了です! そうでない場合 - これらのコマンドは動作しますが、さらなる設定が必要です。詳細については[チュートリアル](TUTORIAL.md)を参照してください。

### ユニットテストの実行

インストールが正常に完了したことを確認するためにユニットテストを実行するには:

```bash
python -m unittest discover tests/
```

## 高度: 複数の設定環境

複数のモデルをトレーニングするユーザーや、異なるデータセットや設定を素早く切り替える必要があるユーザー向けに、起動時に 2 つの環境変数が検査されます。

使用方法:

```bash
simpletuner train env=default config_backend=env
```

- `env` はデフォルトで `default` になり、このガイドで設定した通常の `SimpleTuner/config/` ディレクトリを指します
  - `simpletuner train env=pixart` を使用すると、`SimpleTuner/config/pixart` ディレクトリを使用して `config.env` を見つけます
- `config_backend` はデフォルトで `env` になり、このガイドで設定した通常の `config.env` ファイルを使用します
  - サポートされるオプション: `env`、`json`、`toml`、または `train.py` を手動で実行する場合は `cmd`
  - `simpletuner train config_backend=json` を使用すると、`config.env` の代わりに `SimpleTuner/config/config.json` を検索します
  - 同様に、`config_backend=toml` は `config.env` を使用します

これらの値の一方または両方を含む `config/config.env` を作成できます:

```bash
ENV=default
CONFIG_BACKEND=json
```

これらは次回以降の実行時に記憶されます。これらは[上記](#multiple-gpu-training)で説明したマルチ GPU オプションに加えて追加できることに注意してください。

## トレーニングデータ

約 10,000 枚の画像とファイル名としてのキャプションを含む、SimpleTuner ですぐに使用できる公開データセットが[Hugging Face Hub で利用可能](https://huggingface.co/datasets/bghira/pseudo-camera-10k)です。

画像は単一のフォルダにまとめることも、サブディレクトリに整理することもできます。

### 画像選択ガイドライン

**品質要件:**
- JPEG アーティファクトやぼやけた画像は避ける - 最新のモデルはこれらを検出します
- 粒状の CMOS センサーノイズを避ける(生成される全画像に現れます)
- 透かし、バッジ、署名なし(これらは学習されます)
- 映画のフレームは一般的に圧縮のため機能しません(代わりにプロダクションスチルを使用)

**技術仕様:**
- 64 で割り切れる画像が最適(リサイズなしで再利用可能)
- バランスの取れた機能のために、正方形と非正方形の画像を混在
- 最良の結果のために、多様で高品質なデータセットを使用

### キャプショニング

SimpleTuner はファイルの一括名前変更のための[キャプショニングスクリプト](/scripts/toolkit/README.md)を提供します。サポートされるキャプション形式:
- ファイル名をキャプションとして使用(デフォルト)
- `--caption_strategy=textfile` でテキストファイル
- JSONL、CSV、または高度なメタデータファイル

**推奨キャプショニングツール:**
- **InternVL2**: 最高品質だが低速(小規模データセット向け)
- **BLIP3**: 優れた指示追従性を持つ最良の軽量オプション
- **Florence2**: 最速だが出力を好まない人もいる

### トレーニングバッチサイズ

最大バッチサイズは VRAM と解像度に依存します:
```
vram 使用量 = バッチサイズ * 解像度 + 基本要件
```

**主要原則:**
- VRAM の問題なしで可能な限り最大のバッチサイズを使用
- 高解像度 = より多くの VRAM = より低いバッチサイズ
- 128x128 でバッチサイズ 1 が機能しない場合、ハードウェアが不十分

#### マルチ GPU データセット要件

複数の GPU でトレーニングする場合、データセットは**有効バッチサイズ**に対して十分な大きさである必要があります:
```
有効バッチサイズ = train_batch_size × GPU 数 × gradient_accumulation_steps
```

**例:** 4 GPU で `train_batch_size=4` の場合、アスペクトバケットごとに少なくとも 16 サンプルが必要です。

**小規模データセット用のソリューション:**
- `--allow_dataset_oversubscription` を使用して繰り返しを自動調整
- データローダー設定で手動で `repeats` を設定
- バッチサイズまたは GPU 数を削減

完全な詳細については [DATALOADER.md](DATALOADER.md#multi-gpu-training-and-dataset-sizing) を参照してください。

## Hugging Face Hub への公開

完了時にモデルを Hub に自動的にプッシュするには、`config/config.json` に追加します:

```json
{
  "push_to_hub": true,
  "hub_model_name": "your-model-name"
}
```

トレーニング前にログイン:
```bash
huggingface-cli login
```

## デバッグ

`config/config.env` に追加して詳細なログを有効にします:

```bash
export SIMPLETUNER_LOG_LEVEL=DEBUG
export SIMPLETUNER_TRAINING_LOOP_LOG_LEVEL=DEBUG
```

プロジェクトルートに全てのログエントリを含む `debug.log` ファイルが作成されます。
