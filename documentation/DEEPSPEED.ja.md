# DeepSpeed オフロード / マルチ GPU 学習

SimpleTuner v0.7 で、SDXL を DeepSpeed ZeRO ステージ 1〜3 で学習するための予備的なサポートが導入されました。
v3.0 では、WebUI の設定ビルダー、最適化器サポートの改善、オフロード管理の強化により大幅に改善されています。

> ⚠️ DeepSpeed は macOS（MPS）および ROCm 環境では利用できません。

**VRAM 9237MiB で SDXL 1.0 を学習**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------|
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:08:00.0 Off |                  Off |
|  0%   43C    P2   100W / 450W |   9237MiB / 24564MiB |    100%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A     11500      C   ...uner/.venv/bin/python3.13     9232MiB |
+-----------------------------------------------------------------------------+
```

これらのメモリ削減は DeepSpeed ZeRO Stage 2 のオフロードにより実現しています。これがない場合、SDXL の U-Net は 24G 以上の VRAM を消費し、CUDA Out of Memory が発生します。

## DeepSpeed とは？

ZeRO は **Zero Redundancy Optimizer** の略です。この手法は、モデル学習状態（重み、勾配、オプティマイザ状態）を利用可能なデバイス（GPU と CPU）に分割することで、各 GPU のメモリ消費を削減します。

ZeRO は段階的な最適化として実装され、前段階の最適化は後段階で利用できます。詳細は元論文 [paper](https://arxiv.org/abs/1910.02054v3)（1910.02054v3）を参照してください。

## 既知の問題

### LoRA サポート

DeepSpeed によってモデル保存ルーチンが変更されるため、現時点では DeepSpeed での LoRA 学習はサポートされていません。

将来のリリースで変更される可能性があります。

### 既存チェックポイントでの DeepSpeed の有効/無効切り替え

現在の SimpleTuner では、DeepSpeed を使用せずに学習したチェックポイントから再開する際に DeepSpeed を**有効化**できません。

逆に、DeepSpeed を使用して学習したチェックポイントから再開する際に DeepSpeed を**無効化**することもできません。

回避策として、進行中の学習セッションで DeepSpeed の有効/無効を切り替える前に、学習パイプラインを完全なモデル重みとして書き出してください。

DeepSpeed のオプティマイザが通常の選択肢と大きく異なるため、この機能が実現する可能性は低いと考えられます。

## DeepSpeed ステージ

DeepSpeed は 3 段階の最適化レベルを提供し、段階が上がるほどオーバーヘッドが増えます。

特にマルチ GPU 学習では、CPU との転送が DeepSpeed 内で十分に最適化されていません。

このオーバーヘッドのため、動作する範囲で**最も低い** DeepSpeed レベルを選ぶことを推奨します。

### Stage 1

オプティマイザ状態（例: Adam の 32-bit 重みと一次/二次モーメント推定）をプロセス間で分割し、各プロセスは自身の分割のみを更新します。

### Stage 2

モデル重みを更新するための 32-bit 勾配も分割され、各プロセスはオプティマイザ状態の自分の分割に対応する勾配のみを保持します。

### Stage 3

16-bit のモデルパラメータがプロセス間で分割されます。ZeRO-3 は順伝播と逆伝播の間に自動的に収集・分割します。

## DeepSpeed の有効化

[公式チュートリアル](https://www.deepspeed.ai/tutorials/zero/) は構成が非常に良く、ここで触れていない多くのシナリオを含んでいます。

### 方法 1: WebUI 設定ビルダー（推奨）

SimpleTuner には DeepSpeed 設定用の使いやすい WebUI が用意されています:

1. SimpleTuner WebUI を開く
2. **Hardware** タブに切り替え、**Accelerate & Distributed** セクションを開く
3. `DeepSpeed Config (JSON)` フィールドの **DeepSpeed Builder** ボタンをクリック
4. 対話型 UI で次を設定:
   - ZeRO 最適化ステージ（1/2/3）
   - オフロード設定（CPU、NVMe）
   - オプティマイザとスケジューラ
   - 勾配蓄積とクリッピング設定
5. 生成された JSON 設定をプレビュー
6. 保存して適用

ビルダーは JSON 構造を一貫させ、必要に応じて未サポートのオプティマイザを安全な既定値へ差し替えるため、よくある設定ミスを避けられます。

### 方法 2: JSON を手動で設定

直接編集したい場合は、`config.json` に DeepSpeed 設定を追加します:

```json
{
  "deepspeed_config": {
    "zero_optimization": {
      "stage": 2,
      "offload_param": {
        "device": "cpu",
        "pin_memory": true
      },
      "offload_optimizer": {
        "device": "cpu",
        "pin_memory": true
      }
    },
    "gradient_accumulation_steps": 4,
    "gradient_clipping": 1.0,
    "optimizer": {
      "type": "AdamW",
      "params": {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01
      }
    },
    "scheduler": {
      "type": "WarmupLR",
      "params": {
        "warmup_min_lr": 0,
        "warmup_max_lr": 1e-4,
        "warmup_num_steps": 500
      }
    },
    "train_batch_size": 8,
    "train_micro_batch_size_per_gpu": 2
  }
}
```

**主な設定項目:**

- `zero_optimization.stage`: ZeRO 最適化レベル（1/2/3）
- `offload_param.device`: パラメータのオフロード先（"cpu" または "nvme"）
- `offload_optimizer.device`: オプティマイザのオフロード先（"cpu" または "nvme"）
- `optimizer.type`: サポートされるオプティマイザ（AdamW, Adam, Adagrad, Lamb など）
- `gradient_accumulation_steps`: 勾配を蓄積するステップ数

**NVMe オフロードの例:**
```json
{
  "deepspeed_config": {
    "zero_optimization": {
      "stage": 3,
      "offload_param": {
        "device": "nvme",
        "nvme_path": "/path/to/nvme/storage",
        "buffer_size": 100000000.0,
        "pin_memory": true
      }
    }
  }
}
```

### 方法 3: accelerate config で手動設定

上級者向けに、`accelerate config` を通じて DeepSpeed を有効化することもできます:

```
----------------------------------------------------------------------------------------------------------------------------
In which compute environment are you running?
This machine
----------------------------------------------------------------------------------------------------------------------------
Which type of machine are you using?
No distributed training
Do you want to run your training on CPU only (even if a GPU / Apple Silicon / Ascend NPU device is available)? [yes/NO]:NO
Do you wish to optimize your script with torch dynamo?[yes/NO]:NO
Do you want to use DeepSpeed? [yes/NO]: yes
Do you want to specify a json file to a DeepSpeed config? [yes/NO]: NO
----------------------------------------------------------------------------------------------------------------------------
What should be your DeepSpeed's ZeRO optimization stage?
1
How many gradient accumulation steps you're passing in your script? [1]: 4
Do you want to use gradient clipping? [yes/NO]:
Do you want to enable `deepspeed.zero.Init` when using ZeRO Stage-3 for constructing massive models? [yes/NO]:
How many GPU(s) should be used for distributed training? [1]:
----------------------------------------------------------------------------------------------------------------------------
Do you wish to use FP16 or BF16 (mixed precision)?bf16
accelerate configuration saved at /root/.cache/huggingface/accelerate/default_config.yaml
```

これにより、次の YAML ファイルが生成されます:

```yaml
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 4
  zero3_init_flag: false
  zero_stage: 1
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

## SimpleTuner の設定

SimpleTuner は DeepSpeed の利用に特別な設定を必要としません。

ZeRO stage 2 または 3 を NVMe オフロードと併用する場合は、`--offload_param_path=/path/to/offload` を指定して、パラメータ/オプティマイザのオフロードファイルを専用パーティションに保存できます。理想的には NVMe デバイスですが、他のストレージでも構いません。

### 最近の改善（v0.7+）

#### WebUI 設定ビルダー
SimpleTuner には WebUI に包括的な DeepSpeed 設定ビルダーが追加され、次が可能です:
- 直感的な UI でカスタム DeepSpeed JSON を作成
- 利用可能なパラメータを自動検出
- 適用前に設定の影響を可視化
- 設定テンプレートの保存と再利用

#### オプティマイザサポートの強化
オプティマイザ名の正規化と検証が改善されました:
- **サポートされるオプティマイザ**: AdamW, Adam, Adagrad, Lamb, OneBitAdam, OneBitLamb, ZeroOneAdam, MuAdam, MuAdamW, MuSGD, Lion, Muon
- **未サポートのオプティマイザ**（自動的に AdamW に置換）: cpuadam, fusedadam
- 未サポートの指定があった場合の自動フォールバック警告

#### オフロード管理の改善
- **自動クリーンアップ**: 古い DeepSpeed オフロードのスワップディレクトリを自動削除し、破損した再開状態を防止
- **NVMe サポート強化**: NVMe オフロードパスの扱いが改善され、バッファサイズが自動割り当てされる
- **プラットフォーム検出**: 非対応プラットフォーム（macOS/ROCm）では DeepSpeed を自動的に無効化

#### 設定検証
- 変更適用時にオプティマイザ名と設定構造を自動正規化
- 未サポートのオプティマイザ選択や不正な JSON に対する安全ガード
- トラブルシュート向けのエラーハンドリングとログ改善

### DeepSpeed オプティマイザ / 学習率スケジューラ

DeepSpeed は独自の学習率スケジューラを使い、既定では高最適化された AdamW を使用します（8bit ではありません）。DeepSpeed の場合、処理は CPU に近づく傾向があるため、8bit であることはあまり重要ではありません。

`default_config.yaml` に `scheduler` または `optimizer` が設定されている場合はそれが使用されます。どちらも定義されていない場合、既定の `AdamW` と `WarmUp` がそれぞれオプティマイザとスケジューラとして使用されます。

## 簡易テスト結果

4090 24G GPU を使用:

* 1 メガピクセル（1024^2 ピクセル面積）のフル U-Net 学習を、バッチサイズ 8 で **13102MiB の VRAM** だけで実行可能
  * 1 イテレーションあたり 8 秒で動作。1000 ステップは 2 時間半弱で完了します。
  * DeepSpeed のチュートリアルにあるとおり、バッチサイズを小さく調整して VRAM をパラメータやオプティマイザ状態に回すのが有利な場合があります。
    * ただし SDXL は比較的小規模なモデルであり、性能への影響が許容できるなら一部の推奨は回避できる可能性があります。
* **128x128** 画像、バッチサイズ 8 では、学習は **9237MiB の VRAM** まで低減できます。これはピクセルアート学習（潜在空間と 1:1 対応が必要）における限定的なユースケースです。

これらの条件内で結果はさまざまで、未検証ですが 1024x1024、バッチサイズ 1 なら 8GiB 程度の VRAM でもフル U-Net 学習が可能になる場合があります。

SDXL は多様な解像度とアスペクト比の大規模分布で学習されているため、ピクセル面積を 0.75 メガピクセル（およそ 768x768）まで下げてメモリ最適化をさらに進めることもできます。

# AMD デバイスのサポート

消費者向けまたはワークステーショングレードの AMD GPU を手元で検証していませんが、MI50（現在はサポート終了）や他の上位 Instinct カードで DeepSpeed が動作するという報告があります。AMD は実装のリポジトリを維持しています。

## トラブルシューティング

### よくある問題と解決策

#### 「再開時に DeepSpeed がクラッシュする」
**問題**: DeepSpeed オフロードを有効にしたチェックポイントから再開すると学習がクラッシュする。

**解決策**: SimpleTuner は古い DeepSpeed オフロードのスワップディレクトリを自動でクリーンアップするため、破損した再開状態を防げます。最新の更新で解決済みです。

#### 「未サポートのオプティマイザ」エラー
**問題**: DeepSpeed 設定に未サポートのオプティマイザ名が含まれている。

**解決策**: システムがオプティマイザ名を自動正規化し、未サポート（cpuadam, fusedadam）は AdamW に置き換えます。フォールバック時は警告ログが出ます。

#### 「このプラットフォームでは DeepSpeed を利用できない」
**問題**: DeepSpeed のオプションが無効、または利用できない。

**解決策**: DeepSpeed は CUDA 環境のみサポートです。macOS（MPS）と ROCm では互換性の問題を避けるため設計上自動的に無効化されます。

#### 「NVMe オフロードのパス問題」
**問題**: NVMe オフロードパス設定に関するエラー。

**解決策**: `--offload_param_path` が十分な空き容量を持つ有効なディレクトリを指していることを確認してください。システムはバッファサイズの割り当てとパス検証を自動で処理します。

#### 「設定検証エラー」
**問題**: DeepSpeed 設定パラメータが不正。

**解決策**: WebUI 設定ビルダーを使用して JSON を生成してください。適用前にオプティマイザ選択と構造を正規化します。

### デバッグ情報

DeepSpeed の問題を調査する際は、次を確認してください:
- WebUI の Hardware タブ（Hardware → Accelerate）または `nvidia-smi` によるハードウェア互換性
- 学習ログ内の DeepSpeed 設定
- オフロードパスの権限と空き容量
- プラットフォーム検出ログ

# EMA 学習（指数移動平均）

EMA は勾配を平滑化し、重みの汎化性能を高める有効な手法ですが、メモリ消費が非常に大きくなります。

EMA はモデルパラメータのシャドウコピーをメモリに保持するため、モデルのフットプリントが実質的に 2 倍になります。SimpleTuner では EMA は Accelerator モジュールを通過しないため、DeepSpeed の影響を受けません。つまり、ベース U-Net で得られたメモリ削減は EMA には適用されません。

ただし既定では、EMA モデルは CPU 上に保持されます。
