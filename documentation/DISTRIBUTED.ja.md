# 分散学習（マルチノード）

このドキュメントには、SimpleTuner で 4x 8xH100 クラスターを構成するためのメモ*を記載しています。

> *このガイドはエンドツーエンドのインストール手順ではありません。代わりに、[INSTALL](INSTALL.md) ドキュメントまたは [クイックスタート](QUICKSTART.md) に従う際の考慮事項をまとめています。

## ストレージバックエンド

マルチノード学習では、`output_dir` をノード間で共有ストレージにする必要があります。


### Ubuntu NFS の例

最小限のストレージ例として、まずはこれで動作を確認できます。

#### チェックポイントを書き出す「master」ノード

**1. NFS サーバパッケージをインストール**

```bash
sudo apt update
sudo apt install nfs-kernel-server
```

**2. NFS のエクスポートを設定**

共有ディレクトリを定義するために exports ファイルを編集します:

```bash
sudo nano /etc/exports
```

ファイル末尾に次を追加します（`slave_ip` はスレーブの実 IP に置換）:

```
/home/ubuntu/simpletuner/output slave_ip(rw,sync,no_subtree_check)
```

*複数スレーブやサブネット全体を許可する場合は次を使用できます:*

```
/home/ubuntu/simpletuner/output subnet_ip/24(rw,sync,no_subtree_check)
```

**3. 共有ディレクトリをエクスポート**

```bash
sudo exportfs -a
```

**4. NFS サーバを再起動**

```bash
sudo systemctl restart nfs-kernel-server
```

**5. NFS サーバの状態を確認**

```bash
sudo systemctl status nfs-kernel-server
```

---

#### オプティマイザや各種状態を送信するスレーブノード

**1. NFS クライアントパッケージをインストール**

```bash
sudo apt update
sudo apt install nfs-common
```

**2. マウントポイントディレクトリを作成**

ディレクトリが存在することを確認します（通常はセットアップにより既に存在します）:

```bash
sudo mkdir -p /home/ubuntu/simpletuner/output
```

*注記:* 既存データがある場合はバックアップしてください。マウントすると既存内容は見えなくなります。

**3. NFS 共有をマウント**

マスターの共有ディレクトリをスレーブのローカルディレクトリにマウントします（`master_ip` はマスターの IP に置換）:

```bash
sudo mount master_ip:/home/ubuntu/simpletuner/output /home/ubuntu/simpletuner/output
```

**4. マウントを確認**

```bash
mount | grep /home/ubuntu/simpletuner/output
```

**5. 書き込み権限をテスト**

書き込み権限があることを確認するためにテストファイルを作成します:

```bash
touch /home/ubuntu/simpletuner/output/test_file_from_slave.txt
```

その後、マスター側で `/home/ubuntu/simpletuner/output` にファイルが見えることを確認します。

**6. マウントを永続化**

再起動後もマウントを保持するために `/etc/fstab` に追加します:

```bash
sudo nano /etc/fstab
```

末尾に次の行を追加します:

```
master_ip:/home/ubuntu/simpletuner/output /home/ubuntu/simpletuner/output nfs defaults 0 0
```

---

#### **追加の考慮事項:**

- **ユーザー権限:** `ubuntu` ユーザーの UID/GID が両マシンで一致することを確認してください。`id ubuntu` で確認できます。

- **ファイアウォール設定:** ファイアウォールが有効な場合は NFS 通信を許可してください。マスターで次を実行します:

  ```bash
  sudo ufw allow from slave_ip to any port nfs
  ```

- **時刻同期:** 分散環境では時刻同期が重要です。`ntp` または `systemd-timesyncd` を使用してください。

- **DeepSpeed チェックポイントのテスト:** 小規模な DeepSpeed ジョブで、チェックポイントがマスターのディレクトリに正しく書き込まれることを確認してください。


## データローダ設定

非常に大規模なデータセットは効率的な管理が課題になります。SimpleTuner はデータセットを各ノードに自動シャーディングし、クラスター内のすべての GPU に前処理を分散しつつ、非同期キューとスレッドでスループットを維持します。

### マルチ GPU 学習におけるデータセットサイズ

複数 GPU または複数ノードで学習する場合、データセットは **実効バッチサイズ** を満たすサンプル数が必要です:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

**計算例:**

| 構成 | 計算 | 実効バッチサイズ |
|--------------|-------------|---------------------|
| 1 ノード、8 GPU、batch_size=4、grad_accum=1 | 4 × 8 × 1 | 32 サンプル |
| 2 ノード、16 GPU、batch_size=8、grad_accum=2 | 8 × 16 × 2 | 256 サンプル |
| 4 ノード、32 GPU、batch_size=8、grad_accum=1 | 8 × 32 × 1 | 256 サンプル |

各アスペクト比バケットには（`repeats` を考慮して）この数以上のサンプルが必要で、満たさない場合は詳細なエラーメッセージとともに学習が失敗します。

#### 小さなデータセットの解決策

データセットが実効バッチサイズより小さい場合:

1. **バッチサイズを下げる** - `train_batch_size` を下げてメモリ要件を削減
2. **GPU 数を減らす** - 使用 GPU を減らす（学習は遅くなります）
3. **repeats を増やす** - [データローダ設定](DATALOADER.md#repeats) の `repeats` を設定
4. **自動オーバーサブスクリプション** - `--allow_dataset_oversubscription` を使って repeats を自動調整

`--allow_dataset_oversubscription` フラグ（[OPTIONS.md](OPTIONS.md#--allow_dataset_oversubscription) 参照）は最小必要 repeats を自動計算して適用するため、プロトタイプや小規模データセットの実験に最適です。

### 画像スキャン/ディスカバリが遅い場合

**discovery** バックエンドは現在、アスペクトバケットの収集を 1 ノードに制限しています。非常に大規模なデータセットでは、各画像の幾何情報取得のためにストレージから読み込む必要があるため、**非常に**時間がかかります。

この問題を回避するには、[parquet metadata_backend](DATALOADER.md#parquet-caption-strategy-json-lines-datasets) を使用してください。利用可能な任意の方法で事前処理できるようになります。リンク先のセクションにあるとおり、parquet テーブルには `filename`、`width`、`height`、`caption` 列が含まれ、データを効率よくバケットに分類できます。


### ストレージ容量

巨大データセットでは、特に T5-XXL テキストエンコーダを使う場合、元データ、画像埋め込み、テキスト埋め込みが膨大な容量を消費します。

#### クラウドストレージ

Cloudflare R2 のようなプロバイダを使えば、非常に大規模なデータセットを低コストで保存できます。

`multidatabackend.json` に `aws` タイプを設定する例は [データローダ設定ガイド](DATALOADER.md#local-cache-with-cloud-dataset) を参照してください。

- 画像データはローカルまたは S3 に保存可能
  - 画像が S3 にある場合、前処理速度はネットワーク帯域に依存します
  - 画像がローカルの場合、**学習**時の NVMe スループットを活用しません
- 画像埋め込みとテキスト埋め込みはローカルまたはクラウドに分けて保存可能
  - 埋め込みをクラウドに置いても、並列取得のため学習速度の低下は小さいです

理想的には、すべての画像と埋め込みをクラウドバケットに集約するのが最も簡単で、前処理や再開時のリスクを大幅に減らせます。

#### オンデマンド VAE エンコード

大規模データセットで VAE 潜在のキャッシュ保存が難しい場合、または共有ストレージへのアクセスが遅い場合、`--vae_cache_disable` を使用できます。これにより VAE キャッシュを無効化し、学習中にオンザフライで VAE エンコードを行います。

GPU の計算負荷は増えますが、キャッシュ潜在のストレージ要件とネットワーク I/O を大幅に削減できます。

#### ファイルシステムスキャンキャッシュの保持

データセットが非常に大きく、新規画像のスキャンがボトルネックになる場合、各データローダ設定に `preserve_data_backend_cache=true` を追加すると、新規画像のスキャンを抑止できます。

**注記**: その場合、`image_embeds` のデータバックエンドタイプ（[詳細はこちら](DATALOADER.md#local-cache-with-cloud-dataset)）を使用して、前処理が中断したときにキャッシュリストを別に保持できるようにしてください。これにより起動時の**画像リスト**再スキャンを防げます。

#### データ圧縮

次を `config.json` に追加してデータ圧縮を有効にします:

```json
{
    ...
    "--compress_disk_cache": true,
    ...
}
```

これによりインライン gzip が使われ、大きなテキスト/画像埋め込みが消費するディスク容量を削減できます。

## 🤗 Accelerate での設定

`accelerate config`（`/home/user/.cache/huggingface/accelerate/default_config.yaml`）で SimpleTuner をデプロイする場合、これらの設定は `config/config.env` の内容より優先されます。

DeepSpeed を含まない Accelerate の default_config 例:

```yaml
# this should be updated on EACH node.
machine_rank: 0
# Everything below here is the same on EACH node.
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: MULTI_GPU
downcast_bf16: 'no'
dynamo_config:
  dynamo_backend: NO
enable_cpu_affinity: false
main_process_ip: 10.0.0.100
main_process_port: 8080
main_training_function: main
mixed_precision: bf16
num_machines: 4
num_processes: 32
rdzv_backend: static
same_network: false
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

### DeepSpeed

詳細は [専用ページ](DEEPSPEED.md) を参照してください。

マルチノードで DeepSpeed を最適化するには、可能な限り低い ZeRO レベルを使うことが**重要**です。

例えば、80G の NVIDIA GPU は ZeRO レベル 1 のオフロードで Flux の学習が可能で、オーバーヘッドを大幅に抑えられます。

以下の行を追加します:

```yaml
# Update this from MULTI_GPU to DEEPSPEED
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_multinode_launcher: standard
  gradient_accumulation_steps: 1
  gradient_clipping: 0.01
  zero3_init_flag: false
  zero_stage: 1
```

### torch compile 最適化

互換性の問題という欠点はありますが、追加の性能を得るために、各ノードの YAML 設定に次を追加して torch compile を有効化できます:

```yaml
dynamo_config:
  # Update this from NO to INDUCTOR
  dynamo_backend: INDUCTOR
  dynamo_mode: max-autotune
  dynamo_use_dynamic: false
  dynamo_use_fullgraph: false
```

## 想定パフォーマンス

- ローカルネットワークで接続された 4x H100 SXM5 ノード
- 各ノード 1TB メモリ
- 同一リージョンの共有 S3 互換データバックエンド（Cloudflare R2）から学習キャッシュをストリーミング
- アクセラレータ当たりバッチサイズ **8**、勾配蓄積なし
  - 実効バッチサイズは **256**
- 解像度 1024px、データバケット有効
- **速度**: 1024x1024 データで Flux.1-dev（12B）をフルランク学習した場合、1 ステップ 15 秒

バッチサイズを下げ、解像度を下げ、torch compile を有効化することで **秒あたりのイテレーション** に近づきます:

- 解像度を 512px に下げ、データバケットを無効化（正方形クロップのみ）
- DeepSpeed の AdamW を Lion fused オプティマイザに変更
- torch compile を max-autotune で有効化
- **速度**: 2 イテレーション/秒

## 分散学習の注意点

- すべてのノードで同数のアクセラレータが必要
- 量子化できるのは LoRA/LyCORIS のみで、フルモデルの分散学習には DeepSpeed が必要
- 非常に高コストな操作であり、高バッチサイズは期待以上に遅くなることがあります。予算と GPU 数のバランスを慎重に検討してください。
- （DeepSpeed）ZeRO 3 での学習では検証を無効にする必要がある場合があります
- （DeepSpeed）ZeRO 3 で保存すると分割されたチェックポイントが作られますが、レベル 1 と 2 は想定どおり動作します
- （DeepSpeed）DeepSpeed の CPU ベース最適化器が必要になります。これは最適化状態のシャーディングとオフロードを扱うためです。
