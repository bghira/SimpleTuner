# CaptionFlow 連携

SimpleTuner は [CaptionFlow](https://github.com/bghira/CaptionFlow) を使って、Web UI から画像データセットにキャプションを生成できます。CaptionFlow は vLLM ベースのスケーラブルなキャプション生成システムで、オーケストレータ、GPU ワーカー、チェックポイント付きストレージ、YAML 設定を備えています。SimpleTuner では Datasets ページの **Captioning** サブタブとして表示され、トレーニングやキャッシュと同じローカル GPU ジョブキューを使います。

トレーニング前に SimpleTuner の流れの中でキャプションを生成または更新したい場合に使います。

## インストール

CaptionFlow は任意依存です。SimpleTuner と同じ仮想環境に captioning ターゲットをインストールします。

```bash
pip install "simpletuner[captioning]"
```

CUDA 13 環境では、Web UI に表示される CUDA 13 用ターゲットを使ってください。その runtime に合う vLLM wheel が含まれます。

## SimpleTuner が管理する内容

Captioning ジョブを開始すると、SimpleTuner は次を行います。

- 選択した SimpleTuner データセットを CaptionFlow processor にマップする
- `127.0.0.1` でローカル CaptionFlow オーケストレータを起動する
- ジョブキュー経由で 1 個以上のローカル GPU ワーカーを起動する
- オーケストレータとワーカーのログをジョブ workspace に保存する
- export 前に CaptionFlow storage を安全に checkpoint する
- ローカルデータセットでは `.txt` キャプションを画像のあるディレクトリへ書き戻す
- Hugging Face データセットでは CaptionFlow workspace に JSONL を export する

CaptionFlow 依存が未インストールでもタブは表示されます。その場合はジョブ builder の代わりにインストールコマンドが表示されます。

## Builder モード

**Builder** ビューは一般的な単一ステージの captioning を扱います。

- 有効な dataloader 設定からのデータセット選択
- model、prompt、sampling、batch size、chunk size、GPU memory 設定
- worker 数と queue 動作
- ローカルデータセット向けの text file export

既定モデルは `Qwen/Qwen2.5-VL-3B-Instruct` です。ローカルデータセットでは、フォームで選んだ output field を使って画像の隣に text file を export します。Hugging Face データセットはリモート dataset に書き戻さず、CaptionFlow workspace に JSONL を export します。

## Raw Config モード

**Raw Config** は、builder が扱わない CaptionFlow 機能が必要な場合に使います。例として、multi-stage captioning、stage ごとの model、stage ごとの sampling、ある stage の出力を次の stage の prompt に渡す chain があります。

Raw config は YAML または JSON を受け付けます。`orchestrator:` root を含む完全な設定、または orchestrator object だけを貼り付けられます。

SimpleTuner は runtime で次の項目を上書きします。

- `host`、`port`、`ssl`
- 選択した SimpleTuner dataset に基づく `dataset`
- job workspace 配下の `storage.data_dir` と `storage.checkpoint_dir`
- `auth.worker_tokens` と `auth.admin_tokens`

`chunk_size`、`chunks_per_request`、`storage.caption_buffer_size`、`vllm.sampling`、`vllm.inference_prompts`、`vllm.stages` などの他の設定は、SimpleTuner が default を必要としない限り保持されます。

## Multi-stage 例

この raw config は詳細キャプション stage を実行し、その `{caption}` を短縮 stage に渡します。選択 dataset、storage path、port、auth token は SimpleTuner がジョブ開始時に設定します。

```yaml
orchestrator:
  chunk_size: 1000
  chunks_per_request: 1
  chunk_buffer_multiplier: 2
  min_chunk_buffer: 10
  vllm:
    model: "Qwen/Qwen2.5-VL-3B-Instruct"
    tensor_parallel_size: 1
    max_model_len: 16384
    dtype: "float16"
    gpu_memory_utilization: 0.92
    enforce_eager: true
    disable_mm_preprocessor_cache: true
    limit_mm_per_prompt:
      image: 1
    batch_size: 8
    sampling:
      temperature: 0.7
      top_p: 0.95
      max_tokens: 256
    stages:
      - name: "base_caption"
        prompts:
          - "describe this image in detail"
        output_field: "caption"
      - name: "caption_shortening"
        model: "Qwen/Qwen2.5-VL-7B-Instruct"
        prompts:
          - "Please condense this elaborate caption to only the important details: {caption}"
        output_field: "captions"
        requires: ["base_caption"]
        gpu_memory_utilization: 0.35
```

## 外部ドキュメント

- [CaptionFlow repository](https://github.com/bghira/CaptionFlow)
- [CaptionFlow README](https://github.com/bghira/CaptionFlow#readme)
- [CaptionFlow orchestrator examples](https://github.com/bghira/CaptionFlow/tree/main/examples/orchestrator)

高度な CaptionFlow field については upstream の examples を参照してください。SimpleTuner 経由で実行する場合、dataset routing、local port、storage workspace path、auth token は SimpleTuner が管理します。
