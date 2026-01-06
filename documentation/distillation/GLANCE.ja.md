# Glance 形式の単一サンプル LoRA

Glance は本格的な蒸留パイプラインというより、「スケジュールを分割した単一画像 LoRA」です。同一の画像/キャプションに対して、小さな LoRA を 2 つ学習します。前半ステップ用（「Slow」）と後半ステップ用（「Fast」）を学習し、推論時に連結します。

## 得られるもの

- 単一の画像/キャプションで学習した 2 つの LoRA
- CDF サンプリングではなくカスタム flow タイムステップ（`--flow_custom_timesteps`）
- flow-matching モデルで動作（Flux、SD3 系、Qwen-Image など）

## 前提条件

- Python 3.10–3.12、SimpleTuner をインストール済み（`pip install simpletuner[cuda]`）
- 同じベース名の画像とキャプション（例: `data/glance.png` + `data/glance.txt`）
- flow モデルのチェックポイント（以下の例では `black-forest-labs/FLUX.1-dev`）

## Step 1 – 単一サンプルを dataloader に指定

`config/multidatabackend.json` で単一の画像/テキストペアを参照します。最小例:

```json
[
  {
    "backend": "simple",
    "resolution": 1024,
    "shuffle": true,
    "image_dir": "data",
    "caption_extension": ".txt"
  }
]
```

## Step 2 – Slow LoRA（初期ステップ）を学習

`config/glance-slow.json` を作成し、初期ステップのリストを指定します:

```json
{
  "--model_type": "lora",
  "--model_family": "flux",
  "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
  "--data_backend_config": "config/multidatabackend.json",
  "--train_batch_size": 1,
  "--max_train_steps": 60,
  "--lora_rank": 32,
  "--output_dir": "output/glance-slow",
  "--flow_custom_timesteps": "1000,979.1915,957.5157,934.9171,911.3354"
}
```

実行:

```bash
simpletuner train --config config/glance-slow.json
```

## Step 3 – Fast LoRA（後半ステップ）を学習

設定を `config/glance-fast.json` にコピーし、出力パスとタイムステップを変更します:

```json
{
  "--model_type": "lora",
  "--model_family": "flux",
  "--pretrained_model_name_or_path": "black-forest-labs/FLUX.1-dev",
  "--data_backend_config": "config/multidatabackend.json",
  "--train_batch_size": 1,
  "--max_train_steps": 60,
  "--lora_rank": 32,
  "--output_dir": "output/glance-fast",
  "--flow_custom_timesteps": "886.7053,745.0728,562.9505,320.0802,20.0"
}
```

実行:

```bash
simpletuner train --config config/glance-fast.json
```

注記:
- `--flow_custom_timesteps` は flow-matching の通常の sigmoid/uniform/beta サンプリングを上書きし、各ステップでリストから均一に選択します。
- 他の flow スケジュール関連フラグは既定のままで構いません。カスタムリストがそれらをバイパスします。
- 0–1000 のタイムステップではなくシグマを使いたい場合は、`[0,1]` の値を指定します。

## Step 4 – 2 つの LoRA を併用（正しいシグマで）

Diffusers は学習時と同じシグマスケジュールを見る必要があります。0–1000 タイムステップを 0–1 のシグマに変換して明示的に渡してください。`num_inference_steps` は各リストの長さと一致させます。

```python
import torch
from diffusers import FluxPipeline

device = "cuda"
base_model = "black-forest-labs/FLUX.1-dev"

# Convert the training timesteps to sigmas in [0, 1]
slow_sigmas = [t / 1000.0 for t in (1000.0, 979.1915, 957.5157, 934.9171, 911.3354)]
fast_sigmas = [t / 1000.0 for t in (886.7053, 745.0728, 562.9505, 320.0802, 20.0)]
generator = torch.Generator(device=device).manual_seed(0)

# Early phase
pipe = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
pipe.load_lora_weights("output/glance-slow")
latents = pipe(
    prompt="your prompt",
    num_inference_steps=len(slow_sigmas),
    sigmas=slow_sigmas,
    output_type="latent",
    generator=generator,
).images

# Late phase (continue the same schedule)
pipe_fast = FluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16).to(device)
pipe_fast.load_lora_weights("output/glance-fast")
image = pipe_fast(
    prompt="your prompt",
    num_inference_steps=len(fast_sigmas),
    sigmas=fast_sigmas,
    latents=latents,
    guidance_scale=1.0,
    generator=generator,
).images[0]
image.save("glance.png")
```

同じ prompt と generator/seed を再利用して、Fast LoRA が Slow LoRA の停止位置から正確に再開できるようにし、`--flow_custom_timesteps` で学習したシグマリストと揃えてください。
