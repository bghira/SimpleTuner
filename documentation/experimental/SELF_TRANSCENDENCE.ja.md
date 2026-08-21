# Self-Transcendence

Self-Transcendence は、外部の視覚エンコーダーを使わず、内部ターゲットで拡散 Transformer の浅いブロックを学習します。[Sun ら](https://arxiv.org/abs/2601.07773)の 2 段階方式に基づきます。

潜在 token の中間状態を公開する画像・動画・音声の拡散 Transformer に対応します。UNet、自回帰モデル、LyCORIS には対応しません。フルモデル学習と標準 PEFT LoRA に対応します。

## ステージ 1: VAE 構造ガイダンス

浅い層を VAE 潜在空間におけるモデル系列の拡散ターゲットへ射影します。対象は flow velocity、epsilon、v-prediction、または clean sample です。値を破棄せず、モデルの token グリッドに合わせてパッチ化されます。

```json
{
  "distillation_method": "self_transcendence",
  "distillation_config": {"self_transcendence": {
    "stage": "vae", "student_block": 8, "weight": 0.5,
    "timestep_min": 0.4, "timestep_max": 0.7,
    "projector_hidden_dim": 2048
  }}
}
```

このステージのアダプターまたはチェックポイントを保存し、ステージ 2 の固定教師に使用します。

## ステージ 2: 自己ガイド表現

固定教師を同じノイズ入力に対して通常プロンプトとキャッシュ済み空プロンプトで実行します。深い層の特徴空間 CFG ターゲットで、新しい生徒の浅い層を学習します。

PEFT LoRA では新しい生徒アダプターを作成し、`teacher_adapter_path` にステージ 1 の safetensors を指定します。

```json
{
  "distillation_method": "self_transcendence",
  "distillation_config": {"self_transcendence": {
    "stage": "self", "student_block": 8, "teacher_block": 16,
    "teacher_adapter_path": "output/stage1/pytorch_lora_weights.safetensors",
    "cfg_scale": 30.0, "weight": 0.5,
    "timestep_min": 0.4, "timestep_max": 0.7,
    "stop_step": 5000, "projector_hidden_dim": 2048
  }}
}
```

教師と生徒は同じベースモデル、PEFT rank、対象モジュールを使う必要があります。`teacher_adapter_path` がない場合、再開後の学習可能パラメータを教師としてスナップショットします。これはフルモデルと 1 段階実験を支援しますが、論文の新規生徒設定とは異なります。

ブロック番号は 0 始まりです。生徒は深さの約 1/3、教師は約 2/3 から試してください。`stop_step` 後は教師 forward を停止し、DDP 用にゼロ重みの射影経路だけを維持します。空プロンプト埋め込みは自動でキャッシュされます。

ログは `self_transcendence/loss`、`self_transcendence/weight`、ステージ 2 の `self_transcendence/teacher_cfg_scale` です。他の蒸留方式やテキストエンコーダー学習とは併用できません。
