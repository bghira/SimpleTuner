# Slider LoRA ターゲティング

このガイドでは、SimpleTuner でスライダー方式のアダプタを学習します。Z-Image Turbo を使用します。学習が速く、Apache 2.0 ライセンスで提供され、蒸留済み重みでもサイズの割に優れた結果が得られるためです。

互換性マトリクス（LoRA、LyCORIS、フルランク）の全体像は [documentation/QUICKSTART.md](QUICKSTART.md) の Sliders 列を参照してください。このガイドはすべてのアーキテクチャに適用されます。

スライダーターゲティングは標準 LoRA、LyCORIS（`full` を含む）、ControlNet で動作します。切り替えは CLI と WebUI の両方で利用でき、SimpleTuner に同梱されているため追加インストールは不要です。

## Step 1 — ベース設定に従う

- **CLI**: 環境、インストール、ハードウェア注意点、スターター `config.json` は `documentation/quickstart/ZIMAGE.md` を参照してください。
- **WebUI**: `documentation/webui/TUTORIAL.md` を使ってトレーナーウィザードを実行し、通常どおり Z-Image Turbo を選択してください。

これらのガイドはデータセットを設定するところまでそのまま進められます。スライダーはアダプタの配置とデータサンプリングのみを変更するためです。

## Step 2 — スライダーターゲットを有効化

- CLI: `"slider_lora_target": true` を追加（または `--slider_lora_target true` を渡します）。
- WebUI: Model → LoRA Config → Advanced → “Use slider LoRA targets” をチェック。

LyCORIS では `lora_type: "lycoris"` を維持し、`lycoris_config.json` は下の詳細セクションのプリセットを使用してください。

## Step 3 — スライダー向けデータセットを作る

コンセプトスライダーは「反対」のコントラストデータセットから学習します。小さな before/after ペアを作成してください（最初は 4〜6 ペアで十分。用意できるなら多いほど良いです）。

- **Positive バケット**: 「より強い概念」（例: 目が明るい、笑顔が強い、砂が多い）。`"slider_strength": 0.5`（正の値なら任意）。
- **Negative バケット**: 「より弱い概念」（例: 目が暗い、表情が中立）。`"slider_strength": -0.5`（負の値なら任意）。
- **Neutral バケット（任意）**: 通常の例。`slider_strength` を省略するか `0` に設定します。

正負のフォルダでファイル名を一致させる必要はありません。各バケットのサンプル数を同数にしておけば十分です。

## Step 4 — データローダにバケットを指定

- Z-Image quickstart と同じ dataloader JSON パターンを使用します。
- 各バックエンドエントリに `slider_strength` を追加します。SimpleTuner は次を行います:
  - **positive → negative → neutral** の順でバッチを回し、両方向の学習を新鮮に保ちます。
  - 各バックエンドの確率を引き続き尊重するため、重み付けの調整はそのまま機能します。

追加フラグは不要で、`slider_strength` フィールドだけで十分です。

## Step 5 — 学習

通常のコマンド（`simpletuner train ...`）を使うか、WebUI から開始してください。フラグが有効ならスライダーターゲティングは自動です。

## Step 6 — 検証（任意のスライダー調整）

プロンプトライブラリはプロンプトごとのアダプタ強度を持てるため、A/B チェックが可能です:

```json
{
  "plain": "regular prompt",
  "slider_plus": { "prompt": "same prompt", "adapter_strength": 1.2 },
  "slider_minus": { "prompt": "same prompt", "adapter_strength": 0.5 }
}
```

省略した場合、バリデーションではグローバル強度が使われます。

---

## 参考と詳細

<details>
<summary>なぜこのターゲット？（技術的）</summary>

SimpleTuner は Concept Sliders の「テキストは触らない」ルールを模倣するため、スライダー LoRA を自己注意、conv/proj、time-embedding 層にルーティングします。ControlNet の実行でもスライダーターゲティングは維持されます。Assistant アダプタは固定されたままです。
</details>

<details>
<summary>既定のスライダーターゲット一覧（アーキテクチャ別）</summary>

- 一般（SD1.x、SDXL、SD3、Lumina2、Wan、HiDream、LTXVideo、Qwen-Image、Cosmos、Stable Cascade など）:

  ```json
  [
    "attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0",
    "attn1.to_qkv", "to_qkv",
    "proj_in", "proj_out",
    "conv_in", "conv_out",
    "time_embedding.linear_1", "time_embedding.linear_2"
  ]
  ```

- Flux / Flux2 / Chroma / AuraFlow（ビジュアルストリームのみ）:

  ```json
  ["to_q", "to_k", "to_v", "to_out.0", "to_qkv"]
  ```

  Flux2 変種には `attn.to_q`、`attn.to_k`、`attn.to_v`、`attn.to_out.0`、`attn.to_qkv_mlp_proj` が含まれます。

- Kandinsky 5（画像/動画）:

  ```json
  ["attn1.to_query", "attn1.to_key", "attn1.to_value", "conv_in", "conv_out", "time_embedding.linear_1", "time_embedding.linear_2"]
  ```

</details>

<details>
<summary>LyCORIS プリセット（LoKr 例）</summary>

ほとんどのモデル:

```json
{
  "algo": "lokr",
  "multiplier": 1.0,
  "linear_dim": 4,
  "linear_alpha": 1,
  "apply_preset": {
    "target_module": [
      "attn1.to_q",
      "attn1.to_k",
      "attn1.to_v",
      "attn1.to_out.0",
      "conv_in",
      "conv_out",
      "time_embedding.linear_1",
      "time_embedding.linear_2"
    ]
  }
}
```

Flux/Chroma/AuraFlow: ターゲットを `["attn.to_q","attn.to_k","attn.to_v","attn.to_out.0","attn.to_qkv_mlp_proj"]` に置き換えます（チェックポイントが `attn.` を省略する場合は削除）。テキスト/コンテキストを触らないために `add_*` 投影は避けてください。

Kandinsky 5: `attn1.to_query/key/value` に `conv_*` と `time_embedding.linear_*` を加えて使用します。
</details>

<details>
<summary>サンプリングの動作（技術的）</summary>

`slider_strength` を付与したバックエンドは符号でグルーピングされ、positive → negative → neutral の固定サイクルでサンプリングされます。各グループ内では通常のバックエンド確率が適用されます。枯渇したバックエンドは除外され、残りでサイクルが続きます。
</details>
