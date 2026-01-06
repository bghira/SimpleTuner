# LayerSync（SimpleTuner）

LayerSync は Transformer モデル向けの「自己教師あり」調整です。1 つのレイヤー（学生）がより強いレイヤー（教師）に整合するよう学習します。軽量で自己完結しており、追加の教師モデルをダウンロードする必要はありません。

## 使いどき

- 隠れ状態を公開する Transformer ファミリーを学習している（例: Flux/Flux Kontext/Flux.2、PixArt Sigma、SD3/SDXL、Sana、Wan、Qwen Image/Edit、Hunyuan Video、LTXVideo、Kandinsky5 Video、Chroma、ACE-Step、HiDream、Cosmos/LongCat/Z-Image/Auraflow）。
- 外部教師チェックポイントなしで組み込みの正則化が欲しい。
- 学習中のドリフトや不安定なヘッドを抑えるため、中間レイヤーをより深い教師に引き寄せたい。
- 現在ステップの学生/教師アクティベーションを保持できるだけの VRAM 余裕がある。

## クイック設定（WebUI）

1. **Training → Loss functions** を開く。
2. **LayerSync** を有効化。
3. **Student Block** を中間レイヤーに、**Teacher Block** をより深いレイヤーに設定。24 層 DiT 系（Flux、PixArt、SD3）なら `8` → `16` から開始。短いスタックでは教師を学生より数ブロック深くする。
4. **Weight** は `0.2` のまま（LayerSync 有効時の既定値）。
5. 通常通り学習。ログに `layersync_loss` と `layersync_similarity` が出ます。

## クイック設定（config JSON / CLI）

```json
{
  "layersync_enabled": true,
  "layersync_student_block": 8,
  "layersync_teacher_block": 16,
  "layersync_lambda": 0.2
}
```

## 調整項目

- `layersync_student_block` / `layersync_teacher_block`: 1 始まり互換のインデックス。まず `idx-1`、次に `idx` を試します。
- `layersync_lambda`: コサイン損失のスケール。有効時は > 0 が必要（既定 `0.2`）。
- 教師は未指定時に学生ブロックを使用し、自己類似損失になります。
- VRAM: 補助損失計算まで両レイヤーのアクティベーションを保持。メモリが厳しい場合は LayerSync（または CREPA）を無効化。
- CREPA/TwinFlow と併用可能。同じ隠れ状態バッファを共有します。

<details>
<summary>仕組み（実務者向け）</summary>

- 学生/教師トークンをフラット化し、負のコサイン類似度を計算。重みを上げるほど学生が教師特徴に近づきます。
- 教師トークンは常にデタッチし、勾配が逆流しないようにします。
- 画像/動画 Transformer の 3D `(B, S, D)` と 4D `(B, T, P, D)` の隠れ状態を扱います。
- 上流オプション対応:
  - `--encoder-depth` → `--layersync_student_block`
  - `--gt-encoder-depth` → `--layersync_teacher_block`
  - `--reg-weight` → `--layersync_lambda`
- 既定: 無効。有効時に未設定なら `layersync_lambda=0.2`。

</details>

<details>
<summary>技術詳細（SimpleTuner 内部）</summary>

- 実装: `simpletuner/helpers/training/layersync.py`; `ModelFoundation._apply_layersync_regularizer` から呼び出し。
- 隠れ状態キャプチャ: LayerSync または CREPA が要求すると実行。Transformer は `_store_hidden_state` で `layer_{idx}` として保持。
- レイヤー解決: 1 始まり→0 始まりの順で試し、指定レイヤーがなければエラー。
- 損失経路: 学生/教師トークンを正規化し、平均コサイン類似度を計算。`layersync_loss` と `layersync_similarity` をログし、スケールした損失を主目的に加算。
- 相互作用: CREPA の後に実行され、同じバッファを再利用。処理後にバッファをクリア。

</details>

## よくある落とし穴

- 学生ブロック未設定 → 起動時エラー。`layersync_student_block` を明示的に設定。
- 重み ≤ 0 → 起動時エラー。不明なら既定の `0.2` を使用。
- モデルが持たない深さを指定 → “LayerSync could not find layer” エラー。インデックスを下げる。
- Transformer の隠れ状態を公開しないモデル（Kolors、Lumina2、Stable Cascade C、Kandinsky5 Image、OmniGen）では失敗。Transformer 系に限定してください。
- VRAM スパイク: ブロック番号を下げるか、CREPA/LayerSync を無効化してバッファを解放。

LayerSync は外部教師なしで中間表現を穏やかに整える、低コストな組み込み正則化として使えます。
