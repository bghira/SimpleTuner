# FlexAttention ガイド

**FlexAttention は CUDA デバイスが必要です。**

FlexAttention は PyTorch 2.5.0 で導入されたブロック単位の Attention カーネルです。SDPA 計算をプログラマブルなループに書き換えることで、CUDA を書かずにマスク戦略を表現できます。Diffusers は新しい `attention_backend` ディスパッチャ経由で公開しており、SimpleTuner はこれを `--attention_mechanism=flex` に接続しています。

> ⚠️ FlexAttention は上流でまだ「prototype」扱いです。ドライバ、CUDA バージョン、または PyTorch ビルドを変更すると再コンパイルが必要になる可能性があります。

## 前提条件

1. **Ampere+ GPU** – NVIDIA SM80（A100）、Ada（4090/L40S）、Hopper（H100/H200）に対応しています。古いカードはカーネル登録時の能力チェックで失敗します。
2. **コンパイラツールチェーン** – カーネルは `nvcc` により実行時コンパイルされます。ホイールと一致する `cuda-nvcc` を導入し（現行リリースは CUDA 12.8）、`nvcc` が `$PATH` にあることを確認してください。

## カーネルのビルド

`torch.nn.attention.flex_attention` を最初に import すると、CUDA 拡張が PyTorch の遅延キャッシュにビルドされます。事前に実行して、ビルドエラーを早期に露出させることもできます:

```bash
python - <<'PY'
import torch
from torch.nn.attention import flex_attention

assert torch.__version__ >= "2.5.0", torch.__version__
flex_attention.build_flex_attention_kernels()  # no-op when already compiled
print("FlexAttention kernels installed at", flex_attention.kernel_root)
PY
```

- `AttributeError: flex_attention has no attribute build_flex_attention_kernels` が出る場合は PyTorch をアップグレードしてください（2.5.0+ で提供）。
- キャッシュは `~/.cache/torch/kernels` にあります。CUDA を更新して再ビルドを強制したい場合は削除してください。

## SimpleTuner で FlexAttention を有効化

カーネルが存在する状態で、`config.json` でバックエンドを選択します:

```json
{
  "attention_mechanism": "flex"
}
```

期待される動作:

- Diffusers の `attention_backend` を使う Transformer ブロック（Flux、Wan 2.2、LTXVideo、QwenImage など）のみが経路に乗ります。従来の SD/SDXL UNet は PyTorch SDPA を直接使うため、FlexAttention の影響はありません。
- FlexAttention は現在 BF16/FP16 テンソルのみ対応です。FP32 または FP8 の重みを使うと `ValueError: Query, key, and value must be either bfloat16 or float16` が発生します。
- バックエンドは `is_causal=False` のみを許容します。マスクを渡すとカーネルが期待するブロックマスクに変換されますが、任意のラグドマスクはまだ未対応です（上流と同じ挙動）。

## トラブルシューティングチェックリスト

| 症状 | 対処 |
| --- | --- |
| `RuntimeError: Flex Attention backend 'flex' is not usable because of missing package` | PyTorch が 2.5 未満、または CUDA を含んでいません。新しい CUDA ホイールを導入してください。 |
| `Could not compile flex_attention kernels` | `nvcc` が torch ホイールの期待する CUDA バージョン（12.1+）と一致しているか確認してください。インストーラがヘッダを見つけられない場合は `export CUDA_HOME=/usr/local/cuda-12.4` を設定します。 |
| `ValueError: Query, key, and value must be on a CUDA device` | FlexAttention は CUDA 専用です。Apple/ROCm ではバックエンド設定を外してください。 |
| Training never switches to the backend | Diffusers の `dispatch_attention_fn` を既に使っているモデルファミリー（Flux/Wan/LTXVideo）であることを確認してください。標準 SD UNet はどのバックエンドを選んでも PyTorch SDPA を使い続けます。 |

より詳細な内部仕様と API フラグは上流ドキュメントを参照してください: [PyTorch FlexAttention docs](https://pytorch.org/docs/stable/nn.attention.html#flexattention)。
