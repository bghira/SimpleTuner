# NSFW 分類器チェック

SimpleTuner には、VAE キャッシュ前処理中にサンプルを拒否できる任意の分類器チェックがあります。この機能はローカルのフィルタリングツールです。法律助言、コンプライアンスシステム、または特定の用途でデータセットが合法または許容されることの保証ではありません。

## ユーザーの責任

データセット、学習実行、モデル出力、公開または配布計画が、適用されるルールに従っているかどうかはユーザー自身が判断する必要があります。

それらのルールには、地域、国、プラットフォーム固有の要件が含まれる場合があります。また、同意、年齢、肖像権、プライバシー、パブリシティ権、わいせつ規制、雇用または組織の方針、結果が実在人物を描写またはなりすましているかどうかに依存する場合があります。法律は時間とともに変化し、管轄区域によって異なります。

SimpleTuner はこれを判断しません。ポリシーが不完全であることを警告したり、しきい値が法律に合っているか確認したり、モデル出力が公開して安全かを確認したりしません。不明な場合は、自分の管轄区域とユースケースについて資格のある法律専門家に相談してください。

## プライバシー

NSFW 分類器チェックは、SimpleTuner を実行しているマシン上でローカルに実行されます。

- この機能はデータセットサンプルを第三者のモデレーション API に送信しません。
- 分類器の結果は第三者に転送されません。
- `--report_to` の学習 telemetry オプションは NSFW 分類器結果を受け取りません。
- レポートは `nsfw_classifier_report_rank*.json` として VAE キャッシュディレクトリ内のインスタンス上にローカル保存されます。

想定されるネットワーク動作は、分類器の重みがローカルモデルキャッシュにない場合の通常の Hugging Face モデル読み込みです。モデルがローカルで利用可能になった後、分類処理自体はインスタンス上で実行されます。

## オプトイン動作

この機能は既定では無効です。次のように有効化します。

```bash
--enable_nsfw_check=true
```

チェックは、VAE キャッシュがこれから処理する未キャッシュのサンプルだけに適用されます。既存の VAE キャッシュは信頼され、`skip_file_discovery=vae` は enforcement を回避します。SimpleTuner は、ユーザー自身のポリシーでキャッシュが準備済みであるとみなすためです。

評価データセットはスキャンされません。

## 対応分類器

SimpleTuner は `AutoImageProcessor` と `AutoModelForImageClassification` を通じて、標準 Hugging Face Transformers 画像分類モデルをサポートします。

既定のモデルは次のとおりです。

```text
Falconsai/nsfw_image_detection:threshold=0.5,AdamCodd/vit-base-nsfw-detector:threshold=0.5
```

独自の CSV リストを指定できます。

```bash
--nsfw_check_models="org/model-a:threshold=0.5,org/model-b:threshold=0.7"
```

SimpleTuner はこれらの分類器に対して `trust_remote_code` を有効化せず、この機能のために `timm` を依存関係に追加しません。カスタムコードまたは非 Transformers バックエンドを必要とするモデルは、このスキャナーではサポートされません。

## NSFW 以外の用途

オプション名に NSFW とありますが、この仕組みは性的コンテンツのフィルタリングに限定されません。分類器が SimpleTuner の想定する unsafe/safe ラベルヒントにきれいに対応するラベルとスコアを出力する場合、他の二値またはラベルスコアチェックにも使えます。

例として、禁止された視覚カテゴリ、ブランド上センシティブな内容、その他ローカルに定義したデータセットポリシーを持つサンプルの拒否があります。それでも、分類器ラベル、しきい値、投票設定が自分のポリシーに合っているか検証する責任はユーザーにあります。

## 法的文脈

成人向け性的コンテンツは、すべての場所で自動的に違法になるわけではなく、SimpleTuner も NSFW モデル学習を自動的に禁止しません。ただし、特定のデータセット、出力、またはデプロイが合法であることを意味するものではありません。

高リスク領域には次が含まれます。

- 未成年者または未成年者に見える人物を含むコンテンツ。米国 FBI Internet Crime Complaint Center は、生成 AI や類似ツールで作成された児童性的虐待資料は違法であると述べています。
- 同意のない親密画像、性的搾取、嫌がらせ、恐喝、または許可のない配布。
- 実在人物をなりすまし、再現し、または誤解を招く形で描写する出力。特に性的、詐欺的、または評判を害する目的の場合。FTC は AI によるなりすましと deepfake 詐欺のリスクを強調しています。
- Deepfake の開示および透明性ルール。たとえば EU AI Act Article 50 には、deepfake を構成する特定の AI 生成または操作された画像、音声、動画コンテンツに対する透明性義務が含まれます。
- 契約またはプラットフォーム規則。データセットライセンス、ホスティングプロバイダーポリシー、職場規則、決済事業者規則、モデル配布条件など。

分類器は自分のレビュー手順の一つの制御として扱い、レビュー手順そのものとして扱わないでください。

## 関連オプション

- `--enable_nsfw_check`
- `--nsfw_check_models`
- `--nsfw_check_min_votes`
- `--nsfw_check_backend_types`
- `--nsfw_check_sample_types`
- `--delete_nsfw_images`
- `--nsfw_check_video_frame_count`
- `--nsfw_check_video_frame_selection`
- `--nsfw_check_video_min_flagged_frames`

VAE キャッシュ統合の詳細は [DATALOADER.ja.md#nsfw-classifier-checks-during-vae-caching](DATALOADER.ja.md#nsfw-classifier-checks-during-vae-caching) を参照してください。

## 参考資料

- [FBI IC3: Child Sexual Abuse Material Created by Generative AI and Similar Online Tools is Illegal](https://www.ic3.gov/PSA/2024/PSA240329)
- [FTC: Proposed protections to combat AI impersonation of individuals](https://www.ftc.gov/news-events/news/press-releases/2024/02/ftc-proposes-new-protections-combat-ai-impersonation-individuals)
- [EU AI Act Article 50: transparency obligations](https://ai-act-service-desk.ec.europa.eu/en/ai-act/article-50)
