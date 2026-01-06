# SimpleTuner WebUI 実装詳細

## デザイン概要

SimpleTuner の API は柔軟性を重視して設計されています。

`trainer` モードでは、他サービスへの FastAPI 統合のために単一ポートを開きます。
`unified` モードでは、リモート `trainer` プロセスからのコールバックイベントを WebUI が受け取るための追加ポートが開きます。

## Web フレームワーク

WebUI は FastAPI で構築・配信されます：

- Alpine.js はリアクティブコンポーネントに使用
- HTMX は動的コンテンツ読み込みとインタラクションに使用
- FastAPI（Starlette、SSE-Starlette）で SSE によるリアルタイム更新付きのデータ中心 API を提供
- Jinja2 で HTML テンプレートを作成

Alpine はシンプルで統合しやすく、NodeJS をスタックに持ち込まないため、デプロイや保守が容易になります。

HTMX は軽量で分かりやすい構文を持ち、Alpine と相性が良いです。フルフロントエンドフレームワークなしで、動的コンテンツ、フォーム処理、インタラクションを提供します。

Starlette と SSE-Starlette は、重複を最小限に抑えるために選択しました。アドホックな手続きコードからより宣言的な形へ移行するため、多くのリファクタリングが必要でした。

### データフロー

もともと FastAPI アプリはジョブクラスタ内の「サービスワーカー」としても動作していました。trainer が起動し、狭いコールバック面を公開し、リモートのオーケストレーターが HTTP で状態更新をストリーム送信します。WebUI はこの同じコールバックバスを再利用します。unified モードでは trainer と UI を同一プロセスで動かし、trainer 専用デプロイでは `/callbacks` にイベントを送信し、別の WebUI インスタンスが SSE で購読します。新しいキューやブローカーは不要で、ヘッドレス運用に既に存在する基盤を活用します。

## バックエンドアーキテクチャ

Trainer UI は、疎結合の手続き関数ではなく、明確に定義されたサービス群を公開するコア SDK の上に構築されています。FastAPI は引き続きリクエストを終端しますが、多くのルートはサービスオブジェクトへの薄い委譲です。HTTP 層を単純化し、CLI、設定ウィザード、将来の API で再利用しやすくします。

### ルートハンドラ

`simpletuner/simpletuner_sdk/server/routes/web.py` が `/web/trainer` を束ねます。重要なエンドポイントは 2 つです：

- `trainer_page` – 外枠（ナビゲーション、設定セレクタ、タブ一覧）を描画。`TabService` からメタデータを取得し、`trainer_htmx.html` テンプレートに流し込みます。
- `render_tab` – 汎用 HTMX ターゲット。各タブボタンがこのエンドポイントを叩き、`TabService.render_tab` で対応するレイアウトを解決して HTML 断片を返します。

他の HTTP ルータも `simpletuner/simpletuner_sdk/server/routes/` 配下にあり、同様に、ビジネスロジックはサービスに置き、ルートはパラメータを抽出してサービスを呼び出し、JSON/HTML に変換します。

### TabService

`TabService` は学習フォームの中心的なオーケストレーターです。以下を定義します：

- 各タブの静的メタデータ（タイトル、アイコン、テンプレート、任意のコンテキストフック）
- `render_tab()`：
  1. タブ設定（テンプレート、説明）を取得
  2. `FieldService` にタブ/セクションのフィールドバンドルを要求
  3. タブ固有のコンテキスト（データセット一覧、GPU 在庫、オンボーディング状態）を注入
  4. `form_tab.html`、`datasets_tab.html` などを Jinja で描画して返却

このロジックをクラスに閉じ込めることで、HTMX、CLI ウィザード、テストで同じ描画を再利用できます。テンプレートはグローバル状態に依存せず、すべてコンテキストから明示的に渡されます。

### FieldService と FieldRegistry

`FieldService` はレジストリエントリをテンプレート向け辞書に変換します。責務：

- プラットフォーム/モデル文脈によるフィールドのフィルタ（例: MPS 環境では CUDA 専用項目を隠す）
- 依存関係ルール（`FieldDependency`）の評価により UI で無効化/非表示を制御（例: Dynamo の追加項目はバックエンド選択までグレーアウト）
- ヒント、動的選択肢、表示フォーマット、列クラスの付与

生のフィールドカタログは `FieldRegistry` に委譲され、`simpletuner/simpletuner_sdk/server/services/field_registry` に宣言的に配置されています。各 `ConfigField` は CLI フラグ名、検証ルール、重要度順序、依存メタデータ、既定の UI 文言を記述します。この構成により、CLI パーサ、API、ドキュメント生成が同じソースを共有できます。

### 状態の永続化とオンボーディング

WebUI は `WebUIStateStore` で軽量な設定を保存します。`$SIMPLETUNER_WEB_UI_CONFIG`（または XDG パス）から既定値を読み込み、以下を提供します：

- テーマ、データセットルート、出力ディレクトリのデフォルト
- 機能ごとのオンボーディングチェックリスト状態
- Accelerate 上書きのキャッシュ（`--num_processes`、`--dynamo_backend` などホワイトリストのみ）

これらの値は初回の `/web/trainer` 描画時にページへ注入され、Alpine のストアが追加の往復なしで初期化できるようにします。

### HTMX + Alpine の連携

各設定パネルは `x-data` を持つ HTML 断片です。タブボタンは `/web/trainer/tabs/{tab}` に HTMX GET を送信し、サーバーは描画済みフォームを返し、Alpine は既存のコンポーネント状態を保持します。小さなヘルパー（`trainer-form.js`）が保存済みの値変更を再生し、タブ切り替え時に編集中の変更が失われないようにします。

サーバーからの更新（学習ステータス、GPU テレメトリ）は SSE エンドポイント（`sse_manager.js`）を通じて Alpine ストアに流れ、トースト、進捗バー、システムバナーを更新します。

### ファイル配置チートシート

- `templates/` – Jinja テンプレート。`partials/form_field.html` は個々のコントロールを描画。`partials/form_field_htmx.html` はウィザード用の HTMX 版。
- `static/js/modules/` – Alpine コンポーネント（trainer フォーム、ハードウェア在庫、データセットブラウザ）。
- `static/js/services/` – 共有ヘルパー（依存関係評価、SSE 管理、イベントバス）。
- `simpletuner/simpletuner_sdk/server/services/` – バックエンドのサービス層（fields、tabs、configs、datasets、maintenance、events）。

これにより WebUI はサーバー側をステートレスに保ち、状態（フォームデータ、トーストなど）はブラウザに保持されます。バックエンドは純粋なデータ変換に徹するため、テストが容易で、trainer と Web サーバーが同一プロセスで動く場合のスレッド問題も回避できます。
