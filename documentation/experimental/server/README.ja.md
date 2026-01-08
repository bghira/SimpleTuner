# サーバーとマルチユーザー機能

このディレクトリには、ローカル/クラウドの両方の学習デプロイに適用される SimpleTuner のサーバー側機能のドキュメントが含まれています。

## 内容

- [Worker Orchestration](WORKERS.md) - 分散ワーカー登録、ジョブ配布、GPU フリート管理
- [Enterprise Guide](ENTERPRISE.md) - マルチユーザー運用、SSO、承認、クォータ、ガバナンス
- [External Authentication](EXTERNAL_AUTH.md) - OIDC/LDAP の ID プロバイダー設定
- [Audit Logging](AUDIT.md) - チェーン検証付きのセキュリティ監査ログ

## これらのドキュメントを使う場面

以下のような場合に有用です：

- 複数の GPU マシンに学習を分散する（ワーカーオーケストレーション）
- SimpleTuner を複数ユーザーの共有サービスとして運用する
- 企業の ID プロバイダー（Okta、Azure AD、Keycloak、LDAP）と統合する
- ジョブ送信の承認ワークフローが必要
- 監査/コンプライアンスのためにユーザー操作を追跡する
- チームのクォータやリソース制限を管理する

クラウド固有のドキュメント（Replicate、ジョブキュー、Webhook）については、[Cloud Training](../cloud/README.md) を参照してください。
