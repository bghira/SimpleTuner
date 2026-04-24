# SimpleTuner WebUI チュートリアル

## はじめに

このチュートリアルでは、SimpleTuner Web インターフェースの使い方をご案内します。

## 必要なパッケージのインストール

Ubuntu システムの場合、まず必要なパッケージをインストールします:

```bash
apt -y install python3.13-venv python3.13-dev
apt -y install libopenmpi-dev openmpi-bin cuda-toolkit-12-8 libaio-dev # DeepSpeed を使用する場合
apt -y install ffmpeg # ビデオモデルをトレーニングする場合
```

## ワークスペースディレクトリの作成

ワークスペースには、設定ファイル、出力モデル、検証画像、そして場合によってはデータセットが含まれます。

Vast などのプロバイダーでは、`/workspace/simpletuner` ディレクトリを使用できます:

```bash
mkdir -p /workspace/simpletuner
export SIMPLETUNER_WORKSPACE=/workspace/simpletuner
cd $SIMPLETUNER_WORKSPACE
```

ホームディレクトリに作成する場合は:
```bash
mkdir ~/simpletuner-workspace
export SIMPLETUNER_WORKSPACE=~/simpletuner-workspace
cd $SIMPLETUNER_WORKSPACE
```

## SimpleTuner をワークスペースにインストール

依存関係をインストールするための仮想環境を作成します:

```bash
python3.13 -m venv .venv
. .venv/bin/activate
```

### CUDA 固有の依存関係

NVIDIA ユーザーは、すべての正しい依存関係を取得するために CUDA extras を使用する必要があります:

```bash
pip install -e 'simpletuner[cuda]'
# CUDA 13 / Blackwell users (NVIDIA B-series GPUs):
# pip install -e 'simpletuner[cuda13]' --extra-index-url https://download.pytorch.org/whl/cu130
# または、git 経由でクローンした場合:
# pip install -e '.[cuda]'
```

Apple や ROCm ハードウェアのユーザー向けの他の extras については、[インストール手順](../INSTALL.md)を参照してください。

## サーバーの起動

ポート 8080 で SSL を有効にしてサーバーを起動するには:

```bash
# DeepSpeed の場合、CUDA_HOME を正しい場所に向ける必要があります
export CUDA_HOME=/usr/local/cuda-12.8
export LIBRARY_PATH=$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib/stubs:$LD_LIBRARY_PATH

simpletuner server --ssl --port 8080
```

次に、ウェブブラウザで https://localhost:8080 にアクセスします。

SSH 経由でポートをフォワーディングする必要がある場合があります。例えば:

```bash
ssh -L 8080:localhost:8080 user@remote-server
```

> **ヒント:** 既存の設定環境(例: 以前の CLI 使用から)がある場合、`--env` を使ってサーバーを起動すると、サーバーの準備が整い次第、自動的にトレーニングを開始できます:
>
> ```bash
> simpletuner server --ssl --port 8080 --env my-training-config
> ```
>
> これは、サーバーを起動してから WebUI で手動で「トレーニング開始」をクリックするのと同等ですが、無人起動が可能です。

## 初回セットアップ: 管理者アカウントの作成

初回起動時、SimpleTuner は管理者アカウントの作成を要求します。初めて WebUI にアクセスすると、最初の管理者ユーザーを作成するよう促すセットアップ画面が表示されます。

メールアドレス、ユーザー名、および安全なパスワードを入力してください。このアカウントは完全な管理者権限を持ちます。

### ユーザー管理

セットアップ後、**ユーザー管理**ページ(管理者としてログインしているときにサイドバーからアクセス可能)からユーザーを管理できます:

- **ユーザータブ**: ユーザーアカウントの作成、編集、削除。権限レベル(閲覧者、研究者、リーダー、管理者)の割り当て。
- **レベルタブ**: きめ細かいアクセス制御を持つカスタム権限レベルの定義。
- **認証プロバイダータブ**: シングルサインオンのための外部認証(OIDC、LDAP)の設定。
- **登録タブ**: 新しいユーザーの自己登録を許可するかどうかの制御(デフォルトでは無効)。

### 自動化のための API キー

ユーザーは、プロフィールまたは管理者パネルからスクリプトアクセス用の API キーを生成できます。API キーは `st_` プレフィックスを使用し、`X-API-Key` ヘッダーで使用できます:

```bash
curl -s http://localhost:8080/api/training/status \
  -H 'X-API-Key: st_your_key_here'
```

> **注意:** プライベート/内部デプロイメントの場合、公開登録を無効にしたままにし、管理者パネルを通じて手動でユーザーアカウントを作成してください。

## WebUI の使用

### オンボーディングステップ

ページが読み込まれると、環境をセットアップするためのオンボーディング質問が表示されます。

#### 設定ディレクトリ

特別な設定値 `configs_dir` は、すべての SimpleTuner 設定を含むフォルダを指すために導入されました。サブディレクトリに整理することが推奨されます - **Web UI がこれを自動的に行います**:

```
configs/
├── an-environment-named-something
│   ├── config.json
│   ├── lycoris_config.json
│   └── multidatabackend-DataBackend-Name.json
```

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/4be22081-f13d-4aed-a87c-2313ddefc8a4" />

##### コマンドライン使用からの移行

以前に WebUI なしで SimpleTuner を使用していた場合、既存の config/ フォルダを指すことで、すべての環境が自動的に検出されます。

新規ユーザーの場合、設定とデータセットのデフォルトの場所は `~/.simpletuner/` となり、データセットをより多くのスペースがある場所に移動することが推奨されます:

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/c5b3ab53-654e-4a9b-8e2d-7951f11619ef" />


#### (マルチ-)GPU の選択と設定

デフォルトパスの設定後、マルチ GPU を設定できるステップに到達します(NVIDIA システムで撮影)

<img width="429" height="640" alt="image" src="https://github.com/user-attachments/assets/61d5a7bc-0a02-4a0a-8df0-207cce4b7bc1" />

複数の GPU があり、2 番目のものだけを使用したい場合は、ここで設定できます。

> **マルチ GPU ユーザーへの注意:** 複数の GPU でトレーニングする場合、データセットサイズの要件は比例して増加します。有効なバッチサイズは `train_batch_size × num_gpus × gradient_accumulation_steps` として計算されます。データセットがこの値より小さい場合は、データセット設定で `repeats` 設定を増やすか、詳細設定で `--allow_dataset_oversubscription` オプションを有効にする必要があります。詳細については、以下の[バッチサイズセクション](#マルチ-gpu-バッチサイズの考慮事項)を参照してください。

#### 最初のトレーニング環境の作成

`configs_dir` に既存の設定が見つからなかった場合、**最初のトレーニング環境**を作成するよう求められます:

<img width="500" height="640" alt="image" src="https://github.com/user-attachments/assets/2110287a-16fd-4f87-893b-86d2a555a10f" />

**例から起動**を使用して、開始する例の設定を選択するか、説明的な名前を入力してランダムな環境を作成することもできます。セットアップウィザードを使用したい場合は後者を選択してください。

### トレーニング環境の切り替え

既存の設定環境がある場合、このドロップダウンメニューに表示されます。

それ以外の場合、オンボーディング中に作成したオプションが既に選択されアクティブになっています。

<img width="448" height="225" alt="image" src="https://github.com/user-attachments/assets/66fef6a9-2040-47fd-b22d-918470677992" />

**設定の管理**を使用して、環境、データローダー、その他の設定のリストがある `Environment` タブに移動します。

### 設定ウィザード

最も重要な設定のいくつかを、分かりやすいブートストラップで設定するための包括的なセットアップウィザードを提供するために努力しました。

<img width="470" height="358" alt="image" src="https://github.com/user-attachments/assets/e4bf1a4e-716c-4101-b753-e9e24bb42d8a" />

左上のナビゲーションメニューで、ウィザードボタンをクリックすると選択ダイアログが表示されます:

<img width="448" height="440" alt="image" src="https://github.com/user-attachments/assets/68324fa8-3ca9-45b1-b947-1e7738fd1d8c" />

そして、すべての組み込みモデルバリアントが提供されます。各バリアントは、アテンションマスキングや拡張トークン制限などの必要な設定を事前に有効にします。

#### LoRA モデルオプション

LoRA をトレーニングしたい場合、ここでモデルの量子化オプションを設定できます。

一般的に、Stable Diffusion タイプのモデルをトレーニングしている場合を除き、int8-quanto が推奨されます。品質を損なうことなく、より高いバッチサイズを可能にします。

Cosmos2、Sana、PixArt などの一部の小さなモデルは、量子化を好みません。

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/c2e721f2-b4da-4cd0-84fd-7ac81993e87c" />

#### フルランクトレーニング

フルランクトレーニングは推奨されません。通常、同じデータセットの場合、LoRA/LyCORIS よりもはるかに長い時間がかかり、リソースのコストも高くなります。

ただし、完全なチェックポイントをトレーニングしたい場合は、Auraflow、Flux などの大きなモデルに必要な DeepSpeed ZeRO ステージをここで設定できます。

FSDP2 はサポートされていますが、このウィザードでは設定できません。DeepSpeed を無効のままにして、後で使用したい場合は FSDP2 を手動で設定してください

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/88438f1c-b0a2-4249-afd0-7878aa1abada" />


#### トレーニング期間の設定

トレーニング時間をエポックまたはステップのどちらで測定するかを決定する必要があります。最終的にはほぼ同じですが、人によって好みが分かれることがあります。

<img width="508" height="475" alt="image" src="https://github.com/user-attachments/assets/dcb54279-0ce7-4c66-a9ab-4dd26f87278c" />

#### Hugging Face Hub を介したモデルの共有

オプションで、最終的な*および*中間チェックポイントを [Hugging Face Hub](https://hf.co) に公開できますが、アカウントが必要です - ウィザードまたは公開タブからハブにログインできます。いずれにしても、いつでも気が変わって有効または無効にすることができます。

モデルを公開する場合、モデルを一般公開したくない場合は `Private repo` を選択することを忘れないでください。

<img width="508" height="370" alt="image" src="https://github.com/user-attachments/assets/8d2d282b-e66f-48a8-a40e-4e4ecc2d280b" />

#### チェックポイント頻度

トレーニング中、モデルは定期的にディスクへ保存されます。より多くのチェックポイントを保持するには、より多くのディスク容量が必要です。

チェックポイントがあると、すべてのステップを繰り返さずに後でトレーニングを再開できます。いくつかのチェックポイントを保持しておくと、モデルの複数のバージョンを試し、自分にとって最も良いものを残せます。

10% ごとにチェックポイントを保持することが推奨されますが、トレーニングするデータ量によって異なります。小さなデータセットでは、過学習していないことを確認するために頻繁にチェックポイントを保存するとよいでしょう。

非常に大きなデータセットでは、ディスクへの書き込みに時間を浪費しないよう、チェックポイント間隔を長くする方が有利です。

<img width="508" height="485" alt="image" src="https://github.com/user-attachments/assets/c7b1cd0b-a1b9-47ec-87f9-1ecac2e0841a" />

#### モデルの検証

トレーナーに定期的に画像を生成させたい場合は、ウィザードのこの時点で単一の検証プロンプトを設定できます。複数のプロンプトライブラリは、ウィザード完了後に `Validations & Output` タブで設定できます。

検証を独自のスクリプトやサービスにアウトソースしたいですか? ウィザード後に検証タブで**検証方法**を `external-script` に切り替え、`--validation_external_script` を指定してください。`{local_checkpoint_path}`、`{global_step}`、`{tracker_run_name}`、`{tracker_project_name}`、`{model_family}`、`{huggingface_path}` などのプレースホルダーを使用して、トレーニングコンテキストをスクリプトに渡すことができます。また、任意の `validation_*` 設定値(例: `validation_num_inference_steps`、`validation_guidance`、`validation_noise_scheduler`)も使用できます。トレーニングをブロックせずに実行するには、`--validation_external_background` を有効にしてください。

チェックポイントがディスクに保存された瞬間にフックが必要ですか? 各保存の直後(アップロード開始前)にスクリプトを実行するには、`--post_checkpoint_script` を使用してください。同じプレースホルダーを受け入れ、`{remote_checkpoint_path}` は空のままです。

SimpleTuner の組み込み公開プロバイダー(または Hugging Face Hub アップロード)を保持しながら、リモート URL で独自の自動化をトリガーしたい場合は、代わりに `--post_upload_script` を使用してください。プレースホルダー `{remote_checkpoint_path}`、`{local_checkpoint_path}`、`{global_step}`、`{tracker_run_name}`、`{tracker_project_name}`、`{model_family}`、`{huggingface_path}` を使用して、アップロードごとに 1 回実行されます。SimpleTuner はスクリプトの出力をキャプチャしません—スクリプトから直接トラッカーの更新を発行してください。

フックの例:

```bash
--post_upload_script='/opt/hooks/notify.sh {remote_checkpoint_path} {tracker_project_name} {tracker_run_name}'
```

ここで、`notify.sh` は URL をトラッカーの Web API に投稿します。Slack、カスタムダッシュボード、またはその他の統合に自由に適応させてください。

実際のサンプル: `simpletuner/examples/external-validation/replicate_post_upload.py` は、`{remote_checkpoint_path}`、`{model_family}`、`{model_type}`、`{lora_type}`、`{huggingface_path}` を使用して、アップロード後に Replicate 推論をトリガーする方法を示しています。

別のサンプル: `simpletuner/examples/external-validation/wavespeed_post_upload.py` は WaveSpeed API を呼び出し、同じプレースホルダーを使用して結果をポーリングします。

Flux に焦点を当てたサンプル: `simpletuner/examples/external-validation/fal_post_upload.py` は fal.ai Flux LoRA エンドポイントを呼び出します。`FAL_KEY` が必要で、`model_family` に `flux` が含まれている場合にのみ実行されます。

ローカル GPU サンプル: `simpletuner/examples/external-validation/use_second_gpu.py` は別の GPU(デフォルトは `cuda:1`)で Flux LoRA 推論を実行し、アップロードが発生しない場合でも使用できます。

<img width="508" height="600" alt="image" src="https://github.com/user-attachments/assets/e699ba57-526b-4f60-9e8c-0ba410761c9f" />

#### トレーニング統計のログ記録

SimpleTuner は、トレーニング統計を送信したい場合、複数のターゲット API をサポートしています。

注意: 個人データ、トレーニングログ、キャプション、またはデータは SimpleTuner プロジェクト開発者に**決して**送信されません。データの制御は**あなた**の手にあります。

<img width="508" height="438" alt="image" src="https://github.com/user-attachments/assets/0f8d15c5-456f-4637-af7e-c2f5f31cb968" />

#### データセット設定

この時点で、既存のデータセットを保持するか、データセット作成ウィザードを通じて新しい設定を作成する(他のデータセットはそのまま)かを決定できます。クリックすると表示されます。

<img width="508" height="290" alt="image" src="https://github.com/user-attachments/assets/b5a7f883-e180-4662-b84c-fff609c6b1df" />

##### データセットウィザード

新しいデータセットを作成することを選択した場合、次のウィザードが表示され、ローカルまたはクラウドのデータセットの追加を案内されます。

<img width="508" height="332" alt="image" src="https://github.com/user-attachments/assets/c523930b-563e-4b5d-b104-8e7ce4658b2c" />

<img width="508" height="508" alt="image" src="https://github.com/user-attachments/assets/c263f58e-fd85-437e-811a-967b94e309fd" />

ローカルデータセットの場合、**ディレクトリを参照**ボタンを使用してデータセットブラウザモーダルにアクセスできます。

<img width="396" height="576" alt="image" src="https://github.com/user-attachments/assets/14c51685-3559-4d16-be59-ed4b0959ca32" />

オンボーディング中にデータセットディレクトリを正しく指定した場合、ここにファイルが表示されます。

追加したいディレクトリをクリックして、**ディレクトリを選択**してください。

<img width="454" height="356" alt="image" src="https://github.com/user-attachments/assets/1d482655-158a-4e3f-93b7-ef158396813c" />

その後、解像度値とクロッピングの設定を案内されます。

キャプションを設定するステップに到達したら、どのオプションが正しいか**慎重に検討**してください。

単一のトリガーワードを使用したい場合は、**インスタンスプロンプト**オプションになります。

<img width="1146" height="896" alt="image" src="https://github.com/user-attachments/assets/6252bf9a-5e68-41c6-8a95-906993f2f546" />

##### オプション: ブラウザからデータセットをアップロード

画像とキャプションがまだボックスにない場合、データセットウィザードには**ディレクトリを参照**の横に**アップロード**ボタンが含まれるようになりました。次のことができます:

- 設定されたデータセットディレクトリの下に新しいサブフォルダを作成し、個々のファイルまたは ZIP をアップロードします(画像と .txt/.jsonl/.csv メタデータが受け入れられます)。
- SimpleTuner にそのフォルダに ZIP を展開させます(ローカルバックエンド用にサイズ設定されています。非常に大きなアーカイブは拒否されます)。
- UI を離れることなく、ブラウザで新しくアップロードされたフォルダをすぐに選択し、ウィザードを続行します。

#### 学習率、バッチサイズ、オプティマイザ

データセットウィザードを完了すると(または既存のデータセットを保持することを選択した場合)、オプティマイザ/学習率とバッチサイズのプリセットが提供されます。

これらは、初心者が最初のいくつかのトレーニング実行でやや良い選択をするのに役立つ出発点です - 経験豊富なユーザーは、完全な制御のために**手動設定**を使用してください。

**注意**: 後で DeepSpeed を使用する予定がある場合、ここでのオプティマイザの選択はあまり重要ではありません。

##### マルチ GPU バッチサイズの考慮事項 {#マルチ-gpu-バッチサイズの考慮事項}

複数の GPU でトレーニングする場合、データセットは**有効なバッチサイズ**に対応する必要があることに注意してください:

```
effective_batch_size = train_batch_size × num_gpus × gradient_accumulation_steps
```

データセットがこの値より小さい場合、SimpleTuner は特定のガイダンスを含むエラーを発生させます。次のことができます:
- バッチサイズを減らす
- データセット設定で `repeats` 値を増やす
- 詳細設定で**データセットオーバーサブスクリプションを許可**を有効にして、繰り返しを自動的に調整する

データセットのサイズ設定の詳細については、[DATALOADER.md](../DATALOADER.md#multi-gpu-training-and-dataset-sizing) を参照してください。

<img width="1118" height="1015" alt="image" src="https://github.com/user-attachments/assets/25d5650d-e77b-42fe-b749-06c0ec92b1e2" />

#### メモリ最適化プリセット

コンシューマーハードウェアでのより簡単なセットアップのために、各モデルには、軽量、バランス、または積極的なメモリ節約を選択できるカスタムプリセットが含まれています。

**トレーニング**タブの**メモリ最適化**セクションに、**プリセットの読み込み**ボタンがあります:

<img width="1048" height="940" alt="image" src="https://github.com/user-attachments/assets/804e84f6-7eb8-493e-95d2-a89d930bafa5" />

このインターフェースが表示されます:

<img width="1048" height="940" alt="image" src="https://github.com/user-attachments/assets/775aaee5-c3c0-4659-bbea-ebb39e3eb098" />


#### 確認と保存

選択したすべての値に満足している場合は、ウィザードを**完了**してください。

そして、新しい環境がアクティブに選択され、トレーニングの準備ができていることがわかります!

ほとんどの場合、これらの設定がすべて設定する必要があったものです。追加のデータセットを追加したり、他の設定をいじったりすることもできます。

<img width="1096" height="1403" alt="image" src="https://github.com/user-attachments/assets/29fd0bb3-aab2-4455-9612-583ed949ce64" />

**環境**ページには、新しく設定されたトレーニングジョブと、テンプレートとして使用したい場合に設定をダウンロードまたは複製するボタンが表示されます。

<img width="1881" height="874" alt="image" src="https://github.com/user-attachments/assets/33c0cafa-3fd8-40ee-b6fa-3704b6e698da" />

**注意**: **デフォルト**環境は特別であり、一般的なトレーニング環境として使用することは推奨されません。その設定は、そのオプションを有効にする任意の環境に自動的にマージできます、**環境のデフォルトを使用**:

<img width="1521" height="991" alt="image" src="https://github.com/user-attachments/assets/9d18b0c1-608e-4ab2-be14-65b98907ec69" />
