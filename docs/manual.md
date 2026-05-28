# AbyssEye User Manual

## 1. Overview

AbyssEye は、TIFF 画像やリアルタイム取得画像をもとに、ROI 抽出、推論、DeepScan による確認、手動ラベル修正、細胞数集計、再学習用データの準備を行うためのツールです。

主な用途は次の通りです。

- 保存済み画像をプロジェクト単位で整理して処理する
- 顕微鏡などから送られる最新画像をリアルタイムで確認し、保存または破棄する
- 手動修正済み ROI を学習用データとして蓄積し、再学習に回す

## 2. Startup

バックエンド:

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
python backend/main.py
```

フロントエンド:

```bash
cd frontend
npm install
npm run dev
```

起動後は通常、次を開きます。

- Frontend: `http://localhost:3000`
- API docs: `http://localhost:8000/api/v1/docs`

トップ画面でバックエンド状態が `ok` と表示されていれば、基本的な接続は正常です。

生成される TIFF、DB、再学習結果をソースツリー外に置きたい場合は、バックエンド起動時に `ABYSSEYE_DATA_DIR` を指定します。

```bash
ABYSSEYE_DATA_DIR=./data python backend/main.py
```

## 3. Home

ホーム画面には次の入口があります。

- プロジェクト: 画像をプロジェクト単位で整理します。
- モデル選択: 推論モデルの確認、アップロード、切り替えを行います。
- 再学習: 既存プロジェクトまたは保存済み ZIP から再学習データを準備します。
- Swagger: バックエンド API の一覧を確認します。

## 4. Model Selection

推論を使う前に、モデル選択画面で使用するモデルをアクティブ化してください。

1. ホーム画面で `モデル選択` を開きます。
2. 必要に応じてモデルをアップロードします。
3. 使用したいモデルを `アクティブ化` します。

アクティブモデルは、画像処理、リアルタイム処理、DeepScan の推論、再学習時の比較に使われます。

## 5. Project Workflow

1. ホーム画面で `プロジェクト` を開きます。
2. プロジェクトを作成または選択します。
3. 保存済み画像をアップロードするか、リアルタイムエンジンへ進みます。
4. ROI 抽出と推論を実行します。
5. 必要に応じて DeepScan で manual label や ROI を修正します。
6. セルカウント結果やプロジェクト ZIP を保存します。

単一画像と同視野画像フォルダの両方を扱えます。同視野画像フォルダでは、マージ画像の作成とフォルダ単位の ROI 抽出を実行できます。

## 6. Realtime Workflow

リアルタイムエンジンでは、監視フォルダに追加された TIFF をバックエンドへ送信するスクリプトを生成できます。

1. プロジェクト詳細画面から `リアルタイムエンジン` を開きます。
2. 監視フォルダのパスと送信先 API URL を確認します。
3. OS に応じた監視スクリプトをダウンロードします。
4. カメラ PC などでスクリプトを実行します。
5. 新しい画像が来たら内容を確認し、必要に応じて保存または破棄します。

カメラ PC からバックエンドへ接続する場合、`localhost` ではなく、カメラ PC から到達できるバックエンドの IP アドレスまたはホスト名を指定してください。

## 7. DeepScan

DeepScan では、ROI 抽出と推論が終わった画像を詳細確認できます。

- manual label の付与
- 手動 ROI の追加
- 手動追加 ROI の削除
- 推論結果と手動修正の確認

manual label を付けると、集計や再学習データでは AI 推論より手動ラベルが優先されます。

## 8. Retraining

再学習ページでは、既存プロジェクトまたは保存済み ZIP を再学習データソースとして選択できます。

主な操作:

- 既存プロジェクトを選択する
- 保存済み ZIP をアップロードする
- 対象モデルとデータセット内の `ai_model_name` の一致を確認する
- epoch、batch size、learning rate を指定して再学習を開始する
- 完了したモデルをモデル選択へ追加する

再学習に使う ZIP には `_training_dataset/labels.csv` と `class0` から `class3` の画像が含まれている必要があります。

## 9. Class Labels

- `0`: single cell
- `1`: multiple cells
- `2`: out of focus
- `3`: non-cell particle

## 10. API Docs

Swagger は API の確認と手動テスト用です。

1. ホーム画面で `Swagger` を開きます。
2. API 一覧と request/response schema を確認します。
3. 必要に応じて `別タブで開く` から直接表示します。

## 11. Troubleshooting

- トップ画面でバックエンド接続エラーが出る: バックエンドが起動しているか確認してください。
- 推論できない: モデル選択でアクティブモデルが設定されているか確認してください。
- DeepScan が開けない: 対象画像の ROI 抽出と推論が完了しているか確認してください。
- 再学習できない: manual label または DeepScan 確認済み ROI を含むプロジェクト ZIP を用意してください。
- フロントエンドの変更が反映されない: 開発サーバーまたはブラウザを再読み込みしてください。
