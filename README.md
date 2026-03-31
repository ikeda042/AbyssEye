# AbyssEye

AbyssEye は、顕微鏡画像の ROI 抽出、DeepScan、リアルタイム取り込み、データベース管理を行うためのアプリケーションです。

このリポジトリは、`git clone` 後に `backend` と `frontend` を起動すればそのまま使える構成を前提にしています。

## 必要環境

- Python 3.11
- Node.js 20 以上
- npm

## 1. リポジトリを取得

```bash
git clone <YOUR_REPOSITORY_URL>
cd AbyssEye
```

## 2. Backend を起動

別ターミナルで実行します。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r backend/requirements.txt
python backend/main.py
```

起動先:

- API health check: `http://localhost:8000/api/v1/`
- OpenAPI JSON: `http://localhost:8000/api/v1/openapi.json`
- Swagger UI: `http://localhost:8000/api/v1/docs`

必要に応じて以下の環境変数を使えます。

```bash
APP_HOST=0.0.0.0
APP_PORT=8000
APP_RELOAD=true
```

## 3. Frontend を起動

さらに別ターミナルで実行します。

```bash
cd frontend
npm install
npm run dev
```

起動先:

- Frontend: `http://localhost:3000`

## 開発時の基本動作

1. Backend を起動する
2. Frontend を起動する
3. ブラウザで `http://localhost:3000` を開く

## Backend の確認用 URL

- ヘルスチェック: `http://localhost:8000/api/v1/`
- Swagger UI: `http://localhost:8000/api/v1/docs`
- OpenAPI JSON: `http://localhost:8000/api/v1/openapi.json`

`http://localhost:8000/docs` ではなく、`/api/v1/docs` が正しい URL です。

## Frontend を backend から配信する場合

frontend を build すると、backend 側から配信できます。

```bash
cd frontend
npm install
npm run build
```

この状態で backend を起動すると、`http://localhost:8000/` から build 済み frontend を配信できます。

## ディレクトリ構成

```text
AbyssEye/
├── backend/   # FastAPI, ROI 抽出, 推論, DB 管理
├── frontend/  # React + Vite の UI
└── README.md
```

## 補足

- `frontend/README.md` のような Vite 初期テンプレート説明は削除し、ルートの README に統一しています。
- 試作・個人環境依存のコードは整理済みです。運用対象は `backend/` と `frontend/` です。
