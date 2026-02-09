# AbyssEye

## バックエンドのセットアップ

1. 仮想環境を作成して有効化します（未作成の場合）:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. 依存関係をインストールします:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. API サーバーを起動します:
   ```bash
   python backend/main.py
   ```

## フロントエンドのセットアップ

1. 依存関係をインストールします:
   ```bash
   cd frontend
   npm install
   ```
2. 開発サーバーを起動します:
   ```bash
   npm run dev
   ```