# AbyssEye

## 最短手順

推奨の開発フローは、セットアップを 1 回、起動を 1 コマンドです。

1. セットアップ
   ```bash
   ./scripts/setup-dev.sh
   ```
2. backend / frontend をまとめて起動
   ```bash
   ./scripts/dev-up.sh
   ```

この方法なら、次の問題をまとめて吸収できます。

- `Python 3.14` ではなく `Python 3.11` の `venv` を使う
- 壊れた `venv` を自動で作り直す
- backend / frontend の依存関係をまとめて入れる
- `8000` や `3000` が埋まっているときは空いているポートを使う
- backend の `matplotlib` 初回警告を避ける

## 前提

バックエンドは `Python 3.11` を推奨します。`Python 3.12` でも動く可能性はありますが、`Python 3.13` / `3.14` では TensorFlow がインストールできないため、推論機能は利用できません。

`python3.11` が手元にない場合は、`docker/backend.Dockerfile` と同じく Python 3.11 系の環境を用意してください。

## 手動で起動したい場合

### バックエンド

```bash
source venv/bin/activate
python backend/main.py
```

`8000` が埋まっているときは次のように変えられます。

```bash
APP_PORT=8001 python backend/main.py
```

### フロントエンド

```bash
cd frontend
npm run dev
```

Vite は `3000` が埋まっていても、空いている次のポートで起動します。バックエンドを `8000` 以外で起動した場合だけ、次のように合わせてください。

```bash
cd frontend
VITE_BACKEND_PORT=8001 npm run dev
```

## 補足

- `./scripts/dev-up.sh` は backend の自動リロードをデフォルトで無効にして、終了時にプロセスが残りにくいようにしています。
- backend の自動リロードも欲しい場合は `APP_RELOAD=true ./scripts/dev-up.sh` を使ってください。
