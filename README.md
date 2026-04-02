# AbyssEye

## 1. Clone

試作品は `AbyssEye-Neo` ブランチで公開しています。

```bash
git clone -b AbyssEye-Neo https://github.com/Gashu884/AbyssEye.git
cd AbyssEye
```

## 2. 前提

- バックエンドは `Python 3.11` を推奨します。
- `Python 3.12` でも動く可能性はありますが、`Python 3.13` / `3.14` では TensorFlow がインストールできないため、推論機能は利用できません。
- 自動セットアップを使う場合は `python3.11`、`npm`、`lsof` が必要です。

`python3.11` が手元にない場合は、`docker/backend.Dockerfile` と同じく Python 3.11 系の環境を用意してください。

## 3. 自動でセットアップして起動する方法

いちばん楽なのはこの方法です。セットアップを 1 回実行したあと、起動は 1 コマンドで済みます。

### セットアップ

```bash
./scripts/setup-dev.sh
```

### 起動

```bash
./scripts/dev-up.sh
```

この方法では、次のような点をまとめて吸収します。

- `Python 3.11` の `venv` を使う
- 壊れた `venv` を自動で作り直す
- backend / frontend の依存関係をまとめて入れる
- `8000` や `3000` が埋まっているときは空いているポートを使う
- backend の `matplotlib` 初回警告を避ける

## 4. 手動でセットアップして起動する方法

### バックエンドのセットアップ

1. 仮想環境を作成して有効化します:
   ```bash
   python3.11 -m venv venv
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

`8000` が埋まっているときは次のように変えられます。

```bash
APP_PORT=8001 python backend/main.py
```

### フロントエンドのセットアップ

1. 依存関係をインストールします:
   ```bash
   cd frontend
   npm install
   ```
2. 開発サーバーを起動します:
   ```bash
   npm run dev
   ```

Vite は `3000` が埋まっていても、空いている次のポートで起動します。バックエンドを `8000` 以外で起動した場合だけ、次のように合わせてください。

```bash
cd frontend
VITE_BACKEND_PORT=8001 npm run dev
```

## 5. 補足

- `./scripts/dev-up.sh` は backend の自動リロードをデフォルトで無効にして、終了時にプロセスが残りにくいようにしています。
- backend の自動リロードも欲しい場合は `APP_RELOAD=true ./scripts/dev-up.sh` を使ってください。
