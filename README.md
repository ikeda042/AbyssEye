# AbyssEye-Neo

## 1. 新しいAbyssEyeのインストール

新しいバージョンは `AbyssEye-Neo` ブランチで公開しています。
まず、ターミナルで以下のコマンドでインストールしてください。

```bash
git clone -b AbyssEye-Neo https://github.com/Gashu884/AbyssEye.git
cd AbyssEye
```

## 2. 前提

- バックエンドは `Python 3.11` を推奨します。
- `Python 3.12` でも動く可能性はありますが、`Python 3.13` / `3.14` では TensorFlow がインストールできないため、推論機能は利用できません。
- 手動セットアップには `python3.11` と `npm` が必要です。

`python3.11` が手元にない場合は、`docker/backend.Dockerfile` と同じく Python 3.11 系の環境を用意してください。

## 3. 手動でセットアップして起動する方法

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

## 4. 再起動だけしたいとき

セットアップ済みで、依存関係の入れ直しをせずに再起動だけしたい場合は次だけで大丈夫です。

### バックエンドの再起動

仮想環境を有効化済みのターミナルで実行してください。

```bash
python backend/main.py
```

### フロントエンドの再起動

```bash
cd frontend
npm run dev
```
## 5. 開発者の更新を取り込む

すでに clone 済みの利用者が、開発者の新しい commit を取り込む方法は 2 つあります。

### 方法1. ターミナルで更新する

```bash
git checkout AbyssEye-Neo
git pull --ff-only origin AbyssEye-Neo
```

### 方法2. Swagger から更新する

`AbyssEye` を起動済みなら、HomeのSwagger から `git pull` を実行することもできます。

1. ブラウザで `http://localhost:8000/api/v1/docs` を開く
2. `POST /api/v1/dev/git/pull` を開く
3. `Try it out` を押す
4. 必要なら body を次のように入れる

```json
{
  "branch": "AbyssEye-Neo",
  "remote": "origin"
}
```

5. `Execute` を押す

`branch` を省略した場合も、既定では `AbyssEye-Neo` を更新します。

依存関係も更新されている可能性があるので、必要に応じて手動で入れ直してください。

バックエンド依存関係を更新する場合:

```bash
source venv/bin/activate
pip install -r backend/requirements.txt
```

フロントエンド依存関係を更新する場合:

```bash
cd frontend
npm install
```

`git pull --ff-only` や Swagger の `dev/git/pull` が止まった場合は、ローカルに未コミット変更や独自の commit がある可能性があります。その場合はいったん変更を退避するか commit してから実行してください。

## 6. 補足

- バックエンドを別ポートで起動した場合だけ、フロントエンド側で `VITE_BACKEND_PORT` を合わせてください。
- `python backend/main.py` の前に `source venv/bin/activate` を忘れると、仮想環境外の Python が使われることがあります。
