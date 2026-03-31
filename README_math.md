# フォーカスインジケーター（DeepScan）計算仕様

本ドキュメントは、AbyssEye の DeepScan で表示される「フォーカスインジケーター」に使われる数式と計算手順をまとめたものです。

API は [backend/app/deepscan/router.py](/Users/gashu/Desktop/AbyssEye/backend/app/deepscan/router.py) の  
`GET /deepscan/status` で提供される `focus_profile` / `focus_map` を参照します。

## 1. 前処理（画像読み込み）

- 画像は `cv2.imread(..., cv2.IMREAD_UNCHANGED)` で読み込みます。
- 3チャネル画像なら `cv2.cvtColor(..., COLOR_BGR2GRAY)` でグレースケール化します。
- `dtype != uint8` の場合は `cv2.normalize(..., 0, 255)` して `uint8` 化します。
- `max_side=640` を超える場合は短辺基準で縮小します。  
  （`_load_focus_gray`）
- フォーカス計算はこの処理後のグレースケール画像上で実行します。

## 2. 画素系列に対する基本指標

各画像（またはタイル） `I(x, y)`（0〜255、整数）に対して以下を計算します。

### 2.1 正規化分散（`fnvar`）

まず平均 $\mu$ を取り、分散を平均で割る形で正規化します。

$$
F_{fnvar}
= \frac{\frac{1}{MN}\sum_{x,y}(I(x,y)-\mu)^2}{\mu + \epsilon}
$$

- 実装: [_focus_normalized_variance](/Users/gashu/Desktop/AbyssEye/backend/app/deepscan/crud.py)
- ただし $\mu \le 10^{-9}$ のときは 0.0、また分母に $\epsilon=10^{-12}$ を追加して数値安定化

### 2.2 Tenengrad / 勾配エネルギー（`ften`）

Sobel 微分で勾配を取ります。

$$
g_x = \mathrm{Sobel}(I, \partial_x),\quad
g_y = \mathrm{Sobel}(I, \partial_y)
$$
$$
F_{ften} = \frac{1}{MN}\sum (g_x^2 + g_y^2)
$$

- 実装: [_focus_tenengrad](/Users/gashu/Desktop/AbyssEye/backend/app/deepscan/crud.py)
- `ksize=3` の Sobel を使用

### 2.3 Laplacian分散（`flapvar`）

ノイズ影響を抑えるため、先にガウシアン平滑化した画像を使って Laplacian を適用します。

$$
L = \mathrm{Laplacian}( \mathrm{GaussianBlur}(I) )
,\quad
F_{flapvar} = \mathrm{Var}(L)
$$

- 実装: [_focus_metric_values](/Users/gashu/Desktop/AbyssEye/backend/app/deepscan/crud.py) 内

### 2.4 修正ラプラシアン和（SML, `fsml`）

二次差分の絶対値を使い、隣接差が反転して打ち消し合う問題を避ける形式です。

$$
\Delta_x = \left| I(x,y+2s)-2I(x,y)+I(x,y-2s) \right|
$$
$$
\Delta_y = \left| I(x+2s,y)-2I(x,y)+I(x-2s,y) \right|
$$
$$
F_{fsml} = \mathrm{mean}(\Delta_x + \Delta_y)
$$

- 実装: `_focus_metric_values` の `s = max(1, sml_step)`（既定 1）
- 画像サイズが $2s$ 未満の場合は 0.0

## 3. 指標の選択と重み

`/deepscan/status` では以下パラメータを受け取り、指標を選択します。

- `focus_metric`: `composite`, `fnvar`, `ften`, `flapvar`, `fsml`
- `focus_weight_fnvar`: `fnvar` の重み（既定 0.5）
- `focus_weight_ften`: `ften` の重み（既定 0.5）
- `focus_weight_flapvar`: `flapvar` の重み（既定 0.0）
- `focus_weight_fsml`: `fsml` の重み（既定 0.0）

内部正規化:
- `focus_metric` はエイリアスを吸収（`normalized_variance`→`fnvar` 等）
- 重みは負値を 0 に丸めた後、合計で割って正規化（合計 <= 0 の場合は既定値へ）

## 4. 最小最大正規化（min-max）

画像列内の同一指標値を全深度で正規化します。  
値列 $\{v_i\}$ に対して

$$
\hat{v}_i = \frac{v_i - \min(v)}{\max(v)-\min(v)}
$$

- 差が非常に小さい（$\max-\min \le 10^{-12}$）場合は全て 0.0
- 実装: [_minmax](/Users/gashu/Desktop/AbyssEye/backend/app/deepscan/crud.py)

### 合成スコア

選択が `composite` の場合:
$$
F_{combined} = \sum_{m \in \{fnvar,ften,flapvar,fsml\}} w_m \hat{F}_m
$$
それ以外は、選択指標の正規化値をそのまま採用します。

## 5. `focus_profile`（全体深度プロフィール）

1. 各画像（`available_images` の順）で 4 つの指標を計算
2. 選択した指標群を min-max 正規化
3. `combined_score` を計算（上記）
4. `peak_index = argmax(combined_score)` をピーク深度とする
5. 現在画像 `current_index` と `peak_index` との差:
   - `z_offset_from_peak = index - peak_index`
   - `current_to_peak_ratio = current_score / peak_score`
6. `z_relative = index / (N-1)` を各深度の相対位置として付与

`focus_profile` は `GET /deepscan/status` 応答で返却され、UIでは `focus track` 表示に使用されます。

## 6. `focus_map`（タイル単位フォーカス）

1. 画像群を最大 640px に揃えて同一サイズへリサイズ
2. `tile_size=32` で `rows = floor(H/32)`, `cols = floor(W/32)` でタイル化
3. 各タイル位置で各深度を走査し、各指標を計算
4. 指標ごとに正規化し、合成または単一指標で最適深度を決定
5. 各タイルでスコア差 `best - second_best` を 0〜1 にクリップした `confidence` として付与

`focus_map` は従来のヒートマップ用途ですが、深度推定の補助指標として同時計算されます。

## 7. UI 上での利用（DeepScan）

フロントエンドでは `focus_profile.scores` を `combined_score` で描画し、  
`current_index` を現在位置、`peak_index` をピーク位置として表示しています。  
詳細値はツールチップで表示されます（`combined_score`, `normalized_variance`, `tenengrad`, `laplacian_variance`, `modified_laplacian`）。

## 8. 参照実装

- 集約・数式入口: [backend/app/deepscan/crud.py](/Users/gashu/Desktop/AbyssEye/backend/app/deepscan/crud.py)
- API 経路とパラメータ: [backend/app/deepscan/router.py](/Users/gashu/Desktop/AbyssEye/backend/app/deepscan/router.py)
