# ROI_changed

## 1. 何を目指して変更したか
- 目的は以下の2点です。
  - ROIの重なりによる二重カウントを減らす
  - `Class1(複数細胞)` の細胞数推定を改善する

---

## 2. 現在の基本方針（確定）
- 抽出はセグメンテーションではなく、既存の画像処理ベースを継続
- 抽出時NMSを主対策として使用
- `min_distance` と `nms_iou_threshold` は手動正解CSVで最適化
- 反復抽出はオプション（常時ONにしない）

---

## 3. ROI抽出まわりの主な変更

### 3.1 モデル連動の抽出プロファイル
- ファイル: `models/roi_profiles.json`
- モデル名に応じて抽出パラメータを切替できるようにした
- 主な項目:
  - `roi_width`, `roi_height`
  - `green_rate`, `min_distance`, `min_green`
  - `ratio_primary`, `ratio_secondary`
  - `kernel_size`, `dilate_iterations`
  - `disallow_overlap`, `nms_iou_threshold`, `iterative_passes`

### 3.2 抽出時NMS導入（重なり抑制）
- ファイル: `backend/app/roi_extract/roi_module.py`
- ROI候補をスコア順に採用し、IoUで重なり候補を除外する方式へ変更
- 完全非重複（IoU=0）だと取りこぼしが増えたため、最終的にバランス設定へ調整

### 3.3 現在の採用設定（重要）
- `nms_iou_threshold = 0.15`
- `disallow_overlap = 1`
- `iterative_passes = 1`（通常は反復なし）

### 3.4 反復抽出トグル（Frontend）
- 一括ROI抽出画面に `反復抽出` ON/OFF を追加
- ON: `iterative_passes >= 2`
- OFF: `iterative_passes = 1`
- 変更ファイル:
  - `frontend/src/pages/TiffManagerBulkPage.tsx`
  - `frontend/src/i18n.tsx`
  - `backend/app/tiff_manager_buld/router.py`
  - `backend/app/tiff_manager_buld/crud.py`

---

## 4. Class1(複数細胞)再カウントの変更

### 4.1 再カウント処理の追加
- ファイル: `backend/app/tiff_manager_buld/crud.py`
- `Class1` だけ、ROI画像(`png_blob`)から追加の細胞数推定を実行
- カウント規則:
  - Class0: +1
  - Class1: 再推定値（最低2）
  - Class2/3: +0

### 4.2 144x144黒背景マッピング
- Class1再推定時、ROIを拡大せず `144x144` 黒背景中央に配置して評価
- 背景影響を抑え、ROI情報を維持するため

### 4.3 推論集計時の重複抑制
- 同一画像内でROI候補同士をクラスタ化し、二重加算を抑制するロジックを追加
- 適用箇所:
  - `infer_folder`
  - `infer_single_image`

---

## 5. 手動検証と最適化の導線

### 5.1 Class1 ROIエクスポート
- API: `POST /api/v1/tiff-bulk/infer/export-class1`
- 出力先: `backend/app/databases/<folder>_class1_rois/`
- 出力ファイル:
  - `manifest.csv`（手動記入用）
  - `reconcile_template.csv`（突合テンプレート）

### 5.2 Class1閾値最適化
- API: `POST /api/v1/tiff-bulk/infer/optimize-class1`
- `manifest.csv` の `manual_cell_count` を使って探索
- 出力:
  - `backend/app/databases/<folder>_bulk_class1_tuning.json`
  - `backend/app/databases/<folder>_class1_rois/threshold_search_report.csv`
  - `backend/app/databases/<folder>_class1_rois/reconcile.csv`

### 5.3 抽出パラメータ最適化（ROI数）
- テンプレートAPI: `POST /api/v1/tiff-bulk/extract/export-tuning-template`
- 最適化API: `POST /api/v1/tiff-bulk/extract/optimize`
- `manual_roi_count` から `min_distance` と `nms_iou_threshold` を探索

---

## 6. 直近の実測メモ（Test_data）
- Class1手動正解CSVを取り込み、Class1再カウント閾値を最適化済み
- 採用されたClass1最適パラメータ:
  - `distance_ratio=0.45`
  - `min_contour_area=4.0`
  - `morph_open_iterations=0`
  - `invert_ratio_threshold=0.6`
  - （`canvas_size=144`, `min_cells=2`, `max_cells=12`）
- 評価指標:
  - `MAE=0.5748502994011976`
  - `RMSE=1.1912857241848853`

---

## 7. いま見るべきファイル
- 抽出ロジック本体:
  - `backend/app/roi_extract/roi_module.py`
- バルク抽出/推論/最適化:
  - `backend/app/tiff_manager_buld/crud.py`
  - `backend/app/tiff_manager_buld/router.py`
- モデル連動プロファイル:
  - `models/roi_profiles.json`
- Frontendトグル:
  - `frontend/src/pages/TiffManagerBulkPage.tsx`


---

## 8. DeepScan 手動ROI配置（今回追加）
- 追加API:
  - `POST /api/v1/deepscan/{db_name}/manual-rois`
  - `DELETE /api/v1/deepscan/{db_name}/manual-rois/{record_id}`
- 画面操作:
  - `DeepScan` 右ペインに「手動ROI追加」トグルを追加
  - ON中に画像クリックで `48x48` ROI を追加
  - 「選択ROI削除」で選択中ROIを削除
- 実装ファイル:
  - `backend/app/deepscan/crud.py`
  - `backend/app/deepscan/router.py`
  - `frontend/src/pages/DeepScanPage.tsx`

## 9. DeepScan 手動ROI運用の最終仕様（本日反映）

### 9.1 手動ROIの表示ルール
- 手動追加ROI（`manual_added=true`）は `AI` / `MANUAL` のどちらのフレーム基準でも表示する。
- 自動ROIと区別できるよう、手動ROIは破線スタイルで表示する。

### 9.2 フレーム基準（AI / MANUAL）の適用
- フレーム色の判定は自動ROIと同じルールを適用する。
  - `AI` 選択時: AI推論ラベル色
  - `MANUAL` 選択時: `manual_label` 優先（未設定時はAIラベルへフォールバック）
- これにより、手動ROIでもラベル変更後に枠色が正しく追従する。

### 9.3 削除の安全制約
- 削除できるのは「手動追加ROIのみ」に制限。
- 自動抽出ROIは削除不可（取りこぼし補完のみを目的にするため）。

### 9.4 操作性改善
- 手動追加モード中でもROIクリックで選択可能。
- 背景クリック時のみ新規手動ROIを追加。
- 手動ROI追加時に全ROIが再強調される挙動を抑制し、見やすさを維持。

### 9.5 反映ファイル
- `backend/app/deepscan/crud.py`
- `backend/app/deepscan/router.py`
- `backend/app/realtime/crud.py`
- `frontend/src/pages/DeepScanPage.tsx`
