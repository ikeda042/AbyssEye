import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ReactNode } from "react";

export type Language = "ja" | "en";

type TranslationValue = string | ((params?: Record<string, unknown>) => string);
type TranslationDict = Record<string, TranslationValue>;

const translations: Record<Language, TranslationDict> = {
  en: {
    "common.home": "Home",
    "common.backToList": "Back to list",
    "common.close": "Close",
    "common.unexpectedError": "An unexpected error occurred.",
    "header.logoAlt": "AbyssEye logo",
    "header.logoTitle": "Go to home",
    "header.apiBase": "API Base: {url}",
    "header.toggleTheme": "Toggle color mode",
    "header.languageToggle": "Switch language",
    "header.languageJa": "日本語",
    "header.languageEn": "EN",
    "top.health.checking": "Checking backend status…",
    "top.health.ok": "Backend API is available (status: {status}).",
    "top.health.error": "Unable to reach the backend. Please start the server and try again.",
    "top.cards.roi.title": "ROI Extraction",
    "top.cards.roi.desc": "Upload TIFF stacks, inspect files, and jump into ROI processing.",
    "top.cards.databases.title": "Databases",
    "top.cards.databases.desc": "Browse generated .db files and manage saved experiments.",
    "top.cards.models.title": "Model Manager",
    "top.cards.models.desc": "Upload and review models stored under models/.",
    "top.cards.realtime.title": "Realtime engine",
    "top.cards.realtime.desc": "Automatically shows the latest TIFF and inference results.",
    "tiff.breadcrumb": "ROI Extraction",
    "tiff.title": "ROI Extraction",
    "tiff.uploadCta": "Upload TIFF",
    "tiff.uploading": "Uploading…",
    "tiff.searchPlaceholder": "Search by file name",
    "tiff.clear": "Clear",
    "tiff.download": "Download",
    "tiff.roiExtract": "ROI Extract",
    "tiff.delete": "Delete",
    "tiff.deleting": "Deleting…",
    "tiff.listError": "Failed to fetch TIFF files.",
    "tiff.uploadError": "Failed to upload the TIFF file.",
    "tiff.uploadUnexpected": "An error occurred while uploading.",
    "tiff.uploadSuccess": "{name} uploaded.",
    "tiff.deleteError": "Failed to delete the TIFF file.",
    "tiff.deleteUnexpected": "An error occurred while deleting.",
    "tiff.deleteSuccess": "{name} deleted.",
    "tiff.notFoundTitle": "No files found",
    "tiff.notFoundBody.search": "Change the search query and try again.",
    "tiff.notFoundBody.empty": "Upload a TIFF file first.",
    "tiff.table.filename": "File name",
    "tiff.table.download": "Download",
    "tiff.table.roi": "ROI Extract",
    "tiff.table.delete": "Delete",
    "roi.breadcrumb": "ROI Extract",
    "roi.overline": "ROI Extract",
    "roi.title": "Automated ROI Extraction",
    "roi.description": "Choose a TIFF and start extraction to automatically save ROIs into a SQLite DB.",
    "roi.selectionFromTiff": "You can run extraction for {name} selected in ROI Extraction.",
    "roi.targetLabel": "Target",
    "roi.targetPlaceholder": "Select a TIFF in ROI Extraction before opening this page.",
    "roi.targetMissing": "The specified TIFF file was not found. Check ROI Extraction.",
    "roi.targetUnset": "No TIFF file is specified. Select a file in ROI Extraction first.",
    "roi.run": "Run ROI extraction",
    "roi.running": "Running...",
    "roi.reset": "Reset",
    "roi.backToList": "Back to ROI Extraction",
    "roi.resultTitle": "Generated result",
    "roi.fields.tifName": "TIFF file",
    "roi.fields.dbName": "Saved DB",
    "roi.fields.dbPath": "DB path",
    "roi.fields.roiCount": "ROI count",
    "roi.fields.roiDensity": "ROI density",
    "roi.fields.originalShape": "Original resolution",
    "roi.fields.processedShape": "Processed resolution",
    "roi.fields.patchSize": "ROI patch size",
    "roi.fields.dbSize": "DB file size",
    "roi.fields.savedAt": "Saved at",
    "roi.viewDatabases": "Open in databases",
    "roi.error.fetchTifs": "Failed to fetch TIFF files.",
    "roi.error.list": "Failed to load the list.",
    "roi.error.noSelection": "Select a TIFF file to process.",
    "roi.error.run": "Failed to run ROI extraction.",
    "roi.info.generated": "{name} generated.",
    "databases.breadcrumb": "Databases",
    "databases.title": "Databases",
    "databases.subtitle": "Search generated ROI SQLite files and open their overviews.",
    "databases.searchPlaceholder": "Search by DB name",
    "databases.clear": "Clear",
    "databases.refresh": "Refresh list",
    "databases.refreshing": "Refreshing...",
    "databases.fetchError": "Failed to fetch databases.",
    "databases.fetchInfo": "{count} databases loaded.",
    "databases.emptyTitle": "No databases found",
    "databases.emptySearch": "Change the search query and try again.",
    "databases.emptyNoSearch": "Run ROI extraction to generate a database first.",
    "databases.table.name": "DB name",
    "databases.table.size": "Size",
    "databases.table.updated": "Last updated",
    "databases.table.download": "Download",
    "databases.table.inference": "Inference",
    "databases.table.deepScan": "Deep Scan",
    "databases.table.singleCell": "Single-cell view",
    "databases.table.annotation": "Annotation",
    "databases.table.overview": "Overview",
    "databases.table.delete": "Delete",
    "databases.table.deleting": "Deleting...",
    "databases.table.downloadShort": "DL",
    "databases.table.view": "View",
    "databases.deleteError": "Failed to delete the database.",
    "databases.deleteUnexpected": "An error occurred while deleting.",
    "databases.deleteSuccess": "{name} deleted.",
    "inference.class.0": "Single cell",
    "inference.class.1": "Multiple cells",
    "inference.class.2": "Blurred",
    "inference.class.3": "Non-cell particle",
    "inference.classText": "0=Single cell, 1=Multiple cells, 2=Blurred, 3=Non-cell particle",
    "overview.breadcrumb": "Overview",
    "overview.overline": "Database Overview",
    "overview.records": "{count} ROI records",
    "overview.loading": "Fetching information...",
    "overview.backToList": "Back to list",
    "overview.download": "Download DB",
    "overview.refresh": "Refresh",
    "overview.refreshing": "Refreshing...",
    "overview.previewTitle": "ROI preview",
    "overview.previewCount": "{shown} / {limit} shown",
    "overview.previewReload": "Reload",
    "overview.previewReloading": "Loading...",
    "overview.noRecords": "No ROI records found in this database.",
    "overview.recordLabel": "Record #{id}",
    "overview.infer": "Infer",
    "overview.inferencing": "Running inference...",
    "overview.dialog.title": "ROI inference",
    "overview.dialog.record": "Record #{id}",
    "overview.dialog.running": "Running inference...",
    "overview.dialog.classSummary": "Class {index} ({confidence}%)",
    "overview.dialog.classProbability": "Class {index}{description}: {probability}%",
    "overview.dialog.model": "Model: {path}",
    "overview.dialog.close": "Close",
    "overview.errors.overview": "Failed to fetch database details.",
    "overview.errors.preview": "Failed to fetch ROI records.",
    "overview.errors.inference": "Failed to run ROI inference.",
    "overview.errors.noDb": "No database specified.",
    "overview.missingTitle": "Database is not specified",
    "overview.missingDescription": "Pick a database from the list and press “overview” to return here.",
    "models.breadcrumb": "Model Manager",
    "models.title": "Model Manager",
    "models.subtitle":
      "Upload TensorFlow / Keras model artifacts into the backend {dir} directory and manage the active model used for inference. Single model files or entire SavedModel folders can be uploaded directly.",
    "models.upload": "Upload model",
    "models.uploading": "Uploading...",
    "models.uploadFolder": "Upload folder",
    "models.reload": "Reload",
    "models.supportedFormats": "Supported formats: {formats}",
    "models.fetchError": "Failed to fetch models.",
    "models.uploadError": "Failed to upload the model.",
    "models.uploadUnexpected": "An error occurred while uploading.",
    "models.uploadSuccess": "{name} uploaded.",
    "models.activateError": "Failed to switch the active model.",
    "models.activateSuccess": "{name} set as active model.",
    "models.emptyTitle": "No models found",
    "models.emptyDescription": "Upload a model file first.",
    "models.table.name": "Name",
    "models.table.kind": "Type",
    "models.table.path": "Relative path",
    "models.table.actions": "Actions",
    "models.activate": "Activate",
    "models.active": "Active",
  },
  ja: {
    "common.home": "Home",
    "common.backToList": "一覧へ戻る",
    "common.close": "閉じる",
    "common.unexpectedError": "予期しないエラーが発生しました。",
    "header.logoAlt": "AbyssEye ロゴ",
    "header.logoTitle": "ホームに移動",
    "header.apiBase": "APIベース: {url}",
    "header.toggleTheme": "カラーモードを切り替え",
    "header.languageToggle": "言語を切り替え",
    "header.languageJa": "日本語",
    "header.languageEn": "EN",
    "top.health.checking": "バックエンドの状態を確認中…",
    "top.health.ok": "バックエンドAPIが利用可能です（ステータス: {status}）。",
    "top.health.error": "バックエンドに接続できません。サーバーを起動してから再度お試しください。",
    "top.cards.roi.title": "ROI抽出",
    "top.cards.roi.desc": "TIFFスタックをアップロードし、確認してROI処理に進みます。",
    "top.cards.databases.title": "データベース",
    "top.cards.databases.desc": "生成済みの.dbファイルを参照し、実験を管理します。",
    "top.cards.models.title": "モデル管理",
    "top.cards.models.desc": "models/ 配下に保存されたモデルをアップロード・確認します。",
    "top.cards.realtime.title": "リアルタイムエンジン",
    "top.cards.realtime.desc": "リアルタイムでDeppScanを使用できます。",
    "tiff.breadcrumb": "ROI Extraction",
    "tiff.title": "ROI抽出",
    "tiff.uploadCta": "TIFFをアップロード",
    "tiff.uploading": "アップロード中…",
    "tiff.searchPlaceholder": "ファイル名で検索",
    "tiff.clear": "クリア",
    "tiff.download": "ダウンロード",
    "tiff.roiExtract": "ROI抽出",
    "tiff.delete": "削除",
    "tiff.deleting": "削除中…",
    "tiff.listError": "TIFFファイル一覧の取得に失敗しました。",
    "tiff.uploadError": "TIFFファイルのアップロードに失敗しました。",
    "tiff.uploadUnexpected": "アップロード中にエラーが発生しました。",
    "tiff.uploadSuccess": "{name} をアップロードしました。",
    "tiff.deleteError": "TIFFファイルの削除に失敗しました。",
    "tiff.deleteUnexpected": "削除中にエラーが発生しました。",
    "tiff.deleteSuccess": "{name} を削除しました。",
    "tiff.notFoundTitle": "ファイルが見つかりません",
    "tiff.notFoundBody.search": "検索条件を変更して再度お試しください。",
    "tiff.notFoundBody.empty": "先にTIFFファイルをアップロードしてください。",
    "tiff.table.filename": "ファイル名",
    "tiff.table.download": "ダウンロード",
    "tiff.table.roi": "ROI抽出",
    "tiff.table.delete": "削除",
    "roi.breadcrumb": "ROI Extraction",
    "roi.overline": "ROI Extract",
    "roi.title": "自動ROI抽出",
    "roi.description": "TIFFを選んで抽出を開始すると、自動でSQLite DBにROIが保存されます。",
    "roi.selectionFromTiff": "ROI Extractionで選択した {name} に対して実行できます。",
    "roi.targetLabel": "実行対象",
    "roi.targetPlaceholder": "ROI ExtractionでTIFFファイルを選択してからこのページを開いてください。",
    "roi.targetMissing": "指定されたTIFFファイルが見つかりません。ROI Extractionで状態を確認してください。",
    "roi.targetUnset": "対象となるTIFFファイルが指定されていません。先にROI Extractionでファイルを選択してください。",
    "roi.run": "ROI抽出を実行",
    "roi.running": "抽出中...",
    "roi.reset": "リセット",
    "roi.backToList": "ROI Extractionに戻る",
    "roi.resultTitle": "生成結果",
    "roi.fields.tifName": "TIFFファイル",
    "roi.fields.dbName": "保存DB",
    "roi.fields.dbPath": "DBパス",
    "roi.fields.roiCount": "ROI検出数",
    "roi.fields.roiDensity": "ROI密度",
    "roi.fields.originalShape": "元画像解像度",
    "roi.fields.processedShape": "処理解像度",
    "roi.fields.patchSize": "ROIパッチサイズ",
    "roi.fields.dbSize": "DBファイルサイズ",
    "roi.fields.savedAt": "保存時刻",
    "roi.viewDatabases": "データベース一覧で確認",
    "roi.error.fetchTifs": "TIFFファイルの取得に失敗しました。",
    "roi.error.list": "一覧の取得に失敗しました。",
    "roi.error.noSelection": "処理するTIFFファイルを選択してください。",
    "roi.error.run": "ROI抽出の実行に失敗しました。",
    "roi.info.generated": "{name} を生成しました。",
    "databases.breadcrumb": "Databases",
    "databases.title": "Databases",
    "databases.subtitle": "生成済みのROI SQLiteファイルを検索し、overview画面に遷移して中身を確認できます。",
    "databases.searchPlaceholder": "DB名で検索",
    "databases.clear": "クリア",
    "databases.refresh": "一覧を更新",
    "databases.refreshing": "更新中…",
    "databases.fetchError": "データベース一覧の取得に失敗しました。",
    "databases.fetchInfo": "データベースを {count} 件取得しました。",
    "databases.emptyTitle": "データベースが見つかりません",
    "databases.emptySearch": "検索条件を変更して再度お試しください。",
    "databases.emptyNoSearch": "まずはROI抽出を実行してDBを生成してください。",
    "databases.table.name": "DB名",
    "databases.table.size": "サイズ",
    "databases.table.updated": "最終更新",
    "databases.table.download": "ダウンロード",
    "databases.table.inference": "推論",
    "databases.table.deepScan": "Deep Scan",
    "databases.table.singleCell": "単細胞ビュー",
    "databases.table.annotation": "アノテーション",
    "databases.table.overview": "概要",
    "databases.table.delete": "削除",
    "databases.table.deleting": "削除中…",
    "databases.table.downloadShort": "DL",
    "databases.table.view": "ビュー",
    "databases.deleteError": "データベースの削除に失敗しました。",
    "databases.deleteUnexpected": "削除中にエラーが発生しました。",
    "databases.deleteSuccess": "{name} を削除しました。",
    "inference.class.0": "単一細胞",
    "inference.class.1": "複数細胞",
    "inference.class.2": "ピンぼけ",
    "inference.class.3": "非細胞粒子",
    "inference.classText": "0＝単一細胞、1＝複数細胞、2＝ピンぼけ、3＝非細胞粒子",
    "overview.breadcrumb": "Overview",
    "overview.overline": "Database Overview",
    "overview.records": "{count} 件のROIレコード",
    "overview.loading": "情報を取得しています...",
    "overview.backToList": "一覧へ戻る",
    "overview.download": "DBをダウンロード",
    "overview.refresh": "再取得",
    "overview.refreshing": "更新中…",
    "overview.previewTitle": "ROIプレビュー",
    "overview.previewCount": "{shown} / {limit} 件を表示",
    "overview.previewReload": "再読込",
    "overview.previewReloading": "読込中…",
    "overview.noRecords": "このデータベースにはROIレコードが見つかりません。",
    "overview.recordLabel": "Record #{id}",
    "overview.infer": "推論",
    "overview.inferencing": "推論中…",
    "overview.dialog.title": "ROI推論",
    "overview.dialog.record": "Record #{id}",
    "overview.dialog.running": "推論中…",
    "overview.dialog.classSummary": "クラス {index}（{confidence}%）",
    "overview.dialog.classProbability": "クラス {index}{description}: {probability}%",
    "overview.dialog.model": "モデル: {path}",
    "overview.dialog.close": "閉じる",
    "overview.errors.overview": "データベース情報の取得に失敗しました。",
    "overview.errors.preview": "ROIレコードの取得に失敗しました。",
    "overview.errors.inference": "ROI推論に失敗しました。",
    "overview.errors.noDb": "データベースが指定されていません。",
    "overview.missingTitle": "データベースが指定されていません",
    "overview.missingDescription": "一覧ページからデータベースを選択し、「overview」ボタンを押してこの画面に戻ってください。",
    "models.breadcrumb": "Model Manager",
    "models.title": "Model Manager",
    "models.subtitle":
      "TensorFlow / Kerasのモデルをバックエンドの {dir} ディレクトリにアップロードし、推論で使用するアクティブモデルを管理します。単体ファイルやSavedModelフォルダをそのままアップロードできます。",
    "models.upload": "モデルをアップロード",
    "models.uploading": "アップロード中…",
    "models.uploadFolder": "フォルダごとアップロード",
    "models.reload": "再読み込み",
    "models.supportedFormats": "サポート形式: {formats}",
    "models.fetchError": "モデル一覧を取得できませんでした。",
    "models.uploadError": "モデルのアップロードに失敗しました。",
    "models.uploadUnexpected": "アップロード中にエラーが発生しました。",
    "models.uploadSuccess": "{name} をアップロードしました。",
    "models.activateError": "アクティブモデルの切り替えに失敗しました。",
    "models.activateSuccess": "{name} をアクティブモデルとして設定しました。",
    "models.emptyTitle": "モデルが見つかりません",
    "models.emptyDescription": "先にモデルファイルをアップロードしてください。",
    "models.table.name": "名前",
    "models.table.kind": "種類",
    "models.table.path": "相対パス",
    "models.table.actions": "操作",
    "models.activate": "アクティブ化",
    "models.active": "Active",
  },
};

const FALLBACK_LANGUAGE: Language = "en";
const STORAGE_KEY = "abyssEye:language";

const formatTemplate = (template: string, params?: Record<string, unknown>) => {
  if (!params) return template;
  return template.replace(/\{(\w+)\}/g, (match, key) => {
    const value = params[key];
    return typeof value === "undefined" ? match : String(value);
  });
};

export const translate = (language: Language, key: string, params?: Record<string, unknown>): string => {
  const dict = translations[language] ?? translations[FALLBACK_LANGUAGE];
  const fallbackDict = translations[FALLBACK_LANGUAGE];
  const value = dict[key] ?? fallbackDict[key];
  if (!value) return key;
  if (typeof value === "function") {
    try {
      return value(params);
    } catch {
      return key;
    }
  }
  return formatTemplate(value, params);
};

type I18nContextValue = {
  language: Language;
  setLanguage: (lang: Language) => void;
  toggleLanguage: () => void;
  t: (key: string, params?: Record<string, unknown>) => string;
};

const I18nContext = createContext<I18nContextValue | null>(null);

const loadStoredLanguage = (): Language => {
  if (typeof window === "undefined") return "ja";
  const stored = window.localStorage.getItem(STORAGE_KEY);
  return stored === "en" || stored === "ja" ? stored : "ja";
};

export const I18nProvider = ({ children }: { children: ReactNode }) => {
  const [language, setLanguage] = useState<Language>(() => loadStoredLanguage());

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(STORAGE_KEY, language);
    document.documentElement.lang = language;
  }, [language]);

  const toggleLanguage = useCallback(() => {
    setLanguage((prev) => (prev === "ja" ? "en" : "ja"));
  }, []);

  const translateFn = useCallback((key: string, params?: Record<string, unknown>) => translate(language, key, params), [language]);

  const value = useMemo<I18nContextValue>(
    () => ({
      language,
      setLanguage,
      toggleLanguage,
      t: translateFn,
    }),
    [language, translateFn],
  );

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
};

export const useI18n = (): I18nContextValue => {
  const ctx = useContext(I18nContext);
  if (!ctx) {
    throw new Error("useI18n must be used within I18nProvider");
  }
  return ctx;
};
