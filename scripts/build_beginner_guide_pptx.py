from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "docs" / "slides"
ASSET_DIR = OUTPUT_DIR / "assets"
LOGO_PATH = ROOT / "frontend" / "public" / "logo.png"
CSV_SCREENSHOT_PATH = ASSET_DIR / "cellcount_csv_example.png"
PPTX_PATH = OUTPUT_DIR / "AbyssEye_beginner_guide_ja.pptx"
PDF_PATH = OUTPUT_DIR / "AbyssEye_beginner_guide_ja.pdf"
APPLE_SCRIPT_PATH = OUTPUT_DIR / "build_beginner_guide_tmp.applescript"


def applescript_string(text: str) -> str:
    text = text.replace("\\n", "\n")
    parts = text.splitlines() or [""]
    escaped = ['"' + part.replace("\\", "\\\\").replace('"', '\\"') + '"' for part in parts]
    if len(escaped) == 1:
        return escaped[0]
    return "(" + " & linefeed & ".join(escaped) + ")"


def slide_commands() -> list[str]:
    lines: list[str] = []

    def next_slide() -> None:
        lines.append("my nextBlankSlide()")

    def add_text(text: str, x: int, y: int, w: int, h: int, size: int) -> None:
        lines.append(f"my addTextBox({applescript_string(text)}, {x}, {y}, {w}, {h}, {size})")

    def add_panel(text: str, x: int, y: int, w: int, h: int, size: int) -> None:
        lines.append(f"my addPanel({applescript_string(text)}, {x}, {y}, {w}, {h}, {size})")

    def add_logo(x: int, y: int, w: int, h: int) -> None:
        lines.append(f"my addImage(logoPath, {x}, {y}, {w}, {h})")

    def add_image(path_var: str, x: int, y: int, w: int, h: int) -> None:
        lines.append(f"my addImage({path_var}, {x}, {y}, {w}, {h})")

    def add_slide_header(title: str, subtitle: str) -> None:
        add_text(title, 100, 56, 1180, 70, 34)
        add_text(subtitle, 100, 116, 1160, 34, 18)
        add_logo(1720, 34, 120, 120)

    add_text("AbyssEye 初回利用ガイド", 110, 150, 980, 90, 44)
    add_text("初回利用者向け / ROI抽出・DeepScan・リアルタイムエンジンの基本操作", 110, 250, 1060, 56, 24)
    add_text(
        "この資料では、AbyssEye を初めて使う人が迷いやすいポイントを\\nHome から結果出力まで順番にまとめています。",
        110,
        360,
        840,
        140,
        24,
    )
    add_logo(1350, 120, 420, 420)
    add_panel("1 起動", 120, 730, 320, 120, 24)
    add_panel("2 入口選択", 470, 730, 320, 120, 24)
    add_panel("3 確認と修正", 820, 730, 320, 120, 24)
    add_panel("4 PDF / CSV出力", 1170, 730, 360, 120, 24)
    add_text("推奨 branch: AbyssEye-Neo", 120, 900, 440, 40, 18)

    next_slide()
    add_slide_header("1. 起動前の準備", "README の手順に沿って backend と frontend を起動します。")
    add_text(
        "• Python 3.13 系を推奨\\n• branch は AbyssEye-Neo を使う\\n• backend と frontend は別々に起動\\n• 起動後は http://localhost:3000 を開く\\n• API は通常 http://localhost:8000/api/v1/ を使う",
        100,
        190,
        720,
        620,
        26,
    )
    add_panel(
        "Terminal 1\\ngit clone -b AbyssEye-Neo https://github.com/Gashu884/AbyssEye.git\\ncd AbyssEye\\npython -m venv venv\\nsource venv/bin/activate\\npip install -r backend/requirements.txt\\npython backend/main.py",
        920,
        170,
        860,
        330,
        20,
    )
    add_panel(
        "Terminal 2\\ncd frontend\\nnpm install\\nnpm run dev",
        920,
        540,
        860,
        190,
        22,
    )
    add_panel("Browser\\nhttp://localhost:3000", 920, 770, 860, 120, 24)
    add_text("※ Python 3.14 では TensorFlow の install が通らないことがあります。", 920, 920, 860, 50, 20)

    next_slide()
    add_slide_header("2. Home と入口の選び方", "Home の ROI抽出 から、目的に応じて入口を選びます。")
    add_text(
        "• Home では ROI抽出 と モデル管理 が並ぶ\\n• ROI抽出 を押すと データベース と リアルタイムエンジン に分かれる\\n• 保存済み画像をまとめて扱うなら データベース\\n• 最新 TIFF を見ながら保存するなら リアルタイムエンジン",
        100,
        190,
        720,
        620,
        26,
    )
    add_panel("Home", 980, 170, 720, 90, 26)
    add_panel("ROI抽出", 980, 330, 310, 190, 24)
    add_panel("モデル管理", 1390, 330, 310, 190, 24)
    add_text("↓", 1325, 545, 60, 60, 34)
    add_panel("データベース", 980, 640, 310, 190, 24)
    add_panel("リアルタイムエンジン", 1390, 640, 310, 190, 24)
    add_text("保存済み画像を確認", 1010, 850, 270, 40, 18)
    add_text("最新画像を保存しながら確認", 1400, 850, 300, 40, 18)

    next_slide()
    add_slide_header("3. データベース画面の使い方", "プロジェクト単位で画像と同視野画像ファイルを整理します。")
    add_text(
        "• 先にプロジェクトを作成 / 選択\\n• 画像アップロード または 同視野画像ファイルをアップロード\\n• 上段が画像リスト、下段が同視野フォルダリスト\\n• 単一画像は ROI抽出&推論・DeepScan・保存・セルカウントへ進む\\n• 同視野フォルダは 一覧 / マージ画像ROI抽出 を使う",
        100,
        190,
        720,
        650,
        25,
    )
    add_panel("プロジェクト作成 / 検索 / 画像アップロード / 同視野画像ファイルをアップロード", 920, 170, 860, 120, 20)
    add_panel("画像リスト\\n- ROI抽出&推論\\n- DeepScan\\n- 保存\\n- セルカウント", 920, 340, 860, 250, 24)
    add_panel("同視野フォルダリスト\\n- 一覧\\n- マージ画像ROI抽出", 920, 640, 860, 220, 24)
    add_text("パンくずは Home > ROI抽出 > データベース > [各プロジェクト] で統一", 920, 900, 860, 40, 18)

    next_slide()
    add_slide_header("4. 画像リストで行う操作", "単一画像の処理と安全な削除は、この一覧から進めます。")
    add_text(
        "• ROI抽出&推論 で単一画像の解析をまとめて実行\\n• DeepScan で ROI とラベルを確認\\n• セルカウント後に PDF / CSV を出力\\n• 削除は 2 段階方式\\n  1. 上部の 削除 を押す\\n  2. 削除モードで対象画像を選ぶ\\n  3. 削除 または キャンセル",
        100,
        190,
        720,
        670,
        24,
    )
    add_panel("ROI抽出&推論", 930, 220, 240, 140, 24)
    add_panel("DeepScan", 1230, 220, 240, 140, 24)
    add_panel("セルカウント\\nPDF / CSV", 1530, 220, 240, 140, 24)
    add_panel("削除モード開始\\n上部の 削除 を押す", 930, 500, 260, 170, 24)
    add_text("→", 1210, 555, 60, 60, 34)
    add_panel("対象画像を選択", 1290, 500, 230, 170, 24)
    add_text("→", 1540, 555, 60, 60, 34)
    add_panel("削除 または キャンセル", 1620, 500, 170, 170, 22)
    add_text("各行の直接削除ではなく、上部操作に統一して誤操作を防ぎます。", 1060, 760, 660, 60, 20)

    next_slide()
    add_slide_header("5. DeepScan の見方", "ROI overlay を確認しながら、手動ラベルと手動 ROI を調整します。")
    add_text(
        "• 画像上に ROI 枠が重なって表示される\\n• 右側で AI / Manual の表示基準を切り替える\\n• ROI を選ぶと manual label を更新できる\\n• 手動ROI追加 / 手動ROI削除 が使える\\n• ROI移動 を ON にすると、約 0.45 秒長押し後ドラッグで位置更新",
        100,
        190,
        720,
        690,
        24,
    )
    add_panel("画像ビュー\\nROI overlay", 900, 190, 640, 620, 26)
    add_panel("表示基準\\nAI / Manual\\nDeepScan", 1570, 190, 240, 220, 22)
    add_panel("Manual Label\\n手動ROI追加\\n手動ROI削除\\nROI移動", 1570, 450, 240, 300, 22)
    add_panel("Class 0", 900, 860, 140, 90, 18)
    add_panel("Class 1", 1070, 860, 140, 90, 18)
    add_panel("Class 2", 1240, 860, 140, 90, 18)
    add_panel("Class 3", 1410, 860, 140, 90, 18)

    next_slide()
    add_slide_header("6. リアルタイムエンジンの使い方", "最新 TIFF を見ながら保存し、そのまま DeepScan で確認できます。")
    add_text(
        "• プロジェクトとサンプル名を決めて 保存\\n• 同視野画像ファイルとして保存 を ON にすると、同一視野の画像群として管理\\n• 保存した画像は後から DeepScan で再確認できる\\n• AI / Manual 表示、manual label、手動 ROI の考え方は DeepScan と共通",
        100,
        190,
        720,
        650,
        25,
    )
    add_panel("最新 TIFF 表示\\nDeepScan overlay", 920, 180, 650, 620, 26)
    add_panel("保存パネル\\nプロジェクト\\nサンプル名\\n同視野画像ファイルとして保存\\n保存", 1600, 190, 220, 360, 20)
    add_panel("保存後\\nDeepScan で確認", 1600, 620, 220, 180, 22)
    add_text("説明文の表示が変わっても、ROI overlay 自体はずれないように調整済みです。", 920, 870, 860, 40, 18)

    next_slide()
    add_slide_header("7. 結果の出力と更新", "セルカウント結果は PDF / CSV で保存し、利用者側は ff-only で更新します。")
    add_text(
        "• セルカウント結果画面では、クラスごとに ROI を切り替えて確認できる\\n• ROI 表は 画像順 / 信頼度順 / ROI番号順 で並べ替え可能\\n• A4 出力は PDF出力 に統一\\n• CSV は image_name 中心の列構成で、manual_label は 1 列に整理\\n• 更新時は git pull --ff-only origin AbyssEye-Neo を使う",
        100,
        190,
        720,
        660,
        24,
    )
    add_panel("出力\\nPDF出力\\nCSV出力\\nクラス別確認", 100, 790, 320, 180, 22)
    add_panel("更新\\ngit checkout AbyssEye-Neo\\ngit pull --ff-only origin AbyssEye-Neo", 470, 790, 520, 180, 20)
    add_image("csvScreenshotPath", 980, 220, 760, 72)
    add_text("CSV 出力イメージ", 980, 320, 280, 30, 18)
    add_panel("つまずきやすい点\\n• DeepScan が開かない → 先に ROI抽出&推論\\n• install エラー → Python バージョン確認", 980, 430, 760, 220, 22)

    next_slide()
    add_slide_header("8. まず触るならこの順番", "初回は次の 4 ステップだけ覚えておくと進めやすいです。")
    add_panel("1\\nREADME に沿って起動", 180, 260, 330, 260, 28)
    add_text("→", 560, 355, 70, 60, 40)
    add_panel("2\\nHome から ROI抽出", 640, 260, 330, 260, 28)
    add_text("→", 1020, 355, 70, 60, 40)
    add_panel("3\\nデータベース または\\nリアルタイムエンジン", 1100, 260, 330, 260, 28)
    add_text("→", 1480, 355, 70, 60, 40)
    add_panel("4\\nDeepScan で確認して\\nPDF / CSV 出力", 1560, 260, 240, 260, 26)
    add_text(
        "最初は データベース → ROI抽出&推論 → DeepScan → セルカウント の流れがいちばん分かりやすいです。\\n慣れてきたら、リアルタイムエンジンで保存しながら確認する運用に広げてください。",
        180,
        640,
        1400,
        120,
        24,
    )
    add_logo(1660, 790, 140, 140)

    return lines


def build_applescript() -> str:
    commands = "\n  ".join(slide_commands())
    return f"""tell application "Keynote"
  activate
  set logoPath to "{LOGO_PATH.as_posix()}"
  set csvScreenshotPath to "{CSV_SCREENSHOT_PATH.as_posix()}"
  set outPptxPath to POSIX file "{PPTX_PATH.as_posix()}"
  set outPdfPath to POSIX file "{PDF_PATH.as_posix()}"
  set docRef to make new document

  tell front document to make new slide with properties {{base slide:master slide "空白"}}

  {commands}

  delete first slide of front document
  export docRef to outPptxPath as Microsoft PowerPoint
  export docRef to outPdfPath as PDF
  close docRef saving no
end tell

on nextBlankSlide()
  tell application "Keynote"
    tell front document to make new slide with properties {{base slide:master slide "空白"}}
  end tell
end nextBlankSlide

on addTextBox(txt, xPos, yPos, boxWidth, boxHeight, fontSize)
  tell application "Keynote"
    tell current slide of front document
      set tBox to make new text item with properties {{position:{{xPos, yPos}}, width:boxWidth, height:boxHeight, object text:txt}}
      try
        set size of object text of tBox to fontSize
      end try
      return tBox
    end tell
  end tell
end addTextBox

on addPanel(txt, xPos, yPos, boxWidth, boxHeight, fontSize)
  tell application "Keynote"
    tell current slide of front document
      set panel to make new shape with properties {{position:{{xPos, yPos}}, width:boxWidth, height:boxHeight, object text:txt}}
      try
        set size of object text of panel to fontSize
      end try
      return panel
    end tell
  end tell
end addPanel

on addImage(imagePath, xPos, yPos, boxWidth, boxHeight)
  tell application "Keynote"
    tell current slide of front document
      set imgRef to make new image with properties {{file:(POSIX file imagePath), position:{{xPos, yPos}}, width:boxWidth, height:boxHeight}}
      return imgRef
    end tell
  end tell
end addImage
"""


def main() -> None:
    if not LOGO_PATH.exists():
        raise FileNotFoundError(f"Logo not found: {LOGO_PATH}")
    if not CSV_SCREENSHOT_PATH.exists():
        raise FileNotFoundError(f"CSV screenshot not found: {CSV_SCREENSHOT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    APPLE_SCRIPT_PATH.write_text(build_applescript(), encoding="utf-8")
    subprocess.run(["osascript", str(APPLE_SCRIPT_PATH)], check=True)
    print(f"Generated: {PPTX_PATH}")
    print(f"Generated: {PDF_PATH}")


if __name__ == "__main__":
    main()
