tell application "Keynote"
  activate
  set logoPath to "/Users/gashu/Desktop/AbyssEye/frontend/public/logo.png"
  set csvScreenshotPath to "/Users/gashu/Desktop/AbyssEye/docs/slides/assets/cellcount_csv_example.png"
  set outPptxPath to POSIX file "/Users/gashu/Desktop/AbyssEye/docs/slides/AbyssEye_beginner_guide_ja.pptx"
  set outPdfPath to POSIX file "/Users/gashu/Desktop/AbyssEye/docs/slides/AbyssEye_beginner_guide_ja.pdf"
  set docRef to make new document

  tell front document to make new slide with properties {base slide:master slide "空白"}

  my addTextBox("AbyssEye 初回利用ガイド", 110, 150, 980, 90, 44)
  my addTextBox("初回利用者向け / ROI抽出・DeepScan・リアルタイムエンジンの基本操作", 110, 250, 1060, 56, 24)
  my addTextBox(("この資料では、AbyssEye を初めて使う人が迷いやすいポイントを" & linefeed & "Home から結果出力まで順番にまとめています。"), 110, 360, 840, 140, 24)
  my addImage(logoPath, 1350, 120, 420, 420)
  my addPanel("1 起動", 120, 730, 320, 120, 24)
  my addPanel("2 入口選択", 470, 730, 320, 120, 24)
  my addPanel("3 確認と修正", 820, 730, 320, 120, 24)
  my addPanel("4 PDF / CSV出力", 1170, 730, 360, 120, 24)
  my addTextBox("推奨 branch: AbyssEye-Neo", 120, 900, 440, 40, 18)
  my nextBlankSlide()
  my addTextBox("1. 起動前の準備", 100, 56, 1180, 70, 34)
  my addTextBox("README の手順に沿って backend と frontend を起動します。", 100, 116, 1160, 34, 18)
  my addImage(logoPath, 1720, 34, 120, 120)
  my addTextBox(("• Python 3.13 系を推奨" & linefeed & "• branch は AbyssEye-Neo を使う" & linefeed & "• backend と frontend は別々に起動" & linefeed & "• 起動後は http://localhost:3000 を開く" & linefeed & "• API は通常 http://localhost:8000/api/v1/ を使う"), 100, 190, 720, 620, 26)
  my addPanel(("Terminal 1" & linefeed & "git clone -b AbyssEye-Neo https://github.com/Gashu884/AbyssEye.git" & linefeed & "cd AbyssEye" & linefeed & "python -m venv venv" & linefeed & "source venv/bin/activate" & linefeed & "pip install -r backend/requirements.txt" & linefeed & "python backend/main.py"), 920, 170, 860, 330, 20)
  my addPanel(("Terminal 2" & linefeed & "cd frontend" & linefeed & "npm install" & linefeed & "npm run dev"), 920, 540, 860, 190, 22)
  my addPanel(("Browser" & linefeed & "http://localhost:3000"), 920, 770, 860, 120, 24)
  my addTextBox("※ Python 3.14 では TensorFlow の install が通らないことがあります。", 920, 920, 860, 50, 20)
  my nextBlankSlide()
  my addTextBox("2. Home と入口の選び方", 100, 56, 1180, 70, 34)
  my addTextBox("Home の ROI抽出 から、目的に応じて入口を選びます。", 100, 116, 1160, 34, 18)
  my addImage(logoPath, 1720, 34, 120, 120)
  my addTextBox(("• Home では ROI抽出 と モデル管理 が並ぶ" & linefeed & "• ROI抽出 を押すと データベース と リアルタイムエンジン に分かれる" & linefeed & "• 保存済み画像をまとめて扱うなら データベース" & linefeed & "• 最新 TIFF を見ながら保存するなら リアルタイムエンジン"), 100, 190, 720, 620, 26)
  my addPanel("Home", 980, 170, 720, 90, 26)
  my addPanel("ROI抽出", 980, 330, 310, 190, 24)
  my addPanel("モデル管理", 1390, 330, 310, 190, 24)
  my addTextBox("↓", 1325, 545, 60, 60, 34)
  my addPanel("データベース", 980, 640, 310, 190, 24)
  my addPanel("リアルタイムエンジン", 1390, 640, 310, 190, 24)
  my addTextBox("保存済み画像を確認", 1010, 850, 270, 40, 18)
  my addTextBox("最新画像を保存しながら確認", 1400, 850, 300, 40, 18)
  my nextBlankSlide()
  my addTextBox("3. データベース画面の使い方", 100, 56, 1180, 70, 34)
  my addTextBox("プロジェクト単位で画像と同視野画像ファイルを整理します。", 100, 116, 1160, 34, 18)
  my addImage(logoPath, 1720, 34, 120, 120)
  my addTextBox(("• 先にプロジェクトを作成 / 選択" & linefeed & "• 画像アップロード または 同視野画像ファイルをアップロード" & linefeed & "• 上段が画像リスト、下段が同視野フォルダリスト" & linefeed & "• 単一画像は ROI抽出&推論・DeepScan・保存・セルカウントへ進む" & linefeed & "• 同視野フォルダは 一覧 / マージ画像ROI抽出 を使う"), 100, 190, 720, 650, 25)
  my addPanel("プロジェクト作成 / 検索 / 画像アップロード / 同視野画像ファイルをアップロード", 920, 170, 860, 120, 20)
  my addPanel(("画像リスト" & linefeed & "- ROI抽出&推論" & linefeed & "- DeepScan" & linefeed & "- 保存" & linefeed & "- セルカウント"), 920, 340, 860, 250, 24)
  my addPanel(("同視野フォルダリスト" & linefeed & "- 一覧" & linefeed & "- マージ画像ROI抽出"), 920, 640, 860, 220, 24)
  my addTextBox("パンくずは Home > ROI抽出 > データベース > [各プロジェクト] で統一", 920, 900, 860, 40, 18)
  my nextBlankSlide()
  my addTextBox("4. 画像リストで行う操作", 100, 56, 1180, 70, 34)
  my addTextBox("単一画像の処理と安全な削除は、この一覧から進めます。", 100, 116, 1160, 34, 18)
  my addImage(logoPath, 1720, 34, 120, 120)
  my addTextBox(("• ROI抽出&推論 で単一画像の解析をまとめて実行" & linefeed & "• DeepScan で ROI とラベルを確認" & linefeed & "• セルカウント後に PDF / CSV を出力" & linefeed & "• 削除は 2 段階方式" & linefeed & "  1. 上部の 削除 を押す" & linefeed & "  2. 削除モードで対象画像を選ぶ" & linefeed & "  3. 削除 または キャンセル"), 100, 190, 720, 670, 24)
  my addPanel("ROI抽出&推論", 930, 220, 240, 140, 24)
  my addPanel("DeepScan", 1230, 220, 240, 140, 24)
  my addPanel(("セルカウント" & linefeed & "PDF / CSV"), 1530, 220, 240, 140, 24)
  my addPanel(("削除モード開始" & linefeed & "上部の 削除 を押す"), 930, 500, 260, 170, 24)
  my addTextBox("→", 1210, 555, 60, 60, 34)
  my addPanel("対象画像を選択", 1290, 500, 230, 170, 24)
  my addTextBox("→", 1540, 555, 60, 60, 34)
  my addPanel("削除 または キャンセル", 1620, 500, 170, 170, 22)
  my addTextBox("各行の直接削除ではなく、上部操作に統一して誤操作を防ぎます。", 1060, 760, 660, 60, 20)
  my nextBlankSlide()
  my addTextBox("5. DeepScan の見方", 100, 56, 1180, 70, 34)
  my addTextBox("ROI overlay を確認しながら、手動ラベルと手動 ROI を調整します。", 100, 116, 1160, 34, 18)
  my addImage(logoPath, 1720, 34, 120, 120)
  my addTextBox(("• 画像上に ROI 枠が重なって表示される" & linefeed & "• 右側で AI / Manual の表示基準を切り替える" & linefeed & "• ROI を選ぶと manual label を更新できる" & linefeed & "• 手動ROI追加 / 手動ROI削除 が使える" & linefeed & "• ROI移動 を ON にすると、約 0.45 秒長押し後ドラッグで位置更新"), 100, 190, 720, 690, 24)
  my addPanel(("画像ビュー" & linefeed & "ROI overlay"), 900, 190, 640, 620, 26)
  my addPanel(("表示基準" & linefeed & "AI / Manual" & linefeed & "DeepScan"), 1570, 190, 240, 220, 22)
  my addPanel(("Manual Label" & linefeed & "手動ROI追加" & linefeed & "手動ROI削除" & linefeed & "ROI移動"), 1570, 450, 240, 300, 22)
  my addPanel("Class 0", 900, 860, 140, 90, 18)
  my addPanel("Class 1", 1070, 860, 140, 90, 18)
  my addPanel("Class 2", 1240, 860, 140, 90, 18)
  my addPanel("Class 3", 1410, 860, 140, 90, 18)
  my nextBlankSlide()
  my addTextBox("6. リアルタイムエンジンの使い方", 100, 56, 1180, 70, 34)
  my addTextBox("最新 TIFF を見ながら保存し、そのまま DeepScan で確認できます。", 100, 116, 1160, 34, 18)
  my addImage(logoPath, 1720, 34, 120, 120)
  my addTextBox(("• プロジェクトとサンプル名を決めて 保存" & linefeed & "• 同視野画像ファイルとして保存 を ON にすると、同一視野の画像群として管理" & linefeed & "• 保存した画像は後から DeepScan で再確認できる" & linefeed & "• AI / Manual 表示、manual label、手動 ROI の考え方は DeepScan と共通"), 100, 190, 720, 650, 25)
  my addPanel(("最新 TIFF 表示" & linefeed & "DeepScan overlay"), 920, 180, 650, 620, 26)
  my addPanel(("保存パネル" & linefeed & "プロジェクト" & linefeed & "サンプル名" & linefeed & "同視野画像ファイルとして保存" & linefeed & "保存"), 1600, 190, 220, 360, 20)
  my addPanel(("保存後" & linefeed & "DeepScan で確認"), 1600, 620, 220, 180, 22)
  my addTextBox("説明文の表示が変わっても、ROI overlay 自体はずれないように調整済みです。", 920, 870, 860, 40, 18)
  my nextBlankSlide()
  my addTextBox("7. 結果の出力と更新", 100, 56, 1180, 70, 34)
  my addTextBox("セルカウント結果は PDF / CSV で保存し、利用者側は ff-only で更新します。", 100, 116, 1160, 34, 18)
  my addImage(logoPath, 1720, 34, 120, 120)
  my addTextBox(("• セルカウント結果画面では、クラスごとに ROI を切り替えて確認できる" & linefeed & "• ROI 表は 画像順 / 信頼度順 / ROI番号順 で並べ替え可能" & linefeed & "• A4 出力は PDF出力 に統一" & linefeed & "• CSV は image_name 中心の列構成で、manual_label は 1 列に整理" & linefeed & "• 更新時は git pull --ff-only origin AbyssEye-Neo を使う"), 100, 190, 720, 660, 24)
  my addPanel(("出力" & linefeed & "PDF出力" & linefeed & "CSV出力" & linefeed & "クラス別確認"), 100, 790, 320, 180, 22)
  my addPanel(("更新" & linefeed & "git checkout AbyssEye-Neo" & linefeed & "git pull --ff-only origin AbyssEye-Neo"), 470, 790, 520, 180, 20)
  my addImage(csvScreenshotPath, 980, 220, 760, 72)
  my addTextBox("CSV 出力イメージ", 980, 320, 280, 30, 18)
  my addPanel(("つまずきやすい点" & linefeed & "• DeepScan が開かない → 先に ROI抽出&推論" & linefeed & "• install エラー → Python バージョン確認"), 980, 430, 760, 220, 22)
  my nextBlankSlide()
  my addTextBox("8. まず触るならこの順番", 100, 56, 1180, 70, 34)
  my addTextBox("初回は次の 4 ステップだけ覚えておくと進めやすいです。", 100, 116, 1160, 34, 18)
  my addImage(logoPath, 1720, 34, 120, 120)
  my addPanel(("1" & linefeed & "README に沿って起動"), 180, 260, 330, 260, 28)
  my addTextBox("→", 560, 355, 70, 60, 40)
  my addPanel(("2" & linefeed & "Home から ROI抽出"), 640, 260, 330, 260, 28)
  my addTextBox("→", 1020, 355, 70, 60, 40)
  my addPanel(("3" & linefeed & "データベース または" & linefeed & "リアルタイムエンジン"), 1100, 260, 330, 260, 28)
  my addTextBox("→", 1480, 355, 70, 60, 40)
  my addPanel(("4" & linefeed & "DeepScan で確認して" & linefeed & "PDF / CSV 出力"), 1560, 260, 240, 260, 26)
  my addTextBox(("最初は データベース → ROI抽出&推論 → DeepScan → セルカウント の流れがいちばん分かりやすいです。" & linefeed & "慣れてきたら、リアルタイムエンジンで保存しながら確認する運用に広げてください。"), 180, 640, 1400, 120, 24)
  my addImage(logoPath, 1660, 790, 140, 140)

  delete first slide of front document
  export docRef to outPptxPath as Microsoft PowerPoint
  export docRef to outPdfPath as PDF
  close docRef saving no
end tell

on nextBlankSlide()
  tell application "Keynote"
    tell front document to make new slide with properties {base slide:master slide "空白"}
  end tell
end nextBlankSlide

on addTextBox(txt, xPos, yPos, boxWidth, boxHeight, fontSize)
  tell application "Keynote"
    tell current slide of front document
      set tBox to make new text item with properties {position:{xPos, yPos}, width:boxWidth, height:boxHeight, object text:txt}
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
      set panel to make new shape with properties {position:{xPos, yPos}, width:boxWidth, height:boxHeight, object text:txt}
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
      set imgRef to make new image with properties {file:(POSIX file imagePath), position:{xPos, yPos}, width:boxWidth, height:boxHeight}
      return imgRef
    end tell
  end tell
end addImage
