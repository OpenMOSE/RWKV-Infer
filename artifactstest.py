class TextProcessor:
    def __init__(self, target="<RWKVArtifact"):
        self.target = target
        self.buffer = ""
        self.tag_buffer = ""
        self.is_in_tag = False

    def process_text(self, new_text):
        if not new_text:
            return "", None

        output = ""
        tag_output = None
        
        if self.is_in_tag:
            # タグ処理中の場合
            end_pos = new_text.find('>')
            if end_pos >= 0:
                # タグの終了を見つけた
                self.tag_buffer += new_text[:end_pos + 1]
                tag_output = self.tag_buffer
                self.is_in_tag = False
                self.tag_buffer = ""
                output = new_text[end_pos + 1:]  # タグ後のテキストは即時出力
            else:
                # タグが未完了
                self.tag_buffer += new_text
        else:
            # 通常テキスト処理中の場合
            process_text = self.buffer + new_text
            self.buffer = ""
            
            # targetの検索
            pos = process_text.find(self.target)
            
            if pos >= 0:
                # タグの開始を検出
                output = process_text[:pos]  # タグ前のテキストを出力
                self.is_in_tag = True
                self.tag_buffer = self.target
                remaining = process_text[pos + len(self.target):]
                
                # 残りのテキストでタグの終了を確認
                end_pos = remaining.find('>')
                if end_pos >= 0:
                    self.tag_buffer += remaining[:end_pos + 1]
                    tag_output = self.tag_buffer
                    self.is_in_tag = False
                    self.tag_buffer = ""
                    output += remaining[end_pos + 1:]  # タグ後のテキストも出力
                else:
                    self.tag_buffer += remaining
            else:
                # タグの部分一致をチェック
                for i in range(1, min(len(self.target) + 1, len(process_text) + 1)):
                    if self.target.startswith(process_text[-i:]):
                        # 部分一致を見つけた
                        output = process_text[:-i]
                        self.buffer = process_text[-i:]
                        break
                else:
                    # 部分一致もない場合は全て出力
                    output = process_text

        return output, tag_output

# テストコード
def test_processor():
    processor = TextProcessor()
    
    test_cases = [
        # ケース1: 通常のテキスト
        ["これは", "テストです。"],
        
        # ケース2: 似たタグを含むテキスト
        ["<RWKVThinking>このコードは", "独立した、再利用可能な", "コンテンツであり、", "ユーザーが修正や", "実行を行う可能性が", "高いため、アーティファクトとして", "適切じゃ。", "新しいアーティファクトを作成", "しますけの。</RWKVThinking>"],
        
        # ケース3: 目的のタグ
        ["これは<RWK", "VArtifact type=\"test\">", "です。"],
        
        # ケース4: 混在パターン
        ["<RWKVThinking>考え中</RWKVThinking>", "その後に<RWK", "VArtifact type=\"test\">", "が続きます。"]
    ]

    print("=== テスト開始 ===")
    for i, case in enumerate(test_cases):
        print(f"\nテストケース {i + 1}:")
        for chunk in case:
            print(f"\nチャンク入力: '{chunk}'")
            normal_out, tag_out = processor.process_text(chunk)
            
            if normal_out:
                print(f"通常出力: '{normal_out}'")
            if tag_out:
                print(f"タグ検出: {tag_out}")

if __name__ == "__main__":
    test_processor()