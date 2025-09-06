#!/usr/bin/env python3
"""
RWKV Completion API デバッグ用クライアント
Requests を使用して http://0.0.0.0:9000/completions をテストします
"""

import requests
import json
import time
from typing import Optional, List, Dict, Any

MODENAME = "RWKV-Reka"

class CompletionClient:
    """Completion APIクライアント"""
    
    def __init__(self, base_url: str = "http://0.0.0.0:9000"):
        self.base_url = base_url
        self.completions_url = f"{base_url}/v1/completions"
        
    def create_completion(
        self,
        prompt: str,
        model: str = "RWKV-Reka",
        max_tokens: int = 100,
        temperature: float = 0.3,
        top_p: float = 0.3,
        n: int = 1,
        stream: bool = False,
        echo: bool = False,
        stop: Optional[List[str]] = None,
        presence_penalty: float = 0.3,
        frequency_penalty: float = 0.3,
        logprobs: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Completion APIにリクエストを送信
        
        Args:
            prompt: 入力プロンプト
            model: 使用するモデル名
            max_tokens: 生成する最大トークン数
            temperature: 温度パラメータ（0-2）
            top_p: Top-pサンプリング
            n: 生成する補完の数
            stream: ストリーミングレスポンスを使用するか
            echo: プロンプトを応答に含めるか
            stop: 停止シーケンス
            presence_penalty: 存在ペナルティ
            frequency_penalty: 頻度ペナルティ
            logprobs: ログ確率を返すトークン数
            **kwargs: その他の追加パラメータ
        """
        
        # リクエストボディの構築
        request_data = {
            "prompt": prompt,
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
            "echo": echo,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
        }
        
        # オプションパラメータの追加
        if stop:
            request_data["stop"] = stop
        if logprobs is not None:
            request_data["logprobs"] = logprobs
            
        # 追加のカスタムパラメータ
        request_data.update(kwargs)
        
        print(f"\n{'='*60}")
        print("📤 リクエスト送信")
        print(f"{'='*60}")
        print(f"URL: {self.completions_url}")
        print(f"リクエストボディ:")
        print(json.dumps(request_data, ensure_ascii=False, indent=2))
        
        try:
            if stream:
                return self._handle_stream_response(request_data)
            else:
                return self._handle_normal_response(request_data)
                
        except requests.exceptions.RequestException as e:
            print(f"\n❌ リクエストエラー: {e}")
            return {"error": str(e)}
            
    def _handle_normal_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """通常のレスポンスを処理"""
        
        start_time = time.time()
        response = requests.post(
            self.completions_url,
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print("📥 レスポンス受信")
        print(f"{'='*60}")
        print(f"ステータスコード: {response.status_code}")
        print(f"応答時間: {elapsed_time:.2f}秒")
        
        if response.status_code == 200:
            response_json = response.json()
            print(f"\nレスポンスボディ:")
            print(json.dumps(response_json, ensure_ascii=False, indent=2))
            
            # 生成されたテキストを見やすく表示
            if "choices" in response_json:
                print(f"\n{'='*60}")
                print("🔤 生成されたテキスト")
                print(f"{'='*60}")
                for i, choice in enumerate(response_json["choices"]):
                    if len(response_json["choices"]) > 1:
                        print(f"\n[Choice {i+1}]")
                    print(choice.get("text", ""))
                    
            # Usage情報を表示
            if "usage" in response_json:
                usage = response_json["usage"]
                print(f"\n{'='*60}")
                print("📊 使用統計")
                print(f"{'='*60}")
                print(f"プロンプトトークン: {usage.get('prompt_tokens', 0)}")
                print(f"補完トークン: {usage.get('completion_tokens', 0)}")
                print(f"合計トークン: {usage.get('total_tokens', 0)}")
                
            return response_json
        else:
            print(f"\n❌ エラーレスポンス:")
            print(response.text)
            return {"error": response.text, "status_code": response.status_code}
            
    def _handle_stream_response(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """ストリーミングレスポンスを処理"""
        
        print(f"\n{'='*60}")
        print("📥 ストリーミングレスポンス受信開始")
        print(f"{'='*60}\n")
        
        collected_text = ""
        start_time = time.time()
        
        try:
            response = requests.post(
                self.completions_url,
                json=request_data,
                headers={"Content-Type": "application/json"},
                stream=True
            )
            
            if response.status_code != 200:
                print(f"❌ エラー: ステータスコード {response.status_code}")
                return {"error": response.text, "status_code": response.status_code}
                
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith('data: '):
                        data_str = line_str[6:]  # "data: " を削除
                        
                        if data_str == "[DONE]":
                            print("\n\n✅ ストリーミング完了")
                            break
                            
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                text_chunk = data["choices"][0].get("text", "")
                                collected_text += text_chunk
                                print(text_chunk, end="", flush=True)
                        except json.JSONDecodeError as e:
                            print(f"\n⚠️ JSONパースエラー: {e}")
                            print(f"データ: {data_str}")
                            
            elapsed_time = time.time() - start_time
            
            print(f"\n{'='*60}")
            print("📊 ストリーミング統計")
            print(f"{'='*60}")
            print(f"総応答時間: {elapsed_time:.2f}秒")
            print(f"生成文字数: {len(collected_text)}文字")
            
            return {
                "text": collected_text,
                "streaming": True,
                "elapsed_time": elapsed_time
            }
            
        except Exception as e:
            print(f"\n❌ ストリーミングエラー: {e}")
            return {"error": str(e)}


def run_basic_tests():
    """基本的なテストケースを実行"""
    
    client = CompletionClient()
    
    print("\n" + "="*80)
    print(" RWKV Completion API デバッグテスト")
    print("="*80)
    
    # テスト1: 基本的な補完
    print("\n\n🧪 テスト1: 基本的な補完")
    print("-"*60)
    client.create_completion(
        prompt="The capital of France is",
        model=MODENAME,
        max_tokens=20,
        temperature=0.7
    )
    
    # テスト2: エコーモードのテスト
    print("\n\n🧪 テスト2: エコーモード (プロンプトを含める)")
    print("-"*60)
    client.create_completion(
        prompt="Once upon a time",
        model=MODENAME,
        max_tokens=50,
        temperature=1.0,
        echo=True
    )
    
    # テスト3: 複数の補完を生成
    print("\n\n🧪 テスト3: 複数の補完生成 (n=3)")
    print("-"*60)
    client.create_completion(
        prompt="The best programming language is",
        model=MODENAME,
        max_tokens=30,
        temperature=1.2,
        n=3
    )
    
    # テスト4: ストップシーケンスのテスト
    print("\n\n🧪 テスト4: ストップシーケンス")
    print("-"*60)
    client.create_completion(
        prompt="List three colors:\n1.",
        model=MODENAME,
        max_tokens=100,
        temperature=0.5,
        stop=["\n4.", "Done"]
    )
    
    # テスト5: ストリーミングモード
    print("\n\n🧪 テスト5: ストリーミングモード")
    print("-"*60)
    client.create_completion(
        prompt="Explain quantum computing in simple terms:",
        model=MODENAME,
        max_tokens=150,
        temperature=0.8,
        stream=True
    )
    
    # テスト6: カスタムパラメータ（MRSS等）
    print("\n\n🧪 テスト6: カスタムパラメータテスト")
    print("-"*60)
    client.create_completion(
        prompt="Write a haiku about AI:",
        model=MODENAME,
        max_tokens=50,
        temperature=0.9,
        top_p=0.95,
        presence_penalty=0.5,
        frequency_penalty=0.5,
        penalty_decay=0.99,  # カスタムパラメータ
        mrss_gatingweight=[0.5, 0.3, 0.2]  # カスタムパラメータ
    )


def interactive_mode():
    """インタラクティブモード"""
    
    client = CompletionClient()
    
    print("\n" + "="*80)
    print(" インタラクティブモード")
    print(" 'quit' または 'exit' で終了")
    print("="*80)
    
    while True:
        try:
            print("\n" + "-"*60)
            prompt = input("📝 プロンプトを入力してください: ")
            
            if prompt.lower() in ['quit', 'exit']:
                print("👋 終了します")
                break
                
            if not prompt.strip():
                continue
                
            # パラメータの入力（オプション）
            use_stream = input("ストリーミングを使用しますか？ (y/N): ").lower() == 'y'
            
            temp_input = input("Temperature (デフォルト: 1.0): ").strip()
            temperature = float(temp_input) if temp_input else 1.0
            
            max_tokens_input = input("Max tokens (デフォルト: 100): ").strip()
            max_tokens = int(max_tokens_input) if max_tokens_input else 100
            
            # リクエスト送信
            client.create_completion(
                prompt=prompt,
                model=MODENAME,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=use_stream
            )
            
        except KeyboardInterrupt:
            print("\n👋 中断されました")
            break
        except Exception as e:
            print(f"\n❌ エラー: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--interactive":
        interactive_mode()
    else:
        run_basic_tests()
        print("\n" + "="*80)
        print("✅ すべてのテストが完了しました")
        print("インタラクティブモードを使用するには: python client.py --interactive")
        print("="*80)