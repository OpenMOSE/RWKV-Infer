import json
import requests
import multiprocessing
import math
import sys
import argparse
from typing import List

# OpenAI API互換のエンドポイント
API_URL = "http://127.0.0.1:9000/v1/chat/completions"

def generate_reject(prompt: str) -> str:
    """
    LLMに問い合わせて reject 文字列を生成する。
    """
    payload = {
        "model": "RWKV-x070-1B5-CJE-e12.pth",  # 任意のモデル名に置き換えてください
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 1.0,
        "topp": 0.5
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        # 想定されるレスポンス構造:
        # {
        #   "id": ...,
        #   "object": ...,
        #   "created": ...,
        #   "choices": [
        #       {
        #           "message": {
        #               "role": "assistant",
        #               "content": "生成されたテキスト..."
        #           },
        #           ...
        #       }
        #   ],
        #   ...
        # }
        reject_text = data["choices"][0]["message"]["content"]
        return reject_text
    except Exception as e:
        print(f"Error in generate_reject: {e}", file=sys.stderr)
        return ""

def process_chunk(json_lines: List[str]) -> List[str]:
    """
    JSON文字列リストを受け取り、"reject" フィールドが無いもしくは空の場合に LLM から生成して更新し、
    更新後の各行を JSON 文字列として返す。
    """
    results = []
    for line in json_lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            # JSONとして読み込めない行はスキップ、もしくはログ処理
            continue

        if "reject" not in data or not data["reject"]:
            prompt_text = data.get("prompt", "")
            if prompt_text:
                data["reject"] = generate_reject(prompt_text)
            else:
                data["reject"] = ""

        results.append(json.dumps(data, ensure_ascii=False))
    return results

def chunkify(data_list: List[str], num_chunks: int) -> List[List[str]]:
    """
    data_list を num_chunks 個のチャンクに分割して返す。
    """
    if num_chunks <= 0:
        return [data_list]  # num_chunksが無効なら一括処理にフォールバック

    chunk_size = math.ceil(len(data_list) / num_chunks)
    return [data_list[i:i + chunk_size] for i in range(0, len(data_list), chunk_size)]

def main(input_file: str,
         output_file: str,
         num_processes: int,
         chunk_size: int) -> None:
    """
    ストリーミング方式で input_file を chunk_size 行ずつ読み込みつつ処理し、
    "reject" フィールドを生成して output_file に書き出す。
    """

    pool = multiprocessing.Pool(processes=num_processes)

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        while True:
            # chunk_size 行だけ読み込む
            lines = []
            for _ in range(chunk_size):
                line = fin.readline()
                if not line:
                    break
                lines.append(line)

            # ファイル末尾に達した場合は終了
            if not lines:
                break

            # 読み込んだ lines をさらに num_processes 個に分割して並列処理
            splitted = chunkify(lines, num_processes)
            results_list = pool.map(process_chunk, splitted)

            # 出力ファイルに書き込み
            for results in results_list:
                for line_result in results:
                    fout.write(line_result + "\n")

    pool.close()
    pool.join()

def parse_args():
    parser = argparse.ArgumentParser(description="JSONL の 'reject' フィールドを LLM で生成するスクリプト")
    parser.add_argument(
        "-i", "--input_file",
        type=str,
        default="myfolder/RLHF/qwq.jsonl",
        help="入力 JSONL ファイルパス (デフォルト: input.jsonl)"
    )
    parser.add_argument(
        "-o", "--output_file",
        type=str,
        default="myfolder/RLHF/qwq_reject.jsonl",
        help="出力 JSONL ファイルパス (デフォルト: output.jsonl)"
    )
    parser.add_argument(
        "-n", "--num_processes",
        type=int,
        default=4,
        help="並列処理数 (デフォルト: 4)"
    )
    parser.add_argument(
        "-c", "--chunk_size",
        type=int,
        default=1000,
        help="一度に読み込む行数 (デフォルト: 1000)"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(
        input_file=args.input_file,
        output_file=args.output_file,
        num_processes=args.num_processes,
        chunk_size=args.chunk_size
    )
