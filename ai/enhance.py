import os
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import requests
import dotenv
import argparse
from tqdm import tqdm
from openai import OpenAI

if os.path.exists('.env'):
    dotenv.load_dotenv()

# 读取 prompt 文件
with open("system.txt", "r") as f:
    SYSTEM_PROMPT = f.read()
with open("template.txt", "r") as f:
    USER_TEMPLATE = f.read()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="jsonline data file")
    parser.add_argument("--max_workers", type=int, default=1, help="Maximum number of parallel workers")
    return parser.parse_args()

def check_github_code(content: str) -> Dict:
    code_info = {}
    github_pattern = r"https?://github\.com/([a-zA-Z0-9-_]+)/([a-zA-Z0-9-_\.]+)"
    match = re.search(github_pattern, content)
    if match:
        owner, repo = match.groups()
        repo = repo.rstrip(".git").rstrip(".,)")
        full_url = f"https://github.com/{owner}/{repo}"
        code_info["code_url"] = full_url
        github_token = os.environ.get("TOKEN_GITHUB")
        headers = {"Accept": "application/vnd.github.v3+json"}
        if github_token:
            headers["Authorization"] = f"token {github_token}"
        try:
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            resp = requests.get(api_url, headers=headers, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                code_info["code_stars"] = data.get("stargazers_count", 0)
                code_info["code_last_update"] = data.get("pushed_at", "")[:10]
        except Exception:
            pass
        return code_info
    github_io_pattern = r"https?://[a-zA-Z0-9-_]+\.github\.io(?:/[a-zA-Z0-9-_\.]+)*"
    match_io = re.search(github_io_pattern, content)
    if match_io:
        url = match_io.group(0).rstrip(".,)")
        code_info["code_url"] = url
    return code_info

def process_single_item(client, model_name, system_prompt, user_template, item: Dict, language: str) -> Dict:
    # 检测 GitHub 代码
    code_info = check_github_code(item.get("summary", ""))
    if code_info:
        item.update(code_info)

    default_ai_fields = {
        "tldr": "Summary generation failed",
        "motivation": "Motivation analysis unavailable",
        "method": "Method extraction failed",
        "result": "Result analysis unavailable",
        "conclusion": "Conclusion extraction failed"
    }

    user_prompt = user_template.format(language=language, content=item['summary'])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}},
            max_tokens=2048  # 防止截断
        )
        content = response.choices[0].message.content
        if not content or content.strip() == "":
            raise ValueError("Empty response content")

        parsed = json.loads(content)
        item['AI'] = {**default_ai_fields, **parsed}
    except Exception as e:
        tqdm.write(f"Error for {item.get('id', 'unknown')}: {e}")
        item['AI'] = default_ai_fields

    # 确保所有字段存在
    for field in default_ai_fields:
        if field not in item['AI']:
            item['AI'][field] = default_ai_fields[field]

    return item

def process_all_items(data: List[Dict], model_name: str, language: str, max_workers: int) -> List[Dict]:
    # 创建原生 OpenAI 客户端 (兼容 DeepSeek)
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    )
    tqdm.write(f'Connect to: {model_name}')

    processed_data = [None] * len(data)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(process_single_item, client, model_name, SYSTEM_PROMPT, USER_TEMPLATE, item, language): idx
            for idx, item in enumerate(data)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(data), desc="Processing items"):
            idx = future_to_idx[future]
            try:
                result = future.result()
                processed_data[idx] = result
            except Exception as e:
                tqdm.write(f"Item at index {idx} generated an exception: {e}")
                processed_data[idx] = data[idx]
                processed_data[idx]['AI'] = {
                    "tldr": "Processing failed",
                    "motivation": "Processing failed",
                    "method": "Processing failed",
                    "result": "Processing failed",
                    "conclusion": "Processing failed"
                }
    return processed_data

def main():
    args = parse_args()
    model_name = os.environ.get("MODEL_NAME", 'deepseek-chat')
    language = os.environ.get("LANGUAGE", 'Chinese')

    target_file = args.data.replace('.jsonl', f'_AI_enhanced_{language}.jsonl')
    if os.path.exists(target_file):
        os.remove(target_file)
        tqdm.write(f'Removed existing file: {target_file}')

    data = []
    with open(args.data, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # 去重
    seen_ids = set()
    unique_data = []
    for item in data:
        if item['id'] not in seen_ids:
            seen_ids.add(item['id'])
            unique_data.append(item)
    data = unique_data
    tqdm.write(f'Open: {args.data}')

    processed_data = process_all_items(data, model_name, language, args.max_workers)

    with open(target_file, "w") as f:
        for item in processed_data:
            if item is not None:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()