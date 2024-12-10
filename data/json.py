import json

def trans_json():
    with open('test.txt', 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    with open('test.json', "w", encoding="utf-8") as line:
        json.dump(data, line, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    trans_json()