import json

# 输入和输出文件路径
input_path = './Follower.json'   # 修改为你的原始 JSON 路径
output_path = './DCTP.json'  # 修改为你希望保存的路径（也可以和 input_path 相同）

# 读取 JSON 文件
with open(input_path, 'r') as f:
    data = json.load(f)

# 递归地替换 "algorithm": "Follower"
def replace_algorithm(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "algorithm" and v == "Follower":
                obj[k] = "DCTP"
            else:
                replace_algorithm(v)
    elif isinstance(obj, list):
        for item in obj:
            replace_algorithm(item)

replace_algorithm(data)

# 写回 JSON 文件
with open(output_path, 'w') as f:
    json.dump(data, f, indent=2)
