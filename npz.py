import numpy as np
import torch
from vit_model import vit_huge_patch14_224_in21k as create_model


def map_npz_to_pytorch_key(npz_key):
    """將 .npz 權重名稱映射到 PyTorch 的名稱"""
    # 移除 'Transformer/' 前綴
    npz_key = npz_key.replace("Transformer/", "")
    
    # 轉換層數和對應結構
    npz_key = npz_key.replace("encoderblock_", "blocks.")
    
    # 轉換多頭注意力層
    npz_key = npz_key.replace("MultiHeadDotProductAttention_1/", "attn.")
    npz_key = npz_key.replace("query/kernel", "qkv.weight")
    npz_key = npz_key.replace("key/kernel", "qkv.weight")
    npz_key = npz_key.replace("value/kernel", "qkv.weight")
    npz_key = npz_key.replace("query/bias", "qkv.bias")
    npz_key = npz_key.replace("key/bias", "qkv.bias")
    npz_key = npz_key.replace("value/bias", "qkv.bias")
    
    npz_key = npz_key.replace("out/kernel", "proj.weight")
    npz_key = npz_key.replace("out/bias", "proj.bias")
    
    # 轉換 LayerNorm 和 MLP 層
    npz_key = npz_key.replace("LayerNorm_0/scale", "norm1.weight")
    npz_key = npz_key.replace("LayerNorm_0/bias", "norm1.bias")
    npz_key = npz_key.replace("LayerNorm_2/scale", "norm2.weight")
    npz_key = npz_key.replace("LayerNorm_2/bias", "norm2.bias")
    
    npz_key = npz_key.replace("MlpBlock_3/Dense_0/kernel", "mlp.fc1.weight")
    npz_key = npz_key.replace("MlpBlock_3/Dense_0/bias", "mlp.fc1.bias")
    npz_key = npz_key.replace("MlpBlock_3/Dense_1/kernel", "mlp.fc2.weight")
    npz_key = npz_key.replace("MlpBlock_3/Dense_1/bias", "mlp.fc2.bias")
    
    return npz_key

def load_npz_weights(model, npz_path):
    # 加載 .npz 文件
    npz_weights = np.load(npz_path)

    # PyTorch 模型的 state_dict
    state_dict = model.state_dict()
    new_state_dict = {}

    # 跟踪加載情況
    successfully_loaded_keys = []
    missing_keys = []
    unmatched_keys = []

    # 遍歷 PyTorch 模型中的每個權重名稱
    for pytorch_key in state_dict.keys():
        matched = False
        # 遍歷 .npz 文件中的每個權重名稱
        for npz_key in npz_weights.keys():
            mapped_key = map_npz_to_pytorch_key(npz_key)
            if mapped_key == pytorch_key:
                # 加載 numpy 權重到 PyTorch 模型中
                new_state_dict[pytorch_key] = torch.from_numpy(npz_weights[npz_key])
                successfully_loaded_keys.append(pytorch_key)
                matched = True
                break
        
        if not matched:
            missing_keys.append(pytorch_key)
    
    # 打印每個 PyTorch 權重是否成功加載
    for key in successfully_loaded_keys:
        print(f"Loaded: {key}")
    
    # 打印未加載的 PyTorch 權重名稱
    if missing_keys:
        for key in missing_keys:
            print(f"Warning: {key} not found in .npz file, skipping.")
    
    # 打印 .npz 文件中無法匹配的鍵
    for npz_key in npz_weights.keys():
        mapped_key = map_npz_to_pytorch_key(npz_key)
        if mapped_key not in state_dict:
            unmatched_keys.append(npz_key)

    if unmatched_keys:
        for key in unmatched_keys:
            print(f"Warning: {key} in .npz file does not match any PyTorch model key.")

    # 加載權重到模型
    model.load_state_dict(new_state_dict, strict=False)
    print("Model weights loaded from .npz file.")

# 創建 ViT-Huge Patch14 模型
model = create_model(num_classes=21843)

# 加載 .npz 文件中的權重
npz_path = 'model/ViT-H_14.npz'  # 替換成你的 .npz 文件路徑
load_npz_weights(model, npz_path)

# 檢查模型權重是否已加載成功
print("Model weights successfully loaded.")

# import numpy as np


# npz_weights = np.load(npz_path)

# # 打印 .npz 文件中的權重名稱
# for key in npz_weights.keys():
#     print(key)
