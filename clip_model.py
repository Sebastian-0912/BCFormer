import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

class CLIPImageClassifier(nn.Module):
    def __init__(self, num_classes, model_name="openai/clip-vit-base-patch32"):
        super(CLIPImageClassifier, self).__init__()
        # 加載 CLIP 模型並提取圖片編碼器
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.image_encoder = self.clip_model.vision_model
        
        # 凍結圖片編碼器的權重
        for param in self.image_encoder.parameters():
            param.requires_grad = False

        # CLIP 的圖片編碼器輸出維度為 512 或 768，視模型版本而定
        image_embedding_dim = self.clip_model.config.projection_dim
        

        # 添加分類頭，進行線性分類
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        # 使用 CLIP 圖片編碼器來提取特徵
        with torch.no_grad():  # 凍結圖片編碼器的權重
            image_features = self.image_encoder(pixel_values=x).pooler_output  # [batch_size, embedding_dim]
        
        # print("Image features shape:", image_features.shape) 
        # 將特徵輸入分類頭
        logits = self.classifier(image_features)
        return logits

# # 測試模型定義
# num_classes = 4  # 自訂的分類數量
# model = CLIPImageClassifier(num_classes=num_classes)

# # 設定設備
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model.to(device)

# # 準備處理器
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# # 假設有一張圖片需要分類
# from PIL import Image

# image_path = "path/to/your/image.jpg"  # 你的圖片路徑
# image = Image.open(image_path).convert("RGB")

# # 將圖片處理成 CLIP 模型所需格式
# inputs = processor(images=image, return_tensors="pt").to(device)
# pixel_values = inputs["pixel_values"]

# # 使用模型進行推理
# model.eval()  # 評估模式
# with torch.no_grad():
#     logits = model(pixel_values)
#     predicted_class = logits.argmax(dim=-1).item()
#     print(f"Predicted class: {predicted_class}")
