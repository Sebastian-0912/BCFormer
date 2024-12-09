import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel,AutoTokenizer,AutoProcessor, BlipProcessor, BlipForConditionalGeneration
import numpy as np
import torch.fft

class BFC_Classifier(nn.Module):
    def __init__(self, num_classes=4, hidden_dim=256, model_name="openai/clip-vit-base-patch32"):
        super(BFC_Classifier, self).__init__()
        
        # Load CLIP model for image encoding
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        
        self.clip_processor.feature_extractor.do_rescale = False  # 防止重複縮放
        self.blip_processor.image_processor.do_rescale = False  # 防止重複縮放
        
        # Freeze weights of image and text encoders
        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.blip_model.parameters():
            param.requires_grad = False
        
        # CLIP image embedding dimension
        image_embedding_dim = self.clip_model.config.projection_dim
        text_embedding_dim = self.clip_model.config.projection_dim
        self.fft_dim = 128  # Dimension for FFT phase features

        # Define MLP fusion for combined features
        self.mlp_fusion = nn.Sequential(
            nn.Linear(image_embedding_dim + text_embedding_dim + self.fft_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):

        with torch.no_grad():
            image_features = self.clip_processor(images=x, return_tensors="pt").to(x.device)
            image_features = self.clip_model.get_image_features(**image_features)
        # BLIP caption generation
        blip_inputs = self.blip_processor(images=x, return_tensors="pt").to(x.device)
        with torch.no_grad():
            caption_ids = self.blip_model.generate(**blip_inputs)
            captions = [self.blip_processor.decode(c_id, skip_special_tokens=True) for c_id in caption_ids]



        # print(type(captions), captions)  # 確認輸入類型和內容

        # text_tokens = self.clip_model.get_text_features(captions)
        # with torch.no_grad():
        #     text_features = self.clip_model.get_text_features(**text_tokens) 
        # 使用 clip_tokenizer 對生成的 captions 進行編碼
        captions_encoded = self.clip_tokenizer(captions, padding=True, return_tensors="pt").to(x.device)

        # 使用 CLIP 模型提取文本特徵
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(input_ids=captions_encoded["input_ids"],
                                                            attention_mask=captions_encoded["attention_mask"])
        # print("text_features shape",text_features.shape)#

        # FFT phase extraction
        batch_fft_phase = []
        for i in range(x.size(0)):
            image_np = x[i].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) for FFT
            image_gray = np.mean(image_np, axis=2)  # Convert to grayscale
            fft_result = torch.fft.fft2(torch.tensor(image_gray, dtype=torch.float32))
            phase_info = torch.angle(fft_result).flatten()[:self.fft_dim].to(x.device)  # First 128 elements of phase
            batch_fft_phase.append(phase_info)
        
        # Stack phase info for batch processing
        batch_fft_phase = torch.stack(batch_fft_phase, dim=0)  # [batch_size, fft_dim]
        # print("fft_pahae shape",batch_fft_phase.shape)

        # Concatenate features from image, text, and FFT phase
        combined_features = torch.cat((image_features, text_features, batch_fft_phase), dim=1)

        # Forward pass through MLP fusion
        output = self.mlp_fusion(combined_features)
        return output
