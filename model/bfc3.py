import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel,AutoTokenizer,AutoProcessor, BlipProcessor, BlipForConditionalGeneration
import numpy as np
import torch.fft

class Conv_bn_block(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._conv = torch.nn.Conv2d(*args, **kwargs)
        self._bn = torch.nn.BatchNorm2d(kwargs['out_channels'])

    def forward(self, input):
        return F.relu(self._bn(self._conv(input)))


class Res_block(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self._conv1 = torch.nn.Conv2d(in_channels, in_channels//4, kernel_size=1, stride=1)
        self._conv2 = torch.nn.Conv2d(in_channels//4, in_channels//4, kernel_size=3, stride=1, padding=1)
        self._conv3 = torch.nn.Conv2d(in_channels//4, in_channels, kernel_size=1, stride=1)
        self._bn = torch.nn.BatchNorm2d(in_channels)

    def forward(self, x):
        xin = x
        x = F.relu(self._conv1(x))
        x = F.relu(self._conv2(x))
        x = self._conv3(x)
        x = torch.add(xin, x)
        x = F.relu(self._bn(x))

        return x

class encoder_net(torch.nn.Module):
    def __init__(self, in_channels, get_feature_map=False):
        super().__init__()
        self.cnum = 32
        self.get_feature_map = get_feature_map
        self._conv1_1 = Conv_bn_block(
            in_channels=in_channels,
            out_channels=self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)
        self._conv1_2 = Conv_bn_block(
            in_channels=self.cnum,
            out_channels=self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)

        # --------------------------
        self._pool1 = torch.nn.Conv2d(
            in_channels=self.cnum,
            out_channels=2*self.cnum,
            kernel_size=3,
            stride=2,
            padding=1)
        self._conv2_1 = Conv_bn_block(
            in_channels=2*self.cnum,
            out_channels=2*self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)
        self._conv2_2 = Conv_bn_block(
            in_channels=2*self.cnum,
            out_channels=2*self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)

        # ---------------------------
        self._pool2 = torch.nn.Conv2d(
            in_channels=2*self.cnum,
            out_channels=4*self.cnum,
            kernel_size=3,
            stride=2,
            padding=1)
        self._conv3_1 = Conv_bn_block(
            in_channels=4*self.cnum,
            out_channels=4*self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)

        self._conv3_2 = Conv_bn_block(
            in_channels=4*self.cnum,
            out_channels=4*self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)

        # ---------------------------
        self._pool3 = torch.nn.Conv2d(
            in_channels=4*self.cnum,
            out_channels=8*self.cnum,
            kernel_size=3,
            stride=2,
            padding=1)
        self._conv4_1 = Conv_bn_block(
            in_channels=8*self.cnum,
            out_channels=8*self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)
        self._conv4_2 = Conv_bn_block(
            in_channels=8*self.cnum,
            out_channels=8*self.cnum,
            kernel_size=3,
            stride=1,
            padding=1)

    def forward(self, x):
        x = self._conv1_1(x)
        x = self._conv1_2(x)
        x = F.relu(self._pool1(x))
        x = self._conv2_1(x)
        x = self._conv2_2(x)
        f1 = x
        x = F.relu(self._pool2(x))
        x = self._conv3_1(x)
        x = self._conv3_2(x)
        f2 = x
        x = F.relu(self._pool3(x))
        x = self._conv4_1(x)
        x = self._conv4_2(x)
        if self.get_feature_map:
            return x, [f2, f1]
        else:
            return x
            
class BFC_Classifier(nn.Module):
    def __init__(self, num_classes=3, hidden_dim=256, model_name="openai/clip-vit-base-patch32"):
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

        self.encoder_net = encoder_net(in_channels=1)  # Initialize encoder_net
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
        # FFT phase extraction for the entire batch
        # Convert the batch to grayscale
        x_gray = x.mean(dim=1)  # Average across the color channels, assumes x is [B, C, H, W]
        
        # Perform 2D FFT across the spatial dimensions (H, W) for the batch
        fft_result = torch.fft.fft2(x_gray)  # Output shape: [B, H, W]
        
        # Extract phase information
        phase_info = torch.angle(fft_result)  # Shape: [B, H, W]
        
       # Pass phase_info through encoder_net
        encoded_phase_info = self.encoder_net(phase_info.unsqueeze(1))  # Add channel dimension

        # Flatten the encoded phase information and select the first fft_dim elements
        batch_fft_phase = encoded_phase_info.view(encoded_phase_info.size(0), -1)[:, :self.fft_dim]  # Shape: [B, fft_dim]

        # Concatenate features from image, text, and FFT phase
        combined_features = torch.cat((image_features, text_features, batch_fft_phase), dim=1)

        # Forward pass through MLP fusion
        output = self.mlp_fusion(combined_features)
        return output
