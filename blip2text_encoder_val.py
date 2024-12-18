import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPModel, AutoTokenizer, BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms
import matplotlib.pyplot as plt
from dataloader import MultiLabelImageDataset
from utils import freeze_model_params, compute_metrics
from model.BLIP2CLIP_TextEncoder_MLP import BLIP2CLIP_TextEncoder_MLP

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_blip_model():
    """
    Load and prepare the BLIP model and processor.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
    model.eval()
    processor.image_processor.do_rescale = False  # prevent rescale twice  
    freeze_model_params(model)
    return processor, model

def load_clip_model():
    """
    Load and prepare the CLIP model and tokenizer.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()
    freeze_model_params(model)
    return tokenizer, model

def get_clip_text_embedding(captions, tokenizer, clip_model):
    """
    Get text embeddings from CLIP model for a list of captions.
    """
    inputs = tokenizer(captions, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        text_embeddings = clip_model.get_text_features(**inputs)
    return text_embeddings

def evaluate(dataloader, smoke_model, blip_model, blip_processor, clip_model, clip_tokenizer):
    """
    Evaluate the model on the validation dataset and return the average loss.
    """
    smoke_model.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for images, true_labels in dataloader:
            images = images.to(device)
            true_labels = true_labels.to(device)

            # Generate captions using BLIP
            blip_inputs = blip_processor(images, return_tensors="pt").to(device)
            generated_ids = blip_model.generate(**blip_inputs)
            captions = [blip_processor.decode(g, skip_special_tokens=True) for g in generated_ids]

            # Get CLIP text embeddings
            text_embeddings = get_clip_text_embedding(captions, clip_tokenizer, clip_model)

            # Forward pass through MLP
            outputs = smoke_model(text_embeddings)
            # Collect predictions and targets for metric computation
            all_predictions.append(outputs.sigmoid())  # Use sigmoid to get probabilities
            all_targets.append(true_labels)

    # Stack predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = compute_metrics(all_predictions, all_targets)

    return metrics

# Main Training Loop
def main():
    # Load BLIP and CLIP
    blip_processor, blip_model = load_blip_model()
    clip_tokenizer, clip_model = load_clip_model()

    # Dataset and Dataloader
    images_dir = "./train_val_dataset/val"
    labels_file = "./train_val_dataset/val_labels.txt"
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.ToTensor()])
    val_dataset = MultiLabelImageDataset(images_dir=images_dir, labels_file=labels_file, transform=transform)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)


    # Initialize Model
    hidden_dim = 256
    num_classes = 3
    smoke_model = BLIP2CLIP_TextEncoder_MLP(hidden_dim, num_classes).to(device)
    smoke_model.load_state_dict(torch.load("./model_pth/B2CT_ep300.pth"))
    # Evaluate
    metrics = evaluate(val_dataloader, smoke_model, blip_model, blip_processor, clip_model, clip_tokenizer)
    
    print(f"Metrics:")
    print(f"  mAP:  {metrics['mAP']:.4f}")
    print(f"  CP:   {metrics['CP']:.4f}")
    print(f"  CR:   {metrics['CR']:.4f}")
    print(f"  CF1:  {metrics['CF1']:.4f}")
    print(f"  OP:   {metrics['OP']:.4f}")
    print(f"  OR:   {metrics['OR']:.4f}")
    print(f"  OF1:  {metrics['OF1']:.4f}")


if __name__ == "__main__":
    main()