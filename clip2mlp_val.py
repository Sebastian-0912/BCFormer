import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPModel,AutoTokenizer,AutoProcessor, BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms
from dataloader import MultiLabelImageDataset
from utils import freeze_model_params, compute_metrics
from model.CLIP2MLP import CLIP2MLP

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_clip_model():
    """
    Load and prepare the CLIP model and tokenizer.
    """
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    processor.feature_extractor.do_rescale = False  # prevent rescale twice  
    model.eval()
    freeze_model_params(model)
    return processor,tokenizer, model

def get_clip_image_embedding(images, clip_processor, clip_model):
    """
    Get image embeddings from CLIP model for a batch of images.
    Args:
        images (torch.Tensor): Batch of input images.
        clip_processor: CLIPProcessor for preprocessing images.
        clip_model: CLIP model for generating image embeddings.
    Returns:
        torch.Tensor: Image embeddings from CLIP model.
    """
    with torch.no_grad():
        # Preprocess images using the CLIPProcessor
        processed_images = clip_processor(images=images, return_tensors="pt").to(device)
        image_embeddings = clip_model.get_image_features(**processed_images)
    return image_embeddings


def evaluate(dataloader, smoke_model, clip_model, clip_processor):
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

            image_embeddings = get_clip_image_embedding(images, clip_processor, clip_model)
            
            # Forward pass through MLP
            outputs = smoke_model(image_embeddings)
            
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
    clip_processor, clip_tokenizer, clip_model = load_clip_model()

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
    smoke_model = CLIP2MLP(hidden_dim, num_classes).to(device)
    smoke_model.load_state_dict(torch.load("./model_pth/clip2mlp_ep300.pth"))
    
    # Evaluate
    metrics = evaluate(val_dataloader, smoke_model, clip_model, clip_processor)
    
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