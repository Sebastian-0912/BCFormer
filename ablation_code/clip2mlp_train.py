import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import CLIPModel,AutoTokenizer,AutoProcessor, BlipProcessor, BlipForConditionalGeneration
from torchvision import transforms
import matplotlib.pyplot as plt
from dataloader import MultiLabelImageDataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper Functions
def freeze_model_params(model):
    """
    Freeze the parameters of a model to prevent updates during training.
    """
    for param in model.parameters():
        param.requires_grad = False

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


def train_one_epoch(dataloader, smoke_model, clip_model, clip_processor, criterion, optimizer):
    """
    Train the model for one epoch and return the average loss.
    """
    smoke_model.train()
    running_loss = 0.0

    for images, true_labels in dataloader:
        images = images.to(device)
        true_labels = true_labels.to(device)
        print(1)
            
        # Get CLIP text embeddings
        image_embeddings = get_clip_image_embedding(images, clip_processor, clip_model)
        
        # Forward pass through MLP
        outputs = smoke_model(image_embeddings)
        loss = criterion(outputs, true_labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def evaluate(dataloader, smoke_model, clip_model, clip_processor, criterion):
    """
    Evaluate the model on the validation dataset and return the average loss.
    """
    smoke_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, true_labels in dataloader:
            images = images.to(device)
            true_labels = true_labels.to(device)

            
            image_embeddings = get_clip_image_embedding(images, clip_processor, clip_model)
            
            # Forward pass through MLP
            outputs = smoke_model(image_embeddings)
            val_loss += criterion(outputs, true_labels).item()

    return val_loss / len(dataloader)

def plot_loss_curves(train_losses, val_losses):
    """
    Plot the training and validation loss curves.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.savefig("bclip_loss_curve.png")
    plt.show()

# Model Definition
class BLIP2CLIP_TextImage_MLP(nn.Module):
    def __init__(self, hidden_dim, num_classes):
        super(BLIP2CLIP_TextImage_MLP, self).__init__()
        self.l1 = nn.Linear(512, hidden_dim)  # Text (512) + Image (512)
        self.l2 = nn.Linear(hidden_dim, num_classes)
        self.relu = nn.ReLU()

    def forward(self, combined_embeddings):
        x = self.l1(combined_embeddings)
        x = self.relu(x)
        x = self.l2(x)
        return x


# Main Training Loop
def main():
    # Load BLIP and CLIP
    clip_processor, clip_tokenizer, clip_model = load_clip_model()

    # Dataset and Dataloader
    images_dir = "./final_smoke_datasets"
    labels_file = "./final_smoke_datasets_label/labels.txt"
    transform = transforms.Compose([transforms.Resize((224, 224)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor()])
    dataset = MultiLabelImageDataset(images_dir=images_dir, labels_file=labels_file, transform=transform)
    # val_dataset = ...  # Load validation dataset
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize Model
    hidden_dim = 256
    num_classes = 3
    smoke_model = BLIP2CLIP_TextImage_MLP(hidden_dim, num_classes).to(device)

    # Loss, Optimizer, Scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(smoke_model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Training Loop
    num_epochs = 500
    train_losses, val_losses = [], []

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(dataloader, smoke_model, clip_model, clip_processor, criterion, optimizer)
        # val_loss = evaluate(val_dataloader, smoke_model, clip_model, clip_processor, criterion)

        train_losses.append(train_loss)
        # val_losses.append(val_loss)

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {train_loss:.4f}.")

        if (epoch+1) *10 == 0:
            torch.save(smoke_model.state_dict(), f"./weights/clip2mlp_ep{epoch}.pth")
        
        scheduler.step()

    # Plot Loss Curves
    plot_loss_curves(train_losses, val_losses)

if __name__ == "__main__":
    main()