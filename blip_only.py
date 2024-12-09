import torch
from torch.utils.data import DataLoader, random_split
from transformers import BlipProcessor, BlipForConditionalGeneration
from dataloader import MultiLabelImageDataset
from torchvision import transforms
import matplotlib.pyplot as plt

# Initialize BLIP processor and model
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
blip_processor.image_processor.do_rescale = False  # 防止重複縮放

# Function to evaluate a single image
def evaluate_image(image, true_labels):
    inputs = blip_processor(image, return_tensors="pt").to(device)
    generated_ids = blip_model.generate(**inputs)
    caption = blip_processor.decode(generated_ids[0], skip_special_tokens=True)

    # Infer labels from the caption
    inferred_labels = torch.zeros(len(keywords), device=device)
    for i, keyword in enumerate(keywords):
        if keyword in caption.lower():
            inferred_labels[i] = 1

    # Exact match metric
    exact_match = torch.equal(true_labels, inferred_labels)

    # Label proportion metric
    common_labels = (true_labels == inferred_labels).sum().item()  # Count matching labels
    total_labels = true_labels.numel()  # Total number of labels
    label_proportion = common_labels / total_labels

    return caption, inferred_labels, exact_match, label_proportion

batch_size = 1

# Data Loading
images_dir = "./final_smoke_datasets"  # Specify your directory
labels_file = "./final_smoke_datasets_label/labels.txt"  # Specify your label file
transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
dataset = MultiLabelImageDataset(images_dir=images_dir, labels_file=labels_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Define keywords for labels
keywords = ["fire", "smoke", "cloud"]

# Evaluation metrics
total_samples = 0
exact_match_count = 0
total_label_proportion = 0.0

# Iterate over the dataset and evaluate
for image, true_labels in dataloader:
    total_samples += 1
    image = image.to(device)
    true_labels = true_labels.squeeze().to(device)
    caption, inferred_labels, exact_match, label_proportion = evaluate_image(image, true_labels)
    
    if exact_match:
        exact_match_count += 1
    total_label_proportion += label_proportion

    # Print results for each sample (optional)
    print(f"Caption: {caption}")
    print(f"True Labels: {true_labels.tolist()}")
    print(f"Inferred Labels: {inferred_labels.tolist()}")
    print(f"Exact Match: {exact_match}")
    print(f"Label Proportion: {label_proportion:.2f}")

# Final metrics
exact_match_accuracy = exact_match_count / total_samples
average_label_proportion = total_label_proportion / total_samples

print(f"Exact Match Accuracy: {exact_match_accuracy:.2%}")
print(f"Average Label Proportion Score: {average_label_proportion:.2f}")
