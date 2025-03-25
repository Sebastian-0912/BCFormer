import matplotlib.pyplot as plt
from collections import defaultdict

# Label file path
label_file = 'final_smoke_datasets_label/labels.txt'  # Replace with your label file path

# Initialize a dictionary to count occurrences of each label
label_count = defaultdict(int)

# Read the label file and count occurrences of each label
with open(label_file, 'r') as f:
    for line in f:
        labels = line.strip().split()[1:]  # Skip the filename, and take the labels
        for label in labels:
            label_count[label] += 1

# Define all possible labels
all_labels = ['smoke', 'fire', 'cloud', 'none']

# Prepare data for plotting
counts = [label_count[label] for label in all_labels]

# Define the colors for the bars
colors = ['silver', 'coral', 'skyblue', 'mediumaquamarine']
plt.figure(figsize=(10, 6))

# Plot the histogram (bar chart)
bars = plt.bar(all_labels, counts, color=colors)

# Add the exact number on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

# Title and labels
plt.title('Occurrences of individual Labels')
# plt.xlabel('Label')
plt.ylabel('Frequency')
plt.savefig('individual_distribution.png')
# Show the plot
# plt.show()
