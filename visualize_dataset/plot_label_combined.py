import matplotlib.pyplot as plt
from collections import defaultdict

# Label file path
label_file = 'final_smoke_datasets_label/labels.txt'  # Replace with your label file path

# Initialize a dictionary to count occurrences of each category
category_count = defaultdict(int)

# Read the label file and classify occurrences into categories
with open(label_file, 'r') as f:
    for line in f:
        labels = set(line.strip().split()[1:])  # Take the labels, and use a set to avoid duplicates
        if labels == {'smoke'}:
            category_count['smoke'] += 1
        elif labels == {'fire'}:
            category_count['fire'] += 1
        elif labels == {'cloud'}:
            category_count['cloud'] += 1
        elif labels == {'none'}:
            category_count['none'] += 1
        elif labels == {'fire', 'smoke'}:
            category_count['(fire, smoke)'] += 1
        elif labels == {'smoke', 'cloud'}:
            category_count['(smoke, cloud)'] += 1
        elif labels == {'fire', 'smoke', 'cloud'}:
            category_count['(fire, smoke, cloud)'] += 1

# Define all categories in the desired order
categories = ['smoke', 'fire', 'cloud', 'none', '(fire, smoke)', '(smoke, cloud)', '(fire, smoke, cloud)']

# Prepare data for plotting
counts = [category_count[category] for category in categories]

# Define the colors for the bars
colors = ['silver', 'coral', 'skyblue', 'mediumaquamarine', 'mediumpurple', 'lightpink', 'tomato']
plt.figure(figsize=(13, 7.8))
# Plot the histogram (bar chart)
bars = plt.bar(categories, counts, color=colors)

# Add the exact number on top of each bar
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height, str(height), ha='center', va='bottom')

# Title and labels
plt.title('Label Occurrences in Dataset (Multi-label Categories)')
# plt.xlabel('Category')
plt.ylabel('Frequency')
plt.savefig('combined_distribution.png')

# Show the plot
plt.show()
