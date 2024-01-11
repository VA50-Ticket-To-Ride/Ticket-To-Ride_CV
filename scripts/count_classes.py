from collections import Counter

labels_path = "ignore/board_clean_labels.txt"
with open(labels_path, 'r') as labels_file:
    lines = [line.rstrip() for line in labels_file]

all_classes = lambda line: line.split(", ")[8]
colors_only = lambda line: line.split(", ")[8][:3]

class_count = Counter(map(colors_only, lines))
for class_name, count in class_count.most_common():
    print(class_name + ":    \t" + str(count))
