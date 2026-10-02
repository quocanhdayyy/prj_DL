import numpy as np
from sklearn.utils.class_weight import compute_class_weight

data = np.load("train_data.npz")

y = data["y"]

unique, counts = np.unique(y, return_counts=True)

print("Total samples:", len(y))
print("Number of classes:", len(unique))
print("Smallest class:", counts.min())
print("Largest class:", counts.max())

print("\nClass distribution:")
for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count}")