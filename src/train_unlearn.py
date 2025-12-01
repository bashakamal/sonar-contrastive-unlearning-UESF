import os
import argparse
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn import TripletMarginLoss

from .datasets import create_dataloaders, TripletSonarDataset
from .models import create_model, extract_features, DEVICE
from .metrics import classification_report_and_confusion_matrix


def train_unlearning_model(
    data_dir: str,
    model_name: str = "efficientnet_b0",
    batch_size: int = 16,
    num_epochs: int = 20,
    image_size: int = 224,
    num_workers: int = 4,
    lambda_triplet: float = 1.0,
    save_dir: str = "./saved_models/unlearned",
):
    os.makedirs(save_dir, exist_ok=True)

    # Use the same train/val/test split as baseline (by re-running create_dataloaders)
    train_loader, _, test_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )
    # Under the hood, train_loader.dataset is a Subset -> use its dataset + indices
    base_train_dataset = train_loader.dataset

    triplet_dataset = TripletSonarDataset(base_train_dataset, class_names, seabed_class_name="seafloor")
    triplet_loader = DataLoader(
        triplet_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    num_classes = len(class_names)
    model = create_model(model_name, num_classes, pretrained=True)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_triplet = TripletMarginLoss(margin=1.0, p=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Print training distribution
    if isinstance(base_train_dataset, torch.utils.data.Subset):
        labels = [base_train_dataset.dataset[i][1] for i in base_train_dataset.indices]
    else:
        labels = [base_train_dataset[i][1] for i in range(len(base_train_dataset))]
    label_counts = Counter(labels)
    print("🧾 Class distribution in training set:")
    for idx, count in label_counts.items():
        print(f"   {class_names[idx]}: {count}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for anchor, positive, negative, labels in triplet_loader:
            anchor, positive, negative, labels = (
                anchor.to(DEVICE),
                positive.to(DEVICE),
                negative.to(DEVICE),
                labels.to(DEVICE),
            )

            # Classification loss on anchor
            logits = model(anchor)
            loss_cls = criterion_cls(logits, labels)

            # Feature embeddings for TCU
            feat_anchor = extract_features(model, anchor)
            feat_pos = extract_features(model, positive)
            feat_neg = extract_features(model, negative)

            loss_triplet = criterion_triplet(feat_anchor, feat_pos, feat_neg)

            loss = loss_cls + lambda_triplet * loss_triplet

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        acc = 100.0 * correct / total if total > 0 else 0.0
        print(
            f"📘 Epoch {epoch + 1}/{num_epochs} — "
            f"Loss: {running_loss:.4f}, Accuracy (anchor): {acc:.2f}%"
        )

    save_path = os.path.join(save_dir, f"model_unlearned_{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"✅ Unlearned model saved to: {save_path}")

    # Evaluate on test set
    classification_report_and_confusion_matrix(
        model,
        test_loader,
        class_names,
        device=DEVICE,
        title=f"Confusion Matrix: {model_name} (Unlearned)",
        save_path="./results/confusion_matrices/unlearned_confmat.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_triplet", type=float, default=1.0)

    args = parser.parse_args()

    os.makedirs("./results/confusion_matrices", exist_ok=True)

    train_unlearning_model(
        data_dir=args.data_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        image_size=args.image_size,
        num_workers=args.num_workers,
        lambda_triplet=args.lambda_triplet,
    )
