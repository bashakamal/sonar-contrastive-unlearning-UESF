import os
import argparse

import torch
import torch.nn as nn

from .datasets import create_dataloaders
from .models import create_model, DEVICE
from .metrics import evaluate_accuracy, classification_report_and_confusion_matrix


def train_and_evaluate_baseline(
    data_dir: str,
    model_name: str = "efficientnet_b0",
    batch_size: int = 16,
    num_epochs: int = 20,
    image_size: int = 224,
    num_workers: int = 4,
    save_dir: str = "./saved_models/baseline",
):
    os.makedirs(save_dir, exist_ok=True)

    # Dataloaders
    train_loader, val_loader, test_loader, class_names = create_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=image_size,
        num_workers=num_workers,
    )
    num_classes = len(class_names)

    # Model
    model = create_model(model_name, num_classes, pretrained=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_path = os.path.join(save_dir, f"best_{model_name}.pth")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * correct / total
        val_acc = evaluate_accuracy(model, val_loader, DEVICE)

        print(
            f"📘 Epoch {epoch + 1}/{num_epochs} "
            f"Loss: {running_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_path)
            print(f"✅ New best model saved at val acc {val_acc:.2f}% -> {best_path}")

    print("\n🔍 Final evaluation on test set using best checkpoint...")
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))

    classification_report_and_confusion_matrix(
        model,
        test_loader,
        class_names,
        device=DEVICE,
        title=f"Confusion Matrix: {model_name} (Baseline)",
        save_path="./results/confusion_matrices/baseline_confmat.png",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to ImageFolder-style dataset")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()

    os.makedirs("./results/confusion_matrices", exist_ok=True)

    train_and_evaluate_baseline(
        data_dir=args.data_dir,
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
