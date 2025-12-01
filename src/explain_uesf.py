import os
import argparse

import torch
import torch.nn.functional as F

from .models import create_model, DEVICE
from .lime_utils import preprocess_image, explain_with_lime, compute_e_final


def load_model_checkpoint(model_name: str, num_classes: int, ckpt_path: str):
    model = create_model(model_name, num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Same dataset used for training, for class order")
    parser.add_argument("--model_name", type=str, default="efficientnet_b0")
    parser.add_argument("--baseline_ckpt", type=str,
                        default="./saved_models/baseline/best_efficientnet_b0.pth")
    parser.add_argument("--unlearned_ckpt", type=str,
                        default="./saved_models/unlearned/model_unlearned_efficientnet_b0.pth")
    parser.add_argument("--save_dir", type=str, default="./results/lime")

    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # We need class order; easiest is to reuse dataloader
    from .datasets import create_dataloaders
    _, _, _, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=1,
        image_size=224,
        num_workers=0,
    )
    num_classes = len(class_names)
    print("✅ Class names:", class_names)

    # Load models
    model_f = load_model_checkpoint(args.model_name, num_classes, args.baseline_ckpt)
    model_fu = load_model_checkpoint(args.model_name, num_classes, args.unlearned_ckpt)

    # Image
    input_tensor, image_pil = preprocess_image(args.image_path)

    # Show predictions
    with torch.no_grad():
        out_f = model_f(input_tensor)
        out_fu = model_fu(input_tensor)

        probs_f = F.softmax(out_f, dim=1).squeeze().cpu().numpy()
        probs_fu = F.softmax(out_fu, dim=1).squeeze().cpu().numpy()

    import numpy as np
    idx_f = np.argmax(probs_f)
    idx_fu = np.argmax(probs_fu)

    print(f"\n🧠 Baseline prediction: {class_names[idx_f]} ({probs_f[idx_f] * 100:.2f}%)")
    print(f"🧠 Unlearned prediction: {class_names[idx_fu]} ({probs_fu[idx_fu] * 100:.2f}%)")

    # LIME for each model
    explain_with_lime(image_pil, model_f, f"{args.model_name}_baseline",
                      save_path=os.path.join(args.save_dir, "lime_baseline.png"))
    explain_with_lime(image_pil, model_fu, f"{args.model_name}_unlearned",
                      save_path=os.path.join(args.save_dir, "lime_unlearned.png"))

    # UESF E_final
    compute_e_final(image_pil, model_f, model_fu, save_dir=args.save_dir)
