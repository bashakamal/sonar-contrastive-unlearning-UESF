import os
from typing import Tuple, Callable

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

from .models import DEVICE, create_model


def preprocess_image(image_path: str, image_size: int = 224) -> Tuple[torch.Tensor, Image.Image]:
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor.to(DEVICE), img


def batch_predict(images: np.ndarray, model: torch.nn.Module) -> np.ndarray:
    model.eval()
    images_t = torch.from_numpy(images).permute(0, 3, 1, 2).float()
    images_t = F.interpolate(images_t, size=(224, 224))
    images_t = (images_t / 255.0 - 0.5) / 0.5
    outputs = model(images_t.to(DEVICE))
    return F.softmax(outputs, dim=1).detach().cpu().numpy()


def explain_with_lime(image_pil, model, model_name: str, save_path: str = None):
    explainer = lime_image.LimeImageExplainer()
    image_np = np.array(image_pil.resize((224, 224)))

    explanation = explainer.explain_instance(
        image_np,
        lambda x: batch_predict(x, model),
        top_labels=1,
        hide_color=0,
        num_samples=1000,
    )

    temp, mask = explanation.get_image_and_mask(
        label=explanation.top_labels[0],
        positive_only=True,
        hide_rest=True,
    )

    plt.imshow(mark_boundaries(temp / 255.0, mask))
    plt.title(f"LIME Explanation - {model_name}")
    plt.axis("off")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        print(f"✅ LIME explanation saved to: {save_path}")
    else:
        plt.show()


def compute_e_final(
    image_pil,
    model_f,
    model_fu,
    save_dir: str = None,
):
    """
    Implements the Unlearn-to-Explain Sonar Framework (UESF)
    E_final = clip(M_f - M_fu, 0, 1)
    """
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    explainer = lime_image.LimeImageExplainer()
    image_np = np.array(image_pil.resize((224, 224)))

    def _batch_predict(images, model):
        return batch_predict(images, model)

    exp_f = explainer.explain_instance(
        image_np, lambda x: _batch_predict(x, model_f),
        top_labels=1, num_samples=1000
    )
    exp_fu = explainer.explain_instance(
        image_np, lambda x: _batch_predict(x, model_fu),
        top_labels=1, num_samples=1000
    )

    img_f, mask_f = exp_f.get_image_and_mask(
        exp_f.top_labels[0], positive_only=True, hide_rest=True
    )
    img_fu, mask_fu = exp_fu.get_image_and_mask(
        exp_fu.top_labels[0], positive_only=True, hide_rest=True
    )

    # E_final
    e_final = (mask_f.astype(float) - mask_fu.astype(float)).clip(0, 1)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    axs[0].imshow(image_np, cmap="gray")
    axs[0].set_title("Original")

    axs[1].imshow(mark_boundaries(img_f / 255.0, mask_f))
    axs[1].set_title("LIME: f(x)")

    axs[2].imshow(mark_boundaries(img_fu / 255.0, mask_fu))
    axs[2].set_title("LIME: f_u(x)")

    axs[3].imshow(e_final, cmap="jet")
    axs[3].set_title("E_final = f(x) - f_u(x)")

    for ax in axs:
        ax.axis("off")

    plt.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "uesf_e_final.png")
        plt.savefig(out_path, dpi=300)
        print(f"✅ UESF visualization saved to: {out_path}")
    else:
        plt.show()
