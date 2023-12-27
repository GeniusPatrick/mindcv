"""
Generate a folder containing all the main variables' value.

Example:
    python test.py --mode=0 --model="RN50" --pretrained="openai" --force-quick-gelu=True

P.S. This generated folder can be used by difference.py to calculate the difference statistics.

"""

import argparse
import os

from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer

import mindspore as ms
from mindspore import Tensor, ops


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=int,
        default=0,
        help="Mode of set_context, GRAPH_MODE(0) or PYNATIVE_MODE(1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="A keyword (refer to ./src/open_clip/pretrained.py) or path of ckpt file",
    )
    parser.add_argument(
        "--force-quick-gelu",
        type=bool,
        default=False,
        help="",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    ms.set_context(mode=args.mode)
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        force_quick_gelu=args.force_quick_gelu,
    )
    tokenizer = get_tokenizer(args.model)

    image = Tensor(preprocess_val(Image.open("CLIP.png")))
    text = Tensor(tokenizer(["a diagram", "a dog", "a cat"]))

    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = ops.softmax(100.0 * image_features @ text_features.T, axis=-1)
    print("Label probs:", text_probs)

    # dump results
    root = f"./{args.model}-{args.pretrained}"
    os.makedirs(root, exist_ok=True)
    with open(f"{root}/image.txt", "w+") as f:
        f.write(str(image.numpy().tolist()))
    with open(f"{root}/text.txt", "w+") as f:
        f.write(str(text.numpy().tolist()))
    with open(f"{root}/image_features.txt", "w+") as f:
        f.write(str(image_features.numpy().tolist()))
    with open(f"{root}/text_features.txt", "w+") as f:
        f.write(str(text_features.numpy().tolist()))
    with open(f"{root}/text_probs.txt", "w+") as f:
        f.write(str(text_probs.numpy().tolist()))


if __name__ == "__main__":
    main()
