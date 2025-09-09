# predict.py
import os
import json
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

def load_class_names():
    """
    Priority:
      1) models/class_indices.json (if your train script saved it)
      2) subfolders under samples/train (sorted)
      3) fallback ['car','cat','dog']
    """
    # 1) from json if available
    ci_path = os.path.join("models", "class_indices.json")
    if os.path.exists(ci_path):
        with open(ci_path, "r") as f:
            class_indices = json.load(f)  # e.g. {"car":0,"cat":1,"dog":2}
        # invert to list where index -> class name
        inv = sorted(class_indices.items(), key=lambda kv: kv[1])
        return [k for k, _ in inv]

    # 2) from samples/train
    train_dir = os.path.join("samples", "train")
    if os.path.isdir(train_dir):
        names = [d for d in os.listdir(train_dir)
                 if os.path.isdir(os.path.join(train_dir, d))]
        if names:
            return sorted(names)

    # 3) fallback
    return ["car", "cat", "dog"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True, help="Path to image file")
    parser.add_argument("--model", default="models/cnn_model.keras",
                        help="Path to saved model (.keras or .h5)")
    parser.add_argument("--img-size", type=int, default=128,
                        help="Image size used during training")
    args = parser.parse_args()

    # load model
    model = tf.keras.models.load_model(args.model)
    class_names = load_class_names()

    # prepare image
    img = image.load_img(args.img, target_size=(args.img_size, args.img_size))
    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    # predict
    preds = model.predict(x, verbose=0)
    probs = preds[0]  # shape (num_classes,)
    top3_idx = np.argsort(probs)[::-1][:3]

    # pretty print
    print("\nTop-3 predictions:")
    for rank, idx in enumerate(top3_idx, start=1):
        cname = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        print(f"{rank}) {cname}: {probs[idx]*100:.2f}%")

    top1_name = class_names[top3_idx[0]] if top3_idx[0] < len(class_names) else f"class_{top3_idx[0]}"
    print(f"\n✅ Predicted Class: {top1_name}")

    # save a tiny report
    os.makedirs("reports", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.img))[0]
    out_path = os.path.join("reports", f"prediction_{base}.txt")
    with open(out_path, "w") as f:
        f.write(f"Image: {args.img}\nModel: {args.model}\n\nTop-3 predictions:\n")
        for rank, idx in enumerate(top3_idx, start=1):
            cname = class_names[idx] if idx < len(class_names) else f"class_{idx}"
            f.write(f"{rank}) {cname}: {probs[idx]*100:.2f}%\n")
        f.write(f"\nPredicted Class: {top1_name}\n")
    print(f"📝 Saved: {out_path}")

if __name__ == "__main__":
    main()
