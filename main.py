import sys
from image_to_3d import process_image_to_3d
from text_to_3d import generate_3d_from_text
from utils import visualize_model
def main():
    mode = input("Choose mode (image/text): ").strip().lower()
    if mode == "image":
        image_path = input("Enter image path: ").strip()
        process_image_to_3d(image_path)
    elif mode == "text":
        prompt = input("Enter a short description: ").strip()
        out    = "output/model_from_text.obj"
        generate_3d_from_text(prompt, out)
        visualize_model(out)
    else:
        print("Invalid mode. Choose 'image' or 'text'.")

if __name__ == "__main__":
    main()