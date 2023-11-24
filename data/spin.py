from PIL import Image
import os


def rotate_and_save(image_path, output_directory):
    # Open the image
    original_image = Image.open(image_path)

    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Rotate the image 7 times by 1/7
    for i in range(0, 8):
        rotated_image = original_image.rotate(360 * i / 8)
        rotated_image = rotated_image.convert("RGB")

        img_name = image_path.split("\\")[2][0:-4]

        # Save the rotated image
        output_path = os.path.join(output_directory, f"{img_name}_{i}.jpg")
        rotated_image.save(output_path)


if __name__ == "__main__":
    input_directory = "./Training224"
    output_directory = "./Training224r"

    for folder in os.listdir(input_directory):
        folder_path = os.path.join(input_directory, folder)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            # Iterate through each image file in the folder
            for filename in os.listdir(folder_path):
                if filename.endswith(".png") or filename.endswith(".jpg"):
                    image_path = os.path.join(folder_path, filename)
                    rotate_and_save(image_path, os.path.join(output_directory, folder))
