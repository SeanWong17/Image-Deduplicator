import os
import argparse
from deduplicator import ImageDeduplicator

def find_image_files(directory):
    """查找目录及其子目录下的所有图片文件"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths

def main():
    parser = argparse.ArgumentParser(description="A tool to find and remove duplicate images based on feature similarity.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing images to deduplicate.")
    parser.add_argument("--output-file", type=str, required=True, help="File to save the list of unique image paths.")
    parser.add_argument("--threshold", type=float, default=0.95, help="Similarity threshold for deduplication (0.0 to 1.0). Default is 0.95.")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use. Set to -1 for CPU. Default is 0.")
    parser.add_argument("--threads", type=int, default=8, help="Number of threads for feature extraction. Default is 8.")
    
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    print(f"Scanning for images in '{args.input_dir}'...")
    image_paths = find_image_files(args.input_dir)
    
    if not image_paths:
        print("No images found.")
        return

    deduplicator = ImageDeduplicator(gpu_id=args.gpu_id, num_threads=args.threads)
    unique_paths = deduplicator.deduplicate(image_paths, threshold=args.threshold)

    print(f"\n--- Process Complete ---")
    print(f"Original images found: {len(image_paths)}")
    print(f"Unique images kept: {len(unique_paths)}")
    
    try:
        with open(args.output_file, 'w') as f:
            for path in unique_paths:
                f.write(path + '\n')
        print(f"List of unique image paths saved to '{args.output_file}'")
    except Exception as e:
        print(f"Error writing to output file: {e}")

if __name__ == "__main__":
    main()
