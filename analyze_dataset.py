#!/usr/bin/env python3
"""
Dataset analysis utility for crop disease datasets
This script analyzes the collected dataset and generates statistics and visualizations
"""

import os
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from collections import defaultdict, Counter
import random
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Crop Disease Dataset Analyzer")
    parser.add_argument(
        "--dataset",
        type=str,
        default="./dataset",
        help="Path to the dataset directory (default: ./dataset)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./dataset_analysis",
        help="Output directory for analysis results (default: ./dataset_analysis)",
    )
    return parser.parse_args()


def count_images_by_category(dataset_dir):
    """Count images by category (crop, condition)."""
    stats = defaultdict(lambda: defaultdict(int))
    total = 0

    # Walk through dataset directory
    for crop_name in os.listdir(dataset_dir):
        crop_dir = os.path.join(dataset_dir, crop_name)
        if not os.path.isdir(crop_dir):
            continue

        for condition_name in os.listdir(crop_dir):
            condition_dir = os.path.join(crop_dir, condition_name)
            if not os.path.isdir(condition_dir):
                continue

            # Count images
            image_files = [
                f
                for f in os.listdir(condition_dir)
                if os.path.isfile(os.path.join(condition_dir, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            count = len(image_files)
            stats[crop_name][condition_name] = count
            total += count

    return stats, total


def get_image_size_stats(dataset_dir, sample_size=100):
    """Analyze image sizes (random sampling)."""
    image_sizes = []
    file_paths = []

    # Collect all image paths
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                file_paths.append(os.path.join(root, file))

    # Sample if there are too many images
    if len(file_paths) > sample_size:
        file_paths = random.sample(file_paths, sample_size)

    # Get image sizes
    for file_path in file_paths:
        try:
            with Image.open(file_path) as img:
                image_sizes.append(img.size)
        except Exception:
            pass

    # Calculate statistics
    widths = [size[0] for size in image_sizes]
    heights = [size[1] for size in image_sizes]

    stats = {
        "sample_size": len(image_sizes),
        "width": {
            "min": min(widths) if widths else 0,
            "max": max(widths) if widths else 0,
            "mean": sum(widths) / len(widths) if widths else 0,
        },
        "height": {
            "min": min(heights) if heights else 0,
            "max": max(heights) if heights else 0,
            "mean": sum(heights) / len(heights) if heights else 0,
        },
        "most_common_sizes": Counter(image_sizes).most_common(5),
    }

    return stats, image_sizes


def visualize_dataset(stats, total, image_size_stats, output_dir):
    """Create visualizations of the dataset statistics."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Images by crop
    crop_totals = {crop: sum(conditions.values()) for crop, conditions in stats.items()}

    # Sort crops by number of images
    sorted_crops = sorted(crop_totals.items(), key=lambda x: x[1], reverse=True)
    crops = [item[0] for item in sorted_crops]
    crop_counts = [item[1] for item in sorted_crops]

    plt.figure(figsize=(14, 8))
    plt.bar(crops, crop_counts)
    plt.xticks(rotation=45, ha="right")
    plt.title("Number of Images by Crop")
    plt.xlabel("Crop")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "images_by_crop.png"))
    plt.close()

    # 2. For each crop, create a bar chart of conditions
    for crop, conditions in stats.items():
        if not conditions:
            continue

        # Sort conditions by number of images
        sorted_conditions = sorted(conditions.items(), key=lambda x: x[1], reverse=True)
        condition_names = [item[0] for item in sorted_conditions]
        condition_counts = [item[1] for item in sorted_conditions]

        plt.figure(figsize=(12, 6))
        plt.bar(condition_names, condition_counts)
        plt.xticks(rotation=45, ha="right")
        plt.title(f"Number of Images by Condition for {crop}")
        plt.xlabel("Condition")
        plt.ylabel("Number of Images")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"images_by_condition_{crop}.png"))
        plt.close()

    # 3. Histogram of image widths and heights
    if image_size_stats[1]:  # If we have image sizes
        widths = [size[0] for size in image_size_stats[1]]
        heights = [size[1] for size in image_size_stats[1]]

        plt.figure(figsize=(10, 6))
        plt.hist(widths, bins=20, alpha=0.7, label="Width")
        plt.hist(heights, bins=20, alpha=0.7, label="Height")
        plt.legend()
        plt.title("Distribution of Image Dimensions")
        plt.xlabel("Pixels")
        plt.ylabel("Count")
        plt.savefig(os.path.join(output_dir, "image_dimensions.png"))
        plt.close()

    # 4. Create a summary text file
    with open(os.path.join(output_dir, "dataset_summary.txt"), "w") as f:
        f.write(f"CROP DISEASE DATASET SUMMARY\n")
        f.write(f"=========================\n\n")
        f.write(f"Total images: {total}\n\n")

        f.write(f"Images by crop:\n")
        for crop, count in sorted_crops:
            f.write(f"  {crop}: {count}\n")

        f.write(
            f"\nImage size statistics (from sample of {image_size_stats[0]['sample_size']} images):\n"
        )
        f.write(
            f"  Width: min={image_size_stats[0]['width']['min']}, max={image_size_stats[0]['width']['max']}, mean={image_size_stats[0]['width']['mean']:.1f}\n"
        )
        f.write(
            f"  Height: min={image_size_stats[0]['height']['min']}, max={image_size_stats[0]['height']['max']}, mean={image_size_stats[0]['height']['mean']:.1f}\n"
        )

        f.write(f"\nMost common image dimensions (width x height):\n")
        for size, count in image_size_stats[0]["most_common_sizes"]:
            f.write(f"  {size[0]} x {size[1]}: {count} images\n")

        f.write(f"\nDetailed counts by crop and condition:\n")
        for crop in crops:
            f.write(f"\n  {crop}:\n")
            conditions = stats[crop]
            for condition, count in sorted(
                conditions.items(), key=lambda x: x[1], reverse=True
            ):
                f.write(f"    {condition}: {count}\n")

    # 5. Create a JSON file with all statistics
    with open(os.path.join(output_dir, "dataset_stats.json"), "w") as f:
        json.dump(
            {
                "total_images": total,
                "crops": crop_totals,
                "detailed_stats": stats,
                "image_size_stats": {
                    "width": image_size_stats[0]["width"],
                    "height": image_size_stats[0]["height"],
                    "most_common_sizes": [
                        {"width": size[0], "height": size[1], "count": count}
                        for size, count in image_size_stats[0]["most_common_sizes"]
                    ],
                },
            },
            f,
            indent=2,
        )


def main():
    """Main function to analyze the dataset."""
    args = parse_arguments()

    print(f"Analyzing dataset at: {args.dataset}")
    print(f"Results will be saved to: {args.output}")

    # Count images by category
    print("Counting images...")
    stats, total = count_images_by_category(args.dataset)
    print(f"Found {total} images across {len(stats)} crops")

    # Get image size statistics
    print("Analyzing image sizes (sampling)...")
    image_size_stats = get_image_size_stats(args.dataset)
    print(f"Analyzed dimensions of {image_size_stats[0]['sample_size']} images")

    # Create visualizations
    print("Creating visualizations...")
    visualize_dataset(stats, total, image_size_stats, args.output)
    print("Analysis complete!")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
