# Quick Start Guide

This guide will help you quickly set up and run the crop disease dataset scraper.

## Installation

1. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Running the Script

### Option 1: Demo Run (Recommended for First-Time Users)

Run the demo script to test the scraper with a small sample:

```bash
python demo_scraper.py
```

This will scrape a small sample of images (10 per category) for Cassava and Rice crops to demonstrate functionality.

### Option 2: Full Run

To scrape the complete dataset for all crops:

```bash
python crop_disease_dataset_scraper.py
```

This will scrape images for all 15 crops with a default limit of 2000 images per category.

### Option 3: Custom Run

To customize the scraping process:

```bash
python crop_disease_dataset_scraper.py --crops=Wheat,Rice --max-images=500 --workers=8
```

## Analyzing the Dataset

After scraping, you can analyze the dataset using the analysis script:

```bash
python analyze_dataset.py --dataset=./dataset --output=./dataset_analysis
```

This will generate statistics and visualizations about the collected dataset.

## File Structure

- `crop_disease_dataset_scraper.py`: Main scraper script
- `demo_scraper.py`: Script for running a limited demo
- `analyze_dataset.py`: Script for analyzing the collected dataset
- `requirements.txt`: List of required Python packages
- `README.md`: Detailed documentation

## Common Issues

- **Search Engine Blocking**: If you experience blocking from search engines, try reducing the `--max-images` parameter or spreading your scraping over multiple days.
- **Memory Issues**: If you encounter memory problems, reduce the `--workers` parameter or process fewer crops at a time.

## Next Steps

After collecting your dataset:

1. Analyze the dataset quality with `analyze_dataset.py`
2. Consider augmenting the data if certain categories have too few images
3. Use the train/test split for training your machine learning model
