# Crop Disease Dataset Scraper

A comprehensive Python script for scraping, cleaning, and organizing crop images to train machine learning models for pest and disease detection.

## Features

- Scrapes crop images from Google and Bing image search engines
- Organizes images into a structured dataset for 15 different crops
- Categorizes images by healthy plants, diseases, and pests
- Cleans images by removing duplicates, blurry images, and low-resolution images
- Splits the dataset into training and testing sets
- Parallelized processing for faster execution
- Detailed logging and statistics

## Dataset Structure

```
dataset/
  ├── Groundnut/
  │    ├── healthy/
  │    ├── early_leaf_spot/
  │    ├── aphids/
  │    └── ...
  ├── Cassava/
  │    ├── healthy/
  │    ├── mosaic_virus/
  │    ├── mealybug/
  │    └── ...
  └── ...
```

After splitting:

```
dataset_split/
  ├── train/
  │    ├── Groundnut/
  │    │    ├── healthy/
  │    │    ├── early_leaf_spot/
  │    │    └── ...
  │    └── ...
  └── test/
       ├── Groundnut/
       │    ├── healthy/
       │    ├── early_leaf_spot/
       │    └── ...
       └── ...
```

## Included Crops

1. Groundnut (Peanut)
2. Tobacco
3. Sorghum
4. Millet
5. Cocoa
6. Cassava
7. Scotch bonnet (hot pepper)
8. Bananas / Plantain
9. Leafy greens (spinach)
10. Beans
11. Cowpea (black-eyed peas)
12. Wheat
13. Rice
14. Onion
15. Garlic

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/crop-disease-dataset-scraper.git
   cd crop-disease-dataset-scraper
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

Run the script with default settings:

```bash
python crop_disease_dataset_scraper.py
```

This will scrape all crops with a maximum of 2000 images per category.

### Advanced Usage

Customize the script execution with command-line arguments:

```bash
python crop_disease_dataset_scraper.py --crops=Wheat,Rice --max-images=500 --workers=8
```

### Available Options

- `--crops`: Comma-separated list of crops to scrape (default: all)
- `--max-images`: Maximum number of images per category (default: 2000)
- `--skip-scraping`: Skip the image scraping step
- `--skip-cleaning`: Skip the image cleaning step
- `--skip-splitting`: Skip the train/test splitting step
- `--output-dir`: Output directory for the dataset (default: ./dataset)
- `--split-dir`: Output directory for the train/test split (default: ./dataset_split)
- `--workers`: Number of worker threads for parallel processing (default: 4)

## Example

1. Scrape only Cassava and Rice crops with 1000 images per category:
   ```bash
   python crop_disease_dataset_scraper.py --crops=Cassava,Rice --max-images=1000
   ```

2. Clean and split an existing dataset (skip scraping):
   ```bash
   python crop_disease_dataset_scraper.py --skip-scraping
   ```

3. Resume a previously interrupted scraping session:
   ```bash
   python crop_disease_dataset_scraper.py
   ```
   (The script will automatically skip existing images)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

Please use this tool responsibly and respect the terms of service of search engines. Excessive scraping may result in your IP being temporarily blocked.
