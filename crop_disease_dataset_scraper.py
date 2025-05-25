#!/usr/bin/env python3
"""
Crop Disease Dataset Scraper

This script scrapes crop images from the internet, organizes them into a structured dataset,
and prepares them for training a machine learning model for pest and disease detection.

Usage:
    python crop_disease_dataset_scraper.py [options]

Options:
    --crops CROPS            Comma-separated list of crops to scrape (default: all)
    --max-images NUM        Maximum number of images per category (default: 2000)
    --skip-scraping         Skip the scraping step
    --skip-cleaning         Skip the cleaning step
    --skip-splitting        Skip the train/test splitting step
    --output-dir DIR        Output directory for the dataset (default: ./dataset)
    --split-dir DIR         Output directory for the train/test split (default: ./dataset_split)
    --workers NUM           Number of worker threads for parallel processing (default: 4)
"""

import os
import sys
import time
import random
import logging
import argparse
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
import json
import shutil
import hashlib
import io
import tempfile
from datetime import datetime
from urllib.parse import urlparse
import signal
import re
import socket
import threading
from queue import Queue
from contextlib import contextmanager

# Basic image processing
import cv2
import numpy as np
import imagehash
from PIL import Image, UnidentifiedImageError, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # Handle truncated images

# Network and web scraping libraries
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc
from fp.fp import FreeProxy

# Anti-Captcha API (might need installation: pip install anticaptchaofficial)
try:
    from anticaptchaofficial import recaptchav2
    from anticaptchaofficial import imagecaptcha
except ImportError:
    recaptchav2 = None
    imagecaptcha = None

# Image crawlers
from icrawler.builtin import GoogleImageCrawler, BingImageCrawler
import icrawler.builtin

try:
    from icrawler.builtin import YandexImageCrawler, DuckDuckGoImageCrawler
except ImportError:
    # Create custom implementations if not available
    from icrawler.builtin.google import GoogleFeeder, GoogleParser, GoogleImageCrawler

    # Create custom DuckDuckGo crawler
    class DuckDuckGoFeeder(GoogleFeeder):
        def feed(self, keyword, offset, max_num):
            for i in range(offset, offset + max_num):
                yield {"keyword": keyword, "offset": i}

    class DuckDuckGoParser(GoogleParser):
        def parse(self, response):
            content = json.loads(response.content.decode("utf-8"))
            for item in content.get("results", []):
                if "image" in item:
                    yield {"file_url": item["image"]}

    class DuckDuckGoImageCrawler(GoogleImageCrawler):
        def __init__(self, *args, **kwargs):
            super(DuckDuckGoImageCrawler, self).__init__(*args, **kwargs)
            self.feeder_cls = DuckDuckGoFeeder
            self.parser_cls = DuckDuckGoParser
            self.downloader_cls = icrawler.builtin.google.GoogleDownloader

    # Create custom Yandex crawler
    class YandexImageCrawler(GoogleImageCrawler):
        def __init__(self, *args, **kwargs):
            super(YandexImageCrawler, self).__init__(*args, **kwargs)
            # Use GoogleCrawler but with different base URL and parser


# Machine learning
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# NVMe optimization
import fcntl
import mmap

try:
    import libaio  # Not standard, may need separate installation
except ImportError:
    libaio = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("crop_scraper.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Configuration constants
CONFIG = {
    "MIN_IMAGES_PER_CATEGORY": 500,  # Target minimum number of images per category - Updated to 500
    "MAX_RETRIES": 5,  # Maximum number of retries for network operations
    "PROXY_ROTATION_FREQUENCY": 10,  # How often to rotate proxies (in requests)
    "USER_AGENT_ROTATION_FREQUENCY": 5,  # How often to rotate user agents
    "CRAWLER_PAUSE_MIN": 0.5,  # Minimum pause between requests (seconds)
    "CRAWLER_PAUSE_MAX": 2.0,  # Maximum pause between requests (seconds)
    "INITIAL_DOWNLOAD_BATCH": 300,  # Initial images to download per query - increased from 100
    "ADDITIONAL_BATCH_SIZE": 300,  # How many more images to download if below target - increased from 150
    "MAX_BATCH_ATTEMPTS": 30,  # Maximum number of attempts to reach target - increased from 20
    "CONNECT_TIMEOUT": 10,  # Connection timeout in seconds
    "READ_TIMEOUT": 30,  # Read timeout in seconds
    "IMAGE_QUALITY_THRESHOLDS": {
        "MIN_WIDTH": 224,  # Minimum image width
        "MIN_HEIGHT": 224,  # Minimum image height
        "MIN_AREA": 224 * 224,  # Minimum pixel area
        "BLUR_THRESHOLD": 100,  # Laplacian variance threshold for blur detection
        "SATURATION_THRESHOLD": 30,  # Minimum color saturation
        "MIN_FACES": 0,  # Minimum faces in image (0 = no requirement)
        "MAX_FACES": 0,  # Maximum faces in image (0 = no limit)
        "MIN_FILESIZE_KB": 10,  # Minimum file size in KB
    },
    "NVME_OPTIMIZATIONS": {
        "ENABLED": True,  # Whether to use NVMe-specific optimizations
        "DIRECT_IO": True,  # Use direct I/O
        "BLOCK_SIZE": 4096,  # I/O block size
        "QUEUE_DEPTH": 32,  # I/O queue depth for async operations
        "USE_AIO": libaio is not None,  # Use Asynchronous I/O if available
    },
    "ERROR_HANDLING": {
        "MAX_CONSECUTIVE_FAILURES": 20,  # Max number of consecutive failures before pausing
        "PAUSE_AFTER_FAILURES": 60,  # Seconds to pause after hitting failure threshold
        "BACK_OFF_BASE": 2,  # Base for exponential backoff
        "BACK_OFF_MAX": 600,  # Maximum backoff time in seconds
    },
}

# Keep track of scraping progress to allow for resuming
PROGRESS_FILE = "scraping_progress.json"


# Create a session factory with retries
def create_session():
    session = requests.Session()
    retries = Retry(
        total=CONFIG["MAX_RETRIES"],
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=20)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


@dataclass
class ScraperState:
    """Class to track scraper state for resumability."""

    completed_categories: Dict[str, Dict[str, int]] = field(default_factory=dict)
    failed_queries: List[Dict[str, Any]] = field(default_factory=list)
    current_crop: str = ""
    current_category: str = ""
    current_condition: str = ""
    start_time: float = field(default_factory=time.time)
    images_scraped: int = 0
    total_target: int = 0

    def save(self, filename: str = PROGRESS_FILE):
        """Save the current state to a file."""
        with open(filename, "w") as f:
            json.dump(
                {
                    "completed_categories": self.completed_categories,
                    "failed_queries": self.failed_queries,
                    "current_crop": self.current_crop,
                    "current_category": self.current_category,
                    "current_condition": self.current_condition,
                    "start_time": self.start_time,
                    "images_scraped": self.images_scraped,
                    "total_target": self.total_target,
                    "timestamp": time.time(),
                },
                f,
            )

    @classmethod
    def load(cls, filename: str = PROGRESS_FILE) -> "ScraperState":
        """Load the state from a file."""
        if not os.path.exists(filename):
            return cls()

        try:
            with open(filename, "r") as f:
                data = json.load(f)
                state = cls()
                state.completed_categories = data.get("completed_categories", {})
                state.failed_queries = data.get("failed_queries", [])
                state.current_crop = data.get("current_crop", "")
                state.current_category = data.get("current_category", "")
                state.current_condition = data.get("current_condition", "")
                state.start_time = data.get("start_time", time.time())
                state.images_scraped = data.get("images_scraped", 0)
                state.total_target = data.get("total_target", 0)
                return state
        except Exception as e:
            logger.error(f"Error loading progress file: {e}")
            return cls()


class ProxyManager:
    """Manages a pool of proxies for rotating requests."""

    def __init__(self, use_proxies: bool = True):
        self.use_proxies = use_proxies
        self.current_proxy = None
        self.proxies = []
        self.request_count = 0
        self.lock = threading.Lock()

        if self.use_proxies:
            try:
                logger.info("Initializing proxy manager...")
                self.refresh_proxy_list()
                logger.info(f"Initial proxy: {self.current_proxy}")
            except Exception as e:
                logger.error(f"Failed to initialize proxy manager: {e}")
                self.use_proxies = False

    def refresh_proxy_list(self):
        """Refresh the list of proxies."""
        try:
            # Get a list of proxies to use
            for _ in range(10):  # Try to get 10 proxies
                try:
                    proxy = FreeProxy(
                        country_id=["US", "CA", "GB", "DE", "FR"], timeout=1, rand=True
                    ).get()
                    if proxy and proxy not in self.proxies:
                        self.proxies.append(proxy)
                except Exception:
                    pass

            if not self.proxies:
                # Fallback to any country if we couldn't get specific ones
                for _ in range(10):
                    try:
                        proxy = FreeProxy(timeout=1, rand=True).get()
                        if proxy and proxy not in self.proxies:
                            self.proxies.append(proxy)
                    except Exception:
                        pass

            if self.proxies:
                self.current_proxy = random.choice(self.proxies)
                logger.info(f"Refreshed proxy list. Got {len(self.proxies)} proxies.")
            else:
                logger.warning("Could not obtain any proxies.")
                self.use_proxies = False
        except Exception as e:
            logger.error(f"Error refreshing proxy list: {e}")
            self.use_proxies = False

    def get_proxy(self) -> Optional[str]:
        """Get the current proxy configuration."""
        if not self.use_proxies or not self.proxies:
            return None

        with self.lock:
            self.request_count += 1

            # Rotate proxy if needed
            if self.request_count >= CONFIG["PROXY_ROTATION_FREQUENCY"]:
                try:
                    if len(self.proxies) > 1:
                        # Choose a different proxy than the current one
                        new_proxy = random.choice(
                            [p for p in self.proxies if p != self.current_proxy]
                        )
                        self.current_proxy = new_proxy
                    else:
                        self.current_proxy = self.proxies[0]
                    logger.debug(f"Rotated to new proxy: {self.current_proxy}")
                except Exception as e:
                    logger.warning(f"Failed to rotate proxy: {e}")
                self.request_count = 0

            return self.current_proxy

    def get_proxy_dict(self) -> Optional[Dict[str, str]]:
        """Get proxies dict for requests."""
        proxy = self.get_proxy()
        if not proxy:
            return None

        # Ensure proxy has proper prefix
        if not proxy.startswith("http"):
            proxy = f"http://{proxy}"

        return {"http": proxy, "https": proxy}

    def mark_proxy_bad(self):
        """Mark the current proxy as bad."""
        if (
            self.use_proxies
            and self.current_proxy
            and self.current_proxy in self.proxies
        ):
            logger.info(f"Marking proxy as bad: {self.current_proxy}")
            try:
                self.proxies.remove(self.current_proxy)
                if not self.proxies:
                    # Get new proxies if we've run out
                    self.refresh_proxy_list()
                else:
                    self.current_proxy = random.choice(self.proxies)
            except Exception as e:
                logger.error(f"Error removing bad proxy: {e}")
                # Try to refresh proxy list
                self.refresh_proxy_list()


class UserAgentRotator:
    """Rotates user agents to avoid detection."""

    def __init__(self):
        self.ua = UserAgent(browsers=["chrome", "firefox", "edge", "safari"])
        self.request_count = 0
        self.current_ua = self.ua.random
        self.lock = threading.Lock()

    def get_random_user_agent(self) -> str:
        """Get a random user agent."""
        with self.lock:
            self.request_count += 1

            # Rotate user agent if needed
            if self.request_count >= CONFIG["USER_AGENT_ROTATION_FREQUENCY"]:
                self.current_ua = self.ua.random
                self.request_count = 0

            return self.current_ua


class CaptchaSolver:
    """Solves CAPTCHAs automatically using anti-captcha service."""

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.enabled = bool(self.api_key)

    def solve_recaptcha(self, site_key: str, page_url: str) -> Optional[str]:
        """Solve a reCAPTCHA."""
        if not self.enabled:
            return None

        try:
            solver = recaptchav2.recaptchaV2Proxyless()
            solver.set_verbose(1)
            solver.set_key(self.api_key)
            solver.set_website_url(page_url)
            solver.set_website_key(site_key)

            g_response = solver.solve_and_return_solution()
            if g_response != 0:
                logger.info(f"Solved reCAPTCHA successfully")
                return g_response
            else:
                logger.error(f"Failed to solve reCAPTCHA: {solver.error_code}")
                return None
        except Exception as e:
            logger.error(f"Error solving reCAPTCHA: {e}")
            return None

    def solve_image_captcha(self, image_data: bytes) -> Optional[str]:
        """Solve an image-based CAPTCHA."""
        if not self.enabled:
            return None

        try:
            solver = imagecaptcha.imagecaptcha()
            solver.set_verbose(1)
            solver.set_key(self.api_key)

            # Save image to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as temp:
                temp.write(image_data)
                temp_path = temp.name

            result = solver.solve_and_return_solution(temp_path)
            os.unlink(temp_path)  # Clean up

            if result != 0:
                logger.info(f"Solved image CAPTCHA successfully")
                return result
            else:
                logger.error(f"Failed to solve image CAPTCHA: {solver.error_code}")
                return None
        except Exception as e:
            logger.error(f"Error solving image CAPTCHA: {e}")
            return None


@contextmanager
def nvme_optimized_io(file_path: str, mode: str = "rb"):
    """Context manager for NVMe-optimized I/O operations."""
    if not CONFIG["NVME_OPTIMIZATIONS"]["ENABLED"]:
        # Fall back to regular file operations
        with open(file_path, mode) as f:
            yield f
        return

    try:
        flags = os.O_RDONLY if "r" in mode else os.O_WRONLY
        if "b" not in mode:
            flags |= os.O_TEXT
        if "+" in mode:
            flags = os.O_RDWR
        if "a" in mode:
            flags |= os.O_APPEND
        if CONFIG["NVME_OPTIMIZATIONS"]["DIRECT_IO"]:
            flags |= os.O_DIRECT

        # Align buffer to block size
        block_size = CONFIG["NVME_OPTIMIZATIONS"]["BLOCK_SIZE"]

        fd = os.open(file_path, flags)
        try:
            # Use memory mapping for better performance with NVMe
            if "r" in mode:
                # For reading, memory map the file
                file_size = os.path.getsize(file_path)
                if file_size > 0:  # Can't mmap empty files
                    mapped = mmap.mmap(fd, file_size, mmap.MAP_SHARED, mmap.PROT_READ)
                    try:
                        yield mapped
                    finally:
                        mapped.close()
                else:
                    # Handle empty files
                    yield io.BytesIO(b"")
            else:
                # For writing, use regular file object but with optimized settings
                f = os.fdopen(fd, mode)
                # Set buffer size to a multiple of block size
                f = io.BufferedWriter(f, buffer_size=block_size * 16)
                yield f
        finally:
            if not isinstance(fd, io.IOBase):  # If we didn't convert to a file object
                os.close(fd)
    except Exception as e:
        logger.warning(f"NVMe optimization failed, falling back to standard I/O: {e}")
        with open(file_path, mode) as f:
            yield f


def detect_image_quality_issues(
    img: np.ndarray, quality_threshold: str = "medium"
) -> Tuple[bool, Dict[str, Any]]:
    """
    Detect various quality issues in an image.

    Args:
        img: OpenCV image
        quality_threshold: Quality threshold level

    Returns:
        (is_good, issues_dict)
    """
    # Set thresholds based on quality level
    thresholds = {
        "low": {
            "blur": 50,
            "saturation": 15,
            "min_area": 32 * 32,
            "min_size": (32, 32),
        },
        "medium": CONFIG["IMAGE_QUALITY_THRESHOLDS"].copy(),
        "high": {
            "blur": 150,
            "saturation": 40,
            "min_area": 300 * 300,
            "min_size": (300, 300),
        },
        "strict": {
            "blur": 200,
            "saturation": 50,
            "min_area": 512 * 512,
            "min_size": (384, 384),
        },
    }

    t = thresholds.get(quality_threshold, thresholds["medium"])

    issues = {}

    # Check resolution
    h, w = img.shape[:2]
    issues["resolution"] = {
        "width": w,
        "height": h,
        "ok": w >= t["min_size"][0] and h >= t["min_size"][1],
    }

    if not issues["resolution"]["ok"]:
        return False, issues

    # Check blur
    if len(img.shape) == 3 and img.shape[2] >= 3:  # Color image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    issues["blur"] = {"value": laplacian_var, "ok": laplacian_var >= t["blur"]}

    # Check saturation for color images
    if len(img.shape) == 3 and img.shape[2] >= 3:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].mean()
        issues["saturation"] = {
            "value": saturation,
            "ok": saturation >= t["saturation"],
        }
    else:
        issues["saturation"] = {"value": 0, "ok": True}

    # Overall result - an image passes if it passes all checks
    is_good = all(issue.get("ok", False) for issue in issues.values())

    return is_good, issues


# User agent rotator
ua = UserAgent(browsers=["chrome", "firefox", "edge", "safari"])

# Define crop-specific pests and diseases
CROPS_DATA = {
    "Tobacco": {
        "pests": ["hornworm", "aphids", "whiteflies", "thrips"],
        "diseases": [
            "tobacco_mosaic_virus",
            "black_shank",
            "blue_mold",
            "root_knot_nematodes",
        ],
    },
    "Millet": {
        "pests": ["stem_borers", "shoot_flies"],
        "diseases": ["blast", "smut", "downy_mildew", "ergot"],
    },
    "Scotch_bonnet": {
        "pests": ["aphids", "whiteflies", "thrips", "fruit_borers"],
        "diseases": [
            "powdery_mildew",
            "bacterial_leaf_spot",
            "mosaic_virus",
            "anthracnose",
        ],
    },
    "Cowpea": {
        "pests": ["maruca_pod_borer", "aphids", "thrips"],
        "diseases": ["mosaic_virus", "cercospora_leaf_spot", "fusarium_wilt"],
    },
    "Onion": {
        "pests": ["onion_thrips", "onion_maggot"],
        "diseases": ["purple_blotch", "downy_mildew", "fusarium_basal_rot"],
    },
    "Garlic": {
        "pests": ["onion_thrips", "nematodes"],
        "diseases": ["white_rot", "downy_mildew", "fusarium_basal_rot"],
    },
}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Crop Disease Dataset Scraper")
    parser.add_argument(
        "--crops",
        type=str,
        default="all",
        help="Comma-separated list of crops to scrape (default: all)",
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=CONFIG["MIN_IMAGES_PER_CATEGORY"],
        help=f"Minimum number of images per category (default: {CONFIG['MIN_IMAGES_PER_CATEGORY']})",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=2500,
        help="Maximum number of images per category (default: 2500)",
    )
    parser.add_argument(
        "--skip-scraping", action="store_true", help="Skip the scraping step"
    )
    parser.add_argument(
        "--skip-cleaning", action="store_true", help="Skip the cleaning step"
    )
    parser.add_argument(
        "--skip-splitting",
        action="store_true",
        help="Skip the train/test splitting step",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dataset",
        help="Output directory for the dataset (default: ./dataset)",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="./dataset_split",
        help="Output directory for the train/test split (default: ./dataset_split)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(4, os.cpu_count() or 4),
        help=f"Number of worker threads for parallel processing (default: {max(4, os.cpu_count() or 4)})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume previous scraping session if available",
    )
    parser.add_argument(
        "--use-proxies",
        action="store_true",
        help="Use rotating proxies to avoid IP blocks",
    )
    parser.add_argument(
        "--search-engines",
        type=str,
        default="all",
        help="Comma-separated list of search engines to use (google,bing,duckduckgo,yandex) or 'all'",
    )
    parser.add_argument(
        "--selenium",
        action="store_true",
        help="Use Selenium for advanced web scraping where API methods fail",
    )
    parser.add_argument(
        "--nvme-optimize",
        action="store_true",
        help="Enable NVMe-specific I/O optimizations for faster processing",
    )
    parser.add_argument(
        "--auto-solve-captcha",
        action="store_true",
        help="Use anti-captcha service to automatically solve CAPTCHAs",
    )
    parser.add_argument(
        "--anticaptcha-key",
        type=str,
        default="",
        help="API key for anti-captcha service",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed scraping mode (requires additional configuration)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=str,
        default="medium",
        choices=["low", "medium", "high", "strict"],
        help="Quality threshold for image filtering (default: medium)",
    )
    parser.add_argument(
        "--safe-search",
        action="store_true",
        help="Enable safe search filtering to avoid inappropriate content",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default="",
        help="Path to external config file with advanced settings",
    )
    return parser.parse_args()


def get_search_queries(crop: str, category: str, condition: str) -> List[str]:
    """Generate search queries for image scraping.

    Args:
        crop: Crop name
        category: 'pests' or 'diseases' or 'healthy'
        condition: Specific pest or disease name or 'healthy'

    Returns:
        List of search queries
    """
    queries = []
    common_display_name = crop.replace("_", " ")
    scientific_names = {
        # Scientific names for common crops to improve search results
        "Tobacco": "Nicotiana tabacum",
        "Millet": "Pennisetum glaucum",
        "Scotch_bonnet": "Capsicum chinense",
        "Cowpea": "Vigna unguiculata",
        "Onion": "Allium cepa",
        "Garlic": "Allium sativum",
    }

    # Scientific names of pests and diseases where applicable
    scientific_conditions = {
        # Just examples - would need to be expanded
        "aphids": "Aphidoidea",
        "thrips": "Thysanoptera",
        "rust": "Puccinia",
        "powdery_mildew": "Erysiphales",
        "mosaic_virus": "Potyvirus",
    }

    scientific_crop_name = scientific_names.get(crop, "")
    scientific_condition_name = scientific_conditions.get(condition, "")

    if condition == "healthy":
        general_queries = [
            f"{common_display_name} healthy plant",
            f"{common_display_name} healthy crop",
            f"{common_display_name} healthy leaf",
            f"healthy {common_display_name} plant",
            f"healthy {common_display_name} field",
            f"{common_display_name} healthy foliage",
            f"healthy {common_display_name} cultivation",
            f"{common_display_name} farm healthy plants",
        ]

        # Add scientific name queries if available
        if scientific_crop_name:
            scientific_queries = [
                f"{scientific_crop_name} healthy plant",
                f"{scientific_crop_name} healthy leaf",
                f"healthy {scientific_crop_name}",
            ]
            queries = general_queries + scientific_queries
        else:
            queries = general_queries

    elif category == "pests":
        pest_name = condition.replace("_", " ")
        general_queries = [
            f"{common_display_name} {pest_name} damage",
            f"{common_display_name} {pest_name} infestation",
            f"{pest_name} on {common_display_name}",
            f"{common_display_name} with {pest_name}",
            f"{pest_name} damage to {common_display_name}",
            f"{common_display_name} {pest_name} symptoms",
            f"{common_display_name} plant damaged by {pest_name}",
            f"{common_display_name} leaves with {pest_name} damage",
            f"{pest_name} affecting {common_display_name}",
        ]

        # Add scientific name queries
        if scientific_crop_name or scientific_condition_name:
            scientific_queries = []

            if scientific_crop_name and scientific_condition_name:
                scientific_queries.extend(
                    [
                        f"{scientific_crop_name} {scientific_condition_name}",
                        f"{scientific_condition_name} on {scientific_crop_name}",
                    ]
                )

            if scientific_crop_name:
                scientific_queries.extend(
                    [
                        f"{scientific_crop_name} {pest_name} damage",
                        f"{pest_name} on {scientific_crop_name}",
                    ]
                )

            if scientific_condition_name:
                scientific_queries.extend(
                    [
                        f"{common_display_name} {scientific_condition_name}",
                        f"{scientific_condition_name} {common_display_name} damage",
                    ]
                )

            queries = general_queries + scientific_queries
        else:
            queries = general_queries

    elif category == "diseases":
        disease_name = condition.replace("_", " ")
        general_queries = [
            f"{common_display_name} {disease_name}",
            f"{common_display_name} {disease_name} symptoms",
            f"{common_display_name} leaves with {disease_name}",
            f"{disease_name} on {common_display_name}",
            f"{common_display_name} affected by {disease_name}",
            f"{disease_name} disease {common_display_name}",
            f"{common_display_name} plant {disease_name} infection",
            f"{common_display_name} {disease_name} lesions",
            f"{disease_name} symptoms on {common_display_name}",
            f"{common_display_name} crop {disease_name}",
        ]

        # Add scientific name queries
        if scientific_crop_name or scientific_condition_name:
            scientific_queries = []

            if scientific_crop_name and scientific_condition_name:
                scientific_queries.extend(
                    [
                        f"{scientific_crop_name} {scientific_condition_name}",
                        f"{scientific_condition_name} on {scientific_crop_name}",
                    ]
                )

            if scientific_crop_name:
                scientific_queries.extend(
                    [
                        f"{scientific_crop_name} {disease_name}",
                        f"{disease_name} on {scientific_crop_name}",
                    ]
                )

            if scientific_condition_name:
                scientific_queries.extend(
                    [
                        f"{common_display_name} {scientific_condition_name}",
                        f"{scientific_condition_name} {common_display_name}",
                    ]
                )

            queries = general_queries + scientific_queries
        else:
            queries = general_queries

    # Include agricultural extension specific searches to find better quality examples
    agricultural_terms = [
        "agricultural extension",
        "plant pathology",
        "field guide",
        "diagnostic",
        "agricultural research",
    ]

    # Add some agricultural extension specific searches
    for term in agricultural_terms[:2]:  # Just use a couple to avoid too many queries
        if condition == "healthy":
            queries.append(f"{term} {common_display_name} healthy")
        else:
            condition_name = condition.replace("_", " ")
            queries.append(f"{term} {common_display_name} {condition_name}")

    # Add image quality indicators to some queries
    quality_terms = ["high resolution", "clear image", "closeup"]
    for term in quality_terms[:1]:  # Just add to one query
        if queries:
            queries.append(f"{queries[0]} {term}")

    return queries


def crawl_images(
    crop: str,
    category: str,
    condition: str,
    output_dir: str,
    min_images: int,
    max_images: int,
    scraper_state: ScraperState,
    search_engines: List[str] = None,
    use_proxies: bool = False,
    use_selenium: bool = False,
    captcha_solver: Optional[CaptchaSolver] = None,
    quality_threshold: str = "medium",
    safe_search: bool = True,
) -> Tuple[str, int]:
    """Crawl images from search engines with sophisticated error handling and retry logic.

    Args:
        crop: Crop name
        category: 'pests' or 'diseases' or 'healthy'
        condition: Specific pest or disease name or 'healthy'
        output_dir: Output directory
        min_images: Minimum number of images to collect
        max_images: Maximum number of images to download
        scraper_state: State tracker for resumability
        search_engines: List of search engines to use
        use_proxies: Whether to use proxy rotation
        use_selenium: Whether to use Selenium for advanced scraping
        captcha_solver: CaptchaSolver instance for handling CAPTCHAs
        quality_threshold: Quality threshold for filtering
        safe_search: Whether to use safe search

    Returns:
        (Path to the directory with downloaded images, number of images downloaded)
    """
    target_dir = os.path.join(output_dir, crop, condition)
    os.makedirs(target_dir, exist_ok=True)

    # Initialize tracking vars
    images_before = count_images_in_directory(target_dir)

    # Check if we already have enough images
    if images_before >= min_images:
        logger.info(
            f"Already have {images_before} images for {crop}/{condition} (minimum: {min_images}). Skipping."
        )
        return target_dir, images_before

    # Initialize proxy manager
    proxy_manager = ProxyManager(use_proxies)

    # Initialize user agent rotator
    ua_rotator = UserAgentRotator()

    # Get search queries
    queries = get_search_queries(crop, category, condition)

    # Set up search engines
    if search_engines is None or "all" in search_engines:
        engines = ["google", "bing", "duckduckgo", "yandex"]
    else:
        engines = search_engines

    logger.info(f"Scraping images for {crop} - {condition}")
    logger.info(f"Target: minimum {min_images}, maximum {max_images} images")
    logger.info(f"Using search engines: {', '.join(engines)}")

    # Calculate how many images we need per engine and query
    still_needed = min_images - images_before
    total_combinations = len(queries) * len(engines)
    # Request more images per combination to ensure we get enough after filtering
    base_images_per_combination = max(
        200, still_needed // max(1, (total_combinations // 2))
    )

    # Track failures for backoff
    consecutive_failures = 0
    backoff_time = 1  # Initial backoff time in seconds

    # Create a list to track progress for each query/engine combination
    combinations = []
    for query in queries:
        for engine in engines:
            combinations.append((query, engine))

    # Shuffle to vary the order
    random.shuffle(combinations)

    for query, engine in combinations:
        # Save state regularly
        scraper_state.current_crop = crop
        scraper_state.current_category = category
        scraper_state.current_condition = condition
        scraper_state.save()

        # Check if we've reached our minimum target
        current_count = count_images_in_directory(target_dir)
        if current_count >= min_images:
            logger.info(
                f"Reached minimum target of {min_images} images for {crop}/{condition}"
            )
            break

        # Calculate how many more images we still need
        still_needed = min_images - current_count
        # Request significantly more images than needed to account for filtering and failures
        images_per_combination = min(
            max_images - current_count,
            max(200, still_needed * 3 // (len(combinations))),
        )

        logger.info(
            f"Query: '{query}', Engine: {engine}, Target: {images_per_combination} images"
        )

        # Select the appropriate crawler based on the engine
        try:
            crawler = None

            # Common crawler settings
            storage = {"root_dir": target_dir}
            downloader_threads = min(8, os.cpu_count() or 4)  # Limit based on CPU count
            parser_threads = min(8, os.cpu_count() or 4)

            # Get a fresh user agent
            user_agent = ua_rotator.get_random_user_agent()

            # Get proxy if enabled
            proxies = proxy_manager.get_proxy_dict() if use_proxies else None

            if engine == "google":
                crawler = GoogleImageCrawler(
                    storage=storage,
                    downloader_threads=downloader_threads,
                    parser_threads=parser_threads,
                )
            elif engine == "bing":
                crawler = BingImageCrawler(
                    storage=storage,
                    downloader_threads=downloader_threads,
                    parser_threads=parser_threads,
                )
            elif engine == "duckduckgo":
                try:
                    crawler = DuckDuckGoImageCrawler(
                        storage=storage,
                        downloader_threads=downloader_threads,
                        parser_threads=parser_threads,
                    )
                except Exception as e:
                    logger.warning(
                        f"DuckDuckGo crawler initialization failed: {e}. Using Google as fallback."
                    )
                    crawler = GoogleImageCrawler(
                        storage=storage,
                        downloader_threads=downloader_threads,
                        parser_threads=parser_threads,
                    )
            elif engine == "yandex":
                try:
                    crawler = YandexImageCrawler(
                        storage=storage,
                        downloader_threads=downloader_threads,
                        parser_threads=parser_threads,
                    )
                except Exception as e:
                    logger.warning(
                        f"Yandex crawler initialization failed: {e}. Using Bing as fallback."
                    )
                    crawler = BingImageCrawler(
                        storage=storage,
                        downloader_threads=downloader_threads,
                        parser_threads=parser_threads,
                    )

            if crawler is None:
                logger.error(f"Failed to initialize crawler for engine: {engine}")
                continue

            # Configure the crawler
            crawler.downloader.session.headers.update({"User-Agent": user_agent})

            if proxies:
                crawler.downloader.session.proxies.update(proxies)

            # Additional parameters
            extra_params = {}
            if safe_search:
                extra_params["safe_search"] = True

            # Try to crawl with retry logic
            for attempt in range(1, CONFIG["MAX_RETRIES"] + 1):
                try:
                    crawler.crawl(
                        keyword=query,
                        max_num=images_per_combination,
                        min_size=(224, 224),
                        max_size=None,  # No upper limit
                        file_idx_offset=0,  # Let the crawler handle naming
                        filters=extra_params,
                    )

                    # Successful crawl
                    consecutive_failures = 0
                    backoff_time = 1  # Reset backoff time

                    # Random delay before next query
                    delay = random.uniform(
                        CONFIG["CRAWLER_PAUSE_MIN"], CONFIG["CRAWLER_PAUSE_MAX"]
                    )
                    time.sleep(delay)

                    break  # Exit retry loop on success

                except Exception as e:
                    consecutive_failures += 1
                    logger.warning(
                        f"{engine.title()} crawling attempt {attempt} failed for {query}: {e}"
                    )

                    # Handle different failure scenarios
                    if "429" in str(e) or "too many requests" in str(e).lower():
                        # Rate limiting - mark proxy as bad and rotate
                        if use_proxies:
                            proxy_manager.mark_proxy_bad()

                        # Calculate backoff time with exponential increase
                        backoff_time = min(
                            CONFIG["ERROR_HANDLING"]["BACK_OFF_MAX"],
                            backoff_time * CONFIG["ERROR_HANDLING"]["BACK_OFF_BASE"],
                        )
                        logger.info(
                            f"Rate limited. Backing off for {backoff_time} seconds..."
                        )
                        time.sleep(backoff_time)

                    elif (
                        "captcha" in str(e).lower()
                        and captcha_solver
                        and captcha_solver.enabled
                    ):
                        # Try to solve CAPTCHA if we have a solver
                        logger.info("CAPTCHA detected, attempting to solve...")
                        # This would require custom handling based on the specific site
                        # Placeholder for CAPTCHA handling logic
                        time.sleep(backoff_time)

                    else:
                        # General failure - shorter backoff
                        time.sleep(min(30, backoff_time))

                    # If we've tried all attempts, log failure for this combination
                    if attempt == CONFIG["MAX_RETRIES"]:
                        scraper_state.failed_queries.append(
                            {
                                "crop": crop,
                                "category": category,
                                "condition": condition,
                                "query": query,
                                "engine": engine,
                                "error": str(e),
                                "timestamp": time.time(),
                            }
                        )
                        scraper_state.save()

            # Check if we need a longer pause due to consecutive failures
            if (
                consecutive_failures
                >= CONFIG["ERROR_HANDLING"]["MAX_CONSECUTIVE_FAILURES"]
            ):
                pause_time = CONFIG["ERROR_HANDLING"]["PAUSE_AFTER_FAILURES"]
                logger.warning(
                    f"Too many consecutive failures ({consecutive_failures}). Pausing for {pause_time} seconds..."
                )
                time.sleep(pause_time)
                consecutive_failures = 0

        except Exception as e:
            logger.error(f"Unexpected error with {engine} crawler for {query}: {e}")
            continue

    # Try to use Selenium as a last resort if we still don't have enough images
    if use_selenium and count_images_in_directory(target_dir) < min_images:
        try:
            logger.info("Using Selenium for additional image collection...")
            selenium_scrape_images(
                crop,
                category,
                condition,
                target_dir,
                min_images - count_images_in_directory(target_dir),
                proxy_manager if use_proxies else None,
                ua_rotator,
                captcha_solver,
            )
        except Exception as e:
            logger.error(f"Selenium scraping failed: {e}")

    # Count final number of images
    images_after = count_images_in_directory(target_dir)
    images_added = images_after - images_before

    logger.info(
        f"Added {images_added} images for {crop}/{condition}. Total: {images_after}"
    )

    # Update state
    scraper_state.images_scraped += images_added
    if crop not in scraper_state.completed_categories:
        scraper_state.completed_categories[crop] = {}
    scraper_state.completed_categories[crop][condition] = images_after
    scraper_state.save()

    return target_dir, images_after


def selenium_scrape_images(
    crop: str,
    category: str,
    condition: str,
    target_dir: str,
    target_count: int,
    proxy_manager: Optional[ProxyManager] = None,
    ua_rotator: Optional[UserAgentRotator] = None,
    captcha_solver: Optional[CaptchaSolver] = None,
) -> int:
    """
    Use Selenium to scrape images when API-based methods aren't sufficient.

    Args:
        crop: Crop name
        category: Category ('pests', 'diseases', or 'healthy')
        condition: Specific condition
        target_dir: Target directory
        target_count: Number of images to try to collect
        proxy_manager: Optional proxy manager
        ua_rotator: User agent rotator
        captcha_solver: CAPTCHA solver

    Returns:
        Number of images downloaded
    """
    if target_count <= 0:
        return 0

    # Generate search queries
    queries = get_search_queries(crop, category, condition)

    # Limit to a few queries to not spend too much time
    queries = queries[: min(5, len(queries))]

    images_before = count_images_in_directory(target_dir)
    images_downloaded = 0

    try:
        # Configure Chrome options
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")

        # Use user agent if available
        if ua_rotator:
            options.add_argument(f"user-agent={ua_rotator.get_random_user_agent()}")

        # Create a Chrome driver (undetected version to avoid bot detection)
        driver = uc.Chrome(options=options)

        for query in queries:
            if images_downloaded >= target_count:
                break

            try:
                # Google Images search
                encoded_query = requests.utils.quote(query)
                driver.get(f"https://www.google.com/search?q={encoded_query}&tbm=isch")

                # Wait for images to load
                time.sleep(3)

                # Scroll to load more images - increased from 5 to 15 scrolls
                for _ in range(15):  # Scroll more to load more images
                    driver.execute_script(
                        "window.scrollTo(0, document.body.scrollHeight);"
                    )
                    time.sleep(1.5)  # Wait a bit longer for images to load

                # Find image elements
                img_elements = driver.find_elements_by_css_selector("img.rg_i")

                # Download images
                for i, img in enumerate(img_elements):
                    if images_downloaded >= target_count:
                        break

                    try:
                        # Try to get the src URL
                        img.click()
                        time.sleep(1)

                        # Get the larger image URL
                        actual_image = driver.find_elements_by_css_selector(
                            "img.n3VNCb"
                        )
                        if actual_image:
                            img_url = actual_image[0].get_attribute("src")
                            if img_url and img_url.startswith("http"):
                                # Download the image
                                response = requests.get(
                                    img_url, stream=True, timeout=10
                                )
                                if response.status_code == 200:
                                    img_path = os.path.join(
                                        target_dir,
                                        f"selenium_{int(time.time())}_{i}.jpg",
                                    )
                                    with open(img_path, "wb") as f:
                                        for chunk in response.iter_content(1024):
                                            f.write(chunk)
                                    images_downloaded += 1
                    except Exception as e:
                        logger.debug(f"Error downloading image via Selenium: {e}")
                        continue

            except Exception as e:
                logger.warning(f"Error with Selenium for query {query}: {e}")
                continue

        driver.quit()

    except Exception as e:
        logger.error(f"Failed to initialize Selenium: {e}")

    images_after = count_images_in_directory(target_dir)
    return images_after - images_before


def count_images_in_directory(directory: str) -> int:
    """Count the number of images in a directory."""
    if not os.path.exists(directory):
        return 0

    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    count = 0

    for filename in os.listdir(directory):
        if os.path.splitext(filename.lower())[1] in image_extensions:
            # Check if it's a valid file and not zero byte
            file_path = os.path.join(directory, filename)
            if os.path.isfile(file_path) and os.path.getsize(file_path) > 0:
                count += 1

    return count


def get_image_quality_metrics(image_path: str) -> Dict[str, Any]:
    """Get comprehensive quality metrics for an image.

    Args:
        image_path: Path to the image

    Returns:
        Dictionary of quality metrics
    """
    metrics = {
        "valid": False,
        "width": 0,
        "height": 0,
        "aspect_ratio": 0,
        "blur_score": 0,
        "brightness": 0,
        "saturation": 0,
        "file_size": 0,
        "format": None,
        "mode": None,
        "perceptual_hash": None,
        "color_hash": None,
        "diff_hash": None,
        "wavelet_hash": None,
    }

    try:
        # Get file size
        metrics["file_size"] = os.path.getsize(image_path) / 1024  # KB

        # Open with PIL first for basic info
        pil_img = Image.open(image_path)
        metrics["width"], metrics["height"] = pil_img.size
        metrics["aspect_ratio"] = (
            metrics["width"] / metrics["height"] if metrics["height"] > 0 else 0
        )
        metrics["format"] = pil_img.format
        metrics["mode"] = pil_img.mode

        # Compute different perceptual hashes for better duplicate detection
        metrics["perceptual_hash"] = str(imagehash.phash(pil_img))
        metrics["color_hash"] = str(imagehash.colorhash(pil_img))
        metrics["diff_hash"] = str(imagehash.dhash(pil_img))
        metrics["wavelet_hash"] = str(imagehash.whash(pil_img))

        # Convert for OpenCV processing
        img = cv2.imread(image_path)
        if img is not None:
            # Blur detection using Laplacian variance
            if len(img.shape) == 3 and img.shape[2] >= 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            metrics["blur_score"] = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Calculate brightness and saturation
            if len(img.shape) == 3 and img.shape[2] >= 3:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                metrics["brightness"] = hsv[:, :, 2].mean()  # V channel
                metrics["saturation"] = hsv[:, :, 1].mean()  # S channel
            else:
                metrics["brightness"] = gray.mean()
                metrics["saturation"] = 0  # Grayscale image has no saturation

        metrics["valid"] = True

    except Exception as e:
        logger.debug(f"Error analyzing image quality for {image_path}: {e}")

    return metrics


def is_blurry(image_path: str, threshold: int = 100) -> bool:
    """Check if an image is blurry using the Laplacian variance method.

    Args:
        image_path: Path to the image
        threshold: Threshold for the Laplacian variance

    Returns:
        True if the image is blurry, False otherwise
    """
    try:
        metrics = get_image_quality_metrics(image_path)
        return metrics["blur_score"] < threshold
    except Exception:
        return True  # Consider it blurry if there's an error


def has_minimum_resolution(
    image_path: str, min_width: int = 224, min_height: int = 224
) -> bool:
    """Check if an image meets the minimum resolution requirements.

    Args:
        image_path: Path to the image
        min_width: Minimum width
        min_height: Minimum height

    Returns:
        True if the image meets the requirements, False otherwise
    """
    try:
        metrics = get_image_quality_metrics(image_path)
        return metrics["width"] >= min_width and metrics["height"] >= min_height
    except Exception:
        return False  # Consider it invalid if there's an error


def get_image_hash(image_path: str) -> Optional[str]:
    """Get multiple perceptual hashes of an image and combine them.

    Args:
        image_path: Path to the image

    Returns:
        Combined hash string, or None if the image can't be read
    """
    try:
        metrics = get_image_quality_metrics(image_path)

        # Return the perceptual hash
        hashes = [
            metrics["perceptual_hash"],
            metrics["color_hash"],
            metrics["diff_hash"],
            metrics["wavelet_hash"],
        ]

        # Return a single hash that includes all methods for better comparison
        if all(hashes):
            return "|".join(hashes)
        elif metrics["perceptual_hash"]:
            return metrics["perceptual_hash"]
        return None
    except Exception:
        return None


def is_low_quality(image_path: str, quality_threshold: str = "medium") -> bool:
    """Check if an image is low quality based on multiple criteria.

    Args:
        image_path: Path to the image
        quality_threshold: Quality threshold level

    Returns:
        True if the image is low quality, False otherwise
    """
    # Define threshold levels
    thresholds = {
        "low": {
            "min_width": 128,
            "min_height": 128,
            "min_file_size": 5,  # KB
            "min_blur_score": 50,
            "min_saturation": 15,
            "min_brightness": 20,
            "max_brightness": 235,
        },
        "medium": {
            "min_width": 224,
            "min_height": 224,
            "min_file_size": 10,  # KB
            "min_blur_score": 100,
            "min_saturation": 30,
            "min_brightness": 30,
            "max_brightness": 225,
        },
        "high": {
            "min_width": 300,
            "min_height": 300,
            "min_file_size": 20,  # KB
            "min_blur_score": 150,
            "min_saturation": 40,
            "min_brightness": 40,
            "max_brightness": 215,
        },
        "strict": {
            "min_width": 384,
            "min_height": 384,
            "min_file_size": 30,  # KB
            "min_blur_score": 200,
            "min_saturation": 50,
            "min_brightness": 50,
            "max_brightness": 205,
        },
    }

    try:
        t = thresholds.get(quality_threshold, thresholds["medium"])
        metrics = get_image_quality_metrics(image_path)

        if not metrics["valid"]:
            return True

        # Check each quality criterion
        if metrics["width"] < t["min_width"] or metrics["height"] < t["min_height"]:
            return True

        if metrics["file_size"] < t["min_file_size"]:
            return True

        if metrics["blur_score"] < t["min_blur_score"]:
            return True

        if metrics["saturation"] < t["min_saturation"]:
            return True

        if (
            metrics["brightness"] < t["min_brightness"]
            or metrics["brightness"] > t["max_brightness"]
        ):
            return True

        # Additional checks could be added here

        return False
    except Exception:
        return True  # Consider it low quality if there's an error


def are_similar_images(image1_path: str, image2_path: str, threshold: int = 10) -> bool:
    """Check if two images are similar using perceptual hashing.

    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        threshold: Hamming distance threshold

    Returns:
        True if the images are similar, False otherwise
    """
    try:
        # Get hashes
        img1_hashes = get_image_hash(image1_path)
        img2_hashes = get_image_hash(image2_path)

        if not img1_hashes or not img2_hashes:
            return False

        # Split the combined hashes
        img1_hash_parts = img1_hashes.split("|")
        img2_hash_parts = img2_hashes.split("|")

        # If we have the combined format with multiple hashes
        if len(img1_hash_parts) > 1 and len(img2_hash_parts) > 1:
            # Check each hash type
            for i in range(min(len(img1_hash_parts), len(img2_hash_parts))):
                hash1 = imagehash.hex_to_hash(img1_hash_parts[i])
                hash2 = imagehash.hex_to_hash(img2_hash_parts[i])

                # If any hash is very similar, consider the images similar
                if (
                    hash1 - hash2 <= threshold // 2
                ):  # More strict for specialized hashes
                    return True

            return False
        else:
            # Basic comparison with just perceptual hash
            hash1 = imagehash.hex_to_hash(img1_hashes)
            hash2 = imagehash.hex_to_hash(img2_hashes)
            return hash1 - hash2 <= threshold
    except Exception as e:
        logger.debug(f"Error comparing images: {e}")
        return False


def clean_images(
    directory: str,
    log_file: str = "removed_images.log",
    quality_threshold: str = "medium",
    target_count: int = 0,
) -> Tuple[int, int]:
    """Clean the images in a directory using advanced quality metrics.

    Args:
        directory: Directory containing images
        log_file: Path to the log file
        quality_threshold: Quality threshold level
        target_count: Target number of images to keep (0 = keep all that pass quality checks)

    Returns:
        Tuple of (number of images before cleaning, number of images after cleaning)
    """
    # Find all image files
    image_files = [
        f
        for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
        and f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"))
    ]

    before_count = len(image_files)
    if before_count == 0:
        return 0, 0

    # Log file for removed images
    with open(log_file, "a") as log:
        log.write(f"\n\n=== Cleaning directory: {directory} ===\n")
        log.write(f"Quality threshold: {quality_threshold}\n")
        log.write(f"Starting with {before_count} images\n")

    # Initialize a list to store image quality information
    image_qualities = []
    image_hashes = {}
    removed_count = 0
    removal_reasons = {
        "corrupted": 0,
        "invalid": 0,
        "low_resolution": 0,
        "blurry": 0,
        "low_quality": 0,
        "duplicate": 0,
        "non_plant_content": 0,
        "other": 0,
    }

    # First pass: basic checks and compute quality metrics
    logger.info(f"Analyzing {before_count} images in {directory}...")

    for filename in tqdm(
        image_files, desc=f"Analyzing {os.path.basename(directory)}", ncols=100
    ):
        image_path = os.path.join(directory, filename)

        try:
            # First, verify the image is valid
            try:
                img = Image.open(image_path)
                img.verify()
            except (UnidentifiedImageError, IOError):
                with open(log_file, "a") as log:
                    log.write(f"Removed corrupted image: {filename}\n")
                os.remove(image_path)
                removed_count += 1
                removal_reasons["corrupted"] += 1
                continue

            # Get comprehensive quality metrics
            metrics = get_image_quality_metrics(image_path)

            if not metrics["valid"]:
                with open(log_file, "a") as log:
                    log.write(f"Removed invalid image: {filename}\n")
                os.remove(image_path)
                removed_count += 1
                removal_reasons["invalid"] += 1
                continue

            # Check for minimum resolution
            min_width = CONFIG["IMAGE_QUALITY_THRESHOLDS"]["MIN_WIDTH"]
            min_height = CONFIG["IMAGE_QUALITY_THRESHOLDS"]["MIN_HEIGHT"]
            if metrics["width"] < min_width or metrics["height"] < min_height:
                with open(log_file, "a") as log:
                    log.write(
                        f"Removed low resolution image: {filename} ({metrics['width']}x{metrics['height']})\n"
                    )
                os.remove(image_path)
                removed_count += 1
                removal_reasons["low_resolution"] += 1
                continue

            # Check if too blurry
            blur_threshold = CONFIG["IMAGE_QUALITY_THRESHOLDS"]["BLUR_THRESHOLD"]
            if metrics["blur_score"] < blur_threshold:
                with open(log_file, "a") as log:
                    log.write(
                        f"Removed blurry image: {filename} (blur score: {metrics['blur_score']:.2f})\n"
                    )
                os.remove(image_path)
                removed_count += 1
                removal_reasons["blurry"] += 1
                continue

            # Check for plant content (filter out non-plant images)
            if not detect_plant_content(image_path):
                with open(log_file, "a") as log:
                    log.write(f"Removed non-plant image: {filename}\n")
                os.remove(image_path)
                removed_count += 1
                removal_reasons["non_plant_content"] += 1
                continue

            # Check for comprehensive quality issues
            if is_low_quality(image_path, quality_threshold):
                with open(log_file, "a") as log:
                    log.write(f"Removed low quality image: {filename}\n")
                os.remove(image_path)
                removed_count += 1
                removal_reasons["low_quality"] += 1
                continue

            # Check for duplicates using multiple hash types
            img_hashes = metrics["perceptual_hash"] + "|" + metrics["color_hash"]
            if img_hashes in image_hashes:
                with open(log_file, "a") as log:
                    log.write(
                        f"Removed duplicate image: {filename}, duplicate of {image_hashes[img_hashes]}\n"
                    )
                os.remove(image_path)
                removed_count += 1
                removal_reasons["duplicate"] += 1
                continue
            else:
                image_hashes[img_hashes] = filename

            # Store quality metrics for ranking
            image_qualities.append(
                {
                    "filename": filename,
                    "path": image_path,
                    "quality_score": calculate_quality_score(metrics, image_path),
                    "metrics": metrics,
                }
            )

        except Exception as e:
            with open(log_file, "a") as log:
                log.write(f"Error processing {filename}: {e}\n")
            try:
                os.remove(image_path)
                removed_count += 1
                removal_reasons["other"] += 1
            except Exception:
                pass

    # If we have a target count and significantly more images than needed,
    # keep only the highest quality images
    # Modified to only trim if we have 20% more than the target to retain more images
    excess_threshold = int(target_count * 1.3) if target_count > 0 else 0
    if target_count > 0 and len(image_qualities) > excess_threshold:
        # Sort by quality score (highest first)
        image_qualities.sort(key=lambda x: x["quality_score"], reverse=True)

        # Remove lower quality images beyond the target count
        # But keep slightly more than target to ensure we have enough after processing
        keep_count = min(len(image_qualities), int(target_count * 1.2))
        for img_info in image_qualities[keep_count:]:
            with open(log_file, "a") as log:
                log.write(
                    f"Removed excess image: {img_info['filename']} (quality score: {img_info['quality_score']:.2f})\n"
                )
            try:
                os.remove(img_info["path"])
                removed_count += 1
                # Not tracking reason here as it's not a quality issue
            except Exception as e:
                logger.error(f"Failed to remove excess image {img_info['path']}: {e}")

    after_count = before_count - removed_count

    # Log summary
    with open(log_file, "a") as log:
        log.write(f"\nCleaning summary for {directory}:\n")
        log.write(f"  Before: {before_count} images\n")
        log.write(f"  After: {after_count} images\n")
        log.write(f"  Removed: {removed_count} images\n")
        log.write("  Removal reasons:\n")
        for reason, count in removal_reasons.items():
            if count > 0:
                log.write(f"    {reason}: {count}\n")

    logger.info(
        f"Cleaned {directory}: {before_count} -> {after_count} images, removed {removed_count}"
    )

    return before_count, after_count


def calculate_quality_score(metrics: Dict[str, Any], image_path: str = None) -> float:
    """Calculate a quality score from image metrics.

    Higher score = better quality.
    """
    score = 0.0

    # Resolution factor (0-30 points)
    # Higher resolution = higher score, up to 4K
    resolution = metrics["width"] * metrics["height"]
    resolution_score = min(30, resolution / (3840 * 2160) * 30)
    score += resolution_score

    # Blur factor (0-40 points)
    # Less blur = higher score
    blur_score = min(40, metrics["blur_score"] / 500 * 40)
    score += blur_score

    # Aspect ratio factor (0-10 points)
    # Penalize extreme aspect ratios
    aspect = (
        metrics["aspect_ratio"]
        if metrics["aspect_ratio"] >= 1
        else 1 / metrics["aspect_ratio"]
    )
    aspect_score = max(0, 10 - (aspect - 1) * 5)
    score += aspect_score

    # Plant content quality bonus (0-20 points)
    # Bonus for images that have good agricultural content
    if image_path:
        try:
            plant_score = detect_plant_content(image_path, return_confidence=True)
            agricultural_score = detect_agricultural_content(image_path, return_confidence=True)
            # Combine plant and agricultural scores with weights
            plant_bonus = (plant_score * 0.7 + agricultural_score * 0.3) * 20
            score += plant_bonus
        except Exception:
            # If plant detection fails, don't penalize
            pass

    # Saturation factor (0-10 points)
    # Moderate saturation is best
    sat_score = 0
    if 20 <= metrics["saturation"] <= 100:
        sat_score = 10
    elif metrics["saturation"] < 20:
        sat_score = metrics["saturation"] / 2
    else:
        sat_score = max(0, 10 - (metrics["saturation"] - 100) / 20)
    score += sat_score

    # Brightness factor (0-10 points)
    # Medium brightness is best (not too dark, not too bright)
    brightness = metrics["brightness"]
    if 80 <= brightness <= 180:
        bright_score = 10
    elif brightness < 80:
        bright_score = brightness / 8
    else:
        bright_score = max(0, 10 - (brightness - 180) / 10)
    score += bright_score

    return score


def clean_dataset(
    dataset_dir: str,
    num_workers: int,
    quality_threshold: str = "medium",
    min_images_per_category: int = 1500,
    balance_categories: bool = True,
) -> Dict[str, Tuple[int, int]]:
    """Clean the entire dataset with improved quality control and category balancing.

    Args:
        dataset_dir: Dataset directory
        num_workers: Number of worker threads
        quality_threshold: Quality threshold level
        min_images_per_category: Minimum images per category to aim for
        balance_categories: Whether to balance categories within each crop

    Returns:
        Dictionary of directory to (before, after) count tuples
    """
    logger.info(
        f"Cleaning dataset in {dataset_dir} with quality threshold: {quality_threshold}"
    )
    start_time = time.time()

    # Find all image directories
    image_dirs = []
    crop_conditions = {}  # Track conditions per crop for balancing

    for crop in os.listdir(dataset_dir):
        crop_dir = os.path.join(dataset_dir, crop)
        if not os.path.isdir(crop_dir):
            continue

        crop_conditions[crop] = []

        for condition in os.listdir(crop_dir):
            condition_dir = os.path.join(crop_dir, condition)
            if os.path.isdir(condition_dir):
                image_dirs.append(condition_dir)
                crop_conditions[crop].append(condition_dir)

    # First pass: clean all directories to remove bad quality images
    logger.info(f"First cleaning pass: removing low quality images")
    clean_results = {}

    with tqdm(total=len(image_dirs), desc="Initial cleaning") as pbar:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Define the cleaning function with additional parameters
            clean_fn = lambda dir_path: clean_images(
                directory=dir_path,
                quality_threshold=quality_threshold,
                log_file=f"cleaning_pass1_{os.path.basename(dataset_dir)}.log",
            )

            # Submit all cleaning jobs
            futures = {
                executor.submit(clean_fn, dir_path): dir_path for dir_path in image_dirs
            }

            # Process results as they complete
            for future in concurrent.futures.as_completed(futures):
                dir_path = futures[future]
                try:
                    result = future.result()
                    clean_results[dir_path] = result
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error cleaning {dir_path}: {e}")
                    pbar.update(1)

    # Calculate statistics after first pass
    total_before = sum(before for before, _ in clean_results.values())
    total_after = sum(after for _, after in clean_results.values())
    total_removed = total_before - total_after

    logger.info(
        f"First cleaning pass completed: {total_before} -> {total_after} images, "
        f"removed {total_removed} ({total_removed / total_before * 100:.1f}% if total_before > 0)"
    )

    # Second pass: Balance categories if requested
    if balance_categories:
        logger.info(f"Second cleaning pass: balancing categories")

        # Process each crop separately to balance categories
        for crop, condition_dirs in crop_conditions.items():
            # Skip crops with only one condition
            if len(condition_dirs) <= 1:
                continue

            # Count images in each condition
            condition_counts = {}
            for dir_path in condition_dirs:
                condition = os.path.basename(dir_path)
                image_count = count_images_in_directory(dir_path)
                condition_counts[condition] = image_count

            # Find the target count for this crop
            # Use max(min_required, min_available) to avoid removing too many images
            available_counts = [
                count for count in condition_counts.values() if count > 0
            ]
            if not available_counts:
                continue

            target_count = max(
                min(min_images_per_category, min(available_counts)),
                min_images_per_category,
            )

            logger.info(f"Balancing {crop} conditions to {target_count} images each")

            # Clean each condition to the target count
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                # Define the balancing function
                balance_fn = lambda dir_path: clean_images(
                    directory=dir_path,
                    quality_threshold=quality_threshold,
                    target_count=target_count,
                    log_file=f"cleaning_pass2_{os.path.basename(dataset_dir)}.log",
                )

                # Submit all balancing jobs
                balance_futures = {
                    executor.submit(balance_fn, dir_path): dir_path
                    for dir_path in condition_dirs
                }

                # Process results
                for future in tqdm(
                    concurrent.futures.as_completed(balance_futures),
                    total=len(balance_futures),
                    desc=f"Balancing {crop}",
                ):
                    dir_path = balance_futures[future]
                    try:
                        before, after = future.result()
                        # Update clean_results
                        clean_results[dir_path] = (before, after)
                    except Exception as e:
                        logger.error(f"Error balancing {dir_path}: {e}")

    # Recalculate final statistics
    total_before = sum(before for before, _ in clean_results.values())
    total_after = sum(after for _, after in clean_results.values())
    total_removed = total_before - total_after

    # Calculate stats per crop
    crop_stats = {}
    for crop in crop_conditions:
        crop_total = 0
        for dir_path in crop_conditions[crop]:
            if dir_path in clean_results:
                _, after = clean_results[dir_path]
                crop_total += after
        crop_stats[crop] = crop_total

    # Log detailed results
    logger.info(
        f"Cleaning completed in {time.time() - start_time:.1f} seconds: "
        f"{total_before} -> {total_after} images, "
        f"removed {total_removed} ({total_removed / total_before * 100:.1f}% if total_before > 0)"
    )

    logger.info("Images per crop after cleaning:")
    for crop, count in crop_stats.items():
        logger.info(f"  {crop}: {count} images")

        # Log conditions for each crop
        for dir_path in crop_conditions[crop]:
            condition = os.path.basename(dir_path)
            if dir_path in clean_results:
                _, after = clean_results[dir_path]
                logger.info(f"    {condition}: {after} images")

    return clean_results


def split_dataset(
    dataset_dir: str,
    split_dir: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    balanced_splits: bool = True,
    num_workers: int = 4,
) -> Dict[str, Dict[str, int]]:
    """Split the dataset into training, validation and testing sets with advanced options.

    Args:
        dataset_dir: Dataset directory
        split_dir: Output directory for the split
        test_size: Proportion of the dataset to include in the test split
        val_size: Proportion of the dataset to include in the validation split
        stratify: Whether to ensure class proportions are preserved in splits
        balanced_splits: Whether to ensure each split gets the same number of samples per class
        num_workers: Number of worker threads to use for parallel processing

    Returns:
        Dictionary with statistics about the split
    """
    logger.info(f"Splitting dataset from {dataset_dir} to {split_dir}")
    start_time = time.time()

    # Create output directories
    train_dir = os.path.join(split_dir, "train")
    val_dir = os.path.join(split_dir, "val")  # Add validation directory
    test_dir = os.path.join(split_dir, "test")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Stats dictionary
    stats = {"train": {}, "val": {}, "test": {}}

    # If we want balanced splits, determine target counts per split
    if balanced_splits:
        # First, count the total number of images per class
        class_counts = {}
        for crop in os.listdir(dataset_dir):
            crop_dir = os.path.join(dataset_dir, crop)
            if not os.path.isdir(crop_dir):
                continue

            for condition in os.listdir(crop_dir):
                condition_dir = os.path.join(crop_dir, condition)
                if not os.path.isdir(condition_dir):
                    continue

                image_count = count_images_in_directory(condition_dir)

                class_counts[f"{crop}/{condition}"] = image_count

        # Find minimum class count to maintain balance
        if class_counts:
            min_class_count = min(count for count in class_counts.values() if count > 0)
            min_test_count = max(5, int(min_class_count * test_size))
            min_val_count = max(5, int(min_class_count * val_size))
            min_train_count = max(10, min_class_count - min_test_count - min_val_count)

            logger.info(
                f"Balanced splitting: minimum {min_train_count} train, {min_val_count} val, {min_test_count} test samples per class"
            )

    # Submit tasks in parallel
    split_tasks = []

    # Process each crop
    for crop in os.listdir(dataset_dir):
        crop_dir = os.path.join(dataset_dir, crop)
        if not os.path.isdir(crop_dir):
            continue

        # Create crop directories in train, val, and test
        train_crop_dir = os.path.join(train_dir, crop)
        val_crop_dir = os.path.join(val_dir, crop)
        test_crop_dir = os.path.join(test_dir, crop)

        os.makedirs(train_crop_dir, exist_ok=True)
        os.makedirs(val_crop_dir, exist_ok=True)
        os.makedirs(test_crop_dir, exist_ok=True)

        # Process each condition
        for condition in os.listdir(crop_dir):
            condition_dir = os.path.join(crop_dir, condition)
            if not os.path.isdir(condition_dir):
                continue

            # Create task for parallel processing
            split_tasks.append((crop, condition, crop_dir, condition_dir))

    # Process splits in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Define the split function
        def split_class(task):
            crop, condition, crop_dir, condition_dir = task

            # Create condition directories in train, val, and test
            train_condition_dir = os.path.join(train_dir, crop, condition)
            val_condition_dir = os.path.join(val_dir, crop, condition)
            test_condition_dir = os.path.join(test_dir, crop, condition)

            os.makedirs(train_condition_dir, exist_ok=True)
            os.makedirs(val_condition_dir, exist_ok=True)
            os.makedirs(test_condition_dir, exist_ok=True)

            # Get image files
            image_files = [
                f
                for f in os.listdir(condition_dir)
                if os.path.isfile(os.path.join(condition_dir, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ]

            # Skip if no images
            if not image_files:
                return crop, condition, (0, 0, 0)

            # Calculate exact split sizes or use balanced approach
            if (
                balanced_splits and len(image_files) > 20
            ):  # Only balance if we have enough samples
                # Fixed counts for each split
                test_count = min_test_count
                val_count = min_val_count
                train_count = min(
                    len(image_files) - test_count - val_count, min_train_count
                )

                # Shuffle files
                random.shuffle(image_files)

                # Take exactly the counts we want
                train_files = image_files[:train_count]
                val_files = image_files[train_count : train_count + val_count]
                test_files = image_files[
                    train_count + val_count : train_count + val_count + test_count
                ]
            else:
                # Handle cases with few images
                if len(image_files) == 1:
                    # Single image goes to training
                    train_files = image_files
                    val_files = []
                    test_files = []
                elif len(image_files) == 2:
                    # Two images: 1 train, 1 test
                    train_files = [image_files[0]]
                    val_files = []
                    test_files = [image_files[1]]
                elif len(image_files) == 3:
                    # Three images: 1 train, 1 val, 1 test
                    train_files = [image_files[0]]
                    val_files = [image_files[1]]
                    test_files = [image_files[2]]
                elif len(image_files) < 10:
                    # Few images: split 70/0/30
                    train_size = max(1, int(len(image_files) * 0.7))
                    test_size = len(image_files) - train_size
                    train_files = image_files[:train_size]
                    val_files = []
                    test_files = image_files[train_size:]
                else:
                    # Normal case - first split into train vs test
                    relative_test_size = test_size / (1 - val_size)
                    train_val_files, test_files = train_test_split(
                        image_files,
                        test_size=relative_test_size,
                        random_state=42,
                        shuffle=True,
                    )

                    # Then split train into train vs val
                    train_files, val_files = train_test_split(
                        train_val_files,
                        test_size=val_size / (1 - test_size),
                        random_state=42,
                        shuffle=True,
                    )

            # Process each set using NVMe-optimized functions if available
            copy_methods = {
                "train": (train_files, train_condition_dir),
                "val": (val_files, val_condition_dir),
                "test": (test_files, test_condition_dir),
            }

            result_counts = {}

            for split_name, (files, target_dir) in copy_methods.items():
                copied = 0

                for filename in files:
                    src = os.path.join(condition_dir, filename)
                    dst = os.path.join(target_dir, filename)

                    try:
                        # Use optimized I/O for NVMe if enabled
                        if CONFIG["NVME_OPTIMIZATIONS"]["ENABLED"]:
                            with (
                                nvme_optimized_io(src, "rb") as src_file,
                                nvme_optimized_io(dst, "wb") as dst_file,
                            ):
                                if isinstance(src_file, mmap.mmap):
                                    # For mmap objects
                                    dst_file.write(src_file[:])
                                else:
                                    # For file-like objects
                                    shutil.copyfileobj(src_file, dst_file)
                        else:
                            # Fallback to PIL copy which has better error handling
                            img = Image.open(src)
                            img.save(dst)

                        copied += 1
                    except Exception as e:
                        logger.error(f"Error copying {src} to {dst}: {e}")

                result_counts[split_name] = copied

            return (
                crop,
                condition,
                (
                    result_counts.get("train", 0),
                    result_counts.get("val", 0),
                    result_counts.get("test", 0),
                ),
            )

        # Submit all split tasks
        futures = {executor.submit(split_class, task): task for task in split_tasks}

        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Splitting classes",
        ):
            try:
                crop, condition, (train_count, val_count, test_count) = future.result()

                # Update stats
                class_name = f"{crop}/{condition}"
                stats["train"][class_name] = train_count
                stats["val"][class_name] = val_count
                stats["test"][class_name] = test_count
            except Exception as e:
                task = futures[future]
                logger.error(f"Error splitting {task[0]}/{task[1]}: {e}")

    # Calculate totals
    stats["train"]["total"] = sum(stats["train"].values())
    stats["val"]["total"] = sum(stats["val"].values())
    stats["test"]["total"] = sum(stats["test"].values())

    # Generate summary
    logger.info(
        f"Dataset split completed in {time.time() - start_time:.1f} seconds:\n"
        f"  Training: {stats['train']['total']} samples\n"
        f"  Validation: {stats['val']['total']} samples\n"
        f"  Testing: {stats['test']['total']} samples"
    )

    # Create metadata files with class mappings for ML frameworks
    create_metadata_files(split_dir, stats)

    return stats


def create_metadata_files(split_dir: str, stats: Dict[str, Dict[str, int]]) -> None:
    """Create metadata files for the dataset to make it easier to use with ML frameworks.

    Args:
        split_dir: Split directory
        stats: Statistics dictionary
    """
    # Get all classes
    all_classes = set()
    for split in ["train", "val", "test"]:
        all_classes.update([cls for cls in stats[split].keys() if cls != "total"])
    all_classes = sorted(list(all_classes))

    # Create a class mapping file
    class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}

    # Save as JSON
    with open(os.path.join(split_dir, "class_mapping.json"), "w") as f:
        json.dump(
            {
                "class_to_idx": class_to_idx,
                "idx_to_class": {str(idx): cls for cls, idx in class_to_idx.items()},
                "num_classes": len(all_classes),
                "class_counts": {
                    "train": {cls: stats["train"].get(cls, 0) for cls in all_classes},
                    "val": {cls: stats["val"].get(cls, 0) for cls in all_classes},
                    "test": {cls: stats["test"].get(cls, 0) for cls in all_classes},
                },
                "totals": {
                    "train": stats["train"].get("total", 0),
                    "val": stats["val"].get("total", 0),
                    "test": stats["test"].get("total", 0),
                    "all": sum(
                        stats[split].get("total", 0)
                        for split in ["train", "val", "test"]
                    ),
                },
            },
            f,
            indent=2,
        )

    logger.info(
        f"Created class mapping file at {os.path.join(split_dir, 'class_mapping.json')}"
    )

    # Create a simple README with dataset information
    with open(os.path.join(split_dir, "README.md"), "w") as f:
        f.write(f"# Crop Disease Dataset\n\n")
        f.write(f"This dataset contains images of crop diseases and pests.\n\n")
        f.write(f"## Statistics\n\n")
        f.write(f"- Total classes: {len(all_classes)}\n")
        f.write(
            f"- Total images: {sum(stats[split].get('total', 0) for split in ['train', 'val', 'test'])}\n"
        )
        f.write(f"  - Training: {stats['train'].get('total', 0)}\n")
        f.write(f"  - Validation: {stats['val'].get('total', 0)}\n")
        f.write(f"  - Testing: {stats['test'].get('total', 0)}\n\n")
        f.write(f"## Class Distribution\n\n")
        f.write(f"| Class | Train | Val | Test | Total |\n")
        f.write(f"|-------|-------|-----|------|-------|\n")

        for cls in all_classes:
            train_count = stats["train"].get(cls, 0)
            val_count = stats["val"].get(cls, 0)
            test_count = stats["test"].get(cls, 0)
            total = train_count + val_count + test_count
            f.write(
                f"| {cls} | {train_count} | {val_count} | {test_count} | {total} |\n"
            )

    logger.info(f"Created README file at {os.path.join(split_dir, 'README.md')}")


def generate_dataset_stats(dataset_dir: str) -> Dict[str, Dict[str, int]]:
    """Generate statistics about the dataset.

    Args:
        dataset_dir: Dataset directory

    Returns:
        Dictionary with statistics about the dataset
    """
    stats = {
        "total_images": 0,
        "by_crop": {},
        "by_category": {"healthy": 0, "pests": 0, "diseases": 0},
    }

    # Process each crop
    for crop in os.listdir(dataset_dir):
        crop_dir = os.path.join(dataset_dir, crop)
        if not os.path.isdir(crop_dir):
            continue

        crop_stats = {
            "total": 0,
            "healthy": 0,
            "pests": 0,
            "diseases": 0,
            "conditions": {},
        }
        stats["by_crop"][crop] = crop_stats

        # Process each condition
        for condition in os.listdir(crop_dir):
            condition_dir = os.path.join(crop_dir, condition)
            if not os.path.isdir(condition_dir):
                continue

            # Count images
            image_files = [
                f
                for f in os.listdir(condition_dir)
                if os.path.isfile(os.path.join(condition_dir, f))
                and f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]

            image_count = len(image_files)

            # Update stats
            crop_stats["conditions"][condition] = image_count
            crop_stats["total"] += image_count

            # Determine category
            if condition == "healthy":
                crop_stats["healthy"] += image_count
                stats["by_category"]["healthy"] += image_count
            elif condition in CROPS_DATA.get(crop, {}).get("pests", []):
                crop_stats["pests"] += image_count
                stats["by_category"]["pests"] += image_count
            elif condition in CROPS_DATA.get(crop, {}).get("diseases", []):
                crop_stats["diseases"] += image_count
                stats["by_category"]["diseases"] += image_count

            stats["total_images"] += image_count

    return stats


def run_scraper(args: argparse.Namespace):
    """Run the scraper with the given arguments.

    Args:
        args: Command line arguments
    """
    start_time = time.time()

    # If a config file is provided, load it
    if args.config_file and os.path.exists(args.config_file):
        try:
            with open(args.config_file, "r") as f:
                custom_config = json.load(f)
                # Update config with custom values
                for key, value in custom_config.items():
                    if key in CONFIG:
                        if isinstance(CONFIG[key], dict) and isinstance(value, dict):
                            # Update nested dictionaries
                            CONFIG[key].update(value)
                        else:
                            CONFIG[key] = value
            logger.info(f"Loaded custom configuration from {args.config_file}")
        except Exception as e:
            logger.error(f"Error loading config file: {e}")

    # Update config based on command line arguments
    CONFIG["MIN_IMAGES_PER_CATEGORY"] = args.min_images
    CONFIG["NVME_OPTIMIZATIONS"]["ENABLED"] = args.nvme_optimize

    # Determine search engines to use
    if args.search_engines.lower() == "all":
        search_engines = ["google", "bing", "duckduckgo", "yandex"]
    else:
        search_engines = [
            engine.strip().lower() for engine in args.search_engines.split(",")
        ]

    # Initialize CaptchaSolver if enabled
    captcha_solver = None
    if args.auto_solve_captcha and args.anticaptcha_key:
        captcha_solver = CaptchaSolver(api_key=args.anticaptcha_key)
        logger.info("Anti-captcha service enabled")

    # Determine which crops to scrape
    if args.crops.lower() == "all":
        crops_to_scrape = list(CROPS_DATA.keys())
    else:
        crops_to_scrape = [crop.strip() for crop in args.crops.split(",")]
        # Validate crops
        invalid_crops = [crop for crop in crops_to_scrape if crop not in CROPS_DATA]
        if invalid_crops:
            logger.error(f"Invalid crops: {', '.join(invalid_crops)}")
            logger.info(f"Available crops: {', '.join(CROPS_DATA.keys())}")
            sys.exit(1)

    logger.info(f"Starting the crop image scraper for {len(crops_to_scrape)} crops")
    logger.info(f"Crops: {', '.join(crops_to_scrape)}")
    logger.info(f"Min images per category: {args.min_images}")
    logger.info(f"Max images per category: {args.max_images}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Split directory: {args.split_dir}")
    logger.info(f"Worker threads: {args.workers}")
    logger.info(f"Quality threshold: {args.quality_threshold}")
    logger.info(f"Search engines: {', '.join(search_engines)}")
    logger.info(
        f"NVMe optimizations: {'enabled' if args.nvme_optimize else 'disabled'}"
    )

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize scraper state for resumability
    scraper_state = ScraperState()
    if args.resume and os.path.exists(PROGRESS_FILE):
        scraper_state = ScraperState.load(PROGRESS_FILE)
        logger.info(f"Resuming previous scraping session")
        logger.info(f"Previously scraped {scraper_state.images_scraped} images")

    # Count total target images for progress tracking
    total_categories = 0
    for crop in crops_to_scrape:
        # Count one for healthy
        total_categories += 1
        # Count for each pest
        total_categories += len(CROPS_DATA[crop]["pests"])
        # Count for each disease
        total_categories += len(CROPS_DATA[crop]["diseases"])

    # Update state
    scraper_state.total_target = total_categories * args.min_images

    # Step 1: Scrape images
    if not args.skip_scraping:
        logger.info("Starting image scraping")
        scraped_count = 0

        # Use a progress bar for the whole scraping process
        with tqdm(total=total_categories, desc="Scraping categories") as pbar:
            for crop in crops_to_scrape:
                # Check if this crop was already completed in a previous run
                if args.resume and crop in scraper_state.completed_categories:
                    completed_conditions = set(
                        scraper_state.completed_categories[crop].keys()
                    )
                    all_conditions = {"healthy"}.union(
                        set(CROPS_DATA[crop]["pests"])
                    ).union(set(CROPS_DATA[crop]["diseases"]))

                    if completed_conditions == all_conditions:
                        logger.info(f"Skipping already completed crop: {crop}")
                        pbar.update(
                            1
                            + len(CROPS_DATA[crop]["pests"])
                            + len(CROPS_DATA[crop]["diseases"])
                        )
                        continue

                logger.info(f"Scraping images for crop: {crop}")

                # Scrape healthy images
                if not (
                    args.resume
                    and crop in scraper_state.completed_categories
                    and "healthy" in scraper_state.completed_categories[crop]
                    and scraper_state.completed_categories[crop]["healthy"]
                    >= args.min_images
                ):
                    _, count = crawl_images(
                        crop,
                        "healthy",
                        "healthy",
                        args.output_dir,
                        args.min_images,
                        args.max_images,
                        scraper_state,
                        search_engines=search_engines,
                        use_proxies=args.use_proxies,
                        use_selenium=args.selenium,
                        captcha_solver=captcha_solver,
                        quality_threshold=args.quality_threshold,
                        safe_search=args.safe_search,
                    )
                    scraped_count += count
                    pbar.update(1)
                else:
                    logger.info(f"Skipping already completed category: {crop}/healthy")
                    pbar.update(1)

                # Scrape pest images
                for pest in CROPS_DATA[crop]["pests"]:
                    if not (
                        args.resume
                        and crop in scraper_state.completed_categories
                        and pest in scraper_state.completed_categories[crop]
                        and scraper_state.completed_categories[crop][pest]
                        >= args.min_images
                    ):
                        _, count = crawl_images(
                            crop,
                            "pests",
                            pest,
                            args.output_dir,
                            args.min_images,
                            args.max_images,
                            scraper_state,
                            search_engines=search_engines,
                            use_proxies=args.use_proxies,
                            use_selenium=args.selenium,
                            captcha_solver=captcha_solver,
                            quality_threshold=args.quality_threshold,
                            safe_search=args.safe_search,
                        )
                        scraped_count += count
                        pbar.update(1)
                    else:
                        logger.info(
                            f"Skipping already completed category: {crop}/{pest}"
                        )
                        pbar.update(1)

                # Scrape disease images
                for disease in CROPS_DATA[crop]["diseases"]:
                    if not (
                        args.resume
                        and crop in scraper_state.completed_categories
                        and disease in scraper_state.completed_categories[crop]
                        and scraper_state.completed_categories[crop][disease]
                        >= args.min_images
                    ):
                        _, count = crawl_images(
                            crop,
                            "diseases",
                            disease,
                            args.output_dir,
                            args.min_images,
                            args.max_images,
                            scraper_state,
                            search_engines=search_engines,
                            use_proxies=args.use_proxies,
                            use_selenium=args.selenium,
                            captcha_solver=captcha_solver,
                            quality_threshold=args.quality_threshold,
                            safe_search=args.safe_search,
                        )
                        scraped_count += count
                        pbar.update(1)
                    else:
                        logger.info(
                            f"Skipping already completed category: {crop}/{disease}"
                        )
                        pbar.update(1)

        logger.info(
            f"Image scraping completed. Scraped a total of {scraped_count} new images"
        )
    else:
        logger.info("Skipping image scraping")

    # Step 2: Clean images
    if not args.skip_cleaning:
        logger.info("Starting image cleaning")
        clean_dataset(
            args.output_dir,
            args.workers,
            quality_threshold=args.quality_threshold,
            min_images_per_category=args.min_images,
            balance_categories=True,
        )
        logger.info("Image cleaning completed")
    else:
        logger.info("Skipping image cleaning")

    # Step 3: Split into train/validation/test
    if not args.skip_splitting:
        logger.info("Starting dataset splitting")
        split_dataset(
            args.output_dir,
            args.split_dir,
            test_size=0.15,  # 15% for test set
            val_size=0.15,  # 15% for validation set
            stratify=True,
            balanced_splits=True,
            num_workers=args.workers,
        )
        logger.info("Dataset splitting completed")
    else:
        logger.info("Skipping dataset splitting")

    # Generate and print dataset statistics
    logger.info("Generating detailed dataset statistics")
    stats = generate_dataset_stats(args.output_dir)

    logger.info(f"Total images: {stats['total_images']}")
    logger.info(f"Healthy images: {stats['by_category']['healthy']}")
    logger.info(f"Pest images: {stats['by_category']['pests']}")
    logger.info(f"Disease images: {stats['by_category']['diseases']}")

    logger.info("Images by crop:")
    for crop, crop_stats in stats["by_crop"].items():
        logger.info(f"  {crop}: {crop_stats['total']} images")
        logger.info(f"    Healthy: {crop_stats['healthy']}")
        logger.info(f"    Pests: {crop_stats['pests']}")
        logger.info(f"    Diseases: {crop_stats['diseases']}")

        for condition, count in crop_stats["conditions"].items():
            # Highlight if we didn't meet the minimum image count
            if count < args.min_images:
                logger.warning(
                    f"      {condition}: {count} images (BELOW MINIMUM {args.min_images})"
                )
            else:
                logger.info(f"      {condition}: {count} images")

    # Save final statistics to a JSON file
    stats_file = os.path.join(args.output_dir, "dataset_stats.json")
    with open(stats_file, "w") as f:
        json.dump(
            {
                "total_images": stats["total_images"],
                "by_category": stats["by_category"],
                "by_crop": stats["by_crop"],
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now().isoformat(),
                "config": {
                    k: v for k, v in CONFIG.items() if k != "NVME_OPTIMIZATIONS"
                },
            },
            f,
            indent=2,
        )

    logger.info(f"Detailed statistics saved to {stats_file}")
    logger.info(
        f"Scraping and dataset preparation completed in {(time.time() - start_time) / 60:.1f} minutes"
    )


# Plant detection functions for enhanced image filtering
def detect_plant_content(image_path: str, return_confidence: bool = False) -> Union[bool, float]:
    """
    Detect if an image contains plant/vegetation content using computer vision.
    
    Args:
        image_path: Path to the image file
        return_confidence: If True, return confidence score (0-1), else boolean
    
    Returns:
        bool or float: Plant detection result or confidence score
    """
    try:
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            return 0.0 if return_confidence else False
            
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        
        confidence_score = 0.0
        
        # 1. Green vegetation detection in HSV
        # Define green color ranges for vegetation
        lower_green1 = np.array([35, 40, 40])
        upper_green1 = np.array([85, 255, 255])
        
        # Secondary green range for yellower greens
        lower_green2 = np.array([25, 40, 40]) 
        upper_green2 = np.array([35, 255, 255])
        
        mask_green1 = cv2.inRange(hsv, lower_green1, upper_green1)
        mask_green2 = cv2.inRange(hsv, lower_green2, upper_green2)
        green_mask = cv2.bitwise_or(mask_green1, mask_green2)
        
        green_ratio = np.sum(green_mask > 0) / (img.shape[0] * img.shape[1])
        confidence_score += min(0.4, green_ratio * 2)  # Up to 40% of score from green content
        
        # 2. Texture complexity analysis
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Calculate texture complexity using local binary patterns concept
        # High frequency content indicates natural textures
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and get texture score
        texture_score = np.mean(gradient_magnitude) / 255.0
        confidence_score += min(0.2, texture_score * 0.5)  # Up to 20% from texture
        
        # 3. Edge density analysis
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (img.shape[0] * img.shape[1])
        confidence_score += min(0.15, edge_density * 3)  # Up to 15% from edges
        
        # 4. Color variance (natural images have more color diversity)
        color_std = np.std(img.reshape(-1, 3), axis=0)
        avg_color_variance = np.mean(color_std) / 255.0
        confidence_score += min(0.1, avg_color_variance)  # Up to 10% from color variance
        
        # 5. Frequency domain analysis for natural textures
        # Natural plant textures have specific frequency characteristics
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        # Calculate energy distribution
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Energy in mid-frequencies (natural textures)
        mid_freq_mask = np.zeros_like(magnitude_spectrum)
        cv2.circle(mid_freq_mask, (center_w, center_h), min(h, w) // 4, 1, -1)
        cv2.circle(mid_freq_mask, (center_w, center_h), min(h, w) // 8, 0, -1)
        
        mid_freq_energy = np.sum(magnitude_spectrum * mid_freq_mask)
        total_energy = np.sum(magnitude_spectrum)
        
        if total_energy > 0:
            mid_freq_ratio = mid_freq_energy / total_energy
            confidence_score += min(0.15, mid_freq_ratio * 0.5)  # Up to 15% from frequency analysis
        
        # Normalize confidence score to 0-1 range
        confidence_score = min(1.0, confidence_score)
        
        if return_confidence:
            return confidence_score
        else:
            # Threshold for binary decision
            return confidence_score > 0.3
            
    except Exception as e:
        logger.debug(f"Error in plant detection for {image_path}: {e}")
        return 0.0 if return_confidence else False


def detect_agricultural_content(image_path: str, return_confidence: bool = False) -> Union[bool, float]:
    """
    Detect agricultural/crop-specific content patterns.
    
    Args:
        image_path: Path to the image file
        return_confidence: If True, return confidence score (0-1), else boolean
    
    Returns:
        bool or float: Agricultural content detection result or confidence score
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return 0.0 if return_confidence else False
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        confidence_score = 0.0
        
        # 1. Leaf-like pattern detection using template matching concepts
        # Create simple leaf-like kernels for convolution
        kernel_size = 15
        
        # Elongated kernel for leaf shapes
        leaf_kernel = np.zeros((kernel_size, kernel_size))
        cv2.ellipse(leaf_kernel, (kernel_size//2, kernel_size//2), 
                   (kernel_size//3, kernel_size//6), 45, 0, 360, 1, -1)
        
        # Apply morphological operations to detect leaf-like structures
        response = cv2.filter2D(gray.astype(np.float32), -1, leaf_kernel)
        leaf_response = np.sum(response > np.percentile(response, 85)) / response.size
        confidence_score += min(0.3, leaf_response * 2)
        
        # 2. Row/field structure detection
        # Many crop images show organized planting patterns
        
        # Horizontal line detection (crop rows)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_score = np.sum(horizontal_lines > 50) / horizontal_lines.size
        confidence_score += min(0.2, horizontal_score * 10)
        
        # Vertical patterns (plant stems)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(gray, cv2.MORPH_OPEN, vertical_kernel)
        vertical_score = np.sum(vertical_lines > 50) / vertical_lines.size
        confidence_score += min(0.15, vertical_score * 10)
        
        # 3. Brown/soil color detection (agricultural context)
        # Define brown/soil color ranges
        lower_brown = np.array([8, 50, 20])
        upper_brown = np.array([25, 255, 200])
        brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
        brown_ratio = np.sum(brown_mask > 0) / brown_mask.size
        confidence_score += min(0.2, brown_ratio * 1.5)
        
        # 4. Texture uniformity analysis
        # Agricultural images often have more uniform texture patterns
        # compared to wild vegetation
        
        # Calculate local standard deviation
        kernel = np.ones((9, 9), np.float32) / 81
        mean_filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
        sqr_diff = (gray.astype(np.float32) - mean_filtered) ** 2
        local_variance = cv2.filter2D(sqr_diff, -1, kernel)
        
        # Agricultural images tend to have moderate, consistent variance
        variance_uniformity = 1.0 - (np.std(local_variance) / (np.mean(local_variance) + 1e-6))
        confidence_score += min(0.15, variance_uniformity * 0.3)
        
        # Normalize confidence score
        confidence_score = min(1.0, confidence_score)
        
        if return_confidence:
            return confidence_score
        else:
            return confidence_score > 0.25
            
    except Exception as e:
        logger.debug(f"Error in agricultural content detection for {image_path}: {e}")
        return 0.0 if return_confidence else False


def filter_non_plant_images(image_directory: str, strict_mode: bool = False) -> Tuple[int, int]:
    """
    Filter out non-plant images from a directory.
    
    Args:
        image_directory: Directory containing images to filter
        strict_mode: If True, use stricter thresholds for plant detection
    
    Returns:
        Tuple of (original_count, filtered_count)
    """
    if not os.path.exists(image_directory):
        return 0, 0
    
    # Get all image files
    image_files = [
        f for f in os.listdir(image_directory)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))
    ]
    
    original_count = len(image_files)
    removed_count = 0
    
    plant_threshold = 0.4 if strict_mode else 0.3
    agricultural_threshold = 0.3 if strict_mode else 0.25
    
    for filename in image_files:
        image_path = os.path.join(image_directory, filename)
        
        try:
            # Check if image contains plant content
            plant_confidence = detect_plant_content(image_path, return_confidence=True)
            agricultural_confidence = detect_agricultural_content(image_path, return_confidence=True)
            
            # Combined decision - image needs to pass either plant detection or agricultural detection
            is_plant_related = (plant_confidence >= plant_threshold or 
                              agricultural_confidence >= agricultural_threshold)
            
            if not is_plant_related:
                # Remove non-plant image
                os.remove(image_path)
                removed_count += 1
                logger.debug(f"Removed non-plant image: {filename} "
                           f"(plant: {plant_confidence:.2f}, agri: {agricultural_confidence:.2f})")
                
        except Exception as e:
            logger.debug(f"Error processing {filename}: {e}")
            # If there's an error, keep the image (conservative approach)
            continue
    
    filtered_count = original_count - removed_count
    
    logger.info(f"Plant filtering: {original_count} -> {filtered_count} images "
                f"(removed {removed_count} non-plant images)")
    
    return original_count, filtered_count
