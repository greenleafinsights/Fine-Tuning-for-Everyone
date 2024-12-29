import os
import re
import json
import math
from collections import defaultdict
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytesseract
from PIL import Image
from ultralytics import YOLO


class DocumentLayoutAndOCR:
    """
    Processes an images root directory that contains subdirectories (one per document).
    Each subdirectory is named after the document/PDF name, holding images like page_1.jpg, page_2.jpg, etc.

    Speed-ups:
      - Batch YOLO inference for all images in a subdirectory (instead of per image).
      - Parallel OCR via ThreadPoolExecutor.
    """

    def __init__(
        self,
        model_path: str,
        images_root: str,
        output_root: str,
        tesseract_cmd: Optional[str] = None,
        max_ocr_workers: int = 4
    ):
        """
        Args:
            model_path (str): Path to the YOLO model weights (.pt file).
            images_root (str): Top-level images directory (contains subdirs, each with page images).
            output_root (str): Base output directory to store processed results.
            tesseract_cmd (str, optional): Path to Tesseract executable, if not on system PATH.
            max_ocr_workers (int): Number of parallel workers for OCR tasks.
        """
        self.model_path = model_path
        self.images_root = images_root
        self.output_root = output_root
        self.max_ocr_workers = max_ocr_workers

        # Ensure the output root directory exists
        os.makedirs(self.output_root, exist_ok=True)

        # Load YOLO model once
        self.model = YOLO(self.model_path)

        # Optional: configure Tesseract
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    def process_all_documents(self):
        """
        Finds all subdirectories in `self.images_root`. 
        For each subdirectory (document), processes all .jpg/.png pages in a batch.
        """
        # 1) List subdirectories in images_root
        for entry in os.scandir(self.images_root):
            if entry.is_dir():
                doc_name = entry.name  # e.g. "my_document"
                doc_dir = entry.path   # e.g. "images/my_document"

                print(f"Processing document subdir: {doc_name}")
                self._process_single_document(doc_name, doc_dir)

    def _process_single_document(self, doc_name: str, doc_dir: str):
        """
        1) Gather all .jpg/.png in doc_dir
        2) Batch YOLO inference
        3) For each page image, parallel OCR for bounding boxes
        """
        image_files = [
            f for f in os.listdir(doc_dir)
            if f.lower().endswith((".jpg", ".png"))
        ]
        image_files.sort()

        if not image_files:
            print(f"No images found in {doc_dir}, skipping.")
            return

        # Build full paths
        full_paths = [os.path.join(doc_dir, f) for f in image_files]

        # === 1) Batch YOLO inference ===
        # Each returned item in 'results' corresponds to the same index as 'full_paths'
        print(f"Running batch YOLO inference on {len(full_paths)} images in {doc_dir}...")
        results_list = self.model.predict(source=full_paths, verbose=False)

        # === 2) For each page image, handle bounding boxes + parallel OCR ===
        for img_path, result in zip(full_paths, results_list):
            page_idx = self._extract_page_index(os.path.basename(img_path))
            if page_idx is None:
                print(f"Warning: Could not parse page index from {img_path}. Skipping.")
                continue

            self._process_single_image(doc_name, img_path, page_idx, result)

    def _process_single_image(self, doc_name: str, image_path: str, page_idx: int, detection):
        """
        Crops pictures and runs OCR on text-labeled boxes from a single YOLO result.
        Uses parallel OCR if multiple boxes are text-labeled.
        """
        # Load the original PIL image (for cropping)
        pil_img = Image.open(image_path)

        # Output subfolder for this specific page
        page_folder = os.path.join(self.output_root, doc_name, f"page_{page_idx}")
        os.makedirs(page_folder, exist_ok=True)

        boxes = detection.boxes
        if not boxes or len(boxes) == 0:
            print(f"No detections found in {image_path}")
            return

        # We'll gather tasks for OCR
        ocr_tasks = []
        extracted_text = defaultdict(list)

        # Iterate bounding boxes
        for obj_id, box in enumerate(boxes, start=1):
            cls_id = int(box.cls[0])
            label = detection.names[cls_id]  # e.g., "Text", "Picture", etc.

            # XYXY coords
            x1, y1, x2, y2 = box.xyxy[0]
            crop_box = (int(x1), int(y1), int(x2), int(y2))

            # Crop the region
            cropped_region = pil_img.crop(crop_box)

            # If it's a picture, we save it right away
            if label.lower() in ["picture", "pictures"]:
                picture_filename = f"page_{page_idx}_{obj_id}.jpg"
                picture_path = os.path.join(page_folder, picture_filename)
                cropped_region.save(picture_path, format="JPEG")
            else:
                # It's a text region => queue OCR
                # We'll store the function and the label, so we can map the results later
                ocr_tasks.append((label, cropped_region))

        # === Parallel OCR on text-like regions ===
        if ocr_tasks:
            with ThreadPoolExecutor(max_workers=self.max_ocr_workers) as executor:
                futures = []
                for lbl, crop_img in ocr_tasks:
                    futures.append(executor.submit(self._run_ocr, lbl, crop_img))

                for fut in as_completed(futures):
                    lbl, text = fut.result()
                    if text:
                        extracted_text[lbl].append(text)

        # === Save OCR text to JSON ===
        if extracted_text:
            json_filename = f"page_{page_idx}.json"
            json_path = os.path.join(page_folder, json_filename)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(dict(extracted_text), f, ensure_ascii=False, indent=2)

        print(f"Processed {os.path.basename(image_path)} -> {page_folder}")

    def _run_ocr(self, label: str, pil_image: Image.Image):
        """
        Simple helper function to run Tesseract OCR on a cropped region.
        Returns (label, recognized_text).
        """
        text = pytesseract.image_to_string(pil_image).strip()
        return (label, text)

    def _extract_page_index(self, filename: str) -> Optional[int]:
        """
        Attempt to parse page index from filenames like "page_3.jpg".
        Returns None if no match.
        """
        match = re.search(r"page_(\d+)\.", filename)
        if match:
            return int(match.group(1))
        return None