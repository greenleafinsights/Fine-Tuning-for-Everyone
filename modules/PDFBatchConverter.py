import os
from typing import List
from pdf2image import convert_from_path

class PDFToImageConverter:
    """
    A class to split a SINGLE PDF into multiple images (one per page),
    saving them as .jpg files in a folder named after the PDF.
    """

    def __init__(self, pdf_path: str, output_root: str = "images", dpi: int = 300):
        """
        Args:
            pdf_path (str): Path to the PDF file to be converted.
            output_root (str): Root directory where images are stored.
            dpi (int): Resolution in DPI for rendering the PDF pages to images.
        """
        self.pdf_path = pdf_path
        self.dpi = dpi
        self.output_root = output_root

        # Validate the PDF
        if not os.path.isfile(self.pdf_path):
            raise FileNotFoundError(f"PDF not found at: {self.pdf_path}")

        # Derive the folder name from the PDF filename
        pdf_basename = os.path.basename(self.pdf_path)          # e.g. "some-file.pdf"
        pdf_name_only = os.path.splitext(pdf_basename)[0]       # e.g. "some-file"

        # Build output folder
        self.output_folder = os.path.join(self.output_root, pdf_name_only)
        os.makedirs(self.output_folder, exist_ok=True)

    def convert_all_pages(self) -> List[str]:
        """
        Convert ALL pages in the PDF to images and save them in the output folder.

        Returns:
            List[str]: A list of file paths where each page image was saved.
        """
        images = convert_from_path(self.pdf_path, dpi=self.dpi)

        saved_paths = []
        for i, img in enumerate(images, start=1):
            output_filename = f"page_{i}.jpg"
            output_path = os.path.join(self.output_folder, output_filename)
            img.save(output_path, format="JPEG")
            saved_paths.append(output_path)

        return saved_paths


class PDFBatchConverter:
    """
    A class to process multiple PDF files in a directory,
    converting each PDF into separate images (page_1.jpg, page_2.jpg, etc.)
    inside 'images/<pdf_name>/'
    """

    def __init__(self, pdfs_dir: str = "pdfs", output_root: str = "images", dpi: int = 300):
        """
        Args:
            pdfs_dir (str): Directory containing multiple PDF files.
            output_root (str): Root directory where each PDF's images will be stored.
            dpi (int): Resolution for converting PDFs to images.
        """
        self.pdfs_dir = pdfs_dir
        self.output_root = output_root
        self.dpi = dpi

        if not os.path.isdir(self.pdfs_dir):
            raise NotADirectoryError(f"The directory '{self.pdfs_dir}' does not exist.")

    def convert_all_pdfs(self):
        """
        Iterate over each .pdf in 'pdfs_dir', convert them all, and
        print out the saved image paths for each PDF.
        """
        for filename in os.listdir(self.pdfs_dir):
            # Process only PDF files
            if filename.lower().endswith(".pdf"):
                pdf_path = os.path.join(self.pdfs_dir, filename)

                # Instantiate a converter for this PDF
                converter = PDFToImageConverter(
                    pdf_path=pdf_path,
                    output_root=self.output_root,
                    dpi=self.dpi
                )

                # Convert all pages
                image_paths = converter.convert_all_pages()

                # Print out the saved image paths
                for path in image_paths:
                    print(f"Saved image: {path}")