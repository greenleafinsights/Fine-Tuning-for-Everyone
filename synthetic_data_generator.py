#!/usr/bin/env python3

import os
from modules.PDFBatchConverter import PDFBatchConverter
from modules.DocumentLayoutAndOCR import DocumentLayoutAndOCR
from modules.PageBasedQuestionGenerator import PageBasedQuestionGenerator
from modules.SyntheticQADatasetWithRAG import SyntheticQADatasetWithRAG

def main():
    # --- Directories & paths ---
    pdfs_dir               = "input/pdfs"                # Where the input PDFs live
    imgs_dir               = "input/images"              # Where PDF pages (as images) will be stored
    processed_res_dir      = "input/processed_results"    # Where YOLO+OCR outputs are saved
    layout_model_path      = "models/yolov11s-doclaynet.pt"
    language_model         = "llama3.2"                  # Ollama language model for question/answer calls
    embedding_model        = "granite-embedding"         # Ollama embedding model
    question_generation_prompt = "prompt/question_generation.txt"
    answer_generation_prompt   = "prompt/answer_generation.txt"
    question_json          = "output/questions.jsonl"    # Where generated questions are saved
    qa_json               = "output/Q_and_A.jsonl"       # Final Q&A pairs output

    # --- 1) Convert PDFs to images ---
    batch_converter = PDFBatchConverter(
        pdfs_dir=pdfs_dir,
        output_root=imgs_dir,
        dpi=300
    )
    batch_converter.convert_all_pdfs()
    print("[INFO] Finished converting PDFs to images.\n")

    # --- 2) Document layout & OCR ---
    processor = DocumentLayoutAndOCR(
        model_path=layout_model_path,
        images_root=imgs_dir,
        output_root=processed_res_dir,
        max_ocr_workers=4    # Adjust based on CPU cores
        # tesseract_cmd="/usr/bin/tesseract",  # If Tesseract is not on your PATH
    )
    processor.process_all_documents()
    print("[INFO] Finished YOLO layout + OCR for all images.\n")

    # --- 3) Question Generation from processed results ---
    generator = PageBasedQuestionGenerator(
        processed_results_dir=processed_res_dir,
        prompt_file=question_generation_prompt,
        output_path=question_json,
        model_name=language_model,  # The Ollama language model
        tokens_per_chunk=1000,      # How many tokens per chunk in question generation
        max_workers=2               # Parallel threads for question generation
    )
    generator.run_pipeline()
    print(f"[INFO] Generated questions saved to: {question_json}\n")

    # --- 4) RAG-based Q&A ---
    rag_dataset = SyntheticQADatasetWithRAG(
        questions_file=question_json,
        processed_results_dir=processed_res_dir,
        output_file=qa_json,
        embedding_model=embedding_model,
        model=language_model,
        threshold_sim=0.9,
        top_k=3,
        prompt_file=answer_generation_prompt,
        max_workers=2,     # Parallel threads for final Q&A
        chunk_size=1000    # Character chunk size for doc embeddings
    )
    rag_dataset.run_pipeline()
    print(f"[INFO] Final Q&A pairs written to: {qa_json}\n")

if __name__ == "__main__":
    # Make sure all directories exist
    os.makedirs("input/images", exist_ok=True)
    os.makedirs("input/processed_results", exist_ok=True)
    os.makedirs("output", exist_ok=True)

    main()