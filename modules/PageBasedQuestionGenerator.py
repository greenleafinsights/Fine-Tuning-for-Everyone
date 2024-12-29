import os
import re
import json
import math
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from ollama import chat, ChatResponse


class PageBasedQuestionGenerator:
    """
    1. Recursively scans 'processed_results/' for JSON files in subfolders like:
         processed_results/<some_pdf>/page_1/page_1.json
    2. Treats each page as a single unit to count tokens.
       - If adding the next page to the current chunk <= 2000 tokens, combine them.
       - Otherwise, start a new chunk.
       - If a single page > 2000 tokens, it becomes its own chunk.
    3. Calls Ollama to generate questions using a <content> placeholder prompt,
       in parallel for each chunk to speed up processing.
    4. Extracts the 'questions' array from the JSON in the response and saves them to .jsonl.
    """

    def __init__(
        self,
        processed_results_dir: str = "processed_results",
        prompt_file: str = "prompt/question_generation.txt",
        output_path: str = "output/questions.jsonl",
        model_name: str = "llama3.2",
        tokens_per_chunk: int = 2000,
        max_workers: int = 4
    ):
        """
        Args:
            processed_results_dir (str): Base dir with subfolders for each PDF -> page_{idx}/page_{idx}.json
            prompt_file (str): Template file with <content> placeholder.
            output_path (str): Where to save final .jsonl of questions.
            model_name (str): Ollama model name (e.g. "llama3.2:1b").
            tokens_per_chunk (int): Max chunk size in tokens (pages get combined up to this limit).
            max_workers (int): How many threads to use for parallel question generation.
        """
        self.processed_results_dir = processed_results_dir
        self.prompt_file = prompt_file
        self.output_path = output_path
        self.model_name = model_name
        self.tokens_per_chunk = tokens_per_chunk
        self.max_workers = max_workers

        # Internal data
        self.pages_texts: List[str] = []  # text of each page (one per page)
        self.chunks: List[str] = []       # combined pages
        self.questions: List[str] = []    # final question list

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

    def run_pipeline(self):
        """
        1) Gather page-based text
        2) Create chunks (page-based)
        3) Generate questions (parallel)
        4) Save
        """
        self._collect_pages_text()
        self._create_page_chunks()
        self._generate_questions_for_chunks()
        self._save_questions()

    # -------------------------------------------------------------------------
    # Step 1: Collect page text
    # -------------------------------------------------------------------------
    def _collect_pages_text(self):
        """
        Recursively walk `processed_results_dir` to find page_{idx}.json files.
        For each, gather combined text of that page's JSON (labels: e.g. "Text", "Footer", etc.)
        and store as a single string in `self.pages_texts`.
        The pages_texts list is in ascending order by the "page_X" pattern if possible.
        """
        page_data = []  # store (page_idx, combined_text)

        for root, dirs, files in os.walk(self.processed_results_dir):
            for file in files:
                if file.endswith(".json"):
                    match = re.match(r"page_(\d+)\.json$", file)
                    if match:
                        page_idx = int(match.group(1))
                        json_path = os.path.join(root, file)
                        combined_text = self._merge_json_text(json_path)
                        page_data.append((page_idx, combined_text))

        # Sort by page_idx
        page_data.sort(key=lambda x: x[0])
        # Keep texts in ascending page order
        self.pages_texts = [text for (_, text) in page_data]

    def _merge_json_text(self, json_path: str) -> str:
        """
        Read a single page_{idx}.json, combine text from all labels into one string.
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)  # e.g. {"Text": [...], "Footer": [...], ...}
        lines = []
        for label, text_list in data.items():
            for txt in text_list:
                lines.append(txt)
        return "\n".join(lines)

    # -------------------------------------------------------------------------
    # Step 2: Create page chunks
    # -------------------------------------------------------------------------
    def _create_page_chunks(self):
        """
        Combine page texts so that each chunk is <= tokens_per_chunk.
        If a single page > tokens_per_chunk, it alone forms one chunk.
        """
        def count_tokens(text: str) -> int:
            # Naive whitespace token counting
            return len(text.split())

        current_chunk_text = ""
        current_chunk_tokens = 0

        for page_text in self.pages_texts:
            page_tokens = count_tokens(page_text)

            # If page alone exceeds the chunk limit, it's its own chunk
            if page_tokens > self.tokens_per_chunk:
                # First, if we have anything in the current chunk, finalize it
                if current_chunk_tokens > 0:
                    self.chunks.append(current_chunk_text)
                # Then, the entire page_text is one chunk
                self.chunks.append(page_text)
                # Reset
                current_chunk_text = ""
                current_chunk_tokens = 0
            else:
                # If we can add this page to current chunk
                if current_chunk_tokens + page_tokens <= self.tokens_per_chunk:
                    if current_chunk_tokens == 0:
                        current_chunk_text = page_text
                        current_chunk_tokens = page_tokens
                    else:
                        current_chunk_text += "\n" + page_text
                        current_chunk_tokens += page_tokens
                else:
                    # finalize the current chunk
                    if current_chunk_tokens > 0:
                        self.chunks.append(current_chunk_text)
                    # start a new chunk
                    current_chunk_text = page_text
                    current_chunk_tokens = page_tokens

        # leftover chunk
        if current_chunk_tokens > 0:
            self.chunks.append(current_chunk_text)

    # -------------------------------------------------------------------------
    # Step 3: Generate questions from each chunk
    # -------------------------------------------------------------------------
    def _generate_questions_for_chunks(self):
        """
        For each chunk, load prompt, call Ollama in parallel, parse out 'questions'.
        """
        # 1) Load prompt template
        with open(self.prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()

        # 2) Parallel calls for each chunk
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for idx, chunk_text in enumerate(self.chunks, start=1):
                # Build a prompt for each chunk
                prompt = prompt_template.replace("<content>", chunk_text.strip())
                futures.append(executor.submit(self._call_ollama_for_questions, idx, prompt))

            # 3) Collect results
            for fut in as_completed(futures):
                chunk_idx, questions_or_err = fut.result()
                if isinstance(questions_or_err, list):
                    # It's the list of questions
                    self.questions.extend(questions_or_err)
                    print(f"Chunk_{chunk_idx} processed, found {len(questions_or_err)} questions.")
                else:
                    # It's an error or empty
                    print(f"Chunk_{chunk_idx} had an issue: {questions_or_err}")

    def _call_ollama_for_questions(self, chunk_idx: int, prompt: str):
        """
        Calls Ollama with the given prompt, attempts to parse JSON block for 'questions'.
        Returns (chunk_idx, list_of_questions or error_string).
        """
        try:
            response = chat(model=self.model_name, messages=[{"role": "user", "content": prompt}])
        except Exception as e:
            return (chunk_idx, f"[Error calling Ollama: {e}]")

        raw_output = response.message.content
        # Attempt to find a JSON block
        match = re.search(r"\{.*\}", raw_output, flags=re.DOTALL)
        if not match:
            return (chunk_idx, "[Info] No JSON found in chunk output.")

        json_str = match.group(0)
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            return (chunk_idx, "[Warning] Could not parse JSON from chunk output.")

        # Now, parse out 'questions'. If you expect { "q_1": "xxx", ...} you can do:
        questions_found = []
        for val in parsed.values():
            if isinstance(val, str):
                questions_found.append(val)
            elif isinstance(val, list):
                questions_found.extend(val)
            # etc. adapt to your format

        return (chunk_idx, questions_found)

    # -------------------------------------------------------------------------
    # Step 4: Save questions
    # -------------------------------------------------------------------------
    def _save_questions(self):
        with open(self.output_path, "w", encoding="utf-8") as out_f:
            for q in self.questions:
                record = {"questions": q}
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
