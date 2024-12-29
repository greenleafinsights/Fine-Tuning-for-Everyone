import os
import re
import json
import math
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

# If using Ollama for embeddings & chat
from ollama import embeddings, chat, ChatResponse


class SyntheticQADatasetWithRAG:
    """
    1) Parallel embed questions with error handling & deduplicate them.
    2) Parallel embed doc chunks from processed_results/.
    3) For each question, retrieve top_k docs, build a RAG prompt, call Ollama,
       and write the conversation to output.
    """

    def __init__(
        self,
        questions_file: str = "output/questions.jsonl",
        processed_results_dir: str = "processed_results",
        output_file: str = "output/questions_and_answer.jsonl",
        embedding_model: str = "granite-embedding",
        model: str = "llama3.2",
        threshold_sim: float = 0.9,
        top_k: int = 5,
        prompt_file: str = "prompt/answer_generation.txt",
        max_workers: int = 4,
        chunk_size: int = 1000  # max text chunk length for doc embedding
    ):
        """
        Args:
            questions_file (str): JSONL with questions, one per line: {"questions": "Q?"}
            processed_results_dir (str): Where your doc .json files live
            output_file (str): final output in GPT-like format
            embedding_model (str): Ollama model for embedding
            model (str): Ollama model for final answer (e.g. llama3.2)
            threshold_sim (float): dedup threshold for questions
            top_k (int): how many docs to retrieve for RAG
            prompt_file (str): path to text template with <question> and <retrieved_docs>
            max_workers (int): concurrency for parallel embedding & RAG
            chunk_size (int): naive character limit per doc chunk for embedding
        """
        self.questions_file = questions_file
        self.processed_results_dir = processed_results_dir
        self.output_file = output_file
        self.embedding_model = embedding_model
        self.model = model
        self.threshold_sim = threshold_sim
        self.top_k = top_k
        self.prompt_file = prompt_file
        self.max_workers = max_workers
        self.chunk_size = chunk_size

        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

        # Internal data
        self.questions: List[str] = []
        self.question_embeddings: List[List[float]] = []
        self.unique_questions: List[str] = []

        self.doc_texts: List[str] = []
        self.doc_embeddings: List[List[float]] = []

    # --------------------------------------------------------------------------
    # 1) Public pipeline entry
    # --------------------------------------------------------------------------
    def run_pipeline(self):
        """
        Main pipeline:
          A) Load & parallel-embed questions -> deduplicate
          B) Create doc embeddings
          C) RAG + save
        """
        # A) Load & embed questions
        self._load_and_embed_questions()
        self._deduplicate_questions()

        # B) Create doc embeddings
        self._create_doc_embeddings()

        # C) RAG + save
        self._rag_and_save()

    # --------------------------------------------------------------------------
    # 2) Load & parallel-embed questions
    # --------------------------------------------------------------------------
    def _load_and_embed_questions(self):
        if not os.path.isfile(self.questions_file):
            raise FileNotFoundError(f"{self.questions_file} not found.")

        raw_questions = []
        with open(self.questions_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                q_str = data.get("questions") or data.get("question")
                if isinstance(q_str, dict):
                    q_str = q_str.get("question", "")
                if q_str and isinstance(q_str, str):
                    raw_questions.append(q_str.strip())

        # Parallel embed
        (self.questions, self.question_embeddings) = self._parallel_embed_texts(raw_questions)

    # --------------------------------------------------------------------------
    # 3) Deduplicate questions
    # --------------------------------------------------------------------------
    def _deduplicate_questions(self):
        used_indices = set()
        N = len(self.questions)

        for i in range(N):
            if i in used_indices:
                continue
            emb_i = self.question_embeddings[i]
            q_i = self.questions[i]

            for j in range(i+1, N):
                if j in used_indices:
                    continue
                emb_j = self.question_embeddings[j]
                sim = self._cosine_similarity(emb_i, emb_j)
                if sim > self.threshold_sim:
                    used_indices.add(j)

            used_indices.add(i)
            self.unique_questions.append(q_i)

    # --------------------------------------------------------------------------
    # 4) Create doc embeddings in memory
    # --------------------------------------------------------------------------
    def _create_doc_embeddings(self):
        doc_chunks = []

        for root, _, files in os.walk(self.processed_results_dir):
            for file in files:
                if file.endswith(".json"):
                    path = os.path.join(root, file)
                    with open(path, "r", encoding="utf-8") as ff:
                        data = json.load(ff)

                    # each label => multiple lines
                    lines = []
                    for _, textlist in data.items():
                        lines.extend(textlist)
                    doc_text = "\n".join(lines).strip()

                    # chunk it further if needed
                    splitted = self._split_text(doc_text, self.chunk_size)
                    doc_chunks.extend(splitted)

        if doc_chunks:
            (self.doc_texts, self.doc_embeddings) = self._parallel_embed_texts(doc_chunks)
        else:
            print("No doc chunks found in processed_results/.")

    # --------------------------------------------------------------------------
    # 5) RAG + save
    # --------------------------------------------------------------------------
    def _rag_and_save(self):
        """
        For each question (in parallel):
         1) embed query
         2) retrieve top_k
         3) build prompt
         4) call Ollama
         5) write lines
        """
        with open(self.output_file, "w", encoding="utf-8") as outf:

            def process_question(q_str: str):
                # embed question
                q_emb = self._embed_text_ollama(q_str)
                if q_emb is None:
                    return [
                        {"role": "user", "content": q_str},
                        {"role": "assistant", "content": "[No answer: embedding failed]"}
                    ]

                # retrieve top_k docs
                top_docs = self._retrieve_docs(q_emb)

                # build prompt
                prompt = self._build_rag_prompt(q_str, top_docs)

                # call Ollama
                try:
                    resp = chat(model=self.model, messages=[{"role": "user", "content": prompt}])
                    answer = resp.message.content.strip()
                except Exception as e:
                    answer = f"[Error calling Ollama: {e}]"

                user_obj = {"role": "user", "content": q_str}
                answer_obj = {"role": "assistant", "content": answer}
                return [user_obj, answer_obj]

            # parallel RAG
            results = []
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(process_question, q) for q in self.unique_questions]
                for fut in tqdm(as_completed(futures), total=len(futures), desc="RAG for Qs"):
                    qa_objs = fut.result()  # list of 2 dicts
                    results.extend(qa_objs)

            # write everything
            for obj in results:
                outf.write(json.dumps(obj, ensure_ascii=False) + "\n")

    # --------------------------------------------------------------------------
    # 6) Utility: parallel embed
    # --------------------------------------------------------------------------
    def _parallel_embed_texts(self, texts: List[str]) -> Tuple[List[str], List[List[float]]]:
        """
        Embeds multiple strings in parallel, returning (texts, embeddings) in same order.
        If embedding fails for a text, we skip it or return None. Let's skip them entirely.
        """
        def embed_worker(idx: int, txt: str):
            emb = self._embed_text_ollama(txt)
            return (idx, txt, emb)

        out_texts = []
        out_embs = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(embed_worker, i, t) for i, t in enumerate(texts)]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
                i, t, emb = fut.result()
                if emb is not None:
                    out_texts.append((i, t))
                    out_embs.append((i, emb))

        # sort by original idx
        out_texts.sort(key=lambda x: x[0])
        out_embs.sort(key=lambda x: x[0])

        final_texts = [pair[1] for pair in out_texts]
        final_embs = [pair[1] for pair in out_embs]

        return final_texts, final_embs

    # --------------------------------------------------------------------------
    # 7) Utility: embed text
    # --------------------------------------------------------------------------
    def _embed_text_ollama(self, text: str) -> Optional[List[float]]:
        """
        Attempt to embed 'text' with Ollama. Return None if error or empty response.
        """
        try:
            resp = embeddings(model=self.embedding_model, prompt=text)
            if "embedding" not in resp:
                return None
            return resp["embedding"]
        except Exception as e:
            print(f"[Warning] Embedding failed (len={len(text)}): {e}")
            return None

    # --------------------------------------------------------------------------
    # 8) Utility: retrieve docs
    # --------------------------------------------------------------------------
    def _retrieve_docs(self, query_emb: List[float]) -> List[str]:
        sims = []
        for idx, doc_emb in enumerate(self.doc_embeddings):
            sim = self._cosine_similarity(query_emb, doc_emb)
            sims.append((idx, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        top_indices = [s[0] for s in sims[:self.top_k]]
        return [self.doc_texts[i] for i in top_indices]

    # --------------------------------------------------------------------------
    # 9) Utility: build RAG prompt
    # --------------------------------------------------------------------------
    def _build_rag_prompt(self, question: str, docs: List[str]) -> str:
        if not os.path.isfile(self.prompt_file):
            raise FileNotFoundError(f"{self.prompt_file} not found.")

        with open(self.prompt_file, "r", encoding="utf-8") as pf:
            template = pf.read()

        docs_combined = "\n\n".join(f"- {d}" for d in docs)
        filled_prompt = template.replace("<question>", question)
        filled_prompt = filled_prompt.replace("<retrieved_docs>", docs_combined)
        return filled_prompt

    # --------------------------------------------------------------------------
    # 10) Utility: split large text
    # --------------------------------------------------------------------------
    def _split_text(self, text: str, max_chars: int) -> List[str]:
        """
        Naively splits 'text' every 'max_chars' characters.
        If some chunk is still too large for Ollama, it may fail.
        You can reduce max_chars or handle further.
        """
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end]
            chunks.append(chunk)
            start = end
        return chunks

    # --------------------------------------------------------------------------
    # 11) Utility: cos sim
    # --------------------------------------------------------------------------
    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        arr1 = np.array(v1, dtype=np.float32)
        arr2 = np.array(v2, dtype=np.float32)
        dot = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
