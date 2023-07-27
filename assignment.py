import logging
import os
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List, Optional

import faiss
import torch
from datasets import Features, Sequence, Value, load_dataset

from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser,
    RagRetriever,
    RagSequenceForGeneration,
    RagTokenizer,
)

logger = logging.getLogger(__name__)
torch.set_grad_enabled(False)
device = "cuda" if torch.cuda.is_available() else "cpu"


def split_document_text(text: str, n=100, character=" ") -> List[str]:
    """Split the text every ``n``-th occurrence of ``character``"""
    text_parts = text.split(character)
    return [character.join(text_parts[i : i + n]).strip() for i in range(0, len(text_parts), n)]


def split_documents_into_passages(documents: dict) -> dict:
    """Split documents into passages"""
    passage_titles, passage_texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_document_text(text):
                passage_titles.append(title if title is not None else "")
                passage_texts.append(passage)
    return {"passage_title": passage_titles, "passage_text": passage_texts}


def compute_dpr_embeddings(documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast) -> dict:
    """Compute the DPR embeddings of document passages"""
    input_ids = ctx_tokenizer(
        documents["passage_title"], documents["passage_text"], truncation=True, padding="longest", return_tensors="pt"
    )["input_ids"]
    embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
    return {"passage_embeddings": embeddings.detach().cpu().numpy()}


def main(
    rag_example_args: "RagExampleArguments",
    processing_args: "ProcessingArguments",
    index_hnsw_args: "IndexHnswArguments",
    confidence_threshold: float = 0.5,  # Add a default confidence threshold (you can change this value)
):
    ######################################
    logger.info("Step 1 - Create the dataset")
    ######################################

    # ... (rest of the code remains the same)

    ######################################
    logger.info("Step 2 - Index the dataset")
    ######################################

    # ... (rest of the code remains the same)

    ######################################
    logger.info("Step 3 - Load RAG")
    ######################################

    # ... (rest of the code remains the same)

    ######################################
    logger.info("Step 4 - Have fun")
    ######################################

    question = rag_example_args.question or "What does Moses' rod turn into ?"
    input_ids = tokenizer.question_encoder(question, return_tensors="pt")["input_ids"]
    generated = model.generate(input_ids)
    generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
    logger.info("Q: " + question)

    # Get the logits of the generated answer
    with torch.no_grad():
        answer_logits = model(input_ids=input_ids).logits

    # Compute the probability of the generated answer using softmax
    answer_probabilities = torch.softmax(answer_logits, dim=-1)
    answer_confidence = answer_probabilities[0, generated[0]]

    # Check if the confidence exceeds the threshold
    if answer_confidence >= confidence_threshold:
        logger.info("A: " + generated_string)
    else:
        logger.info("A: Model confidence is below the threshold. The answer may be unreliable.")
        logger.info(f"Confidence: {answer_confidence:.4f}")


# ... (rest of the code remains the same)

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    logger.setLevel(logging.INFO)

    parser = HfArgumentParser((RagExampleArguments, ProcessingArguments, IndexHnswArguments))
    rag_example_args, processing_args, index_hnsw_args = parser.parse_args_into_dataclasses()
    with TemporaryDirectory() as tmp_dir:
        rag_example_args.output_dir = rag_example_args.output_dir or tmp_dir

        # Add the confidence threshold as an argument
        parser.add_argument("--confidence_threshold", type=float, default=0.5, help="Confidence threshold for answer acceptance.")
        args = parser.parse_args()
        
        main(rag_example_args, processing_args, index_hnsw_args, confidence_threshold=args.confidence_threshold)
