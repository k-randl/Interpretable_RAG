# %%
# This is a cell marker, often used in interactive environments like
# VSCode or Jupyter Notebooks to define a code cell.

# --- 1. IMPORTS ---
# Import necessary libraries
import argparse  # For parsing command-line arguments
import os  # For operating system interactions (like file paths)
import pickle  # For loading/saving Python objects (the .pkl files)
from dataclasses import dataclass  # For creating simple data classes
from pathlib import Path  # For object-oriented file system paths
from typing import Iterable, List, Optional, Sequence  # For type hinting

import numpy as np  # For numerical operations (though not explicitly used here, pandas/torch depend on it)
import pandas as pd  # For creating and manipulating dataframes (e.g., for the final CSV)
import torch  # PyTorch library, used for tensor operations (handling the gradient scores)
from tqdm import tqdm  # For displaying progress bars during loops
import textwrap  # For formatting qualitative text snippets


# --- 2. GLOBAL CONSTANTS ---

# A set of special tokens (like padding, separator, start/end of sequence)
# that should be ignored when extracting importance scores.
SPECIAL_TOKENS = {"<pad>", "sep", "<s>", "</s>"}

# A set of valid keys for different types of gradient/attribution scores
# found in the pickle files.
GRAD_KEYS = {"grad", "aGrad", "gradIn", "intGrad"}


# --- 3. DATA STRUCTURE ---

# Defines a simple data class to store information for a single token's importance.
# This makes it easy to collect and manage the data before saving it.
@dataclass
class TokenImportance:
    query_id: int  # The unique identifier for the query
    segment: str  # The type of text: 'query' or 'context'
    segment_index: int  # 0 for 'query', or the index (0, 1, 2...) for 'context'
    grad_type: str  # The attribution method used (e.g., 'gradIn')
    token_position: int  # The token's index in its sequence
    token: str  # The raw token string (e.g., " sub")
    token_display: str  # A cleaned-up version for display (e.g., "sub")
    score: float  # The raw attribution score (can be positive or negative)
    abs_score: float  # The absolute value of the score, used for ranking
    text: str  # The full, detokenized text of the segment this token belongs to


# --- 4. ARGUMENT PARSING ---

# This function defines and parses the command-line arguments
# that the script accepts when run from the terminal.
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract top tokens by attribution score from retrieval importance pickles."
    )
    # Argument for the input directory containing the .pkl files
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("/home/anonymized_user_1/raid/anonymized_user_2/code/Interpretable_RAG/results/retrieval/efra_06_11"),
        help="Directory containing query_*.pkl files.",
    )
    # Argument to specify which gradient type to use
    parser.add_argument(
        "--grad-type",
        type=str,
        choices=sorted(GRAD_KEYS),  # Restrict choice to the defined keys
        default="gradIn",
        help="Which attribution tensor to use.",
    )
    # Argument to specify how many top tokens to save per segment
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of tokens to keep per sequence.",
    )
    # Optional argument to limit the number of query files processed (for testing/debugging)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of queries to process.",
    )
    # Argument for the output CSV file path
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("importance_top_tokens.csv"),
        help="Path to save the aggregated token importance CSV.",
    )
    # Optional directory for saving plots; if not provided, plotting is skipped
    parser.add_argument(
        "--plot-dir",
        type=Path,
        default=None,
        help="If provided, save per-sequence bar plots of token importances here.",
    )
    # How many tokens to show in each plot
    parser.add_argument(
        "--plot-top-k",
        type=int,
        default=20,
        help="Number of tokens to visualize per sequence when plotting.",
    )
    parser.add_argument(
        "--combine-query-context",
        action="store_true",
        help="When plotting, also create joint charts comparing query and context tokens.",
    )
    parser.add_argument(
        "--plot-side-by-side",
        action="store_true",
        help="When plotting, add figures with query and context tokens side by side.",
    )
    parser.add_argument(
        "--qual-query-id",
        type=int,
        default=None,
        help="If set, produce a qualitative report for the given query id.",
    )
    parser.add_argument(
        "--qual-top-contexts",
        type=int,
        default=3,
        help="Number of top-ranked contexts to include in the qualitative example.",
    )
    parser.add_argument(
        "--qual-output",
        type=Path,
        default=Path("qualitative_report.md"),
        help="Destination file (Markdown) for the qualitative example summary.",
    )
    parser.add_argument(
        "--overall-stats",
        action="store_true",
        help="Compute dataset-level overlap metrics between queries and contexts.",
    )
    parser.add_argument(
        "--overall-stats-csv",
        type=Path,
        default=Path("importance_overlap_stats.csv"),
        help="Where to store per-query overlap statistics when --overall-stats is enabled.",
    )
    return parser.parse_args()


# --- 5. HELPER FUNCTIONS (File & Text Processing) ---

# Finds all 'query_*.pkl' files in the specified directory
# and sorts them numerically by the query ID.
def list_query_pickles(results_dir: Path) -> List[Path]:
    # Use a list comprehension to find matching files
    files = [p for p in results_dir.iterdir() if p.suffix == ".pkl" and p.name.startswith("query_")]
    # Sort files based on the integer ID (the part after 'query_')
    files.sort(key=lambda p: int(p.stem.split("_")[-1]))
    return files


# Converts a sequence of tokens back into a clean, readable string.
def detokenize(tokens: Sequence[str]) -> str:
    text = "".join(tokens)
    # Handles BPE/SentencePiece subword tokenization (where '▁' indicates a new word)
    text = text.replace("▁", " ")
    # Remove special sequence tokens
    text = text.replace("<s>", "").replace("</s>", "")
    # Normalize whitespace (e.g., multiple spaces become one)
    text = " ".join(text.split())
    return text.strip()  # Remove leading/trailing whitespace


# Cleans a single token for display.
def clean_token(token: str) -> str:
    # Replaces the BPE/SentencePiece space prefix with a normal space and strips whitespace
    return token.replace("▁", " ").strip()


# This function aggregates multi-dimensional attribution tensors into a
# 1D tensor (one score per token).
# The logic depends on the 'grad_type' because different attribution methods
# might produce tensors with different shapes (e.g., [seq_len, hidden_dim]).
def aggregate_token_scores(tensor: torch.Tensor, grad_type: str) -> torch.Tensor:
    if grad_type == "grad":
        # Handle 3D or 2D tensors by averaging over the last dimension (hidden_dim)
        if tensor.dim() == 3:
            tensor = tensor.mean(dim=-1).squeeze(0)  # [B, S, H] -> [S]
        elif tensor.dim() == 2:
            tensor = tensor.mean(dim=-1)  # [S, H] -> [S]
    elif grad_type == "aGrad":
        # Handle 3D or 2D tensors with different averaging
        if tensor.dim() == 3:
            tensor = tensor.mean(dim=1).squeeze(0)  # [B, S, H] -> [H] or [B, X, Y] -> [B, Y] ? (Check logic)
        elif tensor.dim() == 2:
            tensor = tensor.mean(dim=0)  # [S, H] -> [H]
    else:
        # For other types (like 'gradIn'), assume it's already [1, S] or [S]
        # and just remove extra dimensions.
        tensor = tensor.squeeze()
    return tensor


# --- 6. CORE LOGIC FUNCTIONS ---

# Extracts the top-k most important tokens from a single sequence (query or context).
def extract_top_tokens(
    query_id: int,
    segment: str,
    segment_index: int,
    grad_type: str,
    tokens: Sequence[str],
    scores: torch.Tensor,
    top_k: int,
) -> Iterable[TokenImportance]:
    # Sanity check: ensure tokens and scores have the same length
    assert len(tokens) == scores.shape[0], (
        f"Token length mismatch for query {query_id}, segment {segment}, "
        f"{len(tokens)} tokens vs {scores.shape[0]} scores."
    )

    full_text = detokenize(tokens)  # Detokenize the full sequence once
    entries = []
    # Iterate over each token and its corresponding score
    for idx, (token, score) in enumerate(zip(tokens, scores.tolist())):
        # Skip special tokens
        if token in SPECIAL_TOKENS:
            continue
        
        display = clean_token(token)
        # Skip empty tokens (e.g., if a cleaned token is just an empty string)
        if not display:
            continue
        
        # Create a TokenImportance object for this token
        entries.append(
            TokenImportance(
                query_id=query_id,
                segment=segment,
                segment_index=segment_index,
                grad_type=grad_type,
                token_position=idx,
                token=token,
                token_display=display,
                score=float(score),
                abs_score=float(abs(score)),  # Store abs_score for sorting
                text=full_text,
            )
        )

    # Sort the tokens by their absolute score in descending order (most important first)
    entries.sort(key=lambda x: x.abs_score, reverse=True)
    # Return only the top K tokens
    return entries[:top_k]


# Creates and saves a bar plot showing the most influential tokens for a sequence.
def plot_token_scores(
    query_id: int,
    segment: str,
    segment_index: int,
    grad_type: str,
    tokens: Sequence[str],
    scores: torch.Tensor,
    plot_dir: Path,
    plot_top_k: int,
) -> None:
    # Import matplotlib lazily so the script can run without it when plotting is disabled.
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "Matplotlib is required for plotting. Install a version compatible with your NumPy "
            "setup or run the script without --plot-dir."
        ) from exc

    # Collect the tokens with their scores and positions
    items = _gather_top_items(tokens, scores, plot_top_k, display_order="abs_desc")
    if not items:
        return  # Nothing to plot

    # Reverse so the strongest token appears at the bottom of the bar chart (more readable)
    items = list(reversed(items))

    labels = [f"{label} [{pos}]" for label, _, pos in items]
    values = [score for _, score, _ in items]
    colors = ["#1f77b4" if value >= 0 else "#d62728" for value in values]

    plt.figure(figsize=(10, max(2, 0.5 * len(items) + 1)))
    plt.barh(range(len(items)), values, color=colors)
    plt.yticks(range(len(items)), labels)
    plt.axvline(0, color="black", linewidth=0.8)
    descriptor = f"{segment} {segment_index}" if segment == "context" else segment
    plt.title(f"Query {query_id} • {descriptor} • {grad_type}")
    plt.xlabel("Attribution score")
    plt.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = f"q{query_id:04d}_{segment}_{segment_index}_{grad_type}.png"
    plt.savefig(plot_dir / filename, dpi=200)
    plt.close()


def _gather_top_items(
    tokens: Sequence[str],
    scores: torch.Tensor,
    top_k: int,
    display_order: str = "abs_desc",
) -> List[tuple[str, float, int]]:
    items: List[tuple[str, float, int]] = []
    for position, (token, score) in enumerate(zip(tokens, scores.tolist())):
        if token in SPECIAL_TOKENS:
            continue
        label = clean_token(token)
        if not label:
            continue
        items.append((label, float(score), position))
    if not items:
        return []

    items_sorted = sorted(items, key=lambda x: abs(x[1]), reverse=True)
    if top_k > 0:
        items_sorted = items_sorted[:top_k]

    if display_order == "position":
        return sorted(items_sorted, key=lambda x: x[2])
    if display_order == "abs_desc":
        return sorted(items_sorted, key=lambda x: abs(x[1]), reverse=True)
    return items_sorted


def _build_token_entries(tokens: Sequence[str], scores: Sequence[float]) -> List[tuple[int, str, float, float]]:
    entries: List[tuple[int, str, float, float]] = []
    for idx, (token, score) in enumerate(zip(tokens, scores)):
        if token in SPECIAL_TOKENS:
            continue
        clean = clean_token(token)
        if not clean:
            continue
        score_f = float(score)
        entries.append((idx, clean, score_f, abs(score_f)))
    return entries


def plot_joint_token_scores(
    query_id: int,
    grad_type: str,
    context_index: int,
    tokens_query: Sequence[str],
    scores_query: torch.Tensor,
    tokens_context: Sequence[str],
    scores_context: torch.Tensor,
    plot_dir: Path,
    plot_top_k: int,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive guard
        raise RuntimeError(
            "Matplotlib is required for plotting joint query/context charts. "
            "Install a compatible version or run without --combine-query-context."
        ) from exc

    query_items = _gather_top_items(tokens_query, scores_query, plot_top_k)
    context_items = _gather_top_items(tokens_context, scores_context, plot_top_k)
    if not query_items and not context_items:
        return

    combined = {}
    for label, score, position in query_items:
        combined[label] = {
            "score_query": score,
            "pos_query": position,
            "score_context": 0.0,
            "pos_context": None,
        }
    for label, score, position in context_items:
        entry = combined.setdefault(
            label,
            {
                "score_query": 0.0,
                "pos_query": None,
                "score_context": 0.0,
                "pos_context": None,
            },
        )
        entry["score_context"] = score
        entry["pos_context"] = position

    ordering = sorted(
        combined.items(),
        key=lambda kv: max(abs(kv[1]["score_query"]), abs(kv[1]["score_context"])),
        reverse=True,
    )[:plot_top_k]

    labels = []
    query_scores = []
    context_scores = []
    order_positions = np.arange(len(ordering))
    for label, info in ordering:
        pos_parts = []
        if info["pos_query"] is not None:
            pos_parts.append(f"Q@{info['pos_query']}")
        if info["pos_context"] is not None:
            pos_parts.append(f"C@{info['pos_context']}")
        suffix = " | ".join(pos_parts)
        full_label = f"{label}\n{suffix}" if suffix else label
        labels.append(full_label)
        query_scores.append(info["score_query"])
        context_scores.append(info["score_context"])

    width = 0.4
    plt.figure(figsize=(max(6, 0.6 * len(labels)), 5))
    plt.bar(order_positions - width / 2, query_scores, width, label="Query", color="#1f77b4")
    plt.bar(order_positions + width / 2, context_scores, width, label=f"Context {context_index}", color="#ff7f0e")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(order_positions, labels, rotation=45, ha="right")
    plt.ylabel("Attribution score")
    plt.title(f"Query {query_id} • Context {context_index} • {grad_type}")
    plt.legend()
    plt.tight_layout()

    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = f"q{query_id:04d}_context_{context_index}_{grad_type}_joint.png"
    plt.savefig(plot_dir / filename, dpi=200)
    plt.close()


def write_qualitative_report(
    query_id: int,
    data: dict,
    top_contexts: int,
    output_path: Path,
    grad_type: str,
) -> None:
    selected_contexts = data["contexts"][:top_contexts]
    query_entries = _build_token_entries(data["query_tokens"], data["query_scores"])
    query_entries_sorted = sorted(query_entries, key=lambda x: x[3], reverse=True)

    context_entries_list = [
        (_build_token_entries(ctx["tokens"], ctx["scores"]), ctx) for ctx in selected_contexts
    ]

    context_overlap_tokens = set()
    for entries, _ctx in context_entries_list:
        context_overlap_tokens.update(tok for _, tok, _, _ in entries)
    query_overlap_tokens = {tok for _, tok, _, _ in query_entries if tok in context_overlap_tokens}

    query_map = {}
    for idx, tok, score, abs_score in query_entries:
        if tok not in query_map or abs_score > query_map[tok][2]:
            query_map[tok] = (idx, score, abs_score)

    lines: List[str] = []
    lines.append(f"# Qualitative Analysis for Query {query_id}\n")
    lines.append("## Query\n")
    lines.append(f"**Text:** {data['query_text']}\n")
    lines.append("**Top tokens by absolute attribution:**\n")
    lines.append("| Rank | Token | Pos | Score | |Overlap|\n")
    lines.append("| --- | --- | --- | --- | --- |")
    for rank, (pos, tok, score, abs_score) in enumerate(query_entries_sorted[: min(10, len(query_entries_sorted))], start=1):
        overlap_flag = "✅" if tok in query_overlap_tokens else ""
        lines.append(f"| {rank} | {tok} | {pos} | {score:.4f} | {abs_score:.4f} | {overlap_flag} |")

    for entries, ctx in context_entries_list:
        lines.append("\n---\n")
        lines.append(f"## Context {ctx['index']} (top-{ctx['index'] + 1})\n")
        context_text = textwrap.shorten(ctx["text"], width=500, placeholder=" …")
        lines.append(f"**Excerpt:** {context_text}\n")

        context_sorted = sorted(entries, key=lambda x: x[3], reverse=True)
        lines.append("**Top context tokens by absolute attribution:**\n")
        lines.append("| Rank | Token | Pos | Score | |Overlap|\n")
        lines.append("| --- | --- | --- | --- | --- |")
        for rank, (pos, tok, score, abs_score) in enumerate(context_sorted[: min(10, len(context_sorted))], start=1):
            overlap_flag = "✅" if tok in query_overlap_tokens else ""
            lines.append(f"| {rank} | {tok} | {pos} | {score:.4f} | {abs_score:.4f} | {overlap_flag} |")

        lines.append("\n**Overlapping tokens with query:**\n")
        lines.append("| Token | Query pos | Query score | Context pos | Context score |\n")
        lines.append("| --- | --- | --- | --- | --- |")
        overlap_rows = []
        for pos, tok, score, _ in entries:
            if tok in query_map:
                q_pos, q_score, _q_abs = query_map[tok]
                overlap_rows.append((tok, q_pos, q_score, pos, score))
        if overlap_rows:
            for tok, q_pos, q_score, c_pos, c_score in sorted(
                overlap_rows, key=lambda x: abs(x[4]), reverse=True
            ):
                lines.append(
                    f"| {tok} | {q_pos} | {q_score:.4f} | {c_pos} | {c_score:.4f} |"
                )
        else:
            lines.append("| _No overlapping tokens in this context._ |  |  |  |  |")

    lines.append("\n---\n")
    lines.append(f"_Attribution scores computed with `{grad_type}`._\n")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def compute_overlap_statistics(analysis_data: dict) -> pd.DataFrame:
    records = []
    for query_id, data in analysis_data.items():
        query_entries = _build_token_entries(data["query_tokens"], data["query_scores"])
        contexts = data["contexts"]
        if not query_entries or not contexts:
            continue

        query_token_count = len(query_entries)
        query_total_abs = sum(entry[3] for entry in query_entries)
        query_abs_by_pos = {entry[0]: entry[3] for entry in query_entries}

        query_tokens_set = {entry[1] for entry in query_entries}
        query_overlap_positions = set()
        context_total_tokens = 0
        context_overlap_tokens = 0
        context_total_abs = 0.0
        context_overlap_abs = 0.0

        context_stats = []

        for ctx in contexts:
            entries = _build_token_entries(ctx["tokens"], ctx["scores"])
            if not entries:
                continue
            ctx_total_tokens = len(entries)
            ctx_total_abs = sum(entry[3] for entry in entries)
            ctx_overlap_tokens = 0
            ctx_overlap_abs = 0.0

            for pos, token, score, abs_score in entries:
                context_total_tokens += 1
                context_total_abs += abs_score
                if token in query_tokens_set:
                    context_overlap_tokens += 1
                    context_overlap_abs += abs_score
                    ctx_overlap_tokens += 1
                    ctx_overlap_abs += abs_score

                    # find query entry with same token (highest abs)
                    for q_entry in query_entries:
                        if q_entry[1] == token:
                            query_overlap_positions.add(q_entry[0])
                            break

            context_stats.append(
                {
                    "context_index": ctx["index"],
                    "context_total_tokens": ctx_total_tokens,
                    "context_overlap_tokens": ctx_overlap_tokens,
                    "context_total_abs": ctx_total_abs,
                    "context_overlap_abs": ctx_overlap_abs,
                }
            )

        if not context_stats:
            continue

        query_overlap_count = len(query_overlap_positions)
        query_overlap_abs = sum(query_abs_by_pos[pos] for pos in query_overlap_positions)

        record = {
            "query_id": query_id,
            "num_contexts": len(contexts),
            "query_token_count": query_token_count,
            "query_overlap_token_count": query_overlap_count,
            "query_overlap_token_pct": query_overlap_count / query_token_count if query_token_count else 0.0,
            "query_total_abs": query_total_abs,
            "query_overlap_abs": query_overlap_abs,
            "query_overlap_abs_pct": query_overlap_abs / query_total_abs if query_total_abs else 0.0,
            "context_total_tokens": context_total_tokens,
            "context_overlap_tokens": context_overlap_tokens,
            "context_overlap_token_pct": context_overlap_tokens / context_total_tokens if context_total_tokens else 0.0,
            "context_total_abs": context_total_abs,
            "context_overlap_abs": context_overlap_abs,
            "context_overlap_abs_pct": context_overlap_abs / context_total_abs if context_total_abs else 0.0,
        }
        records.append(record)

    return pd.DataFrame(records)


def plot_side_by_side_token_scores(
    query_id: int,
    grad_type: str,
    context_index: int,
    tokens_query: Sequence[str],
    scores_query: torch.Tensor,
    tokens_context: Sequence[str],
    scores_context: torch.Tensor,
    plot_dir: Path,
    plot_top_k: int,
    highlight_overlap: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Matplotlib is required for side-by-side plotting. Install a compatible version "
            "or omit --plot-side-by-side."
        ) from exc

    query_items = _gather_top_items(tokens_query, scores_query, plot_top_k, display_order="position")
    context_items = _gather_top_items(tokens_context, scores_context, plot_top_k, display_order="position")

    if not query_items and not context_items:
        return

    query_labels = {item[0] for item in query_items}
    context_labels = {item[0] for item in context_items}
    overlaps = query_labels & context_labels

    def _render(ax, items: List[tuple[str, float, int]], title: str) -> None:
        if not items:
            ax.axis("off")
            ax.set_title(f"{title}\n(no tokens)")
            return
        labels = [f"{label} [{pos}]" for label, _, pos in items]
        values = [score for _, score, _ in items]
        colors = []
        for (label, value, _pos) in items:
            base_color = "#1f77b4" if value >= 0 else "#d62728"
            if highlight_overlap and label in overlaps:
                # Brighten the color by blending with yellow
                base_color = "#ffd700"
            colors.append(base_color)
        y_pos = np.arange(len(items))
        ax.barh(y_pos, values, color=colors)
        ax.set_yticks(y_pos, labels)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Attribution score")

    max_len = max(len(query_items), len(context_items), 1)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, max(3, 0.45 * max_len + 1)),
        constrained_layout=True,
        sharex=False,
        sharey=False,
        gridspec_kw={"width_ratios": [0.7, 1.0]},
    )
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    _render(axes[0], query_items, "Query tokens")
    _render(axes[1], context_items, f"Context {context_index} tokens")

    plot_dir.mkdir(parents=True, exist_ok=True)
    filename = f"q{query_id:04d}_context_{context_index}_{grad_type}_side_by_side.png"
    fig.savefig(plot_dir / filename, dpi=200)
    plt.close(fig)


# Processes a single 'query_*.pkl' file.
# It loads the data, extracts top tokens for the query AND
# for all its associated contexts, and returns a single list.
def process_query_file(
    path: Path,
    grad_type: str,
    top_k: int,
    query_id: int,
    plot_dir: Optional[Path],
    plot_top_k: int,
    combine_query_context: bool,
    plot_side_by_side: bool,
    analysis_collector: Optional[dict],
) -> List[TokenImportance]:
    # Load the data from the pickle file
    with open(path, "rb") as f:
        data = pickle.load(f)

    # --- Process the Query ---
    grad_data = data[grad_type]  # Get the specific attribution data (e.g., 'gradIn')
    tokens_query = data["input"]["query"][0]  # Get query tokens
    # Aggregate the query scores from a tensor to a 1D list
    scores_query = aggregate_token_scores(grad_data["query"], grad_type)
    # Extract the top-k tokens for the query
    results: List[TokenImportance] = list(
        extract_top_tokens(query_id, "query", 0, grad_type, tokens_query, scores_query, top_k)
    )
    if analysis_collector is not None:
        analysis_collector[query_id] = {
            "query_tokens": list(tokens_query),
            "query_clean_tokens": [clean_token(tok) for tok in tokens_query],
            "query_scores": [float(x) for x in scores_query.tolist()],
            "query_abs_scores": [float(abs(x)) for x in scores_query.tolist()],
            "query_text": detokenize(tokens_query),
            "contexts": [],
        }
    if plot_dir is not None:
        plot_token_scores(
            query_id,
            "query",
            0,
            grad_type,
            tokens_query,
            scores_query,
            plot_dir,
            plot_top_k,
        )

    # --- Process the Contexts ---
    # Loop over all contexts provided for this query
    for ctx_idx, tokens_context in enumerate(data["input"]["context"]):
        # Aggregate the scores for this specific context
        scores_context = aggregate_token_scores(grad_data["context"][ctx_idx], grad_type)
        # Extract the top-k tokens for this context
        ctx_results = extract_top_tokens(
            query_id, "context", ctx_idx, grad_type, tokens_context, scores_context, top_k
        )
        # Add the context's top tokens to the main list
        results.extend(ctx_results)
        if analysis_collector is not None:
            analysis_collector[query_id]["contexts"].append(
                {
                    "index": ctx_idx,
                    "tokens": list(tokens_context),
                    "clean_tokens": [clean_token(tok) for tok in tokens_context],
                    "scores": [float(x) for x in scores_context.tolist()],
                    "abs_scores": [float(abs(x)) for x in scores_context.tolist()],
                    "text": detokenize(tokens_context),
                }
            )
        if plot_dir is not None:
            plot_token_scores(
                query_id,
                "context",
                ctx_idx,
                grad_type,
                tokens_context,
                scores_context,
                plot_dir,
                plot_top_k,
            )
            if combine_query_context:
                plot_joint_token_scores(
                    query_id,
                    grad_type,
                    ctx_idx,
                    tokens_query,
                    scores_query,
                    tokens_context,
                    scores_context,
                    plot_dir / "combined",
                    plot_top_k,
                )
            if plot_side_by_side:
                plot_side_by_side_token_scores(
                    query_id,
                    grad_type,
                    ctx_idx,
                    tokens_query,
                    scores_query,
                    tokens_context,
                    scores_context,
                    plot_dir / "side_by_side",
                    plot_top_k,
                    highlight_overlap=True,
                )

    return results


# --- 7. OUTPUT FUNCTIONS ---

# Saves the final aggregated list of TokenImportance objects to a CSV file.
def save_results(entries: Sequence[TokenImportance], output_path: Path) -> None:
    # Convert the list of dataclass objects into a pandas DataFrame
    df = pd.DataFrame([entry.__dict__ for entry in entries])
    # Sort the DataFrame for better readability
    df.sort_values(
        ["query_id", "segment", "segment_index", "abs_score"],
        ascending=[True, True, True, False],  # Sort by ID, then segment, then abs_score descending
        inplace=True,
    )
    # Save the DataFrame to a CSV file, without the pandas index
    df.to_csv(output_path, index=False)


# Prints a simple summary of the top tokens for the first few queries to the console.
def print_summary(entries: Sequence[TokenImportance], top_queries: int = 3) -> None:
    if not entries:
        print("No entries to display.")
        return

    # Convert to DataFrame for easy filtering and grouping
    df = pd.DataFrame([entry.__dict__ for entry in entries])
    
    # Loop over the first N unique query IDs
    for query_id in df["query_id"].unique()[:top_queries]:
        print(f"\n=== Query {query_id} ===")
        # Get data just for this query's 'query' segment
        subset_query = df[(df["query_id"] == query_id) & (df["segment"] == "query")]
        # Get data just for this query's 'context' segments
        subset_context = df[(df["query_id"] == query_id) & (df["segment"] == "context")]

        # Print top 5 query tokens
        if not subset_query.empty:
            print(" Top query tokens:")
            for _, row in subset_query.sort_values("abs_score", ascending=False).head(5).iterrows():
                print(f"   {row['token_display']:<20} score={row['score']:.4f}")

        # Print top 3 context tokens for each context
        if not subset_context.empty:
            print(" Top context tokens:")
            # Group by context index (0, 1, 2...)
            for (ctx_idx, ctx_group) in subset_context.groupby("segment_index"):
                # Get top 3 tokens for this context
                ctx_top = ctx_group.sort_values("abs_score", ascending=False).head(3)
                # Format them into a single string
                tokens_str = ", ".join(
                    f"{row['token_display']} ({row['score']:.4f})" for _, row in ctx_top.iterrows()
                )
                print(f"   Context {ctx_idx}: {tokens_str}")


# --- 8. MAIN EXECUTION ---

# The main function that orchestrates the entire script.
def main() -> None:
    # 1. Parse command-line arguments
    args = parse_args()

    # 2. Find all the input .pkl files
    files = list_query_pickles(args.results_dir)
    # 3. Apply the processing limit if one was provided
    if args.limit is not None:
        files = files[: args.limit]

    # Error handling if no files are found
    if not files:
        raise FileNotFoundError(f"No query_*.pkl files found in {args.results_dir}")

    collect_analysis = args.qual_query_id is not None or args.overall_stats
    analysis_data = {} if collect_analysis else None

    # 4. Process all files
    all_entries: List[TokenImportance] = []  # List to store results from all files
    # Loop over all files, using tqdm to show a progress bar
    for idx, path in enumerate(tqdm(files, desc="Processing queries")):
        # Extract the query ID from the filename (e.g., 'query_123.pkl' -> 123)
        query_id = int(path.stem.split("_")[-1])
        # Process the file and get the list of top tokens
        entries = process_query_file(
            path,
            args.grad_type,
            args.top_k,
            query_id,
            args.plot_dir,
            args.plot_top_k,
            args.combine_query_context,
            args.plot_side_by_side,
            analysis_data,
        )
        # Add this file's results to the main list
        all_entries.extend(entries)

    # 5. Save the aggregated results
    save_results(all_entries, args.output_csv)
    print(f"Saved token importance summary to {args.output_csv}")
    if args.plot_dir is not None:
        print(f"Saved token importance plots to {args.plot_dir.resolve()}")
    
    # 6. Print a summary to the console
    print_summary(all_entries)

    if args.qual_query_id is not None:
        if analysis_data is None or args.qual_query_id not in analysis_data:
            print(f"Warning: qualitative query id {args.qual_query_id} not found in the processed data.")
        else:
            write_qualitative_report(
                args.qual_query_id,
                analysis_data[args.qual_query_id],
                args.qual_top_contexts,
                args.qual_output,
                args.grad_type,
            )
            print(f"Wrote qualitative report to {args.qual_output}")

    if args.overall_stats:
        if analysis_data is None:
            print("Warning: overall statistics requested but analysis data was not collected.")
        else:
            stats_df = compute_overlap_statistics(analysis_data)
            if stats_df.empty:
                print("No overlap statistics could be computed (empty dataframe).")
            else:
                args.overall_stats_csv.parent.mkdir(parents=True, exist_ok=True)
                stats_df.to_csv(args.overall_stats_csv, index=False)
                mean_query_overlap = stats_df["query_overlap_token_pct"].mean()
                mean_context_overlap = stats_df["context_overlap_token_pct"].mean()
                mean_context_attr_share = stats_df["context_overlap_abs_pct"].mean()
                mean_query_attr_share = stats_df["query_overlap_abs_pct"].mean()
                print(f"Saved overlap statistics to {args.overall_stats_csv}")
                print(
                    "Dataset-level averages: "
                    f"query token overlap {mean_query_overlap:.2%}, "
                    f"context token overlap {mean_context_overlap:.2%}, "
                    f"context attribution share {mean_context_attr_share:.2%}, "
                    f"query attribution share {mean_query_attr_share:.2%}."
                )


# Standard Python entry point:
# This ensures that the 'main()' function is called only when
# the script is executed directly (not when imported as a module).
if __name__ == "__main__":
    main()