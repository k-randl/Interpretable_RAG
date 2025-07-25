import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from html import escape
from IPython.display import display, HTML

from resources.utils import decode_chat_template

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from typing import List, Literal, Union, Tuple, Callable, Optional

#====================================================================#
# General plotting functions:                                        #
#====================================================================#

def plot_vbars(ax:Axes, scores:NDArray[np.float_], tokens:List[str], *, normalize:bool=False, skip_tokens:List[str]=[], cmap:str='tab10') -> None:
    """Creates a vertical bar plot for attribution scores.

    Args:
        ax (matplotlib.axes.Axes):   
        scores (NDArray[np.float_]): A 2D array of attribution scores with shape `(len(documents), len(tokens))`,
                                     where each row represents a document and each column corresponds to a token's attribution score.
        tokens (List[str]):          A list of token strings corresponding to the scores. Length must match the number of columns in `scores`.
        normalize (bool):            If `True`, normalizes the attribution scores across tokens so that the sum of absolute values equals 1 (default=`False`).
        skip_tokens (List[str]):     An optional list of tokens that will not be printed.
        cmap (str):                  The name of a matplotlib colormap used for highlighting.
    """

    # filter tokens:
    scores = scores[:, [t not in skip_tokens for t in tokens]]
    tokens = [t for t in tokens if t not in skip_tokens]

    # Normalize so that values sum to 1 (if enabled):
    if normalize: scores = scores / np.abs(scores).sum(axis=0)

    # Get colormap:
    colors = cm.get_cmap(cmap)

    s_pos = 0.
    s_neg = 0.
    for i, s in enumerate(scores):
        s_doc_pos = np.maximum(s, 0)
        ax.bar(range(len(s)), s_doc_pos, bottom=s_pos, color=colors(i), label=f'Document {i+1:d}' if len(scores) > 1 else None)
        s_pos += s_doc_pos

        s_doc_neg = np.minimum(s, 0)
        ax.bar(range(len(s)), s_doc_neg, bottom=s_neg, color=colors(i))
        s_neg += s_doc_neg

    # Tick and axis settings:
    ax.set_xticks(range(len(tokens)), tokens, rotation=90)
    if normalize: plt.ylim(-1, 1)

    # Clean frame (keep only left spine):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    # Add horizontal grid line at zero:
    ax.axhline(y=0, color='black', linewidth=0.5)

    # Add vertical grid lines at each tick:
    for tick_loc in range(len(tokens)):
        ax.axvline(x=tick_loc, color='lightgray', linewidth=0.5, zorder=0)

def highlight_dominant_passages(scores:NDArray[np.float_], tokens:List[str], title:str, *, threshold:float=0.0, total:Optional[float]=None, skip_tokens:List[str]=[], token_processor:Optional[Callable[[str],str]]=None, cmap:str='tab10') -> Tuple[str, str]:
    """Highlights tokens in a text sequence where a single document contributes
    at least `threshold` of the total positive attribution scores at that token.

    Args:
        scores (NDArray[np.float_]):    A 2D array of attribution scores with shape `(len(documents), len(tokens))`,
                                        where each row represents a document and each column corresponds to a token's attribution score.
        tokens (List[str]):             A list of token strings corresponding to the scores.
                                        Length must match the number of columns in `scores`.
        title (str):                    The title of the produced html table row.
        threshold (float):              Minimum attribution for highlighting in the intervall `[0., 1.]` (default: `0.0`).
        total (float):                  Optional Maximum attribution for highlighting (default: `None`).
        skip_tokens (List[str]):        An optional list of tokens that will not be printed.
        token_processor ((str) -> str): An otional function applied to each token before printing.
        cmap (str):                     The name of a matplotlib colormap used for highlighting.

    Returns:
        A tuple `(text, legend)`, where both `text` and `legend` are html `<tr>` elements.
    """

    # extract only on positive contributions:
    scores_pos = np.maximum(scores, 0)
    num_docs, num_tokens = scores_pos.shape

    if total == None: total = scores_pos.sum(axis=0) if num_docs > 1 else scores_pos.max()
    if np.all(total > threshold):
        token_docs = scores_pos.argmax(axis=0)
        token_vals = scores_pos[token_docs, np.arange(num_tokens)] / total
        token_vals = np.maximum((token_vals - threshold) / (1. - threshold), 0.)
        token_docs[token_vals == 0.] = -1

    else:
        token_vals = np.zeros(num_tokens, dtype=float)
        token_docs = -np.ones(num_tokens, dtype=int)

    # Prepare color map
    cmap = cm.get_cmap(cmap)
    rgb_colors = [mcolors.to_hex(cmap(i)) for i in range(num_docs)]

    # Build highlighed HTML:
    html_tokens = []
    for tok, doc, val in zip(tokens, token_docs, token_vals):
        if tok in skip_tokens: continue

        if token_processor is not None:
            tok = token_processor(tok)

        html_tokens.append(
            f'<span style="background-color:{rgb_colors[doc]}{int((val)*255):02x}; padding:0px; border-radius:3px;">' +
            escape(tok) +
            '</span>'
        )

    html_text = (
        '<tr style="border-top: 1px solid">\n' +
        '   <td style="text-align:right; vertical-align:top">\n' +
        '       <b style="line-height:2">' + title + ':</b>\n' +
        '   </td>\n' +
        '   <td style="text-align:left; vertical-align:top">\n' +
        '       <div style="line-height:2">' + ''.join(html_tokens) + '</div>\n' +
        '   </td>\n' +
        '</tr>\n'
    )

    # Build legend:
    html_legend = (
        '<tr style="border-top: 1px solid">\n' +
        '   <td style="text-align:right; vertical-align:top">\n' +
        '       <i style="line-height:2">Legend:</i>\n' +
        '   </td>\n' +
        '   <td style="text-align:left; vertical-align:top">\n' +
        '       <div style="line-height:2">' +
                    ' '.join([f'<i style="background-color:{c}; padding:0px; border-radius:3px;">Document {i+1:d}</i>' for i, c in enumerate(rgb_colors)]) +
                '</div>\n' +
        '   </td>\n' +
        '</tr>\n'
    )
    
    return html_text, html_legend

#====================================================================#
# Plots for retrieval:                                               #
#====================================================================#

from resources.retrieval_offline import ExplainableAutoModelForRetrieval as ExplainableAutoModelForOfflineRetrieval
from resources.retrieval_online  import ExplainableAutoModelForRetrieval as ExplainableAutoModelForOnlineRetrieval
ExplainableAutoModelForRetrieval = Union[ExplainableAutoModelForOfflineRetrieval, ExplainableAutoModelForOnlineRetrieval]

def plot_importance_retriever(retriever:ExplainableAutoModelForRetrieval, *, method:Literal['grad', 'gradIn', 'aGrad']='gradIn', absolute:bool=True, cmap:str='tab10', show:bool=True, **kwargs) -> Figure:
    # get scores:
    if   method == 'grad':   scores = {key:[doc.mean(axis=-1) for doc in docs] for key, docs in retriever.grad(**kwargs).items()}
    elif method == 'gradIn': scores = retriever.gradIn(**kwargs)
    elif method == 'aGrad':  scores = {key:[doc.mean(axis=0) for doc in docs] for key, docs in retriever.aGrad(**kwargs).items()}
    else: raise ValueError()

    # compute absolute:
    if absolute: scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}

    # get special tokens:
    special_tokens = set(retriever.tokenizer.special_tokens_map.values())

    fig, axs = plt.subplots(len(scores['query']) + len(scores['context']), 1)

    # plot query:
    plot_vbars(axs[0], 
        scores      = scores['query'][0].numpy()[None,:], 
        tokens      = retriever.in_tokens['query'][0],
        skip_tokens = special_tokens,
        cmap        = cmap
    )
    axs[0].set_title('Query:')

    # plot contexts:
    for i in range(len(scores['context'])):
        plot_vbars(axs[i+1],
            scores      = scores['context'][i].numpy()[None,:], 
            tokens      = retriever.in_tokens['context'][i],
            skip_tokens = special_tokens,
            cmap        = cmap
        )
        axs[i+1].set_title(f'Document {i+1:d}:')

    fig.suptitle(method)
    fig.tight_layout()

    if show: fig.show()
    else: return fig

def higlight_importance_retriever(retriever:ExplainableAutoModelForRetrieval, *, method:Literal['grad', 'gradIn', 'aGrad']='gradIn', threshold:float=0.0, token_processor:Optional[Callable[[str],str]]=None, cmap:str='tab10', **kwargs) -> str:
    # get scores:
    if   method == 'grad':   scores = {key:[doc.mean(axis=-1) for doc in docs] for key, docs in retriever.grad(**kwargs).items()}
    elif method == 'gradIn': scores = retriever.gradIn(**kwargs)
    elif method == 'aGrad':  scores = {key:[doc.mean(axis=0) for doc in docs] for key, docs in retriever.aGrad(**kwargs).items()}
    else: raise ValueError()

    # compute absolute:
    scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}

    # get special tokens:
    special_tokens = set(retriever.tokenizer.special_tokens_map.values())

    # plot query:
    html = highlight_dominant_passages(
        scores          = scores['query'][0].numpy()[None,:], 
        tokens          = retriever.in_tokens['query'][0],
        title           = 'Query',
        threshold       = threshold,
        skip_tokens     = special_tokens,
        token_processor = token_processor,
        cmap            = cmap
    )[0]

    # plot contexts:
    for i in range(len(scores['context'])):
        html += highlight_dominant_passages(
            scores          = scores['context'][i].numpy()[None,:], 
            tokens          = retriever.in_tokens['context'][i],
            title           = f'Document {i+1:d}',
            threshold       = threshold,
            total           = np.concatenate(scores['context']).max(),
            skip_tokens     = special_tokens,
            token_processor = token_processor,
            cmap            = cmap
        )[0]

    # Build, display, and return final HTML:
    html_str = (
        '<!DOCTYPE html>\n' +
        '<html>\n' +
        '<head>\n' +
        '<title>' + method + '</title>\n' +
        '</head>\n' +

        '<body>\n' +
        '<table>\n' +
        html +
        '</table>\n' +
        '</body>\n' +

        '</html>'
    )
    display(HTML(html_str))
    return html_str

#====================================================================#
# Plots for generation:                                              #
#====================================================================#

from resources.generation import ExplainableGeneratorMixin

def plot_attribution_generator(generator:ExplainableGeneratorMixin, *, aggregation:Literal['token', 'sequence', 'bow', 'nucleus']='token', normalize:bool=True, cmap:str='tab10', show:bool=True) -> Figure:
    """
    Plot stacked SHAP attributions for multiple documents as positive and negative bar segments.

    Args:
        generator (ExplainableGeneratorMixin): An object containing the necessary information for plotting.
        aggregation (str):                     Aggregation method for probabilities (default: `'token'`).
        normalize (bool):                      If `True`, normalizes the SHAP values across tokens so that the sum of absolute values equals 1 (default=`True`).
        cmap (str):                            The name of a matplotlib colormap used for highlighting.
        show (bool):                           If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).
    """

    # get attribution scores:
    shap_values = generator.get_shapley_values(aggregation)

    # get tokens:
    if aggregation == 'token':
        tokens = generator.gen_tokens

    elif aggregation == 'sequence':
        tokens = ['']

    elif aggregation == 'bow':
        # get the top-K varying tokens per document:
        K = 50
        indices = np.unique(shap_values.var(axis=0).argsort()[:-K:-1])

        shap_values = shap_values[:,indices]
        tokens = generator.tokenizer.convert_ids_to_tokens(indices[:,None])

    elif aggregation == 'nucleus':
        # get the top-K probable tokens per document: 
        K = 10
        indices = np.unique(shap_values.argsort()[:,:-K:-1].flatten())

        shap_values = shap_values[:,indices]
        tokens = generator.tokenizer.convert_ids_to_tokens(indices[:,None])

    else: raise ValueError(f'Unknown value for parameter `aggregation`: "{aggregation}"')
    
    # get special tokens:
    special_tokens = set(generator.tokenizer.special_tokens_map.values())

    # plot:
    fig, ax = plt.subplots(1, 1)
    plot_vbars(ax,
        shap_values,
        tokens,
        normalize=normalize,
        skip_tokens=special_tokens,
        cmap=cmap
    )
    fig.suptitle(aggregation)
    fig.legend()
    fig.tight_layout()

    if show: fig.show()
    else: return fig

def higlight_attribution_generator(generator:ExplainableGeneratorMixin, *, threshold:float=.5, token_processor:Optional[Callable[[str],str]]=None, cmap:str='tab10') -> str:
    """
    Highlights tokens in a text sequence where a single document contributes
    at least `threshold` of the total positive SHAP value at that token.

    Args:
        generator (ExplainableGeneratorMixin): An object containing the necessary information for plotting.
        threshold (float):                     Minimum attribution for highlighting in the intervall `[0., 1.]` (default: `0.5`).
        token_processor ((str) -> str):        An otional function applied to each token before printing.
        cmap (str):                            The name of a matplotlib colormap used for highlighting.

    Returns:
        An HTML-formatted string with spans highlighting dominant SHAP regions.
    """

    # get indices of the actual content:
    # (excluding chat template specific tokens) 
    response = decode_chat_template(
        generator.gen_tokens,
        generator.config.name_or_path,
        return_indices=True
    )[0]

    # create role:
    role = generator.gen_tokens[response['role']]
    if token_processor is not None: 
        role = [token_processor(t) for t in role]
    role = ''.join(role).strip().capitalize()

    # get special tokens:
    special_tokens = set(generator.tokenizer.special_tokens_map.values())

    # create html:
    html_response, html_legend = highlight_dominant_passages(
        scores          = generator.get_shapley_values('token')[:,response['content']],
        tokens          = generator.gen_tokens[response['content']],
        title           = role,
        threshold       = threshold,
        skip_tokens     = special_tokens,
        token_processor = token_processor,
        cmap            = cmap
    )

    # Build, display, and return final HTML:
    html_str = (
        '<!DOCTYPE html>\n' +
        '<html>\n' +
        '<head>\n' +
        '<title>Generated Text</title>\n' +
        '</head>\n' +

        '<body>\n' +
        '<table>\n' +
        html_response +
        html_legend +
        '</table>\n' +
        '</body>\n' +

        '</html>'
    )
    display(HTML(html_str))
    return html_str