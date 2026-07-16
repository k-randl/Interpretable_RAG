import numpy as np
import pandas as pd
import spacy
import pickle
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import re
from html import escape

_LATEX_ESCAPE_RE = re.compile(r'[\\{}$&#%^_~]')
_LATEX_ESCAPE_MAP = {
    '\\': r'\textbackslash{}', '{': r'\{', '}': r'\}',
    '$': r'\$', '&': r'\&', '#': r'\#', '%': r'\%',
    '^': r'\^{}', '_': r'\_', '~': r'\textasciitilde{}',
}

def _latex_escape(text: str) -> str:
    """Escapes LaTeX special characters (``\\{}$&#%^_~``) in `text`."""
    return _LATEX_ESCAPE_RE.sub(lambda m: _LATEX_ESCAPE_MAP[m.group(0)], text)

_TOK_VIS_RE = re.compile(r'\\tok\{[^}]+\}\{((?:[^}\\]|\\.)*)\}')

def _split_latex_row(row: str, characters_per_line: int) -> str:
    """Splits the ``{\\small ...}`` cell of a single row fragment into multiple
    continuation rows so each contains at most *characters_per_line* visible chars.
    Visible length is taken as the length of the text argument inside ``\\tok{}{text}``
    (or the full line length for un-highlighted plain tokens)."""
    OPEN  = '& {\\small\n'
    CLOSE = '\n}'
    pos = row.find(OPEN)
    if pos == -1:
        return row
    content_end = row.find(CLOSE, pos + len(OPEN))
    if content_end == -1:
        return row
    header      = row[:pos]
    token_lines = [l for l in row[pos + len(OPEN):content_end].split('\n') if l.strip()]
    rest        = row[content_end + len(CLOSE):]
    # group token lines by cumulative visible char count
    groups: list = [[]]
    chars = 0
    for line in token_lines:
        m   = _TOK_VIS_RE.match(line)
        vis = len(m.group(1)) if m else len(line)
        if chars + vis > characters_per_line and groups[-1]:
            groups.append([])
            chars = 0
        groups[-1].append(line)
        chars += vis
    groups = [g for g in groups if g]
    if len(groups) <= 1:
        return row
    result = header + OPEN + '\n'.join(groups[0]) + CLOSE
    for g in groups[1:]:
        result += ' \\\\\n' + OPEN + '\n'.join(g) + CLOSE
    result += rest
    return result

def _wrap_latex_tabular(rows:str, prefix:str) -> str:
    """Collects \\definecolor declarations from all row fragments, deduplicates them,
    places them before the tabular environment, and wraps the content."""

    colors     = []
    cmap       = {}
    clean_rows = []
    
    color_re = re.compile(r'\\definecolor\{([^}]+)\}(\{[^}]+\}\{[^}]+\})')
    for row in rows.split('\n'):
        _match = color_re.match(row)
        if _match:
            k, v = _match.groups()
            try: i = colors.index(v)
            except ValueError:
                i = len(colors)
                colors.append(v)
            cmap[k] = f'{prefix}{i:d}'

        else:
            for old, new in reversed(cmap.items()):
                row = row.replace(old, new)
            clean_rows.append(row)

    color_defs = [
        f'\\definecolor\u007b{prefix}{i:d}\u007d{v}'
        for i,v in enumerate(colors)
    ]

    return (
        '\n'.join(color_defs) + '\n\n'
        '\\setlength{\\tabcolsep}{2pt}\n'
        '\\renewcommand{\\arraystretch}{1.1}\n\n'
        '\\begin{tabular}{rl}\n' +
        '\n'.join(clean_rows) +
        '\\bottomrule\n'
        '\\end{tabular}'
    )

from IPython.display import display, HTML
from src.Interpretable_RAG.utils import match_token_attributions

from .utils import decode_chat_template, nucleus_sample_tokens

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import Colormap
from typing import List, Literal, Tuple, Callable, Optional, Union, Iterable
from .types import FloatArray

#====================================================================#
# Custom colormao specifications:                                    #
#====================================================================#

cmap = mcolors.ListedColormap(["#00dddd", "#dd00dd"], name='ragbin')
mpl.colormaps.register(cmap=cmap, force=True)

#====================================================================#
# General plotting functions:                                        #
#====================================================================#

def plot_token_vbars(ax:Axes, scores:FloatArray, tokens:Iterable[str], document_names:Optional[List[str]]=None, *,
        normalize:bool=False,
        skip_tokens:Iterable[str]=[],
        cmap:str='tab10',
        **kwargs
    ) -> None:
    """Creates a vertical bar plot for attribution scores of tokens.

    Args:
        ax (matplotlib.axes.Axes):      `matplotlib.axes.Axes` object.
        scores (FloatArray):   A 2D array of attribution scores with shape `(len(documents), len(tokens))`,
                                        where each row represents a document and each column corresponds to a token's
                                        attribution score.
        tokens (Iterable[str]):         An itterable of token strings corresponding to the scores. Length must match the number
                                        of columns in `scores`.
        document_names (Iterable[str]): An optional list of names of the documents.
        normalize (bool):               If `True`, normalizes the attribution scores across tokens
                                        so that the sum of absolute values equals 1 (default=`False`).
        skip_tokens (List[str]):        An optional list of tokens that will not be printed.
        cmap (str):                     The name of a matplotlib colormap used for highlighting.
    """

    # filter tokens:
    scores = scores[:, [t not in skip_tokens for t in tokens]]
    tokens = [t for t in tokens if t not in skip_tokens]

    # normalize so that values sum to 1 (if enabled):
    if normalize: scores = scores / np.abs(scores).sum(axis=0)

    # get colormap:
    colors = cm.get_cmap(cmap)

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(len(scores))]
    elif len(document_names) != len(scores): raise ValueError('`len(document_names)` does not match the number of documents!')

    s_pos = 0.
    s_neg = 0.
    for i, s in enumerate(scores):
        s_doc_pos = np.maximum(s, 0)
        ax.bar(range(len(s)), s_doc_pos, bottom=s_pos, color=colors(i), label=document_names[i] if len(scores) > 1 else None, **kwargs)
        s_pos += s_doc_pos

        s_doc_neg = np.minimum(s, 0)
        ax.bar(range(len(s)), s_doc_neg, bottom=s_neg, color=colors(i))
        s_neg += s_doc_neg

    # tick and axis settings:
    ax.set_xticks(range(len(tokens)), tokens, rotation=90)
    if normalize:
        if np.any(s_neg < 0.): ax.set_ylim(-1, 1)
        else: ax.set_ylim(0, 1)

    # clean frame (keep only left spine):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ax.get_ylim()[0] < 0.:
        ax.spines['bottom'].set_visible(False)

        # add horizontal grid line at zero:
        ax.axhline(y=0, color='black', linewidth=0.5)

    # add vertical grid lines at each tick:
    for tick_loc in range(len(tokens)):
        ax.axvline(x=tick_loc, color='lightgray', linewidth=0.5, zorder=0)

def html_legend_discrete(names:List[str], cmap:Colormap, vals:Optional[List[float]]=None) -> str:
    """Builds an HTML legend with one discrete color chip per name.

    Args:
        names (List[str]):        The names to label each chip with.
        cmap (Colormap):          Colormap used to color the chips (indexed by position).
        vals (List[float]):       Optional percentage values shown under each name.

    Returns:
        An HTML fragment containing the legend.
    """
    n = len(names)
    
    # prepare color map:
    rgb_colors = np.array([mcolors.to_rgb(cmap(i)) for i in range(n)])*255.
    
    # format texts:
    if vals is None:
        texts = [f'<i>{n}</i>' for n in names]
    else:
        texts = [f'<i>{n}</i><br><small>({v * 100.:.0f}%)</small>' for n, v in zip(names, vals, strict=True)]

    # return legend:
    return (
        '<small style="float:left;">\n' +
        '   <div style="line-height:1">' +
                '<div style="line-height: 2; text-align: center; padding:3px; margin:3px; float:left;"><i>Legend:</i></div>' +
                ''.join([f'<div style="background-color:#{int(r):02x}{int(g):02x}{int(b):02x}; text-align: center; padding:3px; margin:3px; border-radius:3px; float:left;">{texts[i]}</div>' for i, (r, g, b) in enumerate(rgb_colors)]) +
            '</div>\n' +
        '</small>\n'
    )

def html_legend_continuous(names:List[str], cmap:Colormap, vals:Optional[List[float]]=None) -> str:
    """Builds an HTML legend showing a continuous color gradient between `names`.

    Args:
        names (List[str]):        The names placed at each end of the gradient.
        cmap (Colormap):          Colormap sampled at 0.0 and 1.0 for the gradient endpoints.
        vals (List[float]):       Optional percentage values shown under each name.

    Returns:
        An HTML fragment containing the legend.
    """
    # prepare color map:
    hex_left = mcolors.to_hex(cmap(0.0))
    hex_right = mcolors.to_hex(cmap(1.0))

    # format texts:
    if vals is None:
        texts = [f'<i>{n}</i>' for n in names]
    else:
        texts = [f'<div><i>{n}</i><br><small>({v * 100.:.0f}%)</small></div>' for n, v in zip(names, vals, strict=True)]

    # return a simple horizontal colorbar:
    return (
        '<small style="float:left;">\n' +
        '   <div style="line-height: 1;">' +
                '<div style="text-align: center; padding:3px; margin:3px; float:left;"><i>Legend:</i></div>' +
                f'<div style="background: linear-gradient(90deg, {hex_left}, {hex_right}); text-align: center; padding:3px; margin:3px; border-radius:3px; float:left;">' +
                    f'{"&nbsp;"*5}&#8594;{"&nbsp;"*5}'.join(texts) +
                '</div>' +
            '</div>\n' +
        '</small>\n'
    )

def highlight_dominant_passages(scores:FloatArray, tokens:Iterable[str], title:str, labels:List[str]=[], *,
        threshold:float=0.0,
        max_score:Optional[float]=None,
        skip_tokens:Iterable[str]=[],
        token_processor:Optional[Callable[[str],str]]=None,
        color_mode:Literal['winner_takes_it_all', 'average']='winner_takes_it_all',
        legend:bool=True,
        cmap:str='tab10',
        output_format:Literal['html', 'latex']='html',
        latex_color_prefix:str='cl'
    ) -> str:
    """Highlights tokens in a text sequence where a single document contributes
    at least `threshold` of the total positive attribution scores at that token.

    Args:
        scores (FloatArray):   A 1D or 2D array of attribution scores with shape `([len(documents),] len(tokens))`,
                                        where each row represents a document and each column corresponds to a token's attribution score.
        tokens (Iterable[str]):         A list of token strings corresponding to the scores.
                                        Length must match the number of columns in `scores`.
        labels (List[str]):             An optional list of names of the documents.
        title (str):                    The title of the produced table row.
        threshold (float):              Minimum attribution for highlighting in the intervall `[0., 1.]` (default: `0.0`).
        max_score (float):              Optional Maximum attribution for highlighting (default: `None`).
        skip_tokens (List[str]):        An optional list of tokens that will not be printed.
        token_processor ((str) -> str): An optional function applied to each token before printing.
        legend (bool):                  Whether to create a legend for the plot (default: `True`).
        cmap (str):                     The name of a matplotlib colormap used for highlighting.
        output_format (str):            Output format: ``'html'`` (default) or ``'latex'``.
        latex_color_prefix (str):       Prefix for ``\\definecolor`` names emitted in LaTeX output
                                        (default: ``'cl'``). Colors are named ``<prefix>0``,
                                        ``<prefix>1``, etc. and defined via exact RGB values from
                                        the colormap, so any matplotlib colormap is supported.

    Returns:
        An HTML ``<tr>`` element string when ``output_format='html'``, or a LaTeX table-row
        fragment (``\\definecolor`` declarations + ``\\midrule`` + row) when
        ``output_format='latex'``. Wrap multiple fragments with ``_wrap_latex_tabular()``
        to produce a complete ``tabular`` environment.

    Note:
        LaTeX output requires the following macro definitions in the document preamble::

            \\usepackage{xcolor,colortbl,booktabs}
            \\newcommand{\\tok}[2]{\\colorbox{#1}{\\strut\\vphantom{Xy}\\small #2}}
            \\newcommand{\\docchip}[3]{\\colorbox{#1}{\\textcolor{white}{\\small #2\\;#3}}}
    """

    # check threshold value:
    if (threshold >= 1.) or (threshold < 0.):
        raise ValueError(f'Parameter `threshold` must be in intervall `[0,1))` but is `{threshold:.2f}`.')

    # check dimenisons of `scores` input:
    num_dims = scores.ndim
    if   num_dims == 1: scores = np.stack([scores, -scores])
    elif num_dims != 2: raise ValueError(f"Parameter `scores` must have 2 dimensions abut has {num_dims:d}.")

    # extract only on positive contributions:
    scores_pos = np.maximum(scores, 0)
    num_docs, num_tokens = scores_pos.shape

    # check color mode parameter:
    if color_mode == 'winner_takes_it_all':
        # set non maximal scores to nan:
        token_docs = scores_pos == scores_pos.max(axis=0, keepdims=True)
        scores_pos[~token_docs] = np.nan

    elif color_mode != 'average':
        raise ValueError(f'Unknown `color_mode` "{color_mode}"')

    # compute document influence per token:
    token_totals = np.nansum(scores_pos, initial=0., axis=0, keepdims=True)
    token_totals = np.maximum(token_totals, 1e-9) # make sure that token_sum > 0
    token_vals = scores_pos / token_totals
    token_vals = np.maximum((token_vals - threshold) / (1. - threshold), 0.)

    # compute alpha values:
    alphas = np.nanmean(scores_pos, axis=0)
    alphas /= alphas.max() if max_score == None else max_score
    alphas *= 255.

    # prepare color map
    cmap = cm.get_cmap(cmap)
    rgb_colors = np.array([mcolors.to_rgb(cmap(i)) for i in range(num_docs)])*255.

    if output_format == 'latex':
        # emit \definecolor{<prefix>i}{RGB}{r,g,b} for each document using exact colormap values
        color_defs = [
            f'\\definecolor{{{latex_color_prefix}{i}}}{{RGB}}{{{int(r)},{int(g)},{int(b)}}}'
            for i, (r, g, b) in enumerate(rgb_colors)
        ]

        # build LaTeX token list
        latex_toks = []
        for tok, val, alpha in zip(tokens, token_vals.T, alphas):
            if tok in skip_tokens: continue
            if token_processor is not None:
                tok = token_processor(tok)
            tok = _latex_escape(tok)
            alpha_int = min(100, max(0, int(alpha / 255. * 100)))
            if not np.isnan(val).all():
                dominant_doc = int(np.nanargmax(val))
                color = f'{latex_color_prefix}{dominant_doc}'
                latex_toks.append(f'\\tok{{{color}!{alpha_int}}}{{{tok}}}')
            else:
                latex_toks.append(tok)

        # build legend row
        latex_legend = ''
        if legend:
            if color_mode == 'winner_takes_it_all' and num_dims != 1 and labels:
                vals = scores.mean(axis=1)
                vals /= np.abs(vals).sum()
                chips = [
                    f'\\docchip{{{latex_color_prefix}{i}}}{{{_latex_escape(lbl)}}}{{{v*100:.0f}\\%}}'
                    for i, (lbl, v) in enumerate(zip(labels, vals))
                ]
                latex_legend = (
                    '& {\\scriptsize\\quad\n' +
                    '\\hspace{3pt}\n'.join(chips) + '\n}\n\\\\[-2pt]\n'
                )

        title_esc = _latex_escape(title)
        row = '\n'.join(color_defs) + '\n\\midrule\n\n'
        row += '\\textbf{'+ title_esc + ':} & {\\small\n' + '\n'.join(latex_toks) + '\n}'
        row += '\n' + ('\\\\[-2pt]\n' if latex_legend else '\\\\\n')
        row += latex_legend

        return row

    # build highlighted HTML:
    html_tokens = []
    for tok, val, alpha in zip(tokens, token_vals.T, alphas):
        if tok in skip_tokens: continue

        if token_processor is not None:
            tok = token_processor(tok)

        if not np.isnan(val).all():
            r,g,b = np.nanmean(np.stack([c*v for c,v in zip(rgb_colors, val)]), axis=0).astype(int)
            html_tokens.append(
                f'<span style="background-color:#{r:02x}{g:02x}{b:02x}{int(alpha):02x}; padding:0px; border-radius:3px;">' +
                escape(tok) +
                '</span>'
            )

        else: html_tokens.append(escape(tok))

    html_legend = ''
    if legend:
        if (color_mode == 'average') or (num_dims == 1):
            # add a simple horizontal colorbar:
            html_legend = html_legend_continuous(labels, cmap)

        elif color_mode == 'winner_takes_it_all':
            # extract mean absolute document contributions:
            vals = scores.mean(axis=1)
            vals /= np.abs(vals).sum()

            # build legend:
            html_legend = html_legend_discrete(labels, cmap, vals)

    html_text = (
        '<tr style="border-top: 1px solid">\n' +
        '   <td style="text-align:right; vertical-align:top">\n' +
        '       <b style="line-height:2">' + title + ':</b>\n' +
        '   </td>\n' +
        '   <td style="text-align:left; vertical-align:top">\n' +
        '       <div style="line-height:2">' + ''.join(html_tokens) + '</div>\n' + html_legend +
        '   </td>\n' +
        '</tr>\n'
    )

    return html_text

def plot_document_vbars(ax:Axes, scores:FloatArray, document_names:Optional[List[str]]=None, *,
        normalize:bool=False,
        cmap:str='tab10',
        **kwargs
    ) -> None:
    """Creates a vertical bar plot for attribution scores of documents.

    Args:
        ax (matplotlib.axes.Axes):   `matplotlib.axes.Axes` object.
        scores (FloatArray): A 1D array of attribution scores with shape `(len(documents),)`,
                                     where each item represents a document's attribution score.
        document_names (List[str]):  An optional list of names of the documents.
        normalize (bool):            If `True`, normalizes the attribution scores across documents so
                                     that the sum of absolute values equals 1 (default=`False`).
        cmap (str):                  The name of a matplotlib colormap used for highlighting.
    """
    # normalize so that values sum to 1 (if enabled):
    if normalize: scores = scores / np.abs(scores).sum()

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(len(scores))]
    elif len(document_names) != len(scores): raise ValueError('`len(document_names)` does not match the number of documents!')

    # get color:
    color = kwargs.get('color', cm.get_cmap(cmap)(0))

    # create bar plot:
    bars = ax.bar(range(len(document_names)), scores, color=color, **kwargs)

    # add value labels on bars:
    for bar, value in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
            f'{value:.4f}', ha='center', va='bottom')

    # tick and axis settings:
    ax.set_xticks(range(len(document_names)), document_names, rotation=45, ha='right')
    if normalize: ax.set_ylim(-1, 1)

    # clean frame (keep only left spine):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ax.get_ylim()[0] < 0.:
        ax.spines['bottom'].set_visible(False)

        # add horizontal grid line at zero:
        ax.axhline(y=0, color='black', linewidth=0.5)

    # add horizontal grid lines at each tick:
    for tick_loc in ax.get_yticks(minor=False):
        ax.axhline(y=tick_loc, color='lightgray', linewidth=0.5, zorder=0)

def plot_waterfall(ax:Axes, scores:FloatArray, x_labels:List[str], *,
        base_value:float=0.0,
        normalize:bool=False,
        cmap:str='tab10',
        **kwargs
    ) -> None:
    """Create a waterfall plot showing cumulative scores.
    
    Args:
        ax (matplotlib.axes.Axes):   `matplotlib.axes.Axes` object.
        scores (FloatArray): A 1D array of attribution scores with shape `(len(documents),)`,
                                     where each item represents a document's attribution score.
        x_labels (List[str]):        The list of x-axis labels.
        base_value (float):          Base value to start the waterfall from (default: `0.0`).
        normalize (bool):            If `True`, normalizes the attribution scores across documents so
                                     that the sum of absolute values equals 1 (default=`False`).
        cmap (str):                  The name of a matplotlib colormap used for highlighting.
    """
    # normalize so that values sum to 1 (if enabled):
    if normalize: scores = scores / np.abs(scores).sum()

    # calculate cumulative values:
    cumulative = np.cumsum(np.concatenate([[base_value], scores]))

    # get colormap:
    colors = cm.get_cmap(cmap)

    # plot each contribution:
    for i, value in enumerate(scores):
        color = colors(0) if value > 0 else colors(1)
        ax.bar(i, value, bottom=cumulative[i], color=color, **kwargs)
        
        # add value label:
        ax.text(i, cumulative[i] + value/2, f'{value:.2f}', 
               ha='center', va='center', fontweight='bold', color='white')

    # plot final value:
    ax.bar(len(x_labels), cumulative[-1], color=colors(2), **kwargs)
        
    # add value label:
    ax.text(len(x_labels), cumulative[-1]/2, f'{cumulative[-1]:.2f}', 
            ha='center', va='center', fontweight='bold', color='white')

    # connect bars with lines:
    for i in range(len(x_labels)):
        ax.plot([i + 0.4, i + 1.6], [cumulative[i+1], cumulative[i+1]], 'k--', alpha=0.5)

    # tick and axis settings:
    ax.set_xticks(range(len(x_labels) + 1), x_labels + ['Total'], rotation=45, ha='right')
    if normalize: 
        if ax.get_ylim()[0] < 0: ax.set_ylim(-1, 1)
        else: ax.set_ylim(0, 1)

    # clean frame (keep only left spine):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ax.get_ylim()[0] < 0.:
        ax.spines['bottom'].set_visible(False)

        # add horizontal grid line at zero:
        ax.axhline(y=0, color='black', linewidth=0.5)

    # add horizontal grid lines at each tick:
    for tick_loc in ax.get_yticks(minor=False):
        ax.axhline(y=tick_loc, color='lightgray', linewidth=0.5, zorder=0)

#====================================================================#
# Plots for retrieval:                                               #
#====================================================================#

from .retrieval import RetrieverMethods_t, RetrieverExplanationBase, get_retriever_scores

def plot_importance_retriever(explanation:RetrieverExplanationBase, document_names:Optional[List[str]]=None, *,
        method:RetrieverMethods_t='intGrad',
        absolute:bool=True,
        figsize: Tuple[int, int] = (12, 6),
        cmap:str='tab10',
        show:bool=True,
        **kwargs
    ) -> Union[Figure, None]:
    """Plot tokens in a text sequence that are important for retrieving the document.

    Args:
        explanation (RetrieverExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        method (str):                           The method for calculating token importance.
        aggregation (str):                      Aggregation method for probabilities (default: `'token'`).
        normalize (bool):                       If `True`, normalizes the SHAP values across tokens so that the sum of absolute values equals 1 (default=`True`).
        figsize ((int, int)):                   The size of the figure.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        `matplotlib.figure.Figure` object if `show == False`.
    """

    # get scores:
    scores = get_retriever_scores(explanation, method, **kwargs)

    # compute absolute:
    if absolute: scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(len(scores['context']))]
    elif len(document_names) != len(scores['context']): raise ValueError('`len(document_names)` does not match the number of documents!')

    # get special tokens:
    special_tokens = set(explanation.tokenizer.all_special_tokens)

    fig, axs = plt.subplots(len(scores['query']) + len(scores['context']), 1, figsize=figsize)

    # plot query:
    plot_token_vbars(axs[0],
        scores      = scores['query'][0].numpy()[None,:],
        tokens      = explanation.in_tokens['query'][0],
        skip_tokens = special_tokens,
        cmap        = cmap
    )
    axs[0].set_ylabel('Query:')

    # plot contexts:
    for i, s in enumerate(scores['context']):
        plot_token_vbars(axs[i+1],
            scores      = s.numpy()[None,:], 
            tokens      = explanation.in_tokens['context'][i],
            skip_tokens = special_tokens,
            cmap        = cmap
        )
        axs[i+1].set_ylabel(document_names[i] + ':')

    fig.suptitle(f'Token Importance - {method}', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if show: fig.show()
    else: return fig

def higlight_importance_retriever(explanation:RetrieverExplanationBase, document_names:Optional[List[str]]=None, *,
        method:RetrieverMethods_t='intGrad',
        threshold:float=0.0,
        token_processor:Optional[Callable[[str],str]]=None,
        cmap:str='ragbin',
        show:bool=True,
        output_format:Literal['html', 'latex']='html',
        latex_color_prefix:str='cl',
        characters_per_line:int=150,
        **kwargs
    ) -> Union[str,None]:
    """Highlights tokens in a text sequence that are important for retrieving the document.

    Args:
        explanation (RetrieverExplanationBase):   An object containing the necessary information for plotting.
        document_names (List[str], optional):     An optional list of names of the documents.
        method (str, optional):                   The method for calculating token importance.
        threshold (float, optional):              Minimum importance for highlighting in the intervall `[0., 1.)` (default: `0.`).
        token_processor ((str) -> str, optional): An optional function applied to each token before printing.
        cmap (str, optional):                     The name of a matplotlib colormap used for highlighting.
        show (bool, optional):                    If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).
        output_format (str, optional):            Output format: ``'html'`` (default) or ``'latex'``.
        latex_color_prefix (str, optional):       Prefix for ``\\definecolor`` names in LaTeX output (default: ``'cl'``).
        characters_per_line (int, optional):      Wrap LaTeX rows at this many visible characters (default: ``None`` = no wrapping).

    Returns:
        A formatted string highlighting important tokens if `show == False`.
    """
    # get scores:
    scores = get_retriever_scores(explanation, method, **kwargs)

    # compute absolute:
    scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}

    # fallback for document names:
    if document_names is None:
        sep = '~' if output_format == 'latex' else '&nbsp;'
        document_names = [f'Document{sep}{i+1:d}' for i in range(len(scores['context']))]
    elif len(document_names) != len(scores['context']): raise ValueError('`len(document_names)` does not match the number of documents!')

    # get special tokens:
    special_tokens = set(explanation.tokenizer.all_special_tokens)

    # plot query:
    html = highlight_dominant_passages(
        scores             = scores['query'][0].numpy(),
        tokens             = explanation.in_tokens['query'][0],
        title              = 'Query',
        labels             = ['+', '-'],
        threshold          = threshold,
        skip_tokens        = special_tokens,
        token_processor    = token_processor,
        legend             = True,
        cmap               = cmap,
        output_format      = output_format,
        latex_color_prefix = latex_color_prefix,
    )
    if output_format == 'latex' and characters_per_line is not None:
        html = _split_latex_row(html, characters_per_line)

    # plot contexts:
    for i, s in enumerate(scores['context']):
        _row = highlight_dominant_passages(
            scores             = s.numpy(),
            tokens             = explanation.in_tokens['context'][i],
            title              = document_names[i],
            labels             = ['+', '-'],
            threshold          = threshold,
            max_score          = np.concatenate(scores['context']).max(),
            skip_tokens        = special_tokens,
            token_processor    = token_processor,
            legend             = True,
            cmap               = cmap,
            output_format      = output_format,
            latex_color_prefix = latex_color_prefix,
        )
        if output_format == 'latex' and characters_per_line is not None:
            _row = _split_latex_row(_row, characters_per_line)
        html += _row

    if output_format == 'latex':
        latex_str = _wrap_latex_tabular(html, latex_color_prefix)
        if show: print(latex_str); return None
        else: return latex_str

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

    if show: display(HTML(html_str)); return None
    else: return html_str

def plot_importance_summary_retriever(explanation:RetrieverExplanationBase, document_names:Optional[List[str]]=None, *, 
        method:RetrieverMethods_t='intGrad',
        normalize:bool=True,
        threshold:float=.9,
        figsize: Tuple[int, int] = (12, 6),
        cmap:str='tab10',
        show:bool=True,
        **kwargs
    ) -> Union[Figure, None]:
    """Create a summary plot showing the most important tokens per document.

    Args:
        explanation (RetrieverExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        method (str):                           The method for calculating token importance.
        normalize (bool):                       If `True`, normalizes the importance values across tokens so that the sum of absolute values equals 1 (default=`True`).
        threshold (float):                      The accumulated perctentual impact of shown tokens per document in the interval `[0.,1.]` (default: `.9`).
        figsize ((int, int)):                   The size of the figure.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        `matplotlib.figure.Figure` object if `show == False`.
    """

    # get scores:
    scores = get_retriever_scores(explanation, method, **kwargs)

    # compute absolute:
    scores = {key:[np.abs(doc) for doc in docs] for key, docs in scores.items()}
    
    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(len(scores['context']))]
    elif len(document_names) != len(scores['context']): raise ValueError('`len(document_names)` does not match the number of documents!')

    # create figure:
    fig, axs = plt.subplots(len(scores['query']) + len(scores['context']), 1, figsize=figsize)

    # plot query:
    s, t = nucleus_sample_tokens(scores['query'][0], explanation.in_tokens['query'][0], threshold)
    plot_waterfall(axs[0],
        scores    = s,
        x_labels  = t,
        normalize = normalize,
        cmap      = cmap 
    )
    axs[0].set_ylabel('Query:')

    # plot contexts:
    for i, s in enumerate(scores['context']):
        s, t = nucleus_sample_tokens(s, explanation.in_tokens['context'][i], threshold)
        plot_waterfall(axs[i+1],
            scores    = s,
            x_labels  = t,
            normalize = normalize,
            cmap      = cmap 
        )
        axs[i+1].set_ylabel(document_names[i] + ':')

    # title and x-label:
    fig.suptitle(f'Importance Summary - {method}', fontsize=14, fontweight='bold')
    fig.tight_layout()

    if show: fig.show()
    else: return fig

def visualize_importance_retriever(explanation:RetrieverExplanationBase, document_names:Optional[List[str]]=None, *,
        method:RetrieverMethods_t='intGrad',
        cmap:str='ragbin',
        show:bool=True,
        output_format:Literal['html', 'latex']='html',
        latex_color_prefix:str='cl',
        **kwargs
    ) -> Union[Figure, str, None]:
    """Visualizes token importance while automatically choosing a fitting plot for the chosen `method`.

    Args:
        explanation (RetrieverExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        method (str):                           The method for calculating token importance.
        cmap (str, optional):                   The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).
        output_format (str, optional):          Output format: ``'html'`` (default) or ``'latex'``.
        latex_color_prefix (str, optional):     Prefix for ``\\definecolor`` names in LaTeX output (default: ``'cl'``).

    Returns:
        The illustration object if `show == False`.
    """

    return higlight_importance_retriever(
        explanation        = explanation,
        document_names     = document_names,
        method             = method,
        cmap               = cmap,
        show               = show,
        output_format      = output_format,
        latex_color_prefix = latex_color_prefix,
        **kwargs
    )

#====================================================================#
# Plots for generation:                                              #
#====================================================================#

from .generation import GeneratorExplanationBase, GeneratorMethods_t, get_generator_scores

def plot_attribution_generator(explanation:GeneratorExplanationBase, document_names:Optional[List[str]]=None, *,
        aggregation:Literal['token', 'bow', 'nucleus']='token',
        method:GeneratorMethods_t='shap',
        token_processor:Optional[Callable[[str],str]]=None,
        normalize:bool=True,
        figsize: Tuple[int, int] = (12, 6),
        cmap:str='tab10',
        show:bool=True
    ) -> Union[Figure, None]:
    """Plot stacked SHAP attributions for multiple documents as positive and negative bar segments.

    Args:
        explanation (GeneratorExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        aggregation (str):                      Aggregation method for probabilities (default: `'token'`).
        token_processor ((str) -> str):         An optional function applied to each token before printing.
        normalize (bool):                       If `True`, normalizes the SHAP values across tokens so that the sum of absolute values equals 1 (default=`True`).
        figsize ((int, int)):                   The size of the figure.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        `matplotlib.figure.Figure` object if `show == False`.
    """

    # get attribution scores:
    if   method == 'shap': scores = explanation.shap('context', aggregation)
    elif method == 'lime': scores = explanation.lime('context', aggregation)
    else: raise ValueError(f'Unknown method `{method}`.')

    # get tokens:
    if aggregation == 'token':
        tokens = explanation.gen_tokens

    elif aggregation == 'bow':
        # get the top-K varying tokens per document:
        K = 50
        indices = np.unique(scores.var(axis=0).argsort()[:-K:-1])

        scores = scores[:,indices]
        tokens = explanation.tokenizer.convert_ids_to_tokens(indices[:,None])

    elif aggregation == 'nucleus':
        # get the top-K probable tokens per document: 
        K = 10
        indices = np.unique(scores.argsort()[:,:-K:-1].flatten())

        scores = scores[:,indices]
        tokens = explanation.tokenizer.convert_ids_to_tokens(indices[:,None])

    else: raise ValueError(f'Unknown value for parameter `aggregation`: "{aggregation}"')

    # get special tokens:
    special_tokens = set(explanation.tokenizer.all_special_tokens)

    # preprocess tokens:
    if token_processor is not None:
        tokens = [t if t in special_tokens else token_processor(t) for t in tokens]

    # plot:
    fig, ax = plt.subplots(figsize=figsize)
    plot_token_vbars(ax,
        scores         = scores,
        tokens         = tokens,
        document_names = document_names,
        normalize      = normalize,
        skip_tokens    = special_tokens,
        cmap           = cmap
    )
    ax.set_title(f'SHAP Attribution - {aggregation.capitalize()} Aggregation', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Normalized Attribution' if  normalize else 'Attribution')
    ax.legend()
    fig.tight_layout()

    if show: fig.show()
    else: return fig

def higlight_attribution_generator(explanation:GeneratorExplanationBase, document_names:Optional[List[str]]=None, *,
        method:GeneratorMethods_t='shap',
        threshold:float=.5,
        token_processor:Optional[Callable[[str],str]]=None,
        query_cmap:str='ragbin',
        document_cmap:str='tab10',
        show:bool=True,
        output_format:Literal['html', 'latex']='html',
        latex_color_prefix:str='cl',
        characters_per_line:int=150,
    ) -> str:
    """Highlights tokens in a text sequence where a single document contributes
    at least `threshold` of the total positive SHAP value at that token.

    Args:
        explanation (GeneratorExplanationBase):   An object containing the necessary information for plotting.
        document_names (List[str], optional):     An optional list of names of the documents.
        threshold (float, optional):              Minimum attribution for highlighting in the intervall `[0., 1.)` (default: `0.5`).
        token_processor ((str) -> str, optional): An optional function applied to each token before printing.
        query_cmap (str, optional):               The name of a matplotlib colormap used for highlighting of the query attributions.
        document_cmap (str, optional):            The name of a matplotlib colormap used for highlighting of different documents.
        show (bool, optional):                    If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).
        output_format (str, optional):            Output format: ``'html'`` (default) or ``'latex'``.
        latex_color_prefix (str, optional):       Prefix for ``\\definecolor`` names in LaTeX output (default: ``'cl'``).
        characters_per_line (int, optional):      Wrap LaTeX rows at this many visible characters (default: ``None`` = no wrapping).

    Returns:
        A formatted string highlighting dominant attribution regions if `show == False`.
    """

    # get attribution scores:
    if   method == 'shap': scores = explanation.shap(None, 'token')
    elif method == 'lime': scores = explanation.lime(None, 'token')
    else: raise ValueError(f'Unknown method `{method}`.')

    # get indices of the actual content:
    # (excluding chat template specific tokens) 
    response = decode_chat_template(
        explanation.gen_tokens,
        explanation.model_name_or_path,
        return_indices=True
    )[0]

    # create role:
    role = response['role']
    if token_processor is not None:
        role = [token_processor(t) for t in role]
    role = ''.join(role).strip().capitalize()

    # fallback for document names:
    if document_names is None:
        sep = '~' if output_format == 'latex' else '&nbsp;'
        document_names = [f'Document{sep}{i+1:d}' for i in range(len(scores['context']))]
    elif len(document_names) != len(scores['context']): raise ValueError('`len(document_names)` does not match the number of documents!')

    # get special tokens:
    special_tokens = set(explanation.tokenizer.all_special_tokens)

    # create query row:
    html_query = highlight_dominant_passages(
        scores             = scores['query'].mean(axis=1),
        tokens             = explanation.qry_tokens,
        title              = 'Query',
        labels             = ['+', '-'],
        threshold          = threshold,
        skip_tokens        = special_tokens,
        legend             = True,
        cmap               = query_cmap,
        output_format      = output_format,
        latex_color_prefix = latex_color_prefix,
    )
    if output_format == 'latex' and characters_per_line is not None:
        html_query = _split_latex_row(html_query, characters_per_line)

    # create response row:
    html_response = highlight_dominant_passages(
        scores             = scores['context'][:,response['content']],
        tokens             = explanation.gen_tokens[response['content']],
        title              = role,
        labels             = document_names,
        threshold          = threshold,
        skip_tokens        = special_tokens,
        token_processor    = token_processor,
        legend             = True,
        cmap               = document_cmap,
        output_format      = output_format,
        latex_color_prefix = latex_color_prefix,
    )
    if output_format == 'latex' and characters_per_line is not None:
        html_response = _split_latex_row(html_response, characters_per_line)

    if output_format == 'latex':
        latex_str = _wrap_latex_tabular(html_query + html_response, latex_color_prefix)
        if show: print(latex_str); return None
        else: return latex_str

    # Build, display, and return final HTML:
    html_str = (
        '<!DOCTYPE html>\n' +
        '<html>\n' +
        '<head>\n' +
        '<title>Generated Text</title>\n' +
        '</head>\n' +

        '<body>\n' +
        '<table>\n' +
        html_query +
        html_response +
        '</table>\n' +
        '</body>\n' +

        '</html>'
    )

    if show: display(HTML(html_str)); return None
    else: return html_str

def plot_attribution_summary_generator(explanation:GeneratorExplanationBase, document_names:Optional[List[str]]=None, *,
        aggregation:Literal['token', 'sequence', 'bow', 'nucleus']='token',
        method:GeneratorMethods_t='shap',
        absolute:bool=False,
        normalize:bool=True,
        figsize: Tuple[int, int] = (12, 6),
        cmap:str='tab10',
        show:bool=True
    ) -> Union[Figure, None]:
    """Create a summary plot showing mean absolute SHAP values per document.

    Args:
        explanation (GeneratorExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        aggregation (str):                      Aggregation method for probabilities (default: `'token'`).
        absolute (bool):                        Bar plot tf `True`, else waterfall plot (default=`False`).
        normalize (bool):                       If `True`, normalizes the SHAP values across tokens so that the sum of absolute values equals 1 (default=`True`).
        figsize ((int, int)):                   The size of the figure.
        cmap (str):                             The name of a matplotlib colormap used for highlighting.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        `matplotlib.figure.Figure` object if `show == False`.
    """

    # get attribution scores:
    if   method == 'shap': scores = explanation.shap('context', aggregation)
    elif method == 'lime': scores = explanation.lime('context', aggregation)
    else: raise ValueError(f'Unknown method `{method}`.')

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(len(scores))]
    elif len(document_names) != len(scores): raise ValueError('`len(document_names)` does not match the number of documents!')
    
    # create figure:
    fig, ax = plt.subplots(figsize=figsize)

    if absolute:
        # calculate mean absolute attributions per document:
        scores = np.mean(np.abs(scores), axis=1)
        plot_document_vbars(ax,
            scores         = scores,
            document_names = document_names,
            normalize      = normalize,
            cmap           = cmap 
        )
        ax.set_ylabel('Mean Absolute SHAP Attribution')
        
    else:
        # calculate mean attributions per document
        scores = np.mean(scores, axis=1)
        plot_waterfall(ax,
            scores    = scores,
            x_labels  = document_names,
            normalize = normalize,
            cmap      = cmap 
        )
        ax.set_ylabel('Cumulative Attribution')

    # title and x-label:
    ax.set_xlabel('Documents')
    ax.set_title(f'SHAP Attribution Summary - {aggregation.capitalize()} Aggregation', 
                fontsize=14, fontweight='bold')

    fig.tight_layout()
    if show: fig.show()
    else: return fig

def visualize_attribution_generator(explanation:GeneratorExplanationBase, document_names:Optional[List[str]]=None, *,
        aggregation:Literal['token', 'sequence', 'bow', 'nucleus']='token',
        primary_cmap:str='tab10',
        secondary_cmap:str='ragbin',
        show:bool=True,
        output_format:Literal['html', 'latex']='html',
        latex_color_prefix:str='cl',
        **kwargs
    ) -> Union[Figure, str, None]:
    """Visualizes Shapley attribution values while automatically choosing a fitting plot for the chosen `aggregation`.

    Args:
        explanation (GeneratorExplanationBase): An object containing the necessary information for plotting.
        document_names (List[str]):             An optional list of names of the documents.
        aggregation (str):                      Aggregation method for probabilities (default: `'token'`).
        primary_cmap (str, optional):           The name of a matplotlib colormap used for highlighting of different documents.
        secondary_cmap (str, optional):         The name of a matplotlib colormap used for highlighting of the query attributions.
        show (bool):                            If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).
        output_format (str, optional):          Output format: ``'html'`` (default) or ``'latex'`` (only for ``aggregation='token'``).
        latex_color_prefix (str, optional):     Prefix for ``\\definecolor`` names in LaTeX output (default: ``'cl'``).

    Returns:
        The illustration object if `show == False`.
    """

    if aggregation == 'token':
        return higlight_attribution_generator(
            explanation        = explanation,
            document_names     = document_names,
            query_cmap         = secondary_cmap,
            document_cmap      = primary_cmap,
            show               = show,
            output_format      = output_format,
            latex_color_prefix = latex_color_prefix,
            **kwargs
        )

    elif aggregation == 'sequence':
        return plot_attribution_summary_generator(
            explanation    = explanation,
            document_names = document_names,
            aggregation    = aggregation,
            normalize      = kwargs.pop('normalize', True),
            cmap           = primary_cmap,
            show           = show,
            **kwargs
        )

    else:
        return plot_attribution_generator(
            explanation    = explanation,
            document_names = document_names,
            aggregation    = aggregation,
            cmap           = primary_cmap,
            show           = show,
            **kwargs
        )

#====================================================================#
# Plots for retrieval + generation:                                  #
#====================================================================#

from .rag import ExplainableAutoModelForRAG

def plot_document_importance_rag(explanation:ExplainableAutoModelForRAG, document_names:Optional[List[str]]=None, *,
    figsize: Tuple[int, int] = (12, 6),
    mean_color:str='gray',
    cmap:str='tab10',
    show:bool=True
) -> Union[Figure, None]:
    """Plot document importance.

    Args:
        explanation (ExplainableAutoModelForRAG): An object containing the necessary information for plotting.
        document_names (List[str]):               An optional list of names of the documents.
        figsize ((int, int)):                     The size of the figure.
        mean_color (str):                         The color used for the mean values.
        cmap (str):                               The name of a matplotlib colormap used for highlighting.
        show (bool):                              If `True` shows the plot directly, if `False` the plot is returned instead (default: `True`).

    Returns:
        `matplotlib.figure.Figure` object if `show == False`.
    """

    # get y-values:
    y_ret = explanation.retriever_document_importance
    y_gen = explanation.generator_document_importance
    y_avg = (y_ret + y_gen) / 2.

    # get x-values:
    num_documents = y_avg.shape[0]
    x = np.arange(num_documents)

    # get colormap:
    colors = cm.get_cmap(cmap)

    # fallback for document names:
    if document_names is None:
        document_names = [f'Document {i+1:d}' for i in range(num_documents)]
    elif len(document_names) != num_documents: raise ValueError('`len(document_names)` does not match the number of documents!')

    fig, ax = plt.subplots(figsize=figsize)

    # draw average shade:
    ax.bar(x,    y_avg, .8, color=mean_color, alpha=.5)

    # draw bars:
    ax.bar(x-.2, y_ret, .4, label='Retriever', color=colors(0))
    ax.bar(x+.2, y_gen, .4, label='Generator', color=colors(1))

    # draw average:
    ax.hlines(y_avg, x-.4, x+.4, color=mean_color, label='average')

    # draw texts:
    y_top = np.maximum(y_ret, y_gen)
    for i, s in enumerate(document_names):
        ax.text(x[i], max(y_top[i], 0)+.01, s, ha='center', va='bottom')

    # clean frame (keep only left spine):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    if ax.get_ylim()[0] < 0.:
        ax.spines['bottom'].set_visible(False)

        # add horizontal grid line at zero:
        ax.axhline(y=0, color='black', linewidth=0.5)

    # add horizontal grid lines at each tick:
    for tick_loc in ax.get_yticks(minor=False):
        ax.axhline(y=tick_loc, color='lightgray', linewidth=0.5, zorder=0)

    ax.set_title(f'Overall Document Importance', 
                fontsize=14, fontweight='bold')
    ax.set_xticks([])
    ax.set_ylabel('Normalized Importance')
    ax.legend()

    fig.tight_layout()

    if show: fig.show()
    else: return fig

def higlight_importance_rag(explanation:ExplainableAutoModelForRAG, document_names:Optional[List[str]]=None, *,
        threshold:float=.5,
        retriever_method:RetrieverMethods_t='intGrad',
        generator_method:GeneratorMethods_t='shap',
        retriever_token_processor:Optional[Callable[[str],str]]=None,
        generator_token_processor:Optional[Callable[[str],str]]=None,
        query_cmap:str='ragbin',
        document_cmap:str='tab10',
        show:bool=True,
        output_format:Literal['html', 'latex']='html',
        latex_color_prefix:str='cl',
        characters_per_line:int=150,
        **kwargs
    ) -> str:
    """Highlights tokens in a text sequence where a single document contributes
    at least `threshold` of the total positive SHAP value at that token.

    Args:
        explanation (ExplainableAutoModelForRAG):         An object containing the necessary information for plotting.
        document_names (List[str]):                       An optional list of names of the documents.
        threshold (float, optional):                      Minimum attribution for highlighting in the intervall
                                                          `[0., 1.)` (default: `0.5`).
        retriever_method (str, optional):                 The method for calculating retriever token importance.
        generator_method (str, optional):                 The method for calculating generator token importance.
        retriever_token_processor ((str) -> str, optional): Optional function applied to each retriever token
                                                          before printing.
        generator_token_processor ((str) -> str, optional): Optional function applied to each generator token
                                                          before printing.
        query_cmap (str, optional):                       The name of a matplotlib colormap used for highlighting
                                                          of the query attributions.
        document_cmap (str, optional):                    The name of a matplotlib colormap used for highlighting
                                                          of different documents.
        show (bool, optional):                            If `True` shows the plot directly, if `False` the plot
                                                          is returned instead (default: `True`).
        output_format (str, optional):                    Output format: ``'html'`` (default) or ``'latex'``.
        latex_color_prefix (str, optional):                Prefix for ``\\definecolor`` names in LaTeX output
                                                          (default: ``'cl'``).
        characters_per_line (int, optional):              Wrap LaTeX rows at this many visible characters
                                                          (default: ``None`` = no wrapping).

    Returns:
        A formatted string highlighting dominant attribution regions if `show == False`.
    """

    # get retriever scores:
    retriever_attr = get_retriever_scores(explanation.retriever, retriever_method, **kwargs)

    # compute absolute:
    retriever_attr = {key:[np.abs(doc) for doc in docs] for key, docs in retriever_attr.items()}

    # get generator attribution scores:
    generator_attr = get_generator_scores(explanation.generator, generator_method)

    # get special tokens:
    retriever_special_tokens = set(explanation.retriever.tokenizer.all_special_tokens)
    generator_special_tokens = set(explanation.generator.tokenizer.all_special_tokens)

    # fallback for document names:
    if document_names is None:
        sep = '~' if output_format == 'latex' else '&nbsp;'
        document_names = [f'Document{sep}{i+1:d}' for i in range(len(retriever_attr['context']))]
    elif len(document_names) != len(retriever_attr['context']): raise ValueError('`len(document_names)` does not match the number of documents!')

    # get indices of the actual content:
    # (excluding chat template specific tokens) 
    response = decode_chat_template(
        explanation.generator.gen_tokens,
        explanation.generator.model_name_or_path,
        return_indices=True
    )[0]

    # create role:
    role = response['role']
    if generator_token_processor is not None:
        role = [generator_token_processor(t) for t in role]
    role = ''.join(role).strip().capitalize()

    # create query html:
    cmb_qry, ret_qry_attr, gen_qry_attr = match_token_attributions(
        retriever_attr['query'][0].numpy(),   explanation.retriever.in_tokens['query'][0],
        generator_attr['query'].mean(axis=1), explanation.generator.qry_tokens,
        ret_token_processor = retriever_token_processor,
        gen_token_processor = generator_token_processor
    )
    html = highlight_dominant_passages(
        scores             = np.stack([ret_qry_attr, gen_qry_attr]),
        tokens             = cmb_qry,
        title              = 'Query',
        labels             = ['Retriever', 'Generator'],
        color_mode         = 'average',
        legend             = True,
        cmap               = query_cmap,
        output_format      = output_format,
        latex_color_prefix = latex_color_prefix,
    )
    if output_format == 'latex' and characters_per_line is not None:
        html = _split_latex_row(html, characters_per_line)

    # plot contexts:
    for i, s in enumerate(retriever_attr['context']):
        _row = highlight_dominant_passages(
            scores             = s.numpy(),
            tokens             = explanation.retriever.in_tokens['context'][i],
            title              = document_names[i],
            labels             = ['+', '-'],
            threshold          = threshold,
            max_score          = np.concatenate(retriever_attr['context']).max(),
            skip_tokens        = retriever_special_tokens,
            token_processor    = retriever_token_processor,
            legend             = True,
            cmap               = query_cmap,
            output_format      = output_format,
            latex_color_prefix = latex_color_prefix,
        )
        if output_format == 'latex' and characters_per_line is not None:
            _row = _split_latex_row(_row, characters_per_line)
        html += _row

    # create response row:
    html_response = highlight_dominant_passages(
        scores             = generator_attr['context'][:,response['content']],
        tokens             = explanation.generator.gen_tokens[response['content']],
        title              = role,
        labels             = document_names,
        threshold          = threshold,
        skip_tokens        = generator_special_tokens,
        token_processor    = generator_token_processor,
        legend             = True,
        cmap               = document_cmap,
        output_format      = output_format,
        latex_color_prefix = latex_color_prefix,
    )
    if output_format == 'latex' and characters_per_line is not None:
        html_response = _split_latex_row(html_response, characters_per_line)

    if output_format == 'latex':
        latex_str = _wrap_latex_tabular(html + html_response, latex_color_prefix)
        if show: print(latex_str); return None
        else: return latex_str

    # Build, display, and return final HTML:
    html_str = (
        '<!DOCTYPE html>\n' +
        '<html>\n' +
        '<head>\n' +
        '<title>Generated Text</title>\n' +
        '</head>\n' +

        '<body>\n' +
        '<table>\n' +
        html +
        html_response +
        '</table>\n' +
        '</body>\n' +

        '</html>'
    )

    if show: display(HTML(html_str)); return None
    else: return html_str

#====================================================================#
# (New) Plots for Part-of-Speech (POS) Analysis                      #
#====================================================================#

def tokens_to_pos(tokens, detok="auto"):
    """Converts subword tokens into words, tags them with POS, and maps back to original token indices.

    Args:
        tokens (List[str]): Subword tokens to convert (e.g. RoBERTa/GPT-2 `'Ġ'`-prefixed or BERT `'##'`-prefixed).
        detok (str):         Detokenization strategy. Currently unused; reserved for future tokenizer-specific handling.

    Returns:
        A tuple `(pos_per_token, pos_per_word, mapping, words)`:
        - `pos_per_token`: The POS tag for each original token.
        - `pos_per_word`:  The POS tag for each reconstructed word.
        - `mapping`:       The word index each original token maps to.
        - `words`:         The reconstructed words.

    Note:
        Requires `spacy` and a downloaded model. You may need to run::

            pip install spacy pandas
            python -m spacy download en_core_web_sm
    """

    # load small english model
    nlp = spacy.load("en_core_web_sm")

    # simple detok heuristic (works ok with GPT-style tokens like 'Ġhello')
    words = []
    mapping = []  # map each original token index -> word index
    current_word = ""
    word_idx = -1

    for i, tok in enumerate(tokens):
        # roberta/gpt2 style
        if tok.startswith("Ġ") or tok.startswith(" "):
            # start new word
            words.append(tok[1:])
            word_idx = len(words)-1
            mapping.append(word_idx)
        elif tok.startswith("##"):  # bert/wordpiece
            # continuation
            if words:
                words[-1] += tok[2:]
                mapping.append(word_idx)
            else:
                words.append(tok[2:])
                word_idx = len(words)-1
                mapping.append(word_idx)
        else:
            if words:
                words[-1] += tok
                mapping.append(word_idx)
            else:
                words.append(tok)
                word_idx = len(words)-1
                mapping.append(word_idx)

    text = " ".join(words)
    doc = nlp(text)

    # now assign POS back to each token
    pos_per_word = [token.pos_ for token in doc]  # e.g. "NOUN", "VERB", etc.
    pos_per_token = [pos_per_word[m] if m < len(pos_per_word) else "X" for m in mapping]

    return pos_per_token, pos_per_word, mapping, words

def shap_by_pos(shap_docs_tokens, pos_tags, doc_names=None, absolute=True):
    """Aggregates per-token contributions into per-POS-category contributions.

    Args:
        shap_docs_tokens (NDArray): Attribution scores with shape `(num_documents, num_tokens)`.
        pos_tags (List[str]):      The POS tag for each token; length must match `num_tokens`.
        doc_names (List[str]):     An optional list of names of the documents.
        absolute (bool):           If `True`, aggregates absolute contributions (default: `True`).

    Returns:
        A `POS x Documents` `pandas.DataFrame` of mean contributions per POS category.
    """
    S = np.asarray(shap_docs_tokens)   # (D, T)
    if absolute:
        S = np.abs(S)

    D, T = S.shape
    assert len(pos_tags) == T

    if doc_names is None:
        doc_names = [f"Doc {i+1}" for i in range(D)]

    # unique POS categories
    pos_set = sorted(set(pos_tags))
    agg = {pos: S[:, np.array(pos_tags) == pos].mean(axis=1) for pos in pos_set}

    df = pd.DataFrame(agg, index=doc_names).T
    return df

def plot_shap_by_pos(df, cmap="tab10", figsize=(10,6),dpi=300, normalize=True ,save_path=None,show=True):
    """Plots a stacked bar chart of per-document contributions per POS category.

    Args:
        df (pd.DataFrame):     A `POS x Documents` DataFrame, as returned by `shap_by_pos()`.
        cmap (str):             The name of a matplotlib colormap used for highlighting.
        figsize ((int, int)):   The size of the figure.
        dpi (int):              The resolution of the figure.
        normalize (bool):       If `True`, normalizes each POS row so its values sum to 1 (default: `True`).
        save_path (str):        An optional path to save the figure to.
        show (bool):            If `True`, shows the plot directly (default: `True`).

    Returns:
        The `matplotlib.figure.Figure` object.
    """
    if normalize:
        df = df.div(df.sum(axis=1), axis=0)  # normalize per POS
    fig, ax = plt.subplots(figsize=figsize, dpi = dpi) # Create figure and axes

    # Generate colors consistently with other plot functions
    num_docs = df.shape[1]
    cmap_obj = cm.get_cmap(cmap)
    colors = [cmap_obj(i) for i in range(num_docs)]

    # Pass the explicit list of colors to the plot function
    df.plot(kind="bar", stacked=True, color=colors, ax=ax)

    ax.set_ylabel("Normalized contribution" if normalize else "Mean contribution")
    ax.set_xlabel("POS tag")
    ax.set_title("Document impact by Part-of-Speech")
    plt.xticks(rotation=45, ha="right")
    plt.legend(title="Documents", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    if show:
        plt.show()

    return fig

#====================================================================#
# (New) Plots for Global Importance Analysis                         #
#====================================================================#

def safe_load_pickle(file_path):
    """
    Safely loads a pickle file, handling potential exceptions.

    Args:
        file_path (str): The path to the .pkl file to load.

    Returns:
        dict: The object loaded from the pickle file, or None if an error occurs.
    """
    
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        return obj
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def process_global_importance(pickle_files: list, safe_load_pickle_func) -> pd.DataFrame:
    """
    Processes a list of pickle files to calculate the average importance of each document.

    Args:
        pickle_files (list): List of paths to the .pkl files.
        safe_load_pickle_func (function): The function to safely load the pickle files.

    Returns:
        pd.DataFrame: A DataFrame where each column is a document and each row
                      represents the average importance of that document for a single query.
    """
    all_doc_importances = []

    for file_path in pickle_files:
        try:
            obj = safe_load_pickle_func(file_path)
            if "shapley_values_token" in obj and "context" in obj["shapley_values_token"]:
                S = np.asarray(obj["shapley_values_token"]["context"])
                # Calculate the average (absolute) importance for each document in this file
                mean_importance_per_doc = np.abs(S).mean(axis=1)
                all_doc_importances.append(mean_importance_per_doc)
        except Exception:
            # Ignore files that cannot be loaded or do not have the necessary data
            continue
    
    if not all_doc_importances:
        return pd.DataFrame()

    # Determine the maximum number of documents found
    max_docs = max(len(row) for row in all_doc_importances)
    doc_names = [f"Doc {i+1}" for i in range(max_docs)]

    # Pad rows with fewer documents than max_docs with NaN so every row has the same length
    padded_importances = [
        np.pad(row, (0, max_docs - len(row)), constant_values=np.nan)
        for row in all_doc_importances
    ]

    # Create the DataFrame
    df = pd.DataFrame(padded_importances, columns=doc_names)
    return df

def plot_global_importance_distribution(df: pd.DataFrame, cmap="tab10", figsize=(12, 7), dpi=300):
    """Creates a box plot to visualize the importance distribution of each document across queries.

    Args:
        df (pd.DataFrame):     A `Queries x Documents` DataFrame, as returned by `process_global_importance()`.
        cmap (str):             The name of a matplotlib colormap used for highlighting.
        figsize ((int, int)):   The size of the figure.
        dpi (int):              The resolution of the figure.

    Returns:
        The `matplotlib.figure.Figure` object, or `None` if `df` is empty.
    """
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    num_docs = df.shape[1]
    cmap_obj = cm.get_cmap(cmap)
    colors = [cmap_obj(i) for i in range(num_docs)]

    # Box plot
    box = ax.boxplot(df.values, patch_artist=True, labels=df.columns)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Calculate and plot the average
    means = df.mean().values
    ax.scatter(range(1, num_docs + 1), means, marker='o', color='red', s=50, zorder=3, label='Global Mean')

    ax.set_title(f"Distribution of Document Importance over {len(df)} Queries")
    ax.set_ylabel("Mean Absolute Importance per Query")
    ax.set_xlabel("Source Documents")
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    ax.legend()
    fig.tight_layout()
    
    return fig