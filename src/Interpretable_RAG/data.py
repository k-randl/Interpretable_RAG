import os
import regex as re
from tqdm import tqdm
from html.parser import HTMLParser
from urllib.request import urlretrieve

from typing import Callable, List

#====================================================================================================#
# Data processing:                                                                                   #
#====================================================================================================#

from nltk.tokenize import PunktSentenceTokenizer, word_tokenize
SENT_TOKENIZER = PunktSentenceTokenizer()

class HTMLSplitter(HTMLParser):
    def __init__(self, drop_tables:bool=True):
        super().__init__()
        self.reset()
        self.strict = False
        self.drop_tables = drop_tables
        self.convert_charrefs= True
        self.text = ''
        self.paused = 0

    @property
    def ends_with_space(self) -> bool:
        """Boolean indicating whether the current html ends in a space-character."""
        if len(self.text) == 0: return True
        else: return self.text[-1] in (' ', '\n', '\r', '\t', '>')

    @property
    def ends_with_newline(self) -> bool:
        """Boolean indicating whether the current html ends in a linebreak."""
        if self.text.endswith('\n  - '): return True
        elif len(self.text) == 0: return True
        else: return self.text[-1] in ('\n', '\r', '>')
    
    def handle_starttag(self, tag:str, attrs:list[tuple[str, str | None]]) -> None:
        """Handle the start of an HTML tag by appending an appropriate piece of text/markup to self.text.

        Args:
            tag (str):      The lower-case name of the HTML tag (e.g. 'p', 'b', 'i', 'li', 'table',
                            'tr', 'th', 'td', 'h1', ...).
            attrs (list):   A list of (`name`, `value`) attribute tuples for the tag. This method
                            currently ignores attributes but accepts the parameter to match an
                            HTML parser callback signature.

        Notes:
            Mutates self.text by appending formatting or markup that approximates the appearance of the
            tag in a plain-text / Markdown-like representation. Uses the boolean flags
            `self.ends_with_newline` and `self.ends_with_space` to decide whether to insert leading newlines
            or spaces before the added text. Tag-specific behavior:
            - 'table', 'tr', 'th', 'td':
                    Append an opening tag literal like "<table>" or "<td>".
                    If self.ends_with_newline is True, append directly; otherwise prepend a newline.
            - 'b', 'strong':
                    Append the Markdown bold delimiter "**".
                    If the current text does not end with a space, a single space is inserted
                    before the delimiters to avoid running words together.
            - 'i':
                    Append the Markdown italic delimiter "*", following the same spacing rule as
                    bold (a leading space is inserted when the current text does not end with a space).
            - 'li':
                    Start a list item by appending "  - " at the current position; if not already
                    at a newline, a leading newline is inserted.
            - 'p':
                    Ensure paragraph separation by appending a single newline if the text does not
                    already end with one.
            - Tags starting with 'h' (headings, e.g. 'h1', 'h2'):
                    Append bold markers for a heading line. If already at a newline, append "\n**";
                    otherwise append "\n\n**" to ensure the heading is separated from prior text.
            - Default:
                    If none of the above conditions match and the text does not already end with a
                    space, append a single space.
        """

        if tag in ('head','script','style'):
            self.paused += 1

        elif tag == 'table':
            if self.drop_tables: self.paused += 1
            else: self.text = self.text.strip() + f'\n\n<{tag}>'

        elif self.paused > 0:
            return

        elif tag == 'tr' and not self.drop_tables:
            self.text += f'<{tag}>' if self.ends_with_space else f' <{tag}>'

        elif tag in ('th', 'td') and not self.drop_tables:
            self.text += f'<{tag}>'
        
        elif tag == 'b' or tag == 'strong':
            self.text += '**' if self.ends_with_space else ' **'

        elif tag == 'i':
            if not self.ends_with_space:
                self.text += '*' if self.ends_with_space else ' *'

        elif tag == 'li':
            self.text += '  - ' if self.ends_with_newline else '\n  - '

        elif tag == 'p':
            self.text = self.text.strip() + '\n\n'

        elif tag.startswith('h'):
            self.text = self.text.strip() + '\n\n**'

        elif not self.ends_with_space:
            self.text += ' '

    def handle_endtag(self, tag:str) -> None:
        """Handle the end of an HTML tag by appending appropriate closing markers to the
        parser's text buffer.

        Args:
            tag (str):  The name of the HTML tag that has ended (expected to be lowercase).
                        The method examines this tag to decide what trailing text to append.
        
        Notes:
            This method is intended to be invoked when an HTML closing tag is encountered.
            It updates self.text (and relies on self.ends_with_space to avoid inserting
            duplicate whitespace) to produce a plain-text/markdown-like representation of
            the HTML structure. It does not return a value; it mutates instance state.
            Behavior by tag:
            - 'table', 'tr', 'th', 'td'
                - Append the literal closing HTML tag (e.g. '</table>') to preserve table
                structure in the output.
            - 'b', 'strong'
                - Append a Markdown-style bold closing marker '** ' (two asterisks followed
                by a space).
            - 'i'
                - Append a Markdown-style italic closing marker '* ' (one asterisk plus a
                space).
            - 'p'
                - Append a newline ('\n') if the current buffer does not already end with
                whitespace.
            - tags starting with 'h' (e.g. 'h1', 'h2', ...)
                - Append a bold marker followed by a newline ('**\n') to separate headings.
            - any other tag
                - Append a single space if the buffer does not already end with whitespace.
        """

        if tag in ('head','script','style'):
            self.paused -= 1

        elif tag == 'table':
            if self.drop_tables: self.paused -= 1
            else: self.text = self.text.strip() + f'</{tag}>\n\n'

        elif self.paused > 0:
            return

        elif tag == 'tr' and not self.drop_tables:
            self.text = self.text.strip() + f'</{tag}>\n'

        elif tag in ('th', 'td') and not self.drop_tables:
            self.text = self.text.strip() + f'</{tag}>'

        if tag == 'b' or tag == 'strong':
            self.text = self.text.strip() + '** '

        elif tag == 'i':
            self.text = self.text.strip() + '* '

        elif tag == 'p':
            self.text = self.text.strip() + '\n\n'

        elif tag.startswith('h'):
            self.text = self.text.strip() + '**\n'

        elif not self.ends_with_space:
            self.text += ' '
    
    def handle_data(self, data:str):
        """Append sanitized html to the instance's text buffer.

        Args:
            data (str): The input string to append. All newline ('\\n') and carriage
                        return ('\\r') characters are removed and leading/trailing
                        whitespace is stripped before appending.
        """
        
        if self.paused == 0:
            self.text += data.replace('\n', '').replace('\r', '').strip()

    def get_data(self):
        """Return the instance's text with leading and trailing whitespace removed.
        This method returns a cleaned version of the instance attribute `self.text`
        by calling its `strip()` method. The attribute is expected to be a string.

        Returns:
            str: The value of `self.text` with leading and trailing whitespace removed.

        Raises:
            AttributeError: If the instance has no `text` attribute or if `self.text` is None.
        """

        return self.text.strip()

def load_html(html:str, window:int, *, tokenize:Callable[[str], List[str]]=word_tokenize, output_tokens:bool=False, handle_wiki_tags:bool=False):
    """Load and split an HTML string into fixed-size "parts" of tokenized text. This function strips
    HTML markup, splits the resulting text into paragraphs (by double-newline "\\n\\n"), then splits
    paragraphs into sentence spans via `SENT_TOKENIZER.span_tokenize`.
    Each sentence is tokenized with the provided `tokenize` callable and sentences are grouped into
    consecutive "parts" such that each part contains up to `window` tokens (subject to sentence-level
    truncation). The function returns parallel lists describing the part index, originating paragraph,
    token sequence and the raw text span for each part.

    Args:
        html (str):                     Input HTML content.
        window (int):                   Target maximum number of tokens per part. The algorithm
                                        attempts to pack sentences into parts without exceeding
                                        this window; individual sentences are truncated to at most
                                        `window` tokens if they exceed it.
        tokenize ((str) -> List[str])):  Optional tokenization function applied to each sentence
                                        (default: `nltk.word_tokenize`). It must accept a string
                                        and return a list of token strings.
        output_tokens (bool):           If `True`, return lists of tokens instead of full texts.
        handle_wiki_tags (bool):        If `True`, remove wikipedia refs like "[ 1 ] ".

    Yields:
        A tuple `(text, paragraph, part)`:
        - 'texts'    : List[str] — chunk of the html-document.
        - 'paragraph': List[int] — paragraph indices from which the part was sourced.
        - 'part'     : List[int] — integer indices of the part within each papragraph.
    """
    # parse html:
    parser = HTMLSplitter()
    parser.feed(html)
    html = parser.get_data()

    # handle wikipedia tags:
    bracket_re = re.compile(r"\[\s[^]]+?\s\]\s")
    multispace_re = re.compile(r"\s{2,}")
    punct_re = re.compile(r"\s+([.,;:!?])")

    if handle_wiki_tags:
        html = bracket_re.sub("", html)
    html = multispace_re.sub(" ", html)
    html = punct_re.sub(r"\1", html)

    # split text in paragraphs:
    remaining_tokens = -1
    tokens = []
    for paragraph, text in enumerate(html.split('\n\n')):
        
        part = 0
        cursor = 0
        for i,j in SENT_TOKENIZER.span_tokenize(text):
            sentence = tokenize(text[i:j+1])
            remaining_tokens -= len(sentence) - 1

            if remaining_tokens <= 0:
                yield tokens if output_tokens else text[cursor:j+1].strip(), paragraph, part
                tokens = []
                remaining_tokens = window

                part += 1
                cursor = j+1

            else: tokens.extend(sentence[:window])

        if cursor < j+1:
            yield tokens if output_tokens else text[cursor:j+1].strip(), paragraph, part
        remaining_tokens = -1

def load_data(urls:List[str], window:int, *, tokenize:Callable[[str], List[str]]=word_tokenize, output_tokens:bool=False):
    for url in tqdm(urls):
        # get online content:
        try:
            if not os.path.exists(url):
                path, _ = urlretrieve(url)
            else: path = url
        except Exception as e:
            print(f'Error when retrieving "{url}": {e}')
            continue

        # read contents:
        with open(path, 'r') as file:
            html = file.read()

        # process document:
        for text, i, j in load_html(html=html, window=window, tokenize=tokenize, output_tokens=output_tokens):
            yield text, url, i, j