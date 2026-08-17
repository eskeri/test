"""Lemmatization functions for word normalization.

Lemmatization converts words to their base form (e.g., "running" -> "run").
This helps group different forms of the same word together for analysis.

IMPORTANT: Lemmatization is only applied to unigrams (n=1). Higher n-grams
use the original (non-lemmatized) tokens to preserve word order and context.
"""

from typing import List

# Optional imports for lemmatization (word normalization)
try:
    import spacy
except ImportError:
    spacy = None

try:
    import simplemma
except ImportError:
    simplemma = None


def _should_lemmatize(args) -> bool:
    """Check if lemmatization is enabled in the configuration.
    
    Args:
        args: Command-line arguments
        
    Returns:
        True if lemmatization should be applied, False otherwise
    """
    return getattr(args, 'lemmatize', False)


def _lemmatize_token(token: str, nlp) -> str:
    """Lemmatize a single token using spaCy.
    
    This processes one token at a time, which is slower than batch processing
    but useful for individual tokens. The context is ignored for speed.
    
    Args:
        token: A single word token to lemmatize
        nlp: The spaCy NLP object (or None if lemmatization disabled)
        
    Returns:
        The lemmatized (base form) of the token in lowercase
    """
    if not token or nlp is None:
        return token.lower()
    
    # Process just this single token
    doc = nlp(token)
    if doc and len(doc) > 0:
        lemma = doc[0].lemma_.strip().lower()
        return lemma if lemma else token.lower()
    return token.lower()


def _lemmatize_tokens(tokens: List[str], nlp, batch_size: int = 10000) -> List[str]:
    """Lemmatize a list of tokens using spaCy's batch processing.
    
    This uses spaCy's nlp.pipe() method which is much faster than processing
    tokens one at a time. It processes tokens in batches for efficiency.
    
    Args:
        tokens: List of word tokens to lemmatize
        nlp: The spaCy NLP object (or None if lemmatization disabled)
        batch_size: Number of tokens to process in each batch (default: 10,000)
        
    Returns:
        List of lemmatized tokens in the same order as input
    """
    if not tokens or nlp is None:
        return [t.lower() for t in tokens]

    lemmatized = []
    for i, doc in enumerate(nlp.pipe(tokens, batch_size=batch_size)):
        if doc and len(doc) > 0:
            lemma = doc[0].lemma_.strip().lower()
            lemmatized.append(lemma if lemma else tokens[i].lower())
        else:
            lemmatized.append(tokens[i].lower())

    return lemmatized


def _lemmatize_tokens_chunked(tokens: List[str], nlp, chunk_size: int = 50000) -> List[str]:
    """Lemmatize a large token list by processing in chunks.
    
    For very large token lists (e.g., entire channel transcripts), this function
    breaks the work into smaller chunks to avoid memory issues.
    
    Args:
        tokens: List of tokens to lemmatize (could be very large)
        nlp: The spaCy NLP object, or the string "simplemma" for simplemma
        chunk_size: Maximum tokens per chunk (default: 50,000)
        
    Returns:
        List of lemmatized tokens
    """
    if not tokens or nlp is None:
        return [t.lower() for t in tokens]
    
    # simplemma doesn't need chunking - it's already fast enough
    if nlp == "simplemma":
        return _lemmatize_tokens_simplemma(tokens)
    
    # Process spaCy tokens in chunks to manage memory
    lemmatized = []
    for i in range(0, len(tokens), chunk_size):
        chunk = tokens[i:i + chunk_size]
        lemmatized.extend(_lemmatize_tokens(chunk, nlp))
    
    return lemmatized


def _lemmatize_token_simplemma(token: str, lang: str = 'en') -> str:
    """Lemmatize a single token using the simplemma library.
    
    simplemma is a lightweight alternative to spaCy that's faster but
    potentially less accurate. It doesn't require loading large models.
    
    Args:
        token: A single word token to lemmatize
        lang: Language code (default: 'en' for English)
        
    Returns:
        The lemmatized token in lowercase, or original if lemmatization fails
    """
    if not token or simplemma is None:
        return token.lower()
    
    try:
        lemma = simplemma.lemmatize(token, lang=lang)
        return lemma.lower() if lemma else token.lower()
    except Exception:
        return token.lower()


def _lemmatize_tokens_simplemma(tokens: List[str], lang: str = 'en') -> List[str]:
    """Lemmatize a list of tokens using the simplemma library.
    
    This processes tokens one at a time but is still fast because simplemma
    is lightweight and doesn't require the overhead that spaCy does.
    
    Args:
        tokens: List of word tokens to lemmatize
        lang: Language code (default: 'en' for English)
        
    Returns:
        List of lemmatized tokens in the same order as input
    """
    if not tokens or simplemma is None:
        return [t.lower() for t in tokens]
    
    lemmatized = []
    for token in tokens:
        try:
            lemma = simplemma.lemmatize(token, lang=lang)
            lemmatized.append(lemma.lower() if lemma else token.lower())
        except Exception:
            lemmatized.append(token.lower())
    
    return lemmatized
