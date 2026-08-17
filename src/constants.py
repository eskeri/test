"""Constants and configuration for signature phrase discovery."""

# Cache versioning
# Increment this number when cache semantics change (tokenization, lemmatization, etc.)
CACHE_VERSION = 3

# Cache only n-grams of size 1 through 5 to disk
# Higher-order n-grams are computed on the fly to save disk space
CACHE_MAX_N = 5

# Graph color theme for funnel charts
FOCUS_COLOR = "#00DDFF"           # Color for the focus channel
OTHER_COLOR = "#f0d000"           # Color for other channels
FUNNEL_BG_COLOR = "#1C2A49"      # Background color for funnel chart
FUNNEL_TEXT_COLOR = "#f3f4f6"    # Text color for funnel chart
FUNNEL_LABEL_OUTLINE_COLOR = "#000000"  # Label outline color
FUNNEL_FONT_FAMILY = '"Avenir Next","Segoe UI",Helvetica,Arial,sans-serif'

# Common stopwords to filter out from graph displays
# These are words that appear frequently but don't carry much meaning
GRAPH_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me", "my",
    "no", "not", "of", "on", "or", "our", "ours", "she", "so", "that", "the",
    "their", "theirs", "them", "then", "there", "these", "they", "this", "those",
    "to", "too", "up", "us", "very", "was", "we", "were", "what", "when", "where",
    "which", "who", "why", "will", "with", "you", "your", "yours", "it's", "i'm",
    "i've", "did", "do", "does", "having", "could", "would", "should", "can",
    "may", "might", "you've", "you'll", "he's", "she's", "we're", "they're",
    "who's", "that's", "there's", "i'll", "we've", "you're", "000", "subscribe",
    "channel", "like", "comment", "share", "hit", "smash", "button", "subscribing",
    "subscribed", "don't", "doesn't", "didn't", "can't", "couldn't", "won't",
    "wouldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
    "yes", "no", "we'll", "h", "b", "cr", "let's", "d",
}

# =============================================================================
# HTML EXCLUSION RULES
# =============================================================================
# These rules filter out certain phrases from the HTML output display.
# They are NOT applied to the statistical scoring - only to what gets shown.
#
# Each rule can have:
#   - channels: list of channel names this rule applies to (omit for all channels)
#   - categories: list of result categories to filter (exclusive, distinctive, dominant, mi, delta_p, log_likelihood)
#   - contains_any: phrases containing these substrings (case-insensitive)
#   - exact_any: exact phrase matches (case-insensitive)
#   - word_any: whole-word matches (case-insensitive)
# =============================================================================

HTML_ENTRY_EXCLUSION_RULES = [
    {
        "channels": ["slogo", "finoggin"],
        "categories": ["exclusive", "distinctive", "dominant", "mi", "delta_p", "log_likelihood"],
        "word_any": [
            "jelly", "crainer", "slogo", "slowgo", "josh", "joash", "joshua", "jos",
            "victor", "vtor", "vic", "rishy", "rishi", "richishy", "rish", "rich",
            "reishi", "rashy", "rashi", "richy", "richie", "rissy", "rushi",
            "richish", "richishi", "rushy", "rashid", "stew", "steu", "stu", "ste",
            "stubaroo", "stubaru", "stubu", "finn", "fin", "finnegan", "vinn",
            "fenoggin", "finoggin", "foggen", "finny", "binn", "craner", "crater",
            "crainer", "cran", "crane", "crano", "krona", "krina", "kina", "grina",
            "krana", "rei", "karina", "crer", "crara", "kroner", "crana", "cranor",
            "cranar", "crina", "joshy", "slowo", "st", "wishy", "tiff", "hickey"
        ],
    },
    {
        "channels": ["ctop"],
        "categories": ["exclusive", "distinctive", "dominant", "mi", "delta_p", "log_likelihood"],
        "word_any": [
            "chris", "christopher", "kris", "krisa", "chrisa", "setop", "cindy", "krusty",
            "john", "toop", "ctop", "see top", "seop", "chrises", "chis", "chrise", "krusty's", "crusty's",
            "chr", "seatop", "cris", "crusty", "se", "love", "chris's", "john's",
        ],
    },
    {
        "channels": ["smii7y"],
        "categories": ["exclusive", "distinctive", "dominant", "mi", "delta_p", "log_likelihood"],
        "word_any": [
            "evan", "vanoss", "brian", "wild cat", "marcel", "wildcat", "louie",
            "marel", "noga", "anthony", "lou", "nogle", "nog", "denogla", "puffer",
            "matt", "nogla", "moo", "tyler", "pasta", "spoon", "nola", "venos",
            "john", "smitty", "delirious", "terrorizer", "lui", "louis", "delilous",
            "delious",
        ],
    },
    {
        "channels": ["zud", "sigils", "bifflewiffle", "mitzefy", "helloiamkate", "sigilstv"],
        "categories": ["exclusive", "distinctive", "dominant", "mi", "delta_p", "log_likelihood"],
        "word_any": [
            "sigils", "biffle", "ssundee", "zud", "z", "zed", "bipple", "sig",
            "sigi", "sigu", "ssun", "ssundi", "ssund", "ssunde", "sundey", "sidos",
            "ciddles", "siddles", "bivvle", "bevil", "bivival", "pimple", "bimple",
            "siggles", "cindles", "bivville", "pivvil", "sunny", "sunday",
            "sunundy", "silent", "bibble", "biff", "pipple", "beville", "mitify",
            "bivl", "bibl", "bevel", "bivil", "vivvil", "bivvil", "siddle", "sigo",
            "mify", "kate", "mitz", "mits", "biffl", "biffold", "sigles", "sles",
            "sals", "pival", "sundy", "piffle", "bibbo", "bipp",
        ],
    },
    {
        "channels": ["whackycast"],
        "categories": ["exclusive", "distinctive", "dominant", "mi", "delta_p", "log_likelihood"],
        "word_any": ["whacky", "fudgy", "wack", "wacky", "fugy", "fudy", "whack"],
    },
    {
        "channels": ["blitz"],
        "categories": ["exclusive", "distinctive", "dominant", "mi", "delta_p", "log_likelihood"],
        "word_any": ["intern"],
    },
]
