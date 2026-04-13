import asyncio
import copy
import hashlib
import json
import logging
import math
import os
import random
import re
import threading
import time
from collections import Counter, OrderedDict, deque
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, field_validator

# ==============================================================================
# CONFIGURATION - LOGGER SETUP (before optional dependencies)
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ==============================================================================
# OPTIONAL DEPENDENCIES
# ==============================================================================
try:
    import orjson
    USE_ORJSON = True
except ImportError:
    USE_ORJSON = False
    import json

try:
    import httpx
    USE_HTTPX = True
except ImportError:
    USE_HTTPX = False

try:
    import ahocorasick
    USE_AHOCORASICK = True
except ImportError:
    USE_AHOCORASICK = False
    logger.warning("ahocorasick not installed - using fallback pattern matching")

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)
    vader_analyzer = SentimentIntensityAnalyzer()
    USE_VADER = True
except ImportError:
    USE_VADER = False

# API Keys
VALID_API_KEYS = set()
raw_keys = os.getenv("API_KEYS", "")
if raw_keys:
    VALID_API_KEYS = {k.strip() for k in raw_keys.split(",") if k.strip()}
if not VALID_API_KEYS:
    logger.warning("No API_KEYS configured - running in open mode")

# Cache settings
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_CACHE_SIZE = int(os.getenv("MAX_CACHE_SIZE", "1000"))

# Constants
PLACEHOLDER_API_KEYS = {"your_gemini_api_key_here", "replace_with_real_key"}
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
LEGACY_MODEL_ALIASES = {"gemini-pro": DEFAULT_GEMINI_MODEL}

# Performance timeouts
GEMINI_TIMEOUT = 4
REQUEST_TIMEOUT = 30

# Limits
MAX_POINTS = int(os.getenv("MAX_POINTS", "6"))
MAX_NEUTRAL_POINTS = int(os.getenv("MAX_NEUTRAL_POINTS", "2"))
MAX_REVIEWS = int(os.getenv("MAX_REVIEWS", "100"))
MAX_REVIEWS_TO_ANALYZE = int(os.getenv("MAX_REVIEWS_TO_ANALYZE", "30"))
SENTIMENT_POLARITY_THRESHOLD = float(os.getenv("SENTIMENT_POLARITY_THRESHOLD", "0.1"))

# Rate Limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))
RATE_LIMIT_PER_HOUR = int(os.getenv("RATE_LIMIT_PER_HOUR", "100"))

# Input Limits
MAX_INPUT_SIZE = 10000
MAX_CLAUSES_FOR_AI = int(os.getenv("MAX_CLAUSES_FOR_AI", "20"))
GEMINI_MAX_CLAUSES = 10
GEMINI_MAX_CHARS = 300

# API version prefix
API_V1_PREFIX = "/api/v1"
API_V2_PREFIX = "/api/v2"

# Streaming
STREAM_CHUNK_DELAY = 0.02

# Filler/Connector words (frozenset for O(1) lookup)
FILLER_WORDS = frozenset({
    "honestly", "basically", "actually", "literally", "overall",
    "personally", "maybe", "probably", "frankly", "simply",
    "just", "really", "very", "quite", "somewhat", "kinda",
    "sorta", "fairly", "pretty", "rather", "enough", "almost",
})

CONNECTOR_WORDS = frozenset({
    "but", "however", "although", "though", "while", "whereas",
    "yet", "except", "otherwise", "nonetheless", "nevertheless",
    "alternatively", "instead", "also", "plus", "and then",
})

NEGATIONS = frozenset({
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "seldom", "rarely", "without",
    "lack", "lacking", "doesn't", "don't", "didn't", "won't", "wouldn't",
    "couldn't", "shouldn't", "isn't", "aren't", "wasn't", "weren't",
})

# ==============================================================================
# STRUCTURED WARNINGS MODEL
# ==============================================================================
class WarningDetail(BaseModel):
    type: str
    message: str


# ==============================================================================
# DOMAIN KEYWORDS (OPTIMIZED with frozensets)
# ==============================================================================
DOMAIN_KEYWORDS: dict[str, dict] = {
    "electronics": {
        "single": frozenset({
            "battery", "camera", "screen", "display", "charger", "charging", "usb",
            "processor", "cpu", "ram", "storage", "speaker", "audio", "bluetooth",
            "wifi", "5g", "lte", "sensor", "gps", "fingerprint", "gaming",
            "graphics", "gpu", "refresh", "oled", "lcd", "amoled",
            "pixel", "megapixels", "zoom", "lens", "aperture",
            "waterproof", "ip68", "headphone", "earphone", "touchscreen",
            "phone", "tablet", "laptop", "smartwatch", "earbuds",
        }),
        "phrases": frozenset({
            "battery life", "fast charging", "night mode", "face unlock",
            "wireless charging", "refresh rate", "portrait mode",
            "call quality", "signal strength", "heating issue",
        }),
        "weight": 1,
    },
    "clothing": {
        "single": frozenset({
            "fabric", "material", "cotton", "polyester", "size", "fit", "tight",
            "loose", "stretch", "breathable", "wash", "color",
            "fade", "shrink", "stitch", "seam", "pocket", "zipper", "button",
            "sleeve", "collar", "hem", "inseam", "waist", "hip",
            "jeans", "shirt", "dress", "jacket", "shoes", "boots",
        }),
        "phrases": frozenset({
            "true to size", "runs small", "runs large", "machine wash",
            "color faded", "shrunk after washing", "comfortable fit",
        }),
        "weight": 1,
    },
    "food": {
        "single": frozenset({
            "taste", "flavor", "fresh", "spicy", "sweet", "salty", "bitter",
            "sour", "crispy", "crunchy", "texture", "aroma", "portion",
            "serving", "calorie", "organic", "ingredients", "nutrition", "protein",
            "expired", "stale", "packaging", "ingredient",
        }),
        "phrases": frozenset({
            "expiry date", "best before", "shelf life",
            "taste great", "flavorful", "value for money",
        }),
        "weight": 1,
    },
    "furniture": {
        "single": frozenset({
            "assembly", "stable", "wobbly", "wood", "metal",
            "leather", "cushion", "ergonomic", "backrest", "armrest",
            "drawer", "shelf", "chair", "desk", "bedframe", "mattress",
            "legs", "surface", "scratch",
        }),
        "phrases": frozenset({
            "easy to assemble", "hard to assemble", "assembly instructions",
            "worth the price", "sturdy build", "comfortable seating",
        }),
        "weight": 1,
    },
    "beauty": {
        "single": frozenset({
            "moisturizer", "serum", "foundation", "concealer", "mascara",
            "eyeshadow", "lipstick", "blush", "primer", "breakout",
            "acne", "hypoallergenic", "skin", "cream", "lotion",
        }),
        "phrases": frozenset({
            "setting spray", "cruelty free", "long lasting", "skin tone",
            "easy to apply", "buildable coverage", "blends well",
        }),
        "weight": 1,
    },
    "home_appliances": {
        "single": frozenset({
            "noise", "filter", "capacity", "powerful",
            "efficient", "temperature", "timer", "automatic",
            "vacuum", "blender", "mixer", "coffeemaker", "microwave",
        }),
        "phrases": frozenset({
            "energy efficient", "noise level", "easy to clean",
            "worth the price", "powerful motor", "durable build",
        }),
        "weight": 1,
    },
}


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"\b\w+\b", text.lower()))


def detect_domain(text: str) -> str:
    words = tokenize(text)
    text_lower = text.lower()
    total_words = len(words)
    scores: dict[str, float] = {}

    for domain, config in DOMAIN_KEYWORDS.items():
        score = 0.0
        weight = config.get("weight", 1)
        
        single_hits = words & config["single"]
        score += len(single_hits) * weight * 1.5
        
        for phrase in config.get("phrases", set()):
            if phrase in text_lower:
                score += 3 * weight
        
        scores[domain] = score

    if not scores:
        return "generic"
    
    max_score = max(scores.values())
    normalized_threshold = max(1.5, total_words * 0.1)
    if max_score >= normalized_threshold:
        return max(scores, key=scores.get)
    return "generic"


# ==============================================================================
# NEGATION HANDLING
# ==============================================================================
SPECIAL_NEGATIONS = {
    "not bad": "decent", "not the worst": "acceptable", "not good": "poor",
    "not great": "mediocre", "not happy": "dissatisfied", "not disappointed": "satisfied",
    "not terrible": "acceptable", "not bad at all": "good", "not half bad": "decent",
    "nothing bad": "satisfactory", "no complaints about": "satisfactory",
    "can't complain": "acceptable", "couldn't be better": "excellent",
    "not a problem": "acceptable", "no problem with": "satisfied",
    "not disappointing": "adequate", "not the best": "mediocre",
    "not impressive": "mediocre", "not satisfied": "dissatisfied",
    "not worth": "overpriced", "not recommend": "avoid", "not recommended": "avoid",
    "wouldn't recommend": "avoid", "not buying again": "disappointed",
    "not happy with": "dissatisfied", "not impressed": "mediocre",
    "nothing special": "average", "not worth it": "overpriced",
    "not worth the money": "overpriced", "not a fan": "disappointed",
    "not comfortable": "uncomfortable", "not easy": "complicated",
    "not fast": "slow", "not quiet": "noisy", "not durable": "flimsy",
    "not sturdy": "flimsy", "not reliable": "unreliable", "not accurate": "inaccurate",
    "not sharp": "blurry", "not bright": "dim", "not smooth": "rough",
    "not clear": "unclear", "not responsive": "laggy", "not user friendly": "complicated",
}


def handle_special_negations(text: str) -> str:
    text_lower = text.lower()
    for phrase, replacement in SPECIAL_NEGATIONS.items():
        if phrase in text_lower:
            text = re.sub(r'\b' + re.escape(phrase) + r'\b', replacement, text, flags=re.IGNORECASE)
    return text


# ==============================================================================
# FEATURE DETECTION (OPTIMIZED with Aho-Corasick)
# ==============================================================================
BASE_FEATURES = {
    "battery": frozenset({"battery", "backup", "drain", "mah", "power", "charging", "charge", "charger", "battery life", "drains"}),
    "camera": frozenset({"camera", "photo", "video", "lens", "focus", "zoom", "selfie", "megapixel", "aperture", "photos", "pictures"}),
    "performance": frozenset({"performance", "lag", "slow", "speed", "processor", "ram", "gaming", "gpu", "fast", "responsive"}),
    "design": frozenset({"design", "look", "style", "color", "aesthetic", "sleek", "premium", "appearance"}),
    "display": frozenset({"display", "screen", "brightness", "touch", "oled", "lcd", "amoled", "resolution", "visuals"}),
    "sound": frozenset({"sound", "audio", "speaker", "volume", "bass", "microphone", "mic", "call quality"}),
    "charging": frozenset({"charging", "charge", "charger", "wireless", "fast charging"}),
    "build": frozenset({"build", "quality", "durable", "material", "plastic", "metal", "glass", "waterproof", "sturdy", "flimsy"}),
    "price": frozenset({"price", "cost", "expensive", "value", "worth", "cheap", "affordable", "budget", "money"}),
    "software": frozenset({"software", "ui", "update", "app", "os", "android", "ios", "feature", "apps"}),
    "support": frozenset({"support", "service", "warranty", "help", "response", "customer"}),
    "comfort": frozenset({"comfort", "fit", "pain", "ear", "heavy", "light", "weight", "ergonomic", "comfortable"}),
    "connectivity": frozenset({"wifi", "bluetooth", "signal", "network", "5g", "lte", "gps", "connection"}),
    "delivery": frozenset({"delivery", "shipping", "arrived", "packaging", "damaged", "late", "on time", "fast", "slow"}),
}

DOMAIN_FEATURES = {
    "clothing": {
        "fabric": frozenset({"fabric", "material", "cotton", "polyester", "silk", "denim", "texture"}),
        "fit": frozenset({"fit", "size", "tight", "loose", "true to size", "runs small", "runs large"}),
        "comfort": frozenset({"comfortable", "soft", "breathable", "itchy", "rough", "comfort"}),
        "durability": frozenset({"durability", "fade", "shrink", "stretch", "tear", "pilling", "color faded"}),
    },
    "food": {
        "taste": frozenset({"taste", "flavor", "bland", "delicious", "savory", "sweet", "flavorful"}),
        "texture": frozenset({"texture", "crispy", "crunchy", "chewy", "tender", "dry", "moist", "fresh", "stale"}),
        "value": frozenset({"portion", "serving", "value", "fresh", "expired", "price"}),
    },
    "furniture": {
        "assembly": frozenset({"assembly", "instructions", "difficult", "assemble", "easy to assemble"}),
        "stability": frozenset({"stable", "wobbly", "sturdy", "solid", "flimsy", "shake"}),
        "comfort": frozenset({"comfortable", "cushion", "support", "firm", "soft", "seat", "back"}),
    },
    "beauty": {
        "application": frozenset({"application", "blend", "coverage", "pigmented", "patchy", "apply", "easy to apply"}),
        "wear": frozenset({"wear", "lasting", "smudge", "transfer", "fade", "settle", "long lasting"}),
        "skin": frozenset({"breakout", "irritation", "allergic", "sensitive", "oily", "dry", "skin"}),
    },
    "generic": {
        "quality": frozenset({"quality", "durable", "cheap", "premium", "sturdy", "flimsy", "solid"}),
        "usability": frozenset({"easy", "difficult", "convenient", "complicated", "intuitive", "user friendly"}),
        "value": frozenset({"value", "worth", "price", "expensive", "affordable", "budget", "money"}),
        "packaging": frozenset({"packaging", "arrived", "damaged", "sealed", "wrapped", "shipping"}),
        "delivery": frozenset({"delivery", "shipping", "arrived", "late", "on time", "fast", "slow"}),
    },
}


def get_features_for_domain(domain: str) -> dict:
    features = copy.deepcopy(BASE_FEATURES)
    if domain in DOMAIN_FEATURES:
        features.update(DOMAIN_FEATURES[domain])
    elif domain == "generic":
        features.update(DOMAIN_FEATURES["generic"])
    return features


# ==============================================================================
# AHO-CORASICK AUTOMATONS (Prebuilt at startup - O(n) matching)
# ==============================================================================
_ALIAS_AUTOMATONS: dict[str, 'ahocorasick.Automaton'] = {}
_PHRASE_AUTOMATONS: dict[str, 'ahocorasick.Automaton'] = {}
_ALIAS_TO_FEATURE: dict[str, dict[str, str]] = {}
_AHOCORASICK_BUILT = False


def _build_ahocorasick_automaton(domain: str) -> Tuple['ahocorasick.Automaton', 'ahocorasick.Automaton', dict]:
    """Build Aho-Corasick automatons for a domain - O(n) pattern matching"""
    features = get_features_for_domain(domain)
    
    # Build automaton for single words
    word_automaton = ahocorasick.Automaton()
    word_to_feature = {}
    
    # Build automaton for phrases
    phrase_automaton = ahocorasick.Automaton()
    phrase_to_feature = {}
    
    for feature, aliases in features.items():
        for alias in aliases:
            if " " in alias:  # Multi-word phrase
                phrase_automaton.add_word(alias, (alias, feature))
                phrase_to_feature[alias] = feature
            else:
                word_automaton.add_word(alias, (alias, feature))
                word_to_feature[alias] = feature
    
    word_automaton.make_automaton()
    phrase_automaton.make_automaton()
    
    return word_automaton, phrase_automaton, word_to_feature


def _build_automaton_maps():
    global _ALIAS_AUTOMATONS, _PHRASE_AUTOMATONS, _ALIAS_TO_FEATURE, _AHOCORASICK_BUILT
    if _AHOCORASICK_BUILT:
        return
    
    for domain in ["electronics", "clothing", "food", "furniture", "beauty", "home_appliances", "generic"]:
        word_auto, phrase_auto, word_map = _build_ahocorasick_automaton(domain)
        _ALIAS_AUTOMATONS[domain] = word_auto
        _PHRASE_AUTOMATONS[domain] = phrase_auto
        _ALIAS_TO_FEATURE[domain] = word_map
    
    _AHOCORASICK_BUILT = True
    logger.info("Aho-Corasick automatons built for all domains")


# ==============================================================================
# SENTIMENT WORDS (Precompiled patterns)
# ==============================================================================
STRONG_NEGATIVE = frozenset({
    "bad", "poor", "worst", "waste", "overheat", "lag", "drain", "heats",
    "fails", "failure", "crash", "buggy", "terrible", "horrible", "awful",
    "broken", "disappointing", "frustrating", "useless", "defective",
    "cheaply", "flimsy", "pathetic", "regret", "nightmare", "disaster",
    "avoid", "refuse", "struggles", "slow", "drains", "weak", "unreliable",
    "damaged", "overheating", "overheats", "glitchy", "laggy", "unresponsive",
    "cheap", "brittle", "shoddy", "subpar", "underwhelming",
    "inconsistent", "unstable", "wobbly", "scratches", "scratchy",
    "stains", "faded", "shrunk", "ripped", "leaked", "leaking",
    "stopped working", "died", "dead", "mediocre",
})

STRONG_POSITIVE = frozenset({
    "excellent", "great", "smooth", "easy", "premium", "bright",
    "sharp", "clean", "love", "best", "amazing", "perfect", "outstanding",
    "fantastic", "wonderful", "brilliant", "superb", "impressed",
    "recommend", "exceeded", "delighted", "flawless", "exceptional",
    "remarkable", "solid", "reliable",
})

SOFT_NEGATIVE = frozenset({
    "slow", "weak", "issue", "problem", "expensive", "overpriced", "noisy", "hot",
    "average", "mediocre", "meh", "uncomfortable", "inconvenient", "complicated",
})

SOFT_POSITIVE = frozenset({
    "good", "nice", "fast", "clear", "lightweight", "sleek", "fine", "okay", "decent",
    "satisfactory", "acceptable", "adequate", "pleasant",
})

ALL_POSITIVE = STRONG_POSITIVE | SOFT_POSITIVE
ALL_NEGATIVE = STRONG_NEGATIVE | SOFT_NEGATIVE

# Precompiled patterns for keyword matching
_STRONG_POS_PATTERN = re.compile(r'\b(' + '|'.join(sorted(STRONG_POSITIVE, key=len, reverse=True)) + r')\b')
_SOFT_POS_PATTERN = re.compile(r'\b(' + '|'.join(sorted(SOFT_POSITIVE, key=len, reverse=True)) + r')\b')
_STRONG_NEG_PATTERN = re.compile(r'\b(' + '|'.join(sorted(STRONG_NEGATIVE, key=len, reverse=True)) + r')\b')
_SOFT_NEG_PATTERN = re.compile(r'\b(' + '|'.join(sorted(SOFT_NEGATIVE, key=len, reverse=True)) + r')\b')

# Aho-Corasick for sentiment words
_SENTIMENT_AUTOMATON = None
_SENTIMENT_WORD_MAP = {}


def _build_sentiment_automaton():
    global _SENTIMENT_AUTOMATON, _SENTIMENT_WORD_MAP
    
    auto = ahocorasick.Automaton()
    
    for word in ALL_POSITIVE:
        auto.add_word(word, ("positive", word))
    for word in ALL_NEGATIVE:
        auto.add_word(word, ("negative", word))
    
    auto.make_automaton()
    _SENTIMENT_AUTOMATON = auto
    
    _SENTIMENT_WORD_MAP = {word: "positive" for word in ALL_POSITIVE}
    _SENTIMENT_WORD_MAP.update({word: "negative" for word in ALL_NEGATIVE})


# ==============================================================================
# ADAPTIVE SENTIMENT ANALYSIS (FIX 2: Dynamic weighting)
# ==============================================================================
@lru_cache(maxsize=5000)
def get_sentiment_polarity_cached(text: str) -> Tuple[float, float]:
    """
    FIX 2: Adaptive sentiment with dynamic VADER/keyword weighting.
    Returns (polarity, confidence) tuple.
    """
    text = handle_special_negations(text)
    text_lower = text.lower()
    
    # Step 1: Get VADER score and confidence
    vader_score = 0.0
    vader_confidence = 0.0
    
    if USE_VADER:
        vader_result = vader_analyzer.polarity_scores(text)
        vader_score = vader_result["compound"]
        
        # VADER confidence based on extremity of score
        vader_confidence = abs(vader_score)
        
        # Check for mixed signals (lower confidence)
        pos = vader_result["pos"]
        neg = vader_result["neg"]
        neu = vader_result["neu"]
        
        if max(pos, neg, neu) < 0.5:  # Signals are mixed
            vader_confidence *= 0.7
        elif pos > 0.3 and neg > 0.3:  # Both positive and negative signals
            vader_confidence *= 0.6
    
    # Step 2: Aho-Corasick keyword matching (O(n) for all keywords)
    if _SENTIMENT_AUTOMATON:
        keyword_counts = {"positive": 0, "negative": 0}
        for end_idx, (sentiment, word) in _SENTIMENT_AUTOMATON.iter(text_lower):
            if sentiment == "positive":
                keyword_counts["positive"] += 1
            else:
                keyword_counts["negative"] += 1
        kw_pos = keyword_counts["positive"]
        kw_neg = keyword_counts["negative"]
    else:
        # Fallback to regex matching
        kw_pos = len(_STRONG_POS_PATTERN.findall(text_lower)) + len(_SOFT_POS_PATTERN.findall(text_lower))
        kw_neg = len(_STRONG_NEG_PATTERN.findall(text_lower)) + len(_SOFT_NEG_PATTERN.findall(text_lower))
    
    # Calculate keyword-based score
    kw_score = 0.0
    kw_confidence = min(1.0, (kw_pos + kw_neg) / 5)  # More keywords = higher confidence
    
    for _ in range(kw_pos):
        kw_score += 0.2
    for _ in range(kw_neg):
        kw_score -= 0.2
    
    # Step 3: Check for negation context
    words = text_lower.split()
    for i, word in enumerate(words):
        if word in _SENTIMENT_WORD_MAP:
            if i > 0 and words[i-1] in NEGATIONS:
                if _SENTIMENT_WORD_MAP[word] == "positive":
                    kw_score -= 0.3
                else:
                    kw_score += 0.3
    
    # Step 4: ADAPTIVE WEIGHTING (FIX 2: Dynamic based on confidence)
    if vader_confidence > 0.7:
        # High confidence in VADER - trust it more
        weight_vader = 0.75
        weight_kw = 0.25
    elif vader_confidence > 0.4:
        # Medium confidence - balanced
        weight_vader = 0.6
        weight_kw = 0.4
    else:
        # Low confidence - trust keywords more
        weight_vader = 0.4
        weight_kw = 0.6
    
    # Also adjust based on keyword confidence
    if kw_confidence > 0.8:
        weight_kw = min(0.7, weight_kw + 0.1)
        weight_vader = 1.0 - weight_kw
    
    # Combine scores
    blended = weight_vader * vader_score + weight_kw * kw_score
    
    # Calculate overall confidence
    overall_confidence = max(vader_confidence, kw_confidence)
    
    return max(-1.0, min(1.0, blended)), overall_confidence


def get_sentiment_polarity(text: str) -> float:
    polarity, _ = get_sentiment_polarity_cached(text.lower().strip())
    return polarity


def get_sentiment_confidence(text: str) -> float:
    _, confidence = get_sentiment_polarity_cached(text.lower().strip())
    return confidence


def classify_sentence(sentence: str) -> str:
    """UNIFIED classification using ONLY polarity"""
    polarity = get_sentiment_polarity(sentence)
    if polarity > SENTIMENT_POLARITY_THRESHOLD:
        return "positive"
    if polarity < -SENTIMENT_POLARITY_THRESHOLD:
        return "negative"
    return "neutral"


# ==============================================================================
# MULTI-FEATURE EXTRACTION (FIX 1: Aho-Corasick O(n) matching)
# ==============================================================================
def extract_features_with_context(sentence: str, domain: str = "generic") -> List[str]:
    """
    FIX 1: Use Aho-Corasick automaton for O(n) multi-pattern matching.
    Much faster than iterating through all features.
    """
    sentence_lower = sentence.lower()
    detected_features = set()
    
    # Phase 1: Check for multi-word phrases first (higher priority)
    if domain in _PHRASE_AUTOMATONS:
        for end_idx, (phrase, feature) in _PHRASE_AUTOMATONS[domain].iter(sentence_lower):
            detected_features.add(feature)
    
    # Phase 2: Use automaton for single word matching (O(n) total)
    if domain in _ALIAS_AUTOMATONS:
        for end_idx, (word, feature) in _ALIAS_AUTOMATONS[domain].iter(sentence_lower):
            detected_features.add(feature)
    
    return list(detected_features) if detected_features else []


@lru_cache(maxsize=1000)
def extract_feature_cached(sentence: str, domain: str = "generic") -> Optional[str]:
    features = extract_features_with_context(sentence, domain)
    return features[0] if features else None


# ==============================================================================
# CLAUSE SPLITTING
# ==============================================================================
_CONNECTOR_PATTERN = re.compile(r'\s+((?:but|however|although|though|while|whereas|yet|except|otherwise|nonetheless|nevertheless|alternatively|instead|also|plus|and then))\s+', re.I)
_CONJUNCTION_PATTERN = re.compile(r',\s*(?=(but|however|although|though|and also|plus|and then))\s+', re.I)
_PUNCTUATION_SPLIT = re.compile(r'(?<=[.!?])\s+')


def split_into_clauses(text: str) -> list[dict]:
    segments = []
    
    sentences = _PUNCTUATION_SPLIT.split(text)
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        parts = _CONJUNCTION_PATTERN.split(sentence)
        
        current_connector = None
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            connector_match = _CONNECTOR_PATTERN.match(part)
            if connector_match:
                current_connector = connector_match.group(1).lower()
                part = _CONNECTOR_PATTERN.sub('', part, count=1).strip()
            
            if not part:
                continue
            
            for prefix in ("but ", "however ", "although ", "though ", "while ", "yet ", "except "):
                if part.lower().startswith(prefix):
                    part = part[len(prefix):].strip()
                    break
            
            if part and part[0].islower():
                part = part[0].upper() + part[1:]
            
            part = part.strip(" ,.;")
            
            if part and len(part) >= 5:
                segments.append({
                    "text": part,
                    "connector": current_connector
                })
            
            current_connector = None
    
    return segments


def shorten(text: str, max_length: int = 80) -> str:
    if not text:
        return text
    text = text.strip()
    if "," in text:
        text = text.split(",")[0]
    elif "." in text:
        text = text.split(".")[0]
    if len(text) > max_length:
        text = text[: max_length - 3].rstrip() + "..."
    return text.strip()


# ==============================================================================
# PYDANTIC MODELS
# ==============================================================================
class ReviewRequest(BaseModel):
    reviews: list[str] = Field(..., min_length=1)

    @field_validator("reviews")
    @classmethod
    def clean_reviews(cls, reviews: list[str]) -> list[str]:
        cleaned = [r.strip() for r in reviews if r and len(r.strip()) >= 5]
        if not cleaned:
            raise ValueError("Please provide at least one review with content.")
        if len(cleaned) > MAX_REVIEWS:
            raise ValueError(f"Please provide no more than {MAX_REVIEWS} reviews.")
        return cleaned


class RawAnalyzeRequest(BaseModel):
    raw_text: str = Field(..., min_length=10)
    user_focus: Optional[str] = Field(None)

    @field_validator("raw_text")
    @classmethod
    def validate_size(cls, v: str) -> str:
        if len(v) > MAX_INPUT_SIZE:
            raise ValueError(f"Input too large. Max {MAX_INPUT_SIZE} chars.")
        return v


class SentimentBreakdown(BaseModel):
    positive: float
    neutral: float
    negative: float
    total: int


class ExplainablePoint(BaseModel):
    text: str
    features: List[str]
    sentiment: str
    polarity_score: float
    confidence: float
    impact: str


class AnalysisPoint(BaseModel):
    text: str
    feature: str
    features: List[str] = Field(default_factory=list)
    impact: str


class FeatureScore(BaseModel):
    feature: str
    display_name: str
    positive_count: int
    negative_count: int
    total_mentions: int
    score: float


class AnalyzeResponse(BaseModel):
    summary: str
    pros: list[AnalysisPoint]
    cons: list[AnalysisPoint]
    neutral_points: list[str] = Field(default_factory=list)
    sentiment: SentimentBreakdown
    score: float
    confidence: float
    cached: bool = False
    warnings: list[WarningDetail] = Field(default_factory=list)
    explained_pros: list[ExplainablePoint] | None = None
    explained_cons: list[ExplainablePoint] | None = None
    feature_scores: list[FeatureScore] | None = None
    domain: str | None = None


# ==============================================================================
# FASTAPI APP
# ==============================================================================
app = FastAPI(
    title="AI Product Review Aggregator API",
    description="Production-ready system with enhanced sentiment analysis",
    version="23.0-faang",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    if not VALID_API_KEYS:
        logger.warning("No API_KEYS configured - running in open mode")
        return "dev"
    
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required. Pass 'X-API-Key' header.")
    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return x_api_key


# ==============================================================================
# PROMETHEUS METRICS
# ==============================================================================
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    REQUEST_COUNT = Counter("review_api_requests_total", "Total requests", ["endpoint", "status"])
    REQUEST_LATENCY = Histogram("review_api_request_duration_seconds", "Request latency", ["endpoint"])
    CACHE_HITS = Counter("review_api_cache_hits_total", "Cache hits")
    ACTIVE_REQUESTS = Gauge("review_api_active_requests", "Active requests")
    GEMINI_REQUESTS_ACTIVE = Gauge("review_api_gemini_active", "Active Gemini requests")
    USE_PROMETHEUS = True

    @app.get("/metrics")
    async def metrics():
        return JSONResponse(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
except ImportError:
    USE_PROMETHEUS = False
    class NoOpCounter:
        def __init__(self, *_, **__): pass
        def labels(self, **_): return self
        def inc(self, *_, **__): pass
    class NoOpHistogram:
        def __init__(self, *_, **__): pass
        def labels(self, **_):
            class Ctx:
                def __enter__(self): return self
                def __exit__(self, *_, **__): pass
                def time(self): return self
            return Ctx()
    class NoOpGauge:
        def __init__(self, *_, **__): pass
        def labels(self, **_): return self
        def set(self, *_): pass
    REQUEST_COUNT = NoOpCounter()
    REQUEST_LATENCY = NoOpHistogram()
    CACHE_HITS = NoOpCounter()
    ACTIVE_REQUESTS = NoOpGauge()
    GEMINI_REQUESTS_ACTIVE = NoOpGauge()


# ==============================================================================
# REQUEST TRACKER
# ==============================================================================
class RequestTracker:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._active_http = 0
                    cls._instance._active_gemini = 0
        return cls._instance

    @property
    def active_http(self) -> int:
        return self._active_http

    @active_http.setter
    def active_http(self, value: int):
        self._active_http = max(0, value)
        ACTIVE_REQUESTS.set(self._active_http)

    @property
    def active_gemini(self) -> int:
        return self._active_gemini

    @active_gemini.setter
    def active_gemini(self, value: int):
        self._active_gemini = max(0, value)
        GEMINI_REQUESTS_ACTIVE.set(self._active_gemini)


request_tracker = RequestTracker()


# ==============================================================================
# LRU CACHE (Using orjson)
# ==============================================================================
class LRUCache:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: OrderedDict[str, tuple[str, float]] = OrderedDict()
        self.hit_count = 0
        self.miss_count = 0
        self.size = 0

    def _serialize(self, data: dict) -> str:
        if USE_ORJSON:
            return orjson.dumps(data).decode('utf-8')
        return json.dumps(data)

    def _deserialize(self, data_str: str) -> dict:
        if USE_ORJSON:
            return orjson.loads(data_str)
        return json.loads(data_str)

    def get(self, key: str) -> Optional[dict]:
        if key not in self.cache:
            self.miss_count += 1
            return None
        data_str, expiry = self.cache[key]
        if time.time() > expiry:
            del self.cache[key]
            self.size -= 1
            self.miss_count += 1
            return None
        self.cache.move_to_end(key)
        self.hit_count += 1
        CACHE_HITS.inc()
        return self._deserialize(data_str)

    def set(self, key: str, data: dict, ttl: int = 3600):
        while self.size >= self.max_size:
            self.cache.popitem(last=False)
            self.size -= 1
        data_str = self._serialize(data)
        self.cache[key] = (data_str, time.time() + ttl)
        self.cache.move_to_end(key)
        self.size += 1

    def get_stats(self) -> dict:
        total = self.hit_count + self.miss_count
        hit_rate = (self.hit_count / total * 100) if total > 0 else 0
        return {"hits": self.hit_count, "misses": self.miss_count, "hit_rate": f"{hit_rate:.1f}%"}


class CacheManager:
    def __init__(self):
        self.lru_cache = LRUCache(max_size=MAX_CACHE_SIZE)

    def generate_cache_key(self, reviews: list[str], detailed: bool = False, domain: str = "generic") -> str:
        normalized_reviews = []
        for r in reviews:
            cleaned = r.strip().lower()
            cleaned = re.sub(r'\s+', ' ', cleaned)
            normalized_reviews.append(cleaned)
        
        content_hash = hashlib.sha256("|".join(normalized_reviews).encode()).hexdigest()[:24]
        
        payload = {"reviews": normalized_reviews, "detailed": detailed, "domain": domain, "hash": content_hash}
        cache_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return f"review_cache:{cache_hash}"

    def get(self, key: str) -> Optional[dict]:
        if not ENABLE_CACHE:
            return None
        return self.lru_cache.get(key)

    def set(self, key: str, data: dict, ttl: int = None):
        if not ENABLE_CACHE:
            return
        ttl = ttl or CACHE_TTL_SECONDS
        self.lru_cache.set(key, data, ttl)


cache_manager = CacheManager()


# ==============================================================================
# RATE LIMITER (Using deque to prevent memory leak)
# ==============================================================================
class ScalableRateLimiter:
    def __init__(self):
        self._requests_per_minute: dict[str, deque] = {}
        self._requests_per_hour: dict[str, deque] = {}
        self._lock = threading.Lock()
        self._cleanup_interval = 300
        self._last_cleanup = time.time()

    def _cleanup_old_entries(self):
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        
        with self._lock:
            cutoff_minute = now - 60
            cutoff_hour = now - 3600
            
            for key in list(self._requests_per_minute.keys()):
                while self._requests_per_minute[key] and self._requests_per_minute[key][0] < cutoff_minute:
                    self._requests_per_minute[key].popleft()
                if not self._requests_per_minute[key]:
                    del self._requests_per_minute[key]
            
            for key in list(self._requests_per_hour.keys()):
                while self._requests_per_hour[key] and self._requests_per_hour[key][0] < cutoff_hour:
                    self._requests_per_hour[key].popleft()
                if not self._requests_per_hour[key]:
                    del self._requests_per_hour[key]
            
            self._last_cleanup = now

    def record_request(self, identifier: str):
        now = time.time()
        self._cleanup_old_entries()
        
        with self._lock:
            if identifier not in self._requests_per_minute:
                self._requests_per_minute[identifier] = deque(maxlen=RATE_LIMIT_PER_MINUTE + 10)
            if identifier not in self._requests_per_hour:
                self._requests_per_hour[identifier] = deque(maxlen=RATE_LIMIT_PER_HOUR + 10)
            
            self._requests_per_minute[identifier].append(now)
            self._requests_per_hour[identifier].append(now)

    def get_request_count(self, identifier: str, window_seconds: int = 60) -> int:
        now = time.time()
        with self._lock:
            if window_seconds <= 60:
                if identifier not in self._requests_per_minute:
                    return 0
                cutoff = now - 60
                while self._requests_per_minute[identifier] and self._requests_per_minute[identifier][0] < cutoff:
                    self._requests_per_minute[identifier].popleft()
                return len(self._requests_per_minute[identifier])
            else:
                if identifier not in self._requests_per_hour:
                    return 0
                cutoff = now - 3600
                while self._requests_per_hour[identifier] and self._requests_per_hour[identifier][0] < cutoff:
                    self._requests_per_hour[identifier].popleft()
                return len(self._requests_per_hour[identifier])

    def is_rate_limited(self, identifier: str, per_minute: int = None, per_hour: int = None) -> tuple[bool, str]:
        per_minute = per_minute or RATE_LIMIT_PER_MINUTE
        per_hour = per_hour or RATE_LIMIT_PER_HOUR
        minute_count = self.get_request_count(identifier, 60)
        if minute_count >= per_minute:
            return True, f"Per-minute limit exceeded ({minute_count}/{per_minute})"
        hour_count = self.get_request_count(identifier, 3600)
        if hour_count >= per_hour:
            return True, f"Per-hour limit exceeded ({hour_count}/{per_hour})"
        return False, ""


scalable_limiter = ScalableRateLimiter()


# ==============================================================================
# GEMINI CLIENT MANAGER (Fully async with httpx)
# ==============================================================================
class GeminiKeyConfig:
    def __init__(self, key: str, name: str = ""):
        self._key = key.strip()
        self._hash = hashlib.sha256(self._key.encode()).hexdigest()
        self._name = name or f"key_{self._hash[:6]}"
        self._is_available = True
        self._failure_count = 0
        self._last_failure: Optional[float] = None
        self._success_count = 0
        self._total_requests = 0
        self._failure_threshold = 5
        self._recovery_timeout = 300

    @property
    def key(self) -> str:
        return self._key

    @property
    def id(self) -> str:
        return self._hash[:6]

    @property
    def is_healthy(self) -> bool:
        if not self._is_available:
            if self._last_failure and (time.time() - self._last_failure) > self._recovery_timeout:
                self._is_available = True
                self._failure_count = 0
                return True
            return False
        return True

    def record_success(self):
        self._success_count += 1
        self._total_requests += 1
        if self._failure_count > 0:
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self, is_rate_limit: bool = False):
        self._failure_count += 1
        self._last_failure = time.time()
        self._total_requests += 1
        if self._failure_count >= self._failure_threshold or is_rate_limit:
            self._is_available = False


class GeminiClientManager:
    _instance: Optional["GeminiClientManager"] = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._keys: dict[str, GeminiKeyConfig] = {}
        self._rotation_lock = threading.Lock()
        self._index = 0
        self._round_robin: list[str] = []
        self._max_retries = 2
        self._consecutive_failures = 0
        self._circuit_open_time: Optional[float] = None
        self._http_client: Optional[httpx.AsyncClient] = None
        self._load_keys()
        healthy = self.get_healthy_key_count()
        logger.info(f"GeminiClientManager: {len(self._keys)} keys, {healthy} healthy")

    def has_keys(self) -> bool:
        return bool(self._keys)

    def get_healthy_key_count(self) -> int:
        return sum(1 for cfg in self._keys.values() if cfg.is_healthy)

    def is_circuit_open(self) -> bool:
        if self._circuit_open_time is None:
            return False
        if time.time() - self._circuit_open_time > 60:
            self._circuit_open_time = None
            self._consecutive_failures = 0
            return False
        return True

    def _load_keys(self):
        self._keys.clear()
        raw_keys = os.getenv("GEMINI_API_KEYS", "")
        single_key = os.getenv("GEMINI_API_KEY", "").strip()
        existing_hashes: set[str] = set()

        if raw_keys:
            for entry in raw_keys.split(","):
                entry = entry.strip()
                if not entry:
                    continue
                if ":" in entry:
                    name, key = entry.split(":", 1)
                    self._add_key(key.strip(), name.strip(), existing_hashes)
                else:
                    self._add_key(entry, "", existing_hashes)
        elif single_key and single_key not in PLACEHOLDER_API_KEYS:
            self._add_key(single_key, "default", existing_hashes)
        self._rebuild_rotation()

    def _add_key(self, key: str, name: str, existing_hashes: set[str]):
        if not key:
            return
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        if key_hash in existing_hashes:
            return
        existing_hashes.add(key_hash)
        self._keys[key_hash] = GeminiKeyConfig(key, name)

    def _rebuild_rotation(self):
        with self._rotation_lock:
            healthy = [h for h, cfg in self._keys.items() if cfg.is_healthy]
            if healthy:
                random.shuffle(healthy)
                self._round_robin = healthy
            else:
                self._round_robin = list(self._keys.keys())
            self._index = 0

    def _get_next_key(self) -> Optional[tuple[GeminiKeyConfig, str]]:
        with self._rotation_lock:
            if not self._round_robin or self.is_circuit_open():
                return None
            for _ in range(len(self._round_robin)):
                key_hash = self._round_robin[self._index]
                self._index = (self._index + 1) % len(self._round_robin)
                cfg = self._keys.get(key_hash)
                if cfg and cfg.is_healthy:
                    return cfg, key_hash
            self._rebuild_rotation()
            if not self._round_robin:
                return None
            key_hash = self._round_robin[self._index % len(self._round_robin)]
            cfg = self._keys.get(key_hash)
            return (cfg, key_hash) if cfg else None

    async def _init_http_client(self):
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=GEMINI_TIMEOUT)
        return self._http_client

    async def analyze_reviews(self, reviews: list[str], model_name: str = DEFAULT_GEMINI_MODEL, domain: str = "generic") -> Optional[dict]:
        prompt = build_analysis_prompt(reviews, domain)

        nxt = self._get_next_key()
        if not nxt:
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3 and self._circuit_open_time is None:
                self._circuit_open_time = time.time()
            return None

        key_cfg, _ = nxt
        
        try:
            client = await self._init_http_client()
            
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={key_cfg.key}"
            
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2}
            }
            
            response = await client.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                key_cfg.record_success()
                self._consecutive_failures = 0
                data = response.json()
                if "candidates" in data and data["candidates"]:
                    text = data["candidates"][0]["content"]["parts"][0]["text"]
                    return parse_ai_response(text)
            elif response.status_code == 429:
                key_cfg.record_failure(is_rate_limit=True)
                self._consecutive_failures += 1
            else:
                key_cfg.record_failure()
                self._consecutive_failures += 1
            
            return None
            
        except httpx.TimeoutException:
            key_cfg.record_failure()
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3 and self._circuit_open_time is None:
                self._circuit_open_time = time.time()
            return None
        except Exception as e:
            logger.warning(f"Gemini API error: {e}")
            key_cfg.record_failure()
            self._consecutive_failures += 1
            if self._consecutive_failures >= 3 and self._circuit_open_time is None:
                self._circuit_open_time = time.time()
            return None

    async def close(self):
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    def get_health(self) -> dict:
        return {
            "total_keys": len(self._keys),
            "healthy_keys": self.get_healthy_key_count(),
            "circuit_open": self.is_circuit_open(),
        }


gemini_manager = GeminiClientManager()


# ==============================================================================
# SENTIMENT & ANALYSIS FUNCTIONS
# ==============================================================================
def parse_raw_input(raw_text: str) -> list[str]:
    if not raw_text:
        return []
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    reviews = []

    for line in raw_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for prefix_pattern in [
            r'^[\u2022\u2023\u25E6\u2043\u2219\*\-\+]+\s*',
            r'^\d+[\).\]\:]+\s*',
            r'^[""\'"]+',
            r'^\*\*(.+?)\*\*:',
            r'^[A-Za-z\s]+:\s*',
        ]:
            line = re.sub(prefix_pattern, "", line, flags=re.IGNORECASE)
        line = line.strip('"\' -')
        if len(line) < 5:
            continue
        if len(line) > 150:
            for part in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
                part = part.strip()
                if len(part) >= 10:
                    reviews.append(part)
        else:
            reviews.append(line)

    if len(reviews) < 3 and len(raw_text) > 200:
        for part in re.split(r'\.(?=\s|$)', raw_text):
            part = part.strip()
            if len(part) >= 10:
                is_new = True
                for existing in reviews[:20]:
                    if SequenceMatcher(None, part.lower(), existing.lower()).ratio() > 0.8:
                        is_new = False
                        break
                if is_new:
                    reviews.append(part)

    return [r.strip() for r in reviews if len(r.strip()) >= 5][:MAX_REVIEWS]


def is_valid_fragment(text: str) -> bool:
    if len(text) < 5 or len(text.split()) < 1:
        return False
    if re.search(r"(.)\1{5,}", text):
        return False
    return True


def normalize_point(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip(" -*\t\r\n")
    cleaned = re.sub(r"^\d+[\).\s-]+", "", cleaned)
    
    words = cleaned.lower().split()
    if len(words) <= 3 and any(w in FILLER_WORDS for w in words):
        return ""
    
    if cleaned and cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]
    if len(cleaned) > 120:
        cleaned = cleaned[:117].rstrip() + "..."
    return cleaned


def get_point_signature(text: str) -> str:
    norm = normalize_point(text).lower()
    norm = re.sub(r"[^a-z0-9 ]", "", norm)
    stop_words = frozenset({"a", "an", "and", "as", "at", "for", "from", "in", "is", "of", "the", "to", "very", "with"})
    tokens = [t for t in norm.split() if t not in stop_words]
    return " ".join(tokens)


def points_overlap(a: str, b: str) -> bool:
    norm_a = normalize_point(a).lower()
    norm_b = normalize_point(b).lower()
    if not norm_a or not norm_b:
        return False
    if norm_a == norm_b:
        return True
    if norm_a in norm_b or norm_b in norm_a:
        return True
    if SequenceMatcher(None, norm_a, norm_b).ratio() >= 0.84:
        return True
    return False


def is_useless_point(text: str) -> bool:
    text_lower = text.lower()
    useless_phrases = ("no major", "no clear", "no complaints", "no cons", "no negatives",
                      "no issues", "none", "not mentioned", "n/a")
    return text_lower.startswith(useless_phrases) or any(
        phrase in text_lower for phrase in ("no complaints mentioned", "no major recurring")
    )


def prepare_and_split(reviews: list[str]) -> list[dict]:
    prepared = []
    for raw in reviews:
        clauses = split_into_clauses(raw)
        for clause_dict in clauses:
            if is_valid_fragment(clause_dict["text"]):
                prepared.append(clause_dict)
    return prepared[:MAX_REVIEWS_TO_ANALYZE]


def get_impact_level(polarity_score: float) -> str:
    abs_score = abs(polarity_score)
    if abs_score >= 0.5:
        return "high"
    if abs_score >= 0.2:
        return "medium"
    return "low"


def make_analysis_point(text: str, domain: str, connector: str = None) -> AnalysisPoint:
    features_list = extract_features_with_context(text, domain)
    primary_feature = features_list[0] if features_list else "general"
    polarity = get_sentiment_polarity(text)
    confidence = get_sentiment_confidence(text)
    impact = get_impact_level(polarity)
    return AnalysisPoint(
        text=shorten(text), 
        feature=primary_feature,
        features=features_list,
        impact=impact
    )


def extract_points(clauses: list[dict], domain: str = "generic") -> dict[str, list]:
    pros_raw: List[Tuple[str, float, List[str]]] = []
    cons_raw: List[Tuple[str, float, List[str]]] = []
    neutral_raw: List[str] = []
    
    seen_signatures = set()
    feature_signatures: dict[str, set[str]] = {}

    for clause in clauses:
        if isinstance(clause, dict):
            clause_text = clause.get("text", "")
            connector = clause.get("connector")
        else:
            clause_text = clause
            connector = None

        label = classify_sentence(clause_text)
        normalized = normalize_point(clause_text)
        
        if not normalized:
            continue
        
        sig = get_point_signature(normalized)
        features_list = extract_features_with_context(clause_text, domain)

        if is_useless_point(normalized):
            continue
        
        if sig in seen_signatures:
            continue
        seen_signatures.add(sig)

        for feature in features_list:
            if feature and feature != "general":
                if feature not in feature_signatures:
                    feature_signatures[feature] = set()
                if sig[:20] in feature_signatures[feature]:
                    continue
                feature_signatures[feature].add(sig[:20])

        weight = 1.0
        if connector == "but":
            weight = 2.0
        elif connector in {"however", "although", "though", "yet"}:
            weight = 1.25

        polarity = get_sentiment_polarity(normalized)
        
        if connector == "but":
            if label == "positive":
                polarity = polarity * 1.5
            elif label == "negative":
                polarity = polarity * 1.5
        
        weighted_polarity = abs(polarity) * weight
        
        if label == "positive":
            pros_raw.append((normalized, weighted_polarity, features_list))
        elif label == "negative":
            cons_raw.append((normalized, weighted_polarity, features_list))
        elif abs(polarity) < 0.1:
            neutral_raw.append(normalized)

    pros_raw.sort(key=lambda x: x[1], reverse=True)
    cons_raw.sort(key=lambda x: x[1], reverse=True)
    
    pros = [make_analysis_point(p[0], domain) for p in pros_raw[:MAX_POINTS]]
    cons = [make_analysis_point(c[0], domain) for c in cons_raw[:MAX_POINTS]]
    neutral_points = [shorten(n) for n in neutral_raw[:MAX_NEUTRAL_POINTS]]

    return {
        "pros": pros,
        "cons": cons,
        "neutral_points": neutral_points,
    }


def calculate_sentiment(clauses: list[dict]) -> dict[str, float | int]:
    counts = {"positive": 0, "neutral": 0, "negative": 0, "total": len(clauses)}
    total_confidence = 0.0
    
    for clause in clauses:
        clause_text = clause.get("text", "") if isinstance(clause, dict) else clause
        label = classify_sentence(clause_text)
        counts[label] += 1
        total_confidence += get_sentiment_confidence(clause_text)

    total = counts["total"] or 1
    avg_confidence = total_confidence / total if total > 0 else 0.0
    
    return {
        "positive": round(counts["positive"] / total * 100, 2),
        "neutral": round(counts["neutral"] / total * 100, 2),
        "negative": round(counts["negative"] / total * 100, 2),
        "total": counts["total"],
        "avg_confidence": round(avg_confidence, 3),
    }


def calculate_feature_scores(clauses: list[dict], domain: str = "generic") -> list[FeatureScore]:
    feature_data = {}
    for clause in clauses:
        clause_text = clause.get("text", "") if isinstance(clause, dict) else clause
        if not is_valid_fragment(clause_text):
            continue
        label = classify_sentence(clause_text)
        features_list = extract_features_with_context(clause_text, domain)
        
        normalized = shorten(normalize_point(clause_text))
        if not normalized:
            continue
            
        for feature in features_list:
            if not feature or feature == "general":
                continue
            feature_data.setdefault(feature, {"positive": [], "negative": []})
            if label == "positive":
                feature_data[feature]["positive"].append(normalized)
            elif label == "negative":
                feature_data[feature]["negative"].append(normalized)

    scores = []
    for feature, data in feature_data.items():
        pos = len(data["positive"])
        neg = len(data["negative"])
        total = pos + neg
        if total == 0:
            continue
        score = ((pos - neg) / total) * 100
        scores.append(FeatureScore(
            feature=feature,
            display_name=feature.replace("_", " ").title(),
            positive_count=pos,
            negative_count=neg,
            total_mentions=total,
            score=round(score, 1),
        ))
    scores.sort(key=lambda x: abs(x.score), reverse=True)
    return scores


# ==============================================================================
# AI HELPERS
# ==============================================================================
def resolve_model_name(raw: str) -> str:
    name = raw.strip() or DEFAULT_GEMINI_MODEL
    return LEGACY_MODEL_ALIASES.get(name, name)


def parse_ai_response(raw_text: str) -> Optional[dict]:
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()

    try:
        payload = json.loads(cleaned)
        if isinstance(payload, dict):
            pros = payload.get("pros", [])
            cons = payload.get("cons", [])
            if not isinstance(pros, list) or not isinstance(cons, list):
                return None
            if not pros and not cons:
                return None
            return {
                "summary": str(payload.get("summary", "")).strip(),
                "pros": [shorten(p) for p in pros if isinstance(p, str) and len(p.strip()) >= 5],
                "cons": [shorten(c) for c in cons if isinstance(c, str) and len(c.strip()) >= 5],
                "neutral_points": [shorten(n) for n in payload.get("neutral_points", [])
                                   if isinstance(n, str) and len(n.strip()) >= 5],
            }
    except (json.JSONDecodeError, Exception):
        pass

    sm = re.search(r"summary\s*:\s*(.+?)(?:\n\s*pros?\s*:|\Z)", cleaned, flags=re.I | re.S)
    pm = re.search(r"pros?\s*:\s*(.+?)(?:\n\s*cons?\s*:|\Z)", cleaned, flags=re.I | re.S)
    cm = re.search(r"cons?\s*:\s*(.+?)(?:\n\s*neutral|\Z)", cleaned, flags=re.I | re.S)

    def extract_list(text):
        if not text:
            return []
        return [shorten(l.strip()) for l in text.splitlines()
                if l.strip() and ":" not in l.lower() and len(l.strip()) >= 5]

    pros = extract_list(pm.group(1) if pm else "")
    cons = extract_list(cm.group(1) if cm else "")

    if not pros and not cons:
        return None
    return {"summary": sm.group(1).strip() if sm else "", "pros": pros, "cons": cons, "neutral_points": []}


def build_analysis_prompt(reviews: list[str], domain: str = "generic") -> str:
    domain_context = f"\nProduct domain: {domain.replace('_', ' ')}" if domain != "generic" else ""
    return f"""You are analyzing product reviews. Extract ONLY factual insights.

Domain: {domain}{domain_context}

STRICT RULES:
1. Each point MUST mention a product feature (e.g., battery, camera, display, comfort, taste)
2. Split mixed sentences: "camera good BUT battery bad" → separate points
3. MAX: 3 pros, 3 cons, 1 neutral point
4. NO generic phrases like "good quality" or "nice product"
5. NEVER invent details not in the reviews

Return STRICT JSON:
{{
  "summary": "1-2 sentence verdict mentioning specific features",
  "pros": ["feature + specific positive observation"],
  "cons": ["feature + specific negative observation"],
  "neutral_points": ["neutral observation about a feature"]
}}

Reviews:
{chr(10).join('- ' + r[:200] for r in reviews[:5])}""".strip()


def build_summary(pros: list[AnalysisPoint], cons: list[AnalysisPoint], sentiment: dict, domain: str = "generic") -> str:
    pos_pct = sentiment.get("positive", 0)
    neg_pct = sentiment.get("negative", 0)
    neu_pct = sentiment.get("neutral", 0)

    pro_features = list(dict.fromkeys(p.feature for p in pros if p.feature != "general"))[:2]
    con_features = list(dict.fromkeys(c.feature for c in cons if c.feature != "general"))[:1]

    def fmt(f: str) -> str:
        return f.replace("_", " ")

    if pos_pct >= 60:
        if pro_features:
            feat_str = " and ".join(fmt(f) for f in pro_features)
            suffix = f" Some report issues with the {fmt(con_features[0])}." if con_features else ""
            return f"Users generally love the {feat_str}.{suffix}"
        return "Most users are satisfied with this product."
    elif neg_pct >= 60:
        if con_features:
            feat_str = fmt(con_features[0])
            suffix = f" A few highlight the {fmt(pro_features[0])} as a positive." if pro_features else ""
            return f"Users report notable concerns with the {feat_str}.{suffix}"
        return "Most users are dissatisfied with this product."
    elif pos_pct > neg_pct + 15:
        if pro_features and con_features:
            return f"The {fmt(pro_features[0])} gets praise, though some users flag {fmt(con_features[0])} issues."
        return "Reviews lean positive with a few concerns."
    elif neg_pct > pos_pct + 15:
        if con_features and pro_features:
            return f"Concerns focus on {fmt(con_features[0])}, despite a decent {fmt(pro_features[0])}."
        return "Reviews lean negative with a few bright spots."
    elif neu_pct >= 50:
        return "Reviews are largely mixed — experiences vary significantly across users."
    else:
        if pro_features and con_features:
            return f"A balanced product — {fmt(pro_features[0])} stands out positively, but {fmt(con_features[0])} needs work."
        return f"Feedback is balanced ({pos_pct:.0f}% positive, {neg_pct:.0f}% negative)."


def apply_user_focus(points: list[AnalysisPoint], user_focus: Optional[str]) -> list[AnalysisPoint]:
    if not user_focus:
        return points
    focus = user_focus.lower().strip()
    boosted = [p for p in points if focus in p.feature.lower() or any(focus in f.lower() for f in p.features) or focus in p.text.lower()]
    rest = [p for p in points if p not in boosted]
    return boosted + rest


def select_best_points(points_a: list[str], points_b: list[str], label: str, domain: str = "generic") -> list[AnalysisPoint]:
    seen = set()
    all_points = []

    for point in points_a + points_b:
        sig = get_point_signature(point)
        if sig in seen:
            continue

        text_lower = point.lower()
        score = 0.0

        features_list = extract_features_with_context(point, domain)
        if features_list:
            score += 3.0

        if label == "positive":
            score += 2.0 if any(kw in text_lower for kw in STRONG_POSITIVE) else (
                1.0 if any(kw in text_lower for kw in SOFT_POSITIVE) else 0
            )
        else:
            score += 2.0 if any(kw in text_lower for kw in STRONG_NEGATIVE) else (
                1.0 if any(kw in text_lower for kw in SOFT_NEGATIVE) else 0
            )

        if is_useless_point(point):
            score -= 5.0

        seen.add(sig)
        all_points.append((point, score, features_list))

    all_points.sort(key=lambda x: x[1], reverse=True)

    selected: list[AnalysisPoint] = []
    selected_features = set()

    for point, _, features_list in all_points:
        if any(points_overlap(point, sp.text) for sp in selected):
            continue
        primary_feature = features_list[0] if features_list else "general"
        if primary_feature in selected_features and len(selected) < MAX_POINTS:
            continue
        ap = make_analysis_point(point, domain)
        selected.append(ap)
        if primary_feature and primary_feature != "general":
            selected_features.add(primary_feature)
        if len(selected) >= MAX_POINTS:
            break

    return selected


# ==============================================================================
# MAIN ANALYSIS FUNCTION
# ==============================================================================
async def process_analysis_task(
    reviews: list[str],
    detailed: bool,
    user_focus: Optional[str] = None,
) -> tuple:
    warnings: list[WarningDetail] = []
    start_time = time.time()

    raw_text = " ".join(reviews)
    detected_domain = detect_domain(raw_text)
    domain = detected_domain

    if detected_domain == "generic":
        warnings.append(WarningDetail(type="DOMAIN_UNKNOWN", message="Using general analysis."))

    cache_key = cache_manager.generate_cache_key(reviews, detailed, domain)
    cached = cache_manager.get(cache_key)

    if cached:
        pros = [AnalysisPoint(**p) for p in cached["analysis"].get("pros", [])]
        cons = [AnalysisPoint(**c) for c in cached["analysis"].get("cons", [])]
        pros = apply_user_focus(pros, user_focus)
        cons = apply_user_focus(cons, user_focus)
        cached["analysis"]["pros"] = pros
        cached["analysis"]["cons"] = cons
        logger.info(f"process_analysis_task (cached) completed in {time.time() - start_time:.3f}s")
        return (cached["analysis"], cached["sentiment"], [], True, [], domain)

    clauses = prepare_and_split(reviews)

    if not clauses:
        return (
            {"summary": "No valid review content found.", "pros": [], "cons": [], "neutral_points": []},
            {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "total": 0, "avg_confidence": 0.0},
            [],
            False,
            [WarningDetail(type="NO_CONTENT", message="No valid review content found.")],
            domain,
        )

    rule_based = extract_points(clauses, domain)
    sentiment = calculate_sentiment(clauses)

    ai_result = None
    
    if gemini_manager.has_keys() and 5 <= len(clauses) <= MAX_CLAUSES_FOR_AI:
        try:
            model_name = resolve_model_name(os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
            truncated_clauses = [c["text"][:GEMINI_MAX_CHARS] if isinstance(c, dict) else c[:GEMINI_MAX_CHARS] 
                               for c in clauses[:GEMINI_MAX_CLAUSES]]

            request_tracker.active_gemini += 1
            try:
                ai_result = await gemini_manager.analyze_reviews(truncated_clauses, model_name, domain)
            except asyncio.TimeoutError:
                warnings.append(WarningDetail(type="AI_TIMEOUT", message="AI analysis timed out."))
                ai_result = None
            finally:
                request_tracker.active_gemini -= 1
        except Exception as e:
            logger.warning(f"AI enhancement failed: {e}")
            warnings.append(WarningDetail(type="AI_UNAVAILABLE", message="AI enhancement unavailable."))

    if ai_result and (ai_result.get("pros") or ai_result.get("cons")):
        final_pros = select_best_points(
            ai_result.get("pros", []), [p.text for p in rule_based["pros"]], "positive", domain
        )
        final_cons = select_best_points(
            ai_result.get("cons", []), [c.text for c in rule_based["cons"]], "negative", domain
        )
        final_pros = [p for p in final_pros if not any(points_overlap(p.text, c.text) for c in final_cons)]
        final_cons = [c for c in final_cons if not any(points_overlap(c.text, p.text) for p in final_pros)]

        final_summary = build_summary(final_pros, final_cons, sentiment, domain)
        final_analysis = {
            "summary": final_summary,
            "pros": final_pros[:MAX_POINTS],
            "cons": final_cons[:MAX_POINTS],
            "neutral_points": rule_based["neutral_points"],
        }
    else:
        final_analysis = {
            "summary": build_summary(rule_based["pros"], rule_based["cons"], sentiment, domain),
            "pros": rule_based["pros"],
            "cons": rule_based["cons"],
            "neutral_points": rule_based["neutral_points"],
        }

    final_analysis["pros"] = apply_user_focus(final_analysis["pros"], user_focus)
    final_analysis["cons"] = apply_user_focus(final_analysis["cons"], user_focus)

    cache_data = {
        "analysis": {
            "summary": final_analysis["summary"],
            "pros": [p.model_dump() for p in final_analysis["pros"]],
            "cons": [c.model_dump() for c in final_analysis["cons"]],
            "neutral_points": final_analysis["neutral_points"],
        },
        "sentiment": sentiment,
    }
    if detailed:
        cache_data["feature_scores"] = [fs.model_dump() for fs in calculate_feature_scores(clauses, domain)]

    cache_manager.set(cache_key, cache_data)

    logger.info(f"process_analysis_task completed in {time.time() - start_time:.3f}s")
    return final_analysis, sentiment, cache_data.get("feature_scores", []), False, warnings or [], domain


def calculate_score_and_confidence(sentiment: dict) -> tuple[float, float]:
    total = sentiment.get("total", 1)
    if total == 0:
        return 0.0, 0.0

    pos = sentiment.get("positive", 0) / 100
    neg = sentiment.get("negative", 0) / 100
    neu = sentiment.get("neutral", 0) / 100

    score = round(max(1.0, min(5.0, (pos - neg + 1) * 2.5)), 2)

    dominant = max(pos, neg, neu)
    sample_factor = min(math.sqrt(total / 8), 1.0)
    
    # Use both sentiment and avg_confidence for final confidence
    sentiment_confidence = dominant * (0.25 + 0.75 * sample_factor) * 100
    avg_vader_confidence = sentiment.get("avg_confidence", 0.5) * 100
    
    # Blend sentiment confidence with VADER average confidence
    confidence = (sentiment_confidence * 0.7) + (avg_vader_confidence * 0.3)

    return score, min(100.0, round(confidence, 2))


# ==============================================================================
# SHARED HANDLER
# ==============================================================================
async def _handle_analyze(
    request: Request,
    payload: RawAnalyzeRequest,
    x_api_key: Optional[str],
    endpoint: str,
) -> AnalyzeResponse:
    start_time = time.time()
    request_tracker.active_http += 1

    try:
        api_key = await verify_api_key(x_api_key)
        user_id = hashlib.sha256(api_key.encode()).hexdigest()[:6]

        is_limited, reason = scalable_limiter.is_rate_limited(user_id)
        if is_limited:
            REQUEST_COUNT.labels(endpoint=endpoint, status="rate_limited").inc()
            raise HTTPException(status_code=429, detail=reason)

        if len(payload.raw_text) > MAX_INPUT_SIZE:
            raise HTTPException(status_code=400, detail=f"Input too large. Max {MAX_INPUT_SIZE} chars.")

        reviews = parse_raw_input(payload.raw_text)

        if not reviews:
            raise HTTPException(status_code=400, detail="Could not extract reviews.")

        if len(reviews) > 50:
            raise HTTPException(status_code=400, detail="Too many reviews. Maximum 50 per request.")

        scalable_limiter.record_request(user_id)

        detailed = request.query_params.get("detailed", "false").lower() == "true"
        user_focus = payload.user_focus

        try:
            result = await asyncio.wait_for(
                process_analysis_task(reviews, detailed, user_focus),
                timeout=REQUEST_TIMEOUT
            )
            analysis, sentiment, feature_scores, from_cache, ai_warnings, domain = result
        except asyncio.TimeoutError:
            REQUEST_COUNT.labels(endpoint=endpoint, status="timeout").inc()
            return AnalyzeResponse(
                summary="Request timed out – please try again later.",
                pros=[],
                cons=[],
                neutral_points=[],
                sentiment=SentimentBreakdown(positive=0.0, neutral=0.0, negative=0.0, total=0),
                score=0.0,
                confidence=0.0,
                cached=False,
                warnings=[WarningDetail(type="REQUEST_TIMEOUT", message="Processing took too long.")],
                domain=detect_domain(payload.raw_text),
            )

        score, confidence = calculate_score_and_confidence(sentiment)

        pros = analysis["pros"]
        cons = analysis["cons"]

        response = AnalyzeResponse(
            summary=analysis["summary"],
            pros=pros,
            cons=cons,
            neutral_points=analysis.get("neutral_points", []),
            sentiment=SentimentBreakdown(
                positive=sentiment["positive"],
                neutral=sentiment["neutral"],
                negative=sentiment["negative"],
                total=sentiment["total"],
            ),
            score=score,
            confidence=confidence,
            cached=from_cache,
            warnings=(ai_warnings or []),
            domain=domain,
        )

        if detailed:
            response.explained_pros = [
                ExplainablePoint(
                    text=p.text,
                    features=p.features,
                    sentiment="positive",
                    polarity_score=round(get_sentiment_polarity(p.text), 3),
                    confidence=round(get_sentiment_confidence(p.text), 3),
                    impact=p.impact,
                )
                for p in pros
            ]
            response.explained_cons = [
                ExplainablePoint(
                    text=c.text,
                    features=c.features,
                    sentiment="negative",
                    polarity_score=round(get_sentiment_polarity(c.text), 3),
                    confidence=round(get_sentiment_confidence(c.text), 3),
                    impact=c.impact,
                )
                for c in cons
            ]
            response.feature_scores = feature_scores

        REQUEST_COUNT.labels(endpoint=endpoint, status="success").inc()
        return response

    except HTTPException:
        raise
    except Exception as exc:
        REQUEST_COUNT.labels(endpoint=endpoint, status="error").inc()
        logger.error("ANALYTICS ERROR", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        request_tracker.active_http -= 1


# ==============================================================================
# API ROUTES
# ==============================================================================
@app.post(f"{API_V1_PREFIX}/analyze-raw", response_model=AnalyzeResponse, tags=["v1"])
async def v1_analyze_raw_text(
    request: Request,
    payload: RawAnalyzeRequest,
    x_api_key: Optional[str] = Header(None),
) -> AnalyzeResponse:
    return await _handle_analyze(request, payload, x_api_key, f"{API_V1_PREFIX}/analyze-raw")


@app.post(f"{API_V1_PREFIX}/analyze", response_model=AnalyzeResponse, tags=["v1"])
async def v1_analyze_reviews(
    request: Request,
    payload: ReviewRequest,
    x_api_key: Optional[str] = Header(None),
) -> AnalyzeResponse:
    raw_text = "\n".join(payload.reviews)
    return await _handle_analyze(
        request, RawAnalyzeRequest(raw_text=raw_text), x_api_key, f"{API_V1_PREFIX}/analyze"
    )


@app.post(f"{API_V2_PREFIX}/analyze-raw", response_model=AnalyzeResponse, tags=["v2"])
async def v2_analyze_raw_text(
    request: Request,
    payload: RawAnalyzeRequest,
    x_api_key: Optional[str] = Header(None),
) -> AnalyzeResponse:
    return await _handle_analyze(request, payload, x_api_key, f"{API_V2_PREFIX}/analyze-raw")


@app.post(f"{API_V2_PREFIX}/analyze", response_model=AnalyzeResponse, tags=["v2"])
async def v2_analyze_reviews(
    request: Request,
    payload: ReviewRequest,
    x_api_key: Optional[str] = Header(None),
) -> AnalyzeResponse:
    raw_text = "\n".join(payload.reviews)
    return await _handle_analyze(
        request, RawAnalyzeRequest(raw_text=raw_text), x_api_key, f"{API_V2_PREFIX}/analyze"
    )


@app.post("/analyze-raw", response_model=AnalyzeResponse, tags=["legacy"])
async def analyze_raw_text(
    request: Request,
    payload: RawAnalyzeRequest,
    x_api_key: Optional[str] = Header(None),
) -> AnalyzeResponse:
    return await _handle_analyze(request, payload, x_api_key, "/analyze-raw")


@app.post("/analyze", response_model=AnalyzeResponse, tags=["legacy"])
async def analyze_reviews(
    request: Request,
    payload: ReviewRequest,
    x_api_key: Optional[str] = Header(None),
) -> AnalyzeResponse:
    raw_text = "\n".join(payload.reviews)
    return await _handle_analyze(
        request, RawAnalyzeRequest(raw_text=raw_text), x_api_key, "/analyze"
    )


@app.post("/analyze-stream", tags=["streaming"])
async def analyze_stream(
    request: Request,
    payload: RawAnalyzeRequest,
    x_api_key: Optional[str] = Header(None),
):
    await verify_api_key(x_api_key)
    
    if len(payload.raw_text) > MAX_INPUT_SIZE:
        raise HTTPException(status_code=400, detail=f"Input too large. Max {MAX_INPUT_SIZE} chars.")
    
    reviews = parse_raw_input(payload.raw_text)
    
    if not reviews:
        raise HTTPException(status_code=400, detail="Could not extract reviews.")
    
    if len(reviews) > 50:
        raise HTTPException(status_code=400, detail="Too many reviews. Maximum 50 per request.")
    
    async def stream_results():
        raw_text = " ".join(reviews)
        detected_domain = detect_domain(raw_text)
        yield json.dumps({"type": "domain", "data": detected_domain}) + "\n"
        await asyncio.sleep(STREAM_CHUNK_DELAY)
        
        clauses = prepare_and_split(reviews)
        
        if not clauses:
            yield json.dumps({"type": "error", "data": "No valid review content found."}) + "\n"
            yield "[DONE]\n"
            return
        
        rule_based = extract_points(clauses, detected_domain)
        sentiment = calculate_sentiment(clauses)
        
        summary = build_summary(rule_based["pros"], rule_based["cons"], sentiment, detected_domain)
        yield json.dumps({"type": "summary", "data": summary}) + "\n"
        await asyncio.sleep(STREAM_CHUNK_DELAY)
        
        yield json.dumps({"type": "pros", "data": [p.model_dump() for p in rule_based["pros"][:MAX_POINTS]]}) + "\n"
        await asyncio.sleep(STREAM_CHUNK_DELAY)
        
        yield json.dumps({"type": "cons", "data": [c.model_dump() for c in rule_based["cons"][:MAX_POINTS]]}) + "\n"
        await asyncio.sleep(STREAM_CHUNK_DELAY)
        
        score, confidence = calculate_score_and_confidence(sentiment)
        yield json.dumps({"type": "sentiment", "data": {**sentiment, "score": score, "confidence": confidence}}) + "\n"
        yield "[DONE]\n"
    
    return StreamingResponse(stream_results(), media_type="application/json")


# ==============================================================================
# ADMIN ENDPOINTS
# ==============================================================================
@app.get("/stats")
async def get_stats(x_api_key: Optional[str] = Header(None)):
    await verify_api_key(x_api_key)
    return {
        "cache": cache_manager.lru_cache.get_stats(),
        "polarity_cache": {
            "size": get_sentiment_polarity_cached.cache_info().currsize,
            "hits": get_sentiment_polarity_cached.cache_info().hits,
        },
        "feature_cache": {
            "size": extract_feature_cached.cache_info().currsize,
            "hits": extract_feature_cached.cache_info().hits,
        }
    }


@app.post("/cache/clear")
async def clear_cache(x_api_key: Optional[str] = Header(None)):
    await verify_api_key(x_api_key)
    cache_manager.lru_cache.cache.clear()
    get_sentiment_polarity_cached.cache_clear()
    extract_feature_cached.cache_clear()
    return {"status": "cache cleared"}


@app.get("/admin/gemini-health")
async def get_gemini_health(x_api_key: Optional[str] = Header(None)):
    await verify_api_key(x_api_key)
    return gemini_manager.get_health()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/")
def read_root() -> dict[str, str]:
    return {
        "message": "AI Product Review Aggregator API v23.0-faang",
        "docs": "/docs",
        "v1": f"{API_V1_PREFIX}/analyze-raw",
        "v2": f"{API_V2_PREFIX}/analyze-raw",
        "streaming": "/analyze-stream",
    }


# ==============================================================================
# TESTS
# ==============================================================================
def _run_tests():
    import traceback
    results = []

    def test(name, fn):
        try:
            fn()
            results.append(("PASS", name))
        except AssertionError as e:
            results.append(("FAIL", f"{name}: {e}"))
        except Exception as e:
            results.append(("ERROR", f"{name}: {traceback.format_exc(limit=1)}"))

    def check(condition: bool, message: str):
        if not condition:
            raise AssertionError(message)

    # Test 1: Multi-feature extraction with Aho-Corasick
    features = extract_features_with_context("Camera is good but battery drains fast", "electronics")
    check(len(features) >= 2, f"Should detect multiple features, got {features}")
    
    # Test 2: Negation handling
    result = handle_special_negations("not bad")
    check("decent" in result, f"'not bad' should convert to 'decent'")
    
    # Test 3: Adaptive sentiment returns tuple
    result = get_sentiment_polarity_cached("This is excellent!")
    check(isinstance(result, tuple), "Should return (polarity, confidence) tuple")
    check(len(result) == 2, "Tuple should have 2 elements")
    polarity, confidence = result
    check(isinstance(polarity, float), "Polarity should be float")
    check(isinstance(confidence, float), "Confidence should be float")
    check(polarity > 0, f"Positive text should have positive polarity")
    
    # Test 4: Order-sensitive cache
    cache_key1 = cache_manager.generate_cache_key(["a", "b"], False, "generic")
    cache_key2 = cache_manager.generate_cache_key(["b", "a"], False, "generic")
    check(cache_key1 != cache_key2, "Cache keys should be different for different orderings")
    
    # Test 5: Async process_analysis
    async def quick_test():
        result = await process_analysis_task(["Great product, loved the camera", "Battery drains too fast"], False, None)
        check(isinstance(result, tuple), "Should return tuple")
        check(len(result) == 6, "Should return 6-element tuple")
        analysis, sentiment, _, _, _, _ = result
        check(isinstance(analysis, dict), "First element should be dict")
        check("summary" in analysis, "Should have summary key")
        return True
    
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(quick_test())
        results.append(("PASS", "async_process_analysis"))
    finally:
        loop.close()
    
    # Test 6: Feature scores
    clauses = prepare_and_split(["Camera is amazing", "Battery is terrible"])
    scores = calculate_feature_scores(clauses, "electronics")
    check(len(scores) > 0, "Should have feature scores")
    
    # Test 7: Short clause validation
    check(is_valid_fragment("Battery sucks"), "Short clause should be valid")
    
    # Test 8: Domain detection
    text = "Nice fabric but color faded quickly after washing"
    domain = detect_domain(text)
    check(domain == "clothing", f"Should detect clothing domain")
    
    # Test 9: Rate limiter memory safety
    for i in range(100):
        scalable_limiter.record_request(f"test_user_{i % 10}")
    check(len(scalable_limiter._requests_per_minute) <= 20, "Should clean up old entries")
    
    # Test 10: Confidence in ExplainablePoint
    async def confidence_test():
        polarity, confidence = get_sentiment_polarity_cached("This is absolutely terrible!")
        check(confidence > 0.5, f"Clear negative should have high confidence: {confidence}")
        return True
    
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(confidence_test())
        results.append(("PASS", "adaptive_confidence"))
    finally:
        loop.close()

    passed = sum(1 for r in results if r[0] == "PASS")
    failed = sum(1 for r in results if r[0] != "PASS")
    print("\n" + "=" * 50)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 50)
    for status, name in results:
        icon = "✅" if status == "PASS" else "❌"
        print(f"  {icon} {name}")


# ==============================================================================
# STARTUP
# ==============================================================================
@app.on_event("startup")
async def startup_event():
    _build_automaton_maps()
    _build_sentiment_automaton()
    logger.info("API v23.0-faang started with Aho-Corasick and adaptive sentiment")


@app.on_event("shutdown")
async def shutdown_event():
    await gemini_manager.close()
    logger.info("API shutdown - cleaned up resources")


# ==============================================================================
# ENTRYPOINT
# ==============================================================================
if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        _run_tests()
        sys.exit(0)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
