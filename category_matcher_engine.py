"""
CategoryMatcherEngine
=====================
Hybrid category matching engine for Jumia product validation.

Architecture:
  Layer 1 – Product-type keyword dictionary (fast, deterministic, highest priority)
  Layer 2 – TF-IDF cosine similarity over augmented category corpus
  Layer 3 – JSON boost rules (from category_qc_weighted.json)
  Layer 4 – ML correction classifier (learns from every human approve/reject)

Usage
-----
    from category_matcher_engine import CategoryMatcherEngine, check_wrong_category, get_engine

    engine = get_engine()
    engine.build_tfidf_index(categories_list)          # call once

    # Validation:
    flagged = check_wrong_category(data, categories_list, compiled_rules,
                                   cat_path_to_code, code_to_path)

    # After a human approves a wrong-category item:
    engine.apply_learned_correction("iPhone 15 Pro 256GB",
                                    "Phones & Tablets / Phones / Smartphones")
    engine.save_learning_db()

    # After a human manually rejects something as wrong-category:
    engine.apply_learned_correction("Samsung TV 55 inch",
                                    "Electronics / Television / Smart TVs")
    engine.save_learning_db()
"""

from __future__ import annotations

import json
import logging
import os
import re
import pickle
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────
# File paths (written next to your app files)
# ─────────────────────────────────────────────────────────
LEARNING_DB_PATH = "category_matcher_learning.json"
NAME_SHADOW_PATH = "category_matcher_learning_names.json"
TFIDF_CACHE_PATH = "category_matcher_tfidf.pkl"
CLASSIFIER_PATH  = "category_matcher_clf.pkl"

# ─────────────────────────────────────────────────────────
# Thresholds
# ─────────────────────────────────────────────────────────
_KEYWORD_SCORE    = 0.92   # score assigned when keyword dict fires
_TFIDF_THRESHOLD  = 0.28   # min score to trust a TF-IDF suggestion
_TFIDF_GAP        = 0.10   # suggestion must beat current cat score by this much
_ML_FLOOR         = 0.65   # ML confidence below this is ignored
_RETRAIN_EVERY    = 10     # retrain classifier after this many new corrections

# ─────────────────────────────────────────────────────────
# Keyword → expected-category-path-fragment dictionary
# Sorted longest-first so the most specific pattern wins.
# ─────────────────────────────────────────────────────────
_KW_DICT: Dict[str, List[str]] = {
    # ── Phones & Tablets ─────────────────────────────
    "screen protector":  ["Screen Protectors"],
    "tempered glass":    ["Screen Protectors"],
    "phone case":        ["Cases & Covers"],
    "phone cover":       ["Cases & Covers"],
    "power bank":        ["Powerbanks"],
    "powerbank":         ["Powerbanks"],
    "charging cable":    ["Cables"],
    "usb cable":         ["Cables"],
    "bluetooth speaker": ["Portable Speakers"],
    "smart watch":       ["Smart Watch"],
    "smartwatch":        ["Smart Watch"],
    "airpods":           ["Headsets & Headphones", "Earphones"],
    "earbuds":           ["Headsets & Headphones", "Earphones"],
    "wireless earbuds":  ["Headsets & Headphones", "Earphones"],
    "iphone":            ["Smartphones"],
    "samsung galaxy":    ["Smartphones"],
    "redmi":             ["Smartphones"],
    "infinix hot":       ["Smartphones"],
    "infinix note":      ["Smartphones"],
    "tecno spark":       ["Smartphones"],
    "tecno camon":       ["Smartphones"],
    "tecno pop":         ["Smartphones"],
    "itel a":            ["Smartphones"],
    "ipad":              ["Tablets"],
    "tablet":            ["Tablets"],
    # ── Computing ────────────────────────────────────
    "laptop bag":        ["Laptop Bags & Cases"],
    "laptop stand":      ["Laptop Stands"],
    "macbook":           ["Laptops"],
    "laptop":            ["Laptops"],
    "chromebook":        ["Laptops"],
    "notebook pc":       ["Laptops"],
    "desktop pc":        ["Desktops"],
    "all-in-one pc":     ["Desktops"],
    "mechanical keyboard":["Keyboards"],
    "wireless keyboard": ["Keyboards"],
    "gaming mouse":      ["Keyboards & Mice"],
    "wireless mouse":    ["Keyboards & Mice"],
    "keyboard":          ["Keyboards"],
    "gaming monitor":    ["Monitors"],
    "monitor":           ["Monitors"],
    "external hard":     ["Hard Disk Drives"],
    "hard drive":        ["Hard Disk Drives"],
    "solid state":       ["Solid State Drives"],
    " ssd ":             ["Solid State Drives"],
    "flash drive":       ["Flash Drives"],
    "usb flash":         ["Flash Drives"],
    "wifi router":       ["Routers"],
    "router":            ["Routers"],
    "network switch":    ["Switches"],
    "printer":           ["Printers"],
    "ink cartridge":     ["Ink & Toners"],
    "toner cartridge":   ["Ink & Toners"],
    "hp toner":          ["Ink & Toners"],
    "webcam":            ["Webcams"],
    "ups battery":       ["UPS"],
    # ── TVs & Electronics ────────────────────────────
    "smart tv":          ["Smart TVs"],
    "4k tv":             ["Smart TVs"],
    "oled tv":           ["Smart TVs"],
    "qled tv":           ["Smart TVs"],
    "television":        ["Television"],
    "tv remote":         ["TV Accessories"],
    "projector":         ["Projectors"],
    "soundbar":          ["Soundbars"],
    "home theatre":      ["Home Theatre Systems"],
    "headphones":        ["Headphones"],
    "earphone":          ["Earphones"],
    "speaker":           ["Speakers"],
    "amplifier":         ["Amplifiers"],
    "dslr":              ["DSLR Cameras"],
    "mirrorless camera": ["Mirrorless Cameras"],
    "action camera":     ["Action Cameras"],
    "gopro":             ["Action Cameras"],
    "camera":            ["Digital Cameras"],
    "drone":             ["Drones"],
    "memory card":       ["Memory Cards"],
    "playstation":       ["Consoles"],
    "xbox":              ["Consoles"],
    "nintendo":          ["Consoles"],
    "gaming console":    ["Consoles"],
    "game controller":   ["Controllers"],
    "solar panel":       ["Solar Energy"],
    "inverter":          ["Generator & Inverter"],
    "generator":         ["Generator & Inverter"],
    # ── Home Appliances ──────────────────────────────
    "refrigerator":      ["Refrigerators"],
    "double door fridge":["Refrigerators"],
    "chest freezer":     ["Freezers"],
    "deep freezer":      ["Freezers"],
    "washing machine":   ["Washing Machines"],
    "tumble dryer":      ["Dryers"],
    "air conditioner":   ["Air Conditioners"],
    "split ac":          ["Air Conditioners"],
    "window ac":         ["Air Conditioners"],
    "air fryer":         ["Air Fryers"],
    "microwave oven":    ["Microwaves"],
    "microwave":         ["Microwaves"],
    "electric kettle":   ["Kettles"],
    "blender":           ["Blenders"],
    "food processor":    ["Food Processors"],
    "juicer":            ["Juicers"],
    "rice cooker":       ["Rice Cookers"],
    "pressure cooker":   ["Pressure Cookers"],
    "toaster":           ["Toasters"],
    "sandwich maker":    ["Sandwich Makers"],
    "electric iron":     ["Irons"],
    "steam iron":        ["Irons"],
    "vacuum cleaner":    ["Vacuums"],
    "ceiling fan":       ["Fans"],
    "standing fan":      ["Fans"],
    "table fan":         ["Fans"],
    "water dispenser":   ["Water Dispensers"],
    "dish washer":       ["Dishwashers"],
    # ── Health & Beauty ──────────────────────────────
    "hair dryer":        ["Hair Dryers"],
    "hair straightener": ["Hair Straighteners"],
    "hair clipper":      ["Hair Clippers"],
    "electric shaver":   ["Electric Shavers"],
    "electric razor":    ["Electric Shavers"],
    "body lotion":       ["Body Lotions"],
    "face cream":        ["Face Moisturisers"],
    "moisturiser":       ["Moisturisers"],
    "sunscreen":         ["Sunscreens"],
    "lipstick":          ["Lipstick"],
    "foundation":        ["Foundation"],
    "mascara":           ["Mascara"],
    "shampoo":           ["Shampoos"],
    "conditioner":       ["Conditioners"],
    "hair oil":          ["Hair Oils"],
    "toothbrush":        ["Toothbrushes"],
    "electric toothbrush":["Electric Toothbrushes"],
    "toothpaste":        ["Toothpaste"],
    "deodorant":         ["Deodorants"],
    "perfume":           ["Perfumes"],
    "cologne":           ["Colognes"],
    "protein powder":    ["Protein"],
    "vitamin":           ["Vitamins & Supplements"],
    "supplement":        ["Vitamins & Supplements"],
    "face mask":         ["Face Masks"],
    "blood pressure":    ["Blood Pressure Monitors"],
    "glucometer":        ["Blood Glucose Monitors"],
    # ── Fashion ──────────────────────────────────────
    "football boot":     ["Football Boots"],
    "running shoes":     ["Sports Shoes"],
    "sneakers":          ["Sneakers"],
    "high heels":        ["Heels"],
    "ankle boots":       ["Boots"],
    "handbag":           ["Handbags"],
    "backpack":          ["Backpacks"],
    "laptop backpack":   ["Laptop Bags & Cases", "Backpacks"],
    "wallet":            ["Wallets"],
    "wristwatch":        ["Watches"],
    "sunglasses":        ["Sunglasses"],
    "dress":             ["Dresses"],
    "jeans":             ["Jeans"],
    "t-shirt":           ["T-Shirts"],
    "polo shirt":        ["Polo Shirts"],
    "hoodie":            ["Hoodies"],
    "jacket":            ["Jackets"],
    "jersey":            ["Sports Jerseys"],
    "underwear":         ["Underwear"],
    # ── Food & Grocery ───────────────────────────────
    "cooking oil":       ["Cooking Oils"],
    "vegetable oil":     ["Cooking Oils"],
    "palm oil":          ["Cooking Oils"],
    "basmati rice":      ["Rice"],
    "long grain rice":   ["Rice"],
    "pasta":             ["Pasta & Noodles"],
    "instant noodles":   ["Pasta & Noodles"],
    "coffee":            ["Coffee"],
    "green tea":         ["Tea"],
    "energy drink":      ["Energy Drinks"],
    "chocolate":         ["Chocolate"],
    "biscuits":          ["Biscuits & Cookies"],
    "milk":              ["Milk"],
    "flour":             ["Flours & Baking"],
    "sugar":             ["Sugar & Sweeteners"],
    # ── Home & Office ────────────────────────────────
    "spring mattress":   ["Mattresses"],
    "foam mattress":     ["Mattresses"],
    "mattress":          ["Mattresses"],
    "pillow":            ["Pillows"],
    "bedsheet":          ["Bed Sheets"],
    "duvet":             ["Duvets & Quilts"],
    "sofa":              ["Sofas & Couches"],
    "office chair":      ["Office Chairs"],
    "gaming chair":      ["Gaming Chairs"],
    "dining table":      ["Dining Tables"],
    "desk":              ["Desks"],
    "bookshelf":         ["Bookshelves"],
    "curtains":          ["Curtains & Drapes"],
    "carpet":            ["Rugs & Carpets"],
    "led bulb":          ["Bulbs"],
    "light bulb":        ["Bulbs"],
    "frying pan":        ["Frying Pans"],
    "cooking pot":       ["Pots"],
    "cutlery":           ["Cutlery"],
    # ── Baby ─────────────────────────────────────────
    "baby diaper":       ["Diapers"],
    "pampers":           ["Diapers"],
    "baby formula":      ["Formula"],
    "baby food":         ["Baby Food"],
    "stroller":          ["Strollers"],
    "baby monitor":      ["Baby Monitors"],
    # ── Sports ───────────────────────────────────────
    "treadmill":         ["Treadmills"],
    "dumbbell":          ["Dumbbells"],
    "yoga mat":          ["Yoga Mats"],
    "bicycle":           ["Bicycles"],
    # ── Automobile ───────────────────────────────────
    "car battery":       ["Car Batteries"],
    "car tyre":          ["Tyres"],
    "engine oil":        ["Motor Oils"],
    "car charger":       ["Car Chargers"],
    "dash cam":          ["Dash Cams"],
}

# Build compiled patterns (longest keyword first to prevent short patterns shadowing long ones)
_KW_COMPILED: List[Tuple[re.Pattern, List[str]]] = [
    (re.compile(r"\b" + re.escape(kw.strip()) + r"\b", re.IGNORECASE), cats)
    for kw, cats in sorted(_KW_DICT.items(), key=lambda x: -len(x[0]))
]

# ─────────────────────────────────────────────────────────
# Text helpers
# ─────────────────────────────────────────────────────────
_NOISE = re.compile(
    r"\b(new|sale|original|genuine|authentic|official|premium|quality|best|"
    r"hot|2023|2024|2025|free|deal|promo|special|brand|latest|top|super|ultra|"
    r"pro|max|plus|mini|lite|get|buy|off|offer|price|cheap|good|fast)\b",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _NOISE.sub(" ", t)
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def _path_tokens(path: str) -> str:
    return re.sub(r"[/\-&]", " ", path.lower())


def _name_hash(name: str) -> str:
    return hashlib.md5(name.lower().strip().encode()).hexdigest()


# ─────────────────────────────────────────────────────────
# CategoryMatcherEngine
# ─────────────────────────────────────────────────────────

class CategoryMatcherEngine:
    """Self-improving, four-layer category matcher."""

    def __init__(self) -> None:
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._cat_matrix  = None
        self._cat_paths: List[str] = []
        self._tfidf_built = False

        self._compiled_rules: Dict = {}

        # ML layer
        self.learning_db: Dict[str, str] = {}
        self._clf: Optional[SGDClassifier]       = None
        self._le:  Optional[LabelEncoder]        = None
        self._clf_vec: Optional[TfidfVectorizer] = None
        self._clf_trained = False
        self._corrections_since_train = 0

        self._load_learning_db()
        self._load_classifier()

    # ── Setup ────────────────────────────────────────────

    def set_compiled_rules(self, rules: Dict) -> None:
        self._compiled_rules = rules or {}

    def build_tfidf_index(self, category_paths: List[str]) -> None:
        """Build or reload TF-IDF index. Safe to call multiple times."""
        if not category_paths:
            return

        fp = hashlib.md5("|".join(sorted(category_paths)).encode()).hexdigest()

        if os.path.exists(TFIDF_CACHE_PATH):
            try:
                with open(TFIDF_CACHE_PATH, "rb") as f:
                    cache = pickle.load(f)
                if cache.get("fp") == fp:
                    self._vectorizer  = cache["vec"]
                    self._cat_matrix  = cache["mat"]
                    self._cat_paths   = cache["paths"]
                    self._tfidf_built = True
                    logger.info("TF-IDF loaded from cache (%d cats)", len(self._cat_paths))
                    return
            except Exception as e:
                logger.warning("TF-IDF cache read failed: %s", e)

        logger.info("Building TF-IDF index for %d categories…", len(category_paths))

        # Augment corpus: path tokens + doubled leaf name = precision boost
        augmented = []
        for path in category_paths:
            leaf     = path.split("/")[-1].strip()
            aug_text = f"{_path_tokens(path)} {_clean(leaf)} {_clean(leaf)}"
            augmented.append(aug_text)

        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            min_df=1,
            max_features=60_000,
            sublinear_tf=True,
            analyzer="word",
        )
        self._cat_matrix = self._vectorizer.fit_transform(augmented)
        self._cat_paths  = list(category_paths)
        self._tfidf_built = True

        try:
            with open(TFIDF_CACHE_PATH, "wb") as f:
                pickle.dump({"fp": fp, "vec": self._vectorizer,
                             "mat": self._cat_matrix, "paths": self._cat_paths}, f)
        except Exception as e:
            logger.warning("TF-IDF cache write failed: %s", e)

    # ── Prediction ───────────────────────────────────────

    def get_top_categories(
        self,
        product_name: str,
        brand: str = "",
        n: int = 3,
    ) -> List[Tuple[str, float]]:
        """Return top-n (category_path, blended_score) tuples."""
        if not self._tfidf_built:
            return []

        query_raw = f"{brand} {product_name}".strip()
        query_cln = _clean(query_raw)
        if not query_cln:
            return []

        scores = np.zeros(len(self._cat_paths))

        # ── Layer 1: keyword dictionary ───────────────────
        kw_fragments: List[str] = []
        for pattern, frags in _KW_COMPILED:
            if pattern.search(query_raw):
                kw_fragments.extend(frags)
                break   # longest-matching keyword wins

        if kw_fragments:
            for frag in kw_fragments:
                frag_l = frag.lower()
                for i, path in enumerate(self._cat_paths):
                    if frag_l in path.lower():
                        scores[i] = max(scores[i], _KEYWORD_SCORE)

        # ── Layer 2: TF-IDF cosine similarity ────────────
        q_vec = self._vectorizer.transform([query_cln])
        sims  = cosine_similarity(q_vec, self._cat_matrix).flatten()
        # Keyword wins; TF-IDF only fills gaps
        blend = 0.4 if kw_fragments else 1.0
        scores = np.maximum(scores, sims * blend)

        # ── Layer 3: JSON boost rules ─────────────────────
        for cat_path, rule in self._compiled_rules.items():
            matches = rule["pattern"].findall(query_cln)
            if matches:
                boost = sum(rule["weights"].get(m.lower(), 0) for m in matches)
                try:
                    idx = self._cat_paths.index(cat_path)
                    scores[idx] = min(scores[idx] + boost * 0.15, 0.99)
                except ValueError:
                    pass

        # ── Layer 4: ML classifier ────────────────────────
        ml = self._ml_predict(query_cln)
        if ml:
            ml_path, ml_conf = ml
            if ml_conf >= _ML_FLOOR:
                try:
                    idx = self._cat_paths.index(ml_path)
                    scores[idx] = max(scores[idx], ml_conf * 0.97)
                except ValueError:
                    pass

        top_idx = np.argsort(scores)[::-1][:n]
        return [(self._cat_paths[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def get_category_with_boost(self, product_name: str, brand: str = "") -> Optional[str]:
        tops = self.get_top_categories(product_name, brand, n=1)
        return tops[0][0] if tops else None

    def is_wrong_category(
        self,
        product_name: str,
        current_category_path: str,
        brand: str = "",
        confidence_threshold: float = _TFIDF_THRESHOLD,
    ) -> Tuple[bool, List[Tuple[str, float]]]:
        """Return (is_wrong, suggestions)."""
        tops = self.get_top_categories(product_name, brand, n=3)
        if not tops:
            return False, []

        best_path, best_score = tops[0]
        current_l = current_category_path.strip().lower() if current_category_path else ""

        # Score of the current (listed) category
        current_score = 0.0
        for path, score in tops:
            if path.strip().lower() == current_l:
                current_score = score
                break
        if current_score == 0.0 and current_l and self._tfidf_built:
            q   = self._vectorizer.transform([query_cln := _clean(f"{brand} {product_name}")])
            cat = self._vectorizer.transform([_path_tokens(current_category_path)])
            current_score = float(cosine_similarity(q, cat).flatten()[0])

        best_is_current = best_path.strip().lower() == current_l
        score_gap       = best_score - current_score
        misc_flag       = "miscellaneous" in current_l

        wrong = misc_flag or (
            not best_is_current
            and best_score >= confidence_threshold
            and score_gap >= _TFIDF_GAP
        )
        return wrong, tops

    # ── Learning ──────────────────────────────────────────

    def apply_learned_correction(
        self,
        product_name: str,
        correct_category_path: str,
        auto_save: bool = True,
    ) -> None:
        """Record that product_name belongs in correct_category_path."""
        if not product_name or not correct_category_path:
            return
        key = _name_hash(product_name)
        self.learning_db[key] = correct_category_path.strip()
        self._corrections_since_train += 1
        self._save_name_shadow(key, product_name)
        if auto_save:
            self.save_learning_db()
        if self._corrections_since_train >= _RETRAIN_EVERY:
            self._retrain_correction_classifier()

    def save_learning_db(self) -> None:
        try:
            with open(LEARNING_DB_PATH, "w", encoding="utf-8") as f:
                json.dump(self.learning_db, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("save_learning_db failed: %s", e)

    # ── Internal ──────────────────────────────────────────

    def _save_name_shadow(self, key: str, name: str) -> None:
        try:
            store: Dict[str, str] = {}
            if os.path.exists(NAME_SHADOW_PATH):
                with open(NAME_SHADOW_PATH, "r", encoding="utf-8") as f:
                    store = json.load(f)
            store[key] = name.strip()
            with open(NAME_SHADOW_PATH, "w", encoding="utf-8") as f:
                json.dump(store, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning("Name shadow save failed: %s", e)

    def _load_learning_db(self) -> None:
        if os.path.exists(LEARNING_DB_PATH):
            try:
                with open(LEARNING_DB_PATH, "r", encoding="utf-8") as f:
                    self.learning_db = json.load(f)
                logger.info("Learning DB: %d corrections", len(self.learning_db))
            except Exception as e:
                logger.warning("_load_learning_db: %s", e)

    def _load_classifier(self) -> None:
        if os.path.exists(CLASSIFIER_PATH):
            try:
                with open(CLASSIFIER_PATH, "rb") as f:
                    b = pickle.load(f)
                self._clf     = b["clf"]
                self._le      = b["le"]
                self._clf_vec = b["vec"]
                self._clf_trained = True
                logger.info("Classifier loaded (%d classes)", len(self._le.classes_))
            except Exception as e:
                logger.warning("_load_classifier: %s", e)

    def _ml_predict(self, clean_query: str) -> Optional[Tuple[str, float]]:
        if not self._clf_trained or not clean_query:
            return None
        try:
            x     = self._clf_vec.transform([clean_query])
            idx   = self._clf.predict(x)[0]
            probs = self._clf.predict_proba(x)[0]
            return self._le.inverse_transform([idx])[0], float(probs[idx])
        except Exception:
            return None

    def _retrain_correction_classifier(self) -> None:
        if not os.path.exists(NAME_SHADOW_PATH):
            return
        try:
            with open(NAME_SHADOW_PATH, "r", encoding="utf-8") as f:
                name_store: Dict[str, str] = json.load(f)
        except Exception as e:
            logger.warning("_retrain: name shadow read: %s", e)
            return

        texts, labels = [], []
        for h, cat in self.learning_db.items():
            name = name_store.get(h)
            if name:
                texts.append(_clean(name))
                labels.append(cat)

        if len(set(labels)) < 2 or len(texts) < 5:
            logger.info("Not enough training data (%d samples, %d classes)", len(texts), len(set(labels)))
            return

        try:
            le  = LabelEncoder()
            y   = le.fit_transform(labels)
            vec = TfidfVectorizer(ngram_range=(1, 2), max_features=20_000, sublinear_tf=True)
            X   = vec.fit_transform(texts)
            clf = SGDClassifier(loss="modified_huber", max_iter=1000, tol=1e-3,
                                class_weight="balanced", random_state=42)
            clf.fit(X, y)

            self._clf     = clf
            self._le      = le
            self._clf_vec = vec
            self._clf_trained = True
            self._corrections_since_train = 0

            with open(CLASSIFIER_PATH, "wb") as f:
                pickle.dump({"clf": clf, "le": le, "vec": vec}, f)

            logger.info("Classifier retrained: %d samples, %d classes", len(texts), len(le.classes_))
        except Exception as e:
            logger.error("_retrain failed: %s", e)


# ─────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────
_ENGINE_INSTANCE: Optional[CategoryMatcherEngine] = None


def get_engine() -> CategoryMatcherEngine:
    global _ENGINE_INSTANCE
    if _ENGINE_INSTANCE is None:
        _ENGINE_INSTANCE = CategoryMatcherEngine()
    return _ENGINE_INSTANCE


# ─────────────────────────────────────────────────────────
# check_wrong_category – drop-in replacement for app.py stub
# ─────────────────────────────────────────────────────────

def check_wrong_category(
    data: pd.DataFrame,
    categories_list: List[str] = None,
    compiled_rules: Dict = None,
    cat_path_to_code: Dict[str, str] = None,
    code_to_path: Dict[str, str] = None,
    confidence_threshold: float = _TFIDF_THRESHOLD,
) -> pd.DataFrame:
    """
    Scan `data` for products whose CATEGORY doesn't match NAME/BRAND.

    Returned DataFrame has the same columns as `data` plus:
      Comment_Detail  – human-readable explanation shown in the QC UI
      Suggested_Cat_1 – best matching category path
      Suggested_Cat_2 – second best
      Suggested_Cat_3 – third best
      Match_Score     – confidence of top suggestion (0-1)
    """
    required = {"PRODUCT_SET_SID", "NAME"}
    if not required.issubset(data.columns):
        return pd.DataFrame(columns=data.columns)

    engine = get_engine()
    if compiled_rules:
        engine.set_compiled_rules(compiled_rules)
    if categories_list and not engine._tfidf_built:
        engine.build_tfidf_index(categories_list)

    if not engine._tfidf_built:
        if "CATEGORY" not in data.columns:
            return pd.DataFrame(columns=data.columns)
        flagged = data[
            data["CATEGORY"].astype(str).str.contains("miscellaneous", case=False, na=False)
        ].copy()
        if not flagged.empty:
            flagged["Comment_Detail"] = "Category contains 'Miscellaneous'"
        return flagged.drop_duplicates(subset=["PRODUCT_SET_SID"])

    extra        = ["Comment_Detail", "Suggested_Cat_1", "Suggested_Cat_2", "Suggested_Cat_3", "Match_Score"]
    flagged_rows = []

    for _, row in data.iterrows():
        name     = str(row.get("NAME",          "")).strip()
        brand    = str(row.get("BRAND",         "")).strip()
        cat_code = str(row.get("CATEGORY_CODE", "")).strip().split(".")[0]

        current_path = ""
        if code_to_path and cat_code:
            current_path = code_to_path.get(cat_code, "")
        if not current_path and "CATEGORY" in data.columns:
            current_path = str(row.get("CATEGORY", "")).strip()

        if not name:
            continue

        wrong, suggestions = engine.is_wrong_category(
            name, current_path, brand, confidence_threshold,
        )

        if wrong:
            r    = row.to_dict()
            top3 = suggestions[:3]
            r["Suggested_Cat_1"] = top3[0][0] if len(top3) > 0 else ""
            r["Suggested_Cat_2"] = top3[1][0] if len(top3) > 1 else ""
            r["Suggested_Cat_3"] = top3[2][0] if len(top3) > 2 else ""
            r["Match_Score"]     = round(top3[0][1], 3) if top3 else 0.0

            cur_leaf  = current_path.split("/")[-1].strip() if current_path else cat_code
            sugg_leaf = r["Suggested_Cat_1"].split("/")[-1].strip() if r["Suggested_Cat_1"] else "?"
            r["Comment_Detail"] = (
                f"Current: '{cur_leaf}' | Suggested: '{sugg_leaf}' "
                f"(confidence {r['Match_Score']:.0%})"
            )
            flagged_rows.append(r)

    if not flagged_rows:
        return pd.DataFrame(columns=list(data.columns) + extra)

    return pd.DataFrame(flagged_rows).drop_duplicates(subset=["PRODUCT_SET_SID"])
