import os
import re
import json
import pickle
import numpy as np
import pandas as pd
import logging
import traceback
import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

class CategoryMatcherEngine:
    def __init__(self, db_path="cat_learning.db"):
        self.db_path = db_path
        self.vectorizer = TfidfVectorizer(
            analyzer='word',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        self.tfidf_matrix = None
        self.categories = []
        self._tfidf_built = False
        self.learning_db = {}
        self.compiled_rules = {}  # Store JSON rules directly in the engine
        self.correction_classifier = None
        self.correction_vectorizer = None
        self._init_db()
        self.load_learning_db()

    def set_compiled_rules(self, rules: dict):
        """Loads the JSON heuristic rules into the engine."""
        self.compiled_rules = rules or {}

    def _init_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute('''
                    CREATE TABLE IF NOT EXISTS category_corrections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        category TEXT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                conn.commit()
        except Exception as e:
            logger.warning(f"Failed to init category learning DB: {e}")

    def load_learning_db(self):
        try:
            with sqlite3.connect(self.db_path) as conn:
                df = pd.read_sql_query("SELECT name, category FROM category_corrections", conn)
                if not df.empty:
                    self.learning_db = df.groupby('name')['category'].last().to_dict()
                    self._retrain_correction_classifier(df)
        except Exception as e:
            logger.warning(f"Failed to load category learning DB: {e}")

    def _retrain_correction_classifier(self, df=None):
        if df is None:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    df = pd.read_sql_query("SELECT name, category FROM category_corrections", conn)
            except Exception:
                return
        if df is None or df.empty or len(df['category'].unique()) < 2:
            return
        try:
            df['clean_name'] = df['name'].apply(clean_text)
            self.correction_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
            X = self.correction_vectorizer.fit_transform(df['clean_name'])
            y = df['category']
            self.correction_classifier = LogisticRegression(class_weight='balanced', max_iter=1000)
            self.correction_classifier.fit(X, y)
        except Exception as e:
            logger.warning(f"Failed to retrain correction classifier: {e}")

    def apply_learned_correction(self, name: str, category: str, auto_save=True):
        clean_n = clean_text(name)
        if not clean_n or not category: return
        self.learning_db[clean_n] = category
        if auto_save:
            try:
                with sqlite3.connect(self.db_path) as conn:
                    c = conn.cursor()
                    c.execute("INSERT INTO category_corrections (name, category) VALUES (?, ?)", (clean_n, category))
                    conn.commit()
                self._retrain_correction_classifier()
            except Exception as e:
                logger.warning(f"Failed to save correction to DB: {e}")

    def save_learning_db(self):
        if not self.learning_db: return
        try:
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("BEGIN TRANSACTION")
                for name, cat in self.learning_db.items():
                    c.execute("INSERT INTO category_corrections (name, category) VALUES (?, ?)", (name, cat))
                conn.commit()
            self._retrain_correction_classifier()
        except Exception as e:
            logger.warning(f"Failed to batch save learning DB: {e}")

    def build_tfidf_index(self, categories_list: list):
        if not categories_list: return
        self.categories = [str(c).strip() for c in categories_list if str(c).strip() and str(c).strip().lower() != 'nan']
        if not self.categories: return
        clean_cats = [clean_text(c) for c in self.categories]
        try:
            self.tfidf_matrix = self.vectorizer.fit_transform(clean_cats)
            self._tfidf_built = True
        except Exception as e:
            logger.warning(f"Failed to build TF-IDF index: {e}")

    def predict_category_from_learning(self, name: str) -> str:
        clean_n = clean_text(name)
        if clean_n in self.learning_db:
            return self.learning_db[clean_n]
        if self.correction_classifier and self.correction_vectorizer:
            try:
                vec = self.correction_vectorizer.transform([clean_n])
                probs = self.correction_classifier.predict_proba(vec)[0]
                max_prob_idx = np.argmax(probs)
                if probs[max_prob_idx] > 0.6: 
                    return self.correction_classifier.classes_[max_prob_idx]
            except Exception:
                pass
        return None

    def get_category_with_fallback(self, name: str, kw_map: dict = None, categories_list: list = None) -> str:
        learned = self.predict_category_from_learning(name)
        if learned: return learned
        
        if self._tfidf_built:
            try:
                name_clean = clean_text(name)
                name_vec = self.vectorizer.transform([name_clean])
                similarities = cosine_similarity(name_vec, self.tfidf_matrix).flatten()
                best_idx = np.argmax(similarities)
                if similarities[best_idx] > 0.15:
                    return self.categories[best_idx]
            except Exception:
                pass
                
        if kw_map:
            name_lower = str(name).lower()
            for kw, cat in kw_map.items():
                if re.search(r'\b' + re.escape(kw) + r'\b', name_lower):
                    return cat
        return ""

    def get_category_with_boost(self, name: str, top_n: int = 20) -> str:
        """
        Gets the top N TF-IDF predictions, applies internal JSON heuristic boosts, 
        and returns the highest scoring category.
        """
        learned = self.predict_category_from_learning(name)
        if learned: 
            return learned

        if not getattr(self, '_tfidf_built', False):
            return ""
        
        try:
            name_clean = clean_text(name)
            name_vec = self.vectorizer.transform([name_clean])
            similarities = cosine_similarity(name_vec, self.tfidf_matrix).flatten()
            
            top_indices = similarities.argsort()[-top_n:][::-1]
            
            best_category = ""
            best_score = -1.0
            name_lower = str(name).lower()
            
            for idx in top_indices:
                cat_path = self.categories[idx]
                base_score = float(similarities[idx])
                boost = 0.0
                
                # 1. Convert the engine's category path to lowercase
                cat_path_lower = cat_path.lower()
                
                # 2. Check the JSON rules using the lowercase path
                if cat_path_lower in self.compiled_rules:
                    rule = self.compiled_rules[cat_path_lower]
                    matches = rule['pattern'].findall(name_lower)
                    if matches:
                        boost = sum(rule['weights'].get(m.lower(), 0.0) for m in set(matches))
                
                final_score = base_score + (boost * 0.6) # Increased multiplier slightly for more authority
                
                if final_score > best_score:
                    best_score = final_score
                    best_category = cat_path
            
            # Reject garbage matches if confidence is too low
            if best_score < 0.12:
                return ""
                
            return best_category
            
        except Exception as e:
            logger.warning(f"Boosted prediction failed: {e}")
            return ""

    def build_keyword_to_category_mapping(self) -> dict:
        kw_map = {}
        for cat in self.categories:
            parts = [p.strip().lower() for p in cat.split('>')]
            if len(parts) > 1:
                kw_map[parts[-1]] = cat
        return kw_map


_engine_instance = None

def get_engine(db_path="cat_learning.db"):
    global _engine_instance
    if _engine_instance is None:
        try:
            _engine_instance = CategoryMatcherEngine(db_path)
        except Exception as e:
            logger.error(f"Failed to initialize CategoryMatcherEngine: {e}")
            logger.error(traceback.format_exc())
            _engine_instance = None
    return _engine_instance

def check_wrong_category(data: pd.DataFrame, categories_list: list, compiled_rules: dict = None, cat_path_to_code: dict = None, code_to_path: dict = None, confidence_threshold: float = 0.0):
    if not {'NAME', 'CATEGORY'}.issubset(data.columns) or not categories_list:
        return pd.DataFrame(columns=data.columns)
        
    engine = get_engine()
    if engine is None:
        return pd.DataFrame(columns=data.columns)

    if not engine._tfidf_built:
        engine.build_tfidf_index(categories_list)

    # CRITICAL: Feed the engine the JSON rules so it can use them!
    if compiled_rules:
        engine.set_compiled_rules(compiled_rules)

    if cat_path_to_code is None: cat_path_to_code = {}
    if code_to_path is None: code_to_path = {}

    d = data.copy()
    d['_cat_clean'] = d['CATEGORY'].astype(str).str.strip()
    
    if 'CATEGORY_CODE' in d.columns and code_to_path:
        for idx, row in d.iterrows():
            if not row['_cat_clean'] or row['_cat_clean'].lower() in ('nan', 'none', 'miscellaneous'):
                code = str(row.get('CATEGORY_CODE', '')).strip().split('.')[0]
                if code in code_to_path:
                    d.at[idx, '_cat_clean'] = code_to_path[code]

    d['_cat_lower'] = d['_cat_clean'].str.lower()
    d['_name_clean'] = d['NAME'].astype(str).str.strip()
    
    flagged_indices = []
    comment_map = {}
    kw_map = engine.build_keyword_to_category_mapping()

    for idx, row in d.iterrows():
        current_cat = row['_cat_clean']
        name = row['_name_clean']
        
        if not current_cat or current_cat.lower() in ('nan', 'none', ''):
            continue

        if 'miscellaneous' in current_cat.lower():
            flagged_indices.append(idx)
            comment_map[idx] = "Category is 'Miscellaneous'"
            continue

        # ---> CRITICAL FIX: The automated scanner now uses the JSON boost!
        predicted = engine.get_category_with_boost(name)
        
        # Fallback to standard tf-idf if boost rejected it for low confidence
        if not predicted:
            predicted = engine.get_category_with_fallback(name, kw_map, categories_list)
        
        if predicted and predicted.lower() != current_cat.lower():
            # Extract just the "leaf" category (the final word in the path) for a cleaner comparison
            p_leaf = predicted.split('>')[-1].strip().lower()
            c_leaf = current_cat.split('>')[-1].strip().lower()
            
            if p_leaf != c_leaf:
                flagged_indices.append(idx)
                comment_map[idx] = f"Wrong Category. Suggested: {predicted}"

    if not flagged_indices:
        return pd.DataFrame(columns=data.columns)

    res = data.loc[flagged_indices].copy()
    res['Comment_Detail'] = res.index.map(comment_map)
    return res.drop_duplicates(subset=['PRODUCT_SET_SID'])
