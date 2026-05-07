from __future__ import annotations


from functools import lru_cache
from typing import Sequence


import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin


from process_skills import (
   SkillVocabulary,
   build_skill_texts,
   compute_group_binary,
   compute_group_counts,
   get_skill_counts,
   parse_skills,
)




def _as_series(X) -> pd.Series:
   if isinstance(X, pd.Series):
       return X
   if isinstance(X, pd.DataFrame):
       if X.shape[1] != 1:
           raise ValueError("Expected a single-column DataFrame for skill input.")
       return X.iloc[:, 0]
   return pd.Series(X)




class BinarySkillTransformer(BaseEstimator, TransformerMixin):
   """
   Transform HaveWorkedWith strings into a sparse binary skill matrix.
   """


   def __init__(self, min_freq: int = 2, delimiter: str = ";") -> None:
       self.min_freq = min_freq
       self.delimiter = delimiter


   def fit(self, X, y=None):
       X = _as_series(X)
       skill_lists = parse_skills(X, self.delimiter)


       self.vocabulary_ = SkillVocabulary(min_freq=self.min_freq)
       self.vocabulary_.fit(skill_lists)


       self.feature_names_ = [
           self.vocabulary_.index_to_skill_[i]
           for i in range(self.vocabulary_.n_skills)
       ]
       return self


   def transform(self, X):
       X = _as_series(X)
       skill_lists = parse_skills(X, self.delimiter)
       index_lists = self.vocabulary_.transform(skill_lists)


       rows = []
       cols = []


       for row_idx, indices in enumerate(index_lists):
           for col_idx in indices:
               rows.append(row_idx)
               cols.append(col_idx)


       data = np.ones(len(rows), dtype=np.float32)
       return csr_matrix(
           (data, (rows, cols)),
           shape=(len(index_lists), self.vocabulary_.n_skills),
           dtype=np.float32,
       )


   def get_feature_names_out(self):
       return np.array(self.feature_names_, dtype=object)




class SkillCountTransformer(BaseEstimator, TransformerMixin):
   """
   Return one feature: total distinct skills per row.
   """


   def __init__(self, delimiter: str = ";") -> None:
       self.delimiter = delimiter


   def fit(self, X, y=None):
       return self


   def transform(self, X):
       X = _as_series(X)
       skill_lists = parse_skills(X, self.delimiter)
       return get_skill_counts(skill_lists).to_frame(name="total_skills")


   def get_feature_names_out(self):
       return np.array(["total_skills"], dtype=object)




class GroupedSkillTransformer(BaseEstimator, TransformerMixin):
   """
   Map skills to grouped technology categories.


   mode:
     - "count": counts per group
     - "binary": 0/1 per group
     - "both": concatenate both
   """


   def __init__(self, delimiter: str = ";", mode: str = "both") -> None:
       self.delimiter = delimiter
       self.mode = mode


   def fit(self, X, y=None):
       return self


   def transform(self, X):
       X = _as_series(X)
       skill_lists = parse_skills(X, self.delimiter)


       if self.mode == "count":
           return compute_group_counts(skill_lists)
       if self.mode == "binary":
           return compute_group_binary(skill_lists)
       if self.mode == "both":
           counts = compute_group_counts(skill_lists)
           binary = compute_group_binary(skill_lists)
           return pd.concat([counts, binary], axis=1)


       raise ValueError("mode must be one of: 'count', 'binary', 'both'")


   def get_feature_names_out(self):
       from process_skills import TECH_GROUPS


       groups = list(TECH_GROUPS.keys())
       if self.mode == "count":
           return np.array([f"count_{g}" for g in groups], dtype=object)
       if self.mode == "binary":
           return np.array([f"has_{g}" for g in groups], dtype=object)
       return np.array(
           [f"count_{g}" for g in groups] + [f"has_{g}" for g in groups],
           dtype=object,
       )




@lru_cache(maxsize=1)
def _get_sentence_model(model_name: str):
   try:
       from sentence_transformers import SentenceTransformer
   except ImportError as exc:
       raise ImportError(
           "sentence-transformers is required for embeddings. "
           "Install it with: pip install sentence-transformers"
       ) from exc


   return SentenceTransformer(model_name)




def generate_embeddings(
   skill_strings: Sequence[str],
   model_name: str = "all-MiniLM-L6-v2",
   batch_size: int = 64,
   show_progress_bar: bool = False,
) -> np.ndarray:
   """
   Generate dense embeddings for a list of skill strings.
   """
   model = _get_sentence_model(model_name)
   embeddings = model.encode(
       list(skill_strings),
       batch_size=batch_size,
       show_progress_bar=show_progress_bar,
       convert_to_numpy=True,
       normalize_embeddings=True,
   )
   return embeddings.astype(np.float32)




class EmbeddingTransformer(BaseEstimator, TransformerMixin):
   """
   Sklearn-style transformer for skill embeddings.
   Stateless: nothing is fitted.
   """


   def __init__(
       self,
       model_name: str = "all-MiniLM-L6-v2",
       delimiter: str = ";",
       batch_size: int = 64,
       show_progress_bar: bool = False,
   ) -> None:
       self.model_name = model_name
       self.delimiter = delimiter
       self.batch_size = batch_size
       self.show_progress_bar = show_progress_bar


   def fit(self, X, y=None):
       return self


   def transform(self, X):
       X = _as_series(X)
       skill_lists = parse_skills(X, self.delimiter)
       skill_strings = build_skill_texts(skill_lists)
       return generate_embeddings(
           skill_strings,
           model_name=self.model_name,
           batch_size=self.batch_size,
           show_progress_bar=self.show_progress_bar,
       )


   def get_feature_names_out(self):
       return np.array([], dtype=object)




def combine_features(*arrays):
   """
   Horizontally stack dense or sparse feature blocks.
   Returns sparse if any input is sparse.
   """
   if any(sparse.issparse(a) for a in arrays):
       converted = [
           a if sparse.issparse(a) else sparse.csr_matrix(np.asarray(a))
           for a in arrays
       ]
       return sparse.hstack(converted).tocsr()


   dense_parts = []
   for arr in arrays:
       if isinstance(arr, pd.DataFrame):
           part = arr.to_numpy()
       elif isinstance(arr, pd.Series):
           part = arr.to_numpy().reshape(-1, 1)
       else:
           part = np.asarray(arr)
           if part.ndim == 1:
               part = part.reshape(-1, 1)
       dense_parts.append(part)


   return np.hstack(dense_parts).astype(np.float32)




def build_skill_representations(
   X_train,
   X_val,
   X_test,
   skill_col: str = "HaveWorkedWith",
   min_freq: int = 2,
   delimiter: str = ";",
   use_embeddings: bool = True,
):
   """
   Build train/val/test feature sets for the skills pipeline.
   """
   X_train_skill = X_train[skill_col]
   X_val_skill = X_val[skill_col]
   X_test_skill = X_test[skill_col]


   outputs = {}


   binary_tf = BinarySkillTransformer(min_freq=min_freq, delimiter=delimiter)
   X_train_bin = binary_tf.fit_transform(X_train_skill)
   X_val_bin = binary_tf.transform(X_val_skill)
   X_test_bin = binary_tf.transform(X_test_skill)


   outputs["binary"] = {
       "transformer": binary_tf,
       "X_train": X_train_bin,
       "X_val": X_val_bin,
       "X_test": X_test_bin,
   }


   count_tf = SkillCountTransformer(delimiter=delimiter)
   X_train_cnt = count_tf.fit_transform(X_train_skill)
   X_val_cnt = count_tf.transform(X_val_skill)
   X_test_cnt = count_tf.transform(X_test_skill)


   outputs["counts"] = {
       "transformer": count_tf,
       "X_train": X_train_cnt,
       "X_val": X_val_cnt,
       "X_test": X_test_cnt,
   }


   group_tf = GroupedSkillTransformer(delimiter=delimiter, mode="both")
   X_train_grp = group_tf.fit_transform(X_train_skill)
   X_val_grp = group_tf.transform(X_val_skill)
   X_test_grp = group_tf.transform(X_test_skill)


   outputs["grouped"] = {
       "transformer": group_tf,
       "X_train": X_train_grp,
       "X_val": X_val_grp,
       "X_test": X_test_grp,
   }


   if use_embeddings:
       emb_tf = EmbeddingTransformer(delimiter=delimiter)
       X_train_emb = emb_tf.fit_transform(X_train_skill)
       X_val_emb = emb_tf.transform(X_val_skill)
       X_test_emb = emb_tf.transform(X_test_skill)


       outputs["embeddings"] = {
           "transformer": emb_tf,
           "X_train": X_train_emb,
           "X_val": X_val_emb,
           "X_test": X_test_emb,
       }


       outputs["binary_plus_embeddings"] = {
           "transformer": None,
           "X_train": combine_features(X_train_bin, X_train_emb),
           "X_val": combine_features(X_val_bin, X_val_emb),
           "X_test": combine_features(X_test_bin, X_test_emb),
       }


   return outputs
