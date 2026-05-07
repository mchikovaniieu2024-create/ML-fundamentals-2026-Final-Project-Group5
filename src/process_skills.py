from __future__ import annotations


import re
from dataclasses import dataclass, field
from typing import Sequence


import numpy as np
import pandas as pd




# Normalization tokens to lower case,
# and expand meanings of abbreviations








def normalize_skill(token: str) -> str:


   if token is None:
       return ""


   token = str(token).strip().lower()
   token = re.sub(r"\s+", " ", token)


   if not token:
       return ""
   # no splitting on punctuation, normalize spaces
   return token




def parse_skills(values: Sequence[str] | pd.Series, delimiter: str = ";") -> list[list[str]]:


   if isinstance(values, pd.Series):
       raw_iter = values.fillna("").astype(str).tolist()
   else:
       raw_iter = ["" if v is None else str(v) for v in values]


   parsed: list[list[str]] = []
   for raw in raw_iter:
       raw = raw.strip()
       if not raw:
           parsed.append([])
           continue


       row_skills: list[str] = []
       seen = set()


       for part in raw.split(delimiter):
           skill = normalize_skill(part)
           if not skill or skill in seen:
               continue
           seen.add(skill)
           row_skills.append(skill)


       parsed.append(row_skills)


   # Produce one list of canonical skill tokens per row seperated by semicolon ;


   return parsed




def get_skill_counts(skill_lists: Sequence[Sequence[str]]) -> pd.Series:
   return pd.Series([len(set(skills)) for skills in skill_lists], name="total_skills")




def build_skill_text(skill_list: Sequence[str]) -> str:
   #Create a text from ONE USERS skill list into a string separated by space
   return " ".join(sorted(set(skill_list))) if skill_list else ""




def build_skill_texts(skill_lists: Sequence[Sequence[str]]) -> list[str]:
   # Create a text from ALL USERS skill list into a string separated by space
   return [build_skill_text(skills) for skills in skill_lists]




# Dataset-specific grouped features


TECH_GROUPS = {


   "frontend": {
       "javascript", "typescript", "html/css",
       "react.js", "angular", "angular.js",
       "vue.js", "next.js", "jquery"
   },


   "backend": {
       "node.js", "express",
       "django", "flask", "laravel",
       "asp.net", "asp.net core",
       "spring", "java", "php", "c#"
   },


   "databases": {
       "sql", "postgresql", "mysql",
       "sqlite", "mongodb", "redis",
       "microsoft sql server", "mariadb",
       "elasticsearch", "oracle"
   },


   "cloud": {
       "aws", "microsoft azure", "google cloud platform",
       "heroku", "digitalocean"
   },


   "devops": {
       "docker", "kubernetes", "terraform",
       "ansible", "bash/shell", "git",
       "npm", "yarn"
   },


   "programming_languages": {
       "python", "java", "c", "c++", "c#", "go",
       "kotlin", "php", "javascript", "typescript"
   },


   "data_science": {
       "python", "pandas", "numpy",
       "scikit-learn", "tensorflow", "pytorch"
   },


   "mobile": {
       "android", "ios", "flutter",
       "react native", "kotlin", "swift"
   }
}




def _count_groups(skill_lists: Sequence[Sequence[str]]) -> pd.DataFrame:
   rows = []
   for skills in skill_lists:
       skill_set = set(skills)
       row = {}
       for group_name, group_skills in TECH_GROUPS.items():
           row[group_name] = sum(1 for skill in skill_set if skill in group_skills)
       rows.append(row)


   return pd.DataFrame(rows)




def compute_group_counts(skill_lists: Sequence[Sequence[str]]) -> pd.DataFrame:


   return _count_groups(skill_lists).astype(np.int32)




def compute_group_binary(skill_lists: Sequence[Sequence[str]]) -> pd.DataFrame:
   #Return binary group from skills list


   counts = _count_groups(skill_lists)
   binary = (counts > 0).astype(np.int32)
   binary.columns = [f"has_{c}" for c in binary.columns]
   return binary




# Train-only vocabulary


@dataclass
class SkillVocabulary:


   min_freq: int = 2
   skill_to_index_: dict[str, int] = field(default_factory=dict, init=False)
   index_to_skill_: dict[int, str] = field(default_factory=dict, init=False)
   counts_: dict[str, int] = field(default_factory=dict, init=False)
   n_skills: int = field(default=0, init=False)


   def fit(self, skill_lists: Sequence[Sequence[str]]) -> "SkillVocabulary":
       counts: dict[str, int] = {}
       for skills in skill_lists:
           for skill in set(skills):
               counts[skill] = counts.get(skill, 0) + 1


       kept = sorted([skill for skill, freq in counts.items() if freq >= self.min_freq])


       self.counts_ = counts
       self.skill_to_index_ = {skill: i for i, skill in enumerate(kept)}
       self.index_to_skill_ = {i: skill for skill, i in self.skill_to_index_.items()}
       self.n_skills = len(kept)
       return self


   def transform(self, skill_lists: Sequence[Sequence[str]]) -> list[list[int]]:
       if not self.skill_to_index_:
           raise ValueError("SkillVocabulary is not fitted yet.")


       transformed: list[list[int]] = []
       for skills in skill_lists:
           indices: list[int] = []
           seen = set()
           for skill in skills:
               idx = self.skill_to_index_.get(skill)
               if idx is None or idx in seen:
                   continue
               seen.add(idx)
               indices.append(idx)
           transformed.append(indices)
       return transformed


   def fit_transform(self, skill_lists: Sequence[Sequence[str]]) -> list[list[int]]:
       self.fit(skill_lists)
       return self.transform(skill_lists)
