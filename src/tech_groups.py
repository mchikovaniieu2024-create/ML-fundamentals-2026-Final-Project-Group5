from data_utils import load_data, clean_data
from process_skills import parse_skills
from collections import Counter


# Load and clean
df = clean_data(load_data())


# Parse skills
skills = parse_skills(df["HaveWorkedWith"])


# Flatten
all_skills = [s for row in skills for s in row]


# Count frequencies
counts = Counter(all_skills)


# Print top skills
print(counts.most_common(50))
