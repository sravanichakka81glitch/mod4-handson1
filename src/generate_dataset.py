"""
Generate the Telugu movies dataset as a CSV file.
"""

import os
import pandas as pd

MOVIES = [
    ("Bahubali", "hit", "positive", 5, "Epic masterpiece with stunning visuals and powerful storytelling"),
    ("Radhe Shyam", "flop", "negative", 2, "Weak screenplay and disappointing execution"),
    ("RRR", "hit", "positive", 5, "Phenomenal action and brilliant performances by lead actors"),
    ("Acharya", "flop", "negative", 2, "Weak story and poor screenplay wasted talented cast"),
    ("Pushpa", "hit", "positive", 5, "Raw and rustic performance with mass appeal"),
    ("Bheemla Nayak", "hit", "positive", 4, "Powerful performances and engaging drama"),
    ("Liger", "flop", "negative", 1, "Confusing plot and poor execution throughout"),
    ("KGF Chapter 2", "hit", "positive", 5, "Grand scale action with goosebumps moments"),
    ("Sarkaru Vaari Paata", "hit", "positive", 4, "Entertaining commercial film with good mass elements"),
    ("Thank You", "flop", "negative", 2, "Predictable storyline and weak narration"),
    ("Vikram", "hit", "positive", 5, "Brilliant screenplay and outstanding performances"),
    ("Ravanasura", "flop", "negative", 2, "Confused narrative and weak character development"),
    ("Waltair Veerayya", "hit", "positive", 4, "Mass entertainer with energetic performances"),
    ("Veera Simha Reddy", "hit", "positive", 4, "Powerful mass dialogues and action sequences"),
    ("Agent", "flop", "negative", 1, "Poor storyline and disappointing action sequences"),
    ("Salaar", "hit", "positive", 5, "Intense action drama with stellar performances"),
    ("Animal", "hit", "positive", 5, "Bold and gripping narrative with powerful acting"),
    ("Hi Nanna", "hit", "positive", 4, "Emotional and heartwarming family entertainer"),
    ("Guntur Kaaram", "flop", "negative", 2, "Weak screenplay and outdated treatment"),
    ("Eagle", "flop", "negative", 2, "Disjointed plot and poor execution"),
    ("Hanuman", "hit", "positive", 5, "Refreshing superhero film with great VFX"),
    ("Tillu Square", "hit", "positive", 4, "Entertaining comedy with engaging screenplay"),
    ("Family Star", "flop", "negative", 2, "Weak story and predictable narration"),
    ("Operation Valentine", "flop", "negative", 2, "Average execution and weak emotional connect"),
    ("Kalki 2898 AD", "hit", "positive", 5, "Visual spectacle with engaging narrative"),
    ("Pushpa 2", "hit", "positive", 5, "Massive entertainer with powerful performance"),
    ("Devara", "hit", "positive", 4, "Impressive action and strong screen presence"),
    ("Saripodhaa Sanivaaram", "hit", "positive", 4, "Unique concept with engaging execution"),
    ("Mathu Vadalara 2", "hit", "positive", 4, "Hilarious comedy with smart writing"),
    ("Lucky Baskhar", "hit", "positive", 5, "Brilliant performances and engaging screenplay"),
    ("KA", "hit", "positive", 4, "Suspenseful thriller with good twists"),
    ("Mechanic Rocky", "flop", "negative", 2, "Predictable plot and weak execution"),
    ("Bhaje Vaayu Vegam", "flop", "negative", 2, "Confused narrative and poor pacing"),
    ("Gam Gam Ganesha", "flop", "negative", 2, "Weak comedy and underdeveloped characters"),
    ("Krishnamma", "flop", "negative", 2, "Average storyline and weak emotional depth"),
    ("Committee Kurrollu", "hit", "positive", 4, "Fresh narrative with relatable characters"),
    ("35 Chinna Katha Kaadu", "hit", "positive", 5, "Heartwarming story with excellent performances"),
    ("Aay", "hit", "positive", 4, "Fun entertainer with good comedy timing"),
    ("Kotha Bangaru Lokam", "hit", "positive", 4, "Emotional coming of age story with good music"),
    ("Magadheera", "hit", "positive", 5, "Epic romantic action with brilliant visuals"),
    ("Eega", "hit", "positive", 5, "Innovative concept with stunning execution"),
    ("Baahubali 2", "hit", "positive", 5, "Grand conclusion with emotional highs and spectacular visuals"),
    ("Arjun Reddy", "hit", "positive", 5, "Raw and intense performance with bold storytelling"),
    ("Jersey", "hit", "positive", 5, "Emotional sports drama with brilliant acting"),
    ("Mahanati", "hit", "positive", 5, "Bio-pic with outstanding performances and direction"),
    ("C/o Kancharapalem", "hit", "positive", 5, "Beautiful anthology with realistic storytelling"),
    ("Vedam", "hit", "positive", 5, "Powerful anthology with strong social message"),
    ("Leader", "hit", "positive", 4, "Political drama with strong narrative"),
    ("Manam", "hit", "positive", 5, "Heartwarming family drama with excellent performances"),
    ("Athadu", "hit", "positive", 5, "Stylish action entertainer with brilliant screenplay"),
    ("Pokiri", "hit", "positive", 5, "Iconic mass entertainer with powerful performance"),
    ("Dookudu", "hit", "positive", 5, "Entertaining action comedy with great timing"),
    ("Gabbar Singh", "hit", "positive", 5, "Mass entertainer with powerful dialogues and action"),
]


def generate_dataset(output_path: str) -> None:
    df = pd.DataFrame(MOVIES, columns=["movie_name", "verdict", "sentiment", "stars", "review"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset generated with {len(df)} movies at: {output_path}")


if __name__ == "__main__":
    generate_dataset(os.path.join("data", "telugu_movies.csv"))