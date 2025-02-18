import pandas as pd
import re
import chardet
from collections import Counter
from autocorrect import Speller

# Initialize the spell corrector
spell = Speller(lang='en')

def detect_encoding(file_path):
    """Detects the file encoding using chardet."""
    with open(file_path, "rb") as file:
        raw_data = file.read()
        result = chardet.detect(raw_data)
        return result["encoding"]

def extract_common_words_from_csv(file_path, output_file, top_n=1000):
    # Detect encoding for the CSV file
    encoding = detect_encoding(file_path)

    # Read the CSV file into a DataFrame (skip bad lines)
    df = pd.read_csv(file_path, encoding=encoding, on_bad_lines='skip')

    # Combine text from all columns into one string
    all_text = " ".join(df.astype(str).values.flatten())

    # Remove punctuation and non-word characters (keeping letters, numbers, and underscores)
    cleaned_text = re.sub(r'[^a-zA-Z0-9_]+', ' ', all_text)

    # Convert text to lowercase and split into individual words
    words = cleaned_text.lower().split()

    # Correct typos in each word (using a cache to improve performance)
    cache = {}
    def correct_word(word):
        if word in cache:
            return cache[word]
        corrected = spell(word)
        cache[word] = corrected
        return corrected

    corrected_words = [correct_word(word) for word in words]

    # Define a set of common stopwords (English & Hinglish) to exclude
    common_stopwords = {
        # Basic English stopwords
        "the", "is", "a", "an", "to", "this", "that", "it", "of", "in", "at", "with", "as",
        "for", "was", "were", "has", "had", "have", "be", "but", "by", "so", "or", "nor",
        "on", "off", "out", "up", "down", "not", "do", "does", "did", "been", "just", "being",
        "i", "you", "he", "she", "we", "they", "me", "him", "her", "them", "my", "your", "our",
        "their", "mine", "yours", "ours", "theirs",
        # Additional common English words
        "if", "then", "because", "about", "into", "over", "after", "before", "again", "further",
        "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "only", "own", "same",
        "than", "too", "very",
        # Common Hinglish/Indian stopwords
        "hai", "ka", "ki", "ke", "ko", "se", "aur", "mein", "par", "bhi", "ye", "woh", "nahi",
        "ho", "tha", "thi", "hen", "ab", "lekin", "magar", "bas", "jai", "raha", "rahi", "rahe",
        "kuch", "aisa", "aisi", "itna", "itni", "itne", "samajh", "samajhna", "kya"
    }

    # Filter out stopwords from the corrected words
    filtered_words = [word for word in corrected_words if word not in common_stopwords]

    # Count the frequency of each word
    word_counts = Counter(filtered_words)

    # Get the top N most common words
    common_words = word_counts.most_common(top_n)

    # Write the result to the output file
    with open(output_file, "w", encoding="utf-8") as f:
        for word, count in common_words:
            f.write(f"{word}: {count}\n")

    print(f"Top {top_n} common words have been saved to '{output_file}'.")

if __name__ == "__main__":
    # Update the file names as needed
    input_csv_file = "data/crime_report.csv"  # Path to your CSV file
    output_text_file = "common_words.txt"       # Output file for the common words
    extract_common_words_from_csv(input_csv_file, output_text_file)
