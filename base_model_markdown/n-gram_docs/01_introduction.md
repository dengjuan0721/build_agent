```markdown
# Introduction

## What Are N-grams?

N-grams are fundamental building blocks in statistical language modeling. Simply put, an n-gram is a contiguous sequence of *n* items (words, characters, or symbols) extracted from a given text. For example:  
- **Unigram (1-gram):** "language"  
- **Bigram (2-gram):** "statistical model"  
- **Trigram (3-gram):** "natural language processing"  

These sequences help machines predict or generate text by analyzing patterns in large datasets.

## Why Are N-grams Important?

N-grams power many real-world applications:  
- **Autocomplete** (predicting the next word in a search query)  
- **Spell-checking** (identifying likely corrections)  
- **Machine translation** (mapping phrases between languages)  

Their simplicity and efficiency make them a go-to tool for modeling language statistically, even in the era of advanced AI.

## How Do N-grams Work?

1. **Frequency Counting:** N-gram models learn by counting how often sequences appear in training data.  
   - Example: If "the cat" occurs 50 times in a corpus, its probability is higher than rare bigrams.  
2. **Prediction:** The model uses these counts to estimate the likelihood of the next item in a sequence.  

A key challenge is handling unseen n-grams (solved later via techniques like *smoothing*).

## Conclusion

This chapter introduced n-grams as a core concept for statistical language modeling. By breaking text into manageable sequences, n-grams provide a straightforward yet powerful way to capture linguistic patterns. In the next chapters, we’ll dive deeper into their mathematical foundations, limitations, and optimizations.  
```