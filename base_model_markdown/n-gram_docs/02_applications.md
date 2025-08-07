```markdown
# Applications

N-gram models, as fundamental statistical language tools, power diverse real-world applications by capturing local linguistic patterns. Below are key use cases demonstrating their practical impact:

## 1. **Autocomplete & Predictive Text**
- **Case Study**: Google Keyboard  
  Trigram models predict next-word candidates by analyzing preceding two words (e.g., "happy birthday _to_"). User-specific n-gram frequencies personalize suggestions over time.

## 2. **Spell Checking & Correction**
- **Case Study**: GNU Aspell  
  Bigrams/trigrams identify contextual typos by comparing P("their plan") vs. P("they plan"). Low-probability sequences trigger corrections.

## 3. **Machine Translation**
- **Case Study**: Early Statistical MT (e.g., IBM Candide)  
  Bilingual n-grams aligned phrase pairs ("house" ↔ "maison") to generate translations, later refined with neural approaches.

## 4. **Speech Recognition**
- **Case Study**: CMU Sphinx  
  Acoustic models combined with trigram language models to resolve homophones (e.g., "two" vs. "too") based on preceding words.

## 5. **Genre Detection**
- **Case Study**: Email Filtering  
  Character 4-grams distinguish technical reports (high "tion") from casual chats (high "lol"), achieving 92% accuracy in spam classification.

## 6. **DNA Sequence Analysis**
- **Case Study**: BLAST Genome Search  
  3-grams (codons) identify conserved protein regions by matching statistical patterns across species.

## Conclusion
From keystrokes to genomes, n-grams provide a lightweight yet powerful framework for modeling sequential data. Their computational efficiency makes them indispensable in latency-sensitive applications like real-time typing aids, while their interpretability aids debugging in complex systems. As we'll explore next, modern neural models often build upon these statistical foundations.
```