# Text Summarizer

This text summarizer generates abstractive summaries from .txt files using the transformers library and the pre-trained facebook/bart-large-cnn model. It condenses text into concise, newly phrased summaries.

## Requirements

- Python 3.x
- transformers
- torch

## Setup
1. In root directory activate virutal env `source venv/bin/activate`
2. Move into the correct working directory by running `cd Day_06_TextSummarizer`
3. Install required dependencies `pip install -r requirements.txt`
4. Run the program: `python3 text_summarizer.py`