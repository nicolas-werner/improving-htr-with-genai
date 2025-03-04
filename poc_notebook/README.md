# Multimodal Medieval Handwriting Transcription

This folder contains Marimo notebooks for transcribing handwritten medieval texts using multimodal large language models.

## Available Notebooks

### 1. lot-books.py
- Original proof of concept using OpenAI's GPT-4o model
- Supports zero-shot, one-shot, and HTR improvement transcription

### 2. multilingual-transcription.py
- Enhanced version with support for both OpenAI and Google Gemini models
- Allows switching between model providers
- Supports zero-shot, one-shot, and HTR improvement transcription

## Getting Started

### Prerequisites
- Python 3.12 or higher
- Required packages (can be installed via `pip install -r requirements.txt` from the root directory)
- API keys for OpenAI and/or Google Gemini

### Running the Notebooks

1. Clone the repository
2. Navigate to the poc_notebook directory
3. Run the notebook of your choice:

```bash
python -m marimo run multilingual-transcription.py
```

4. Open the provided URL in your browser (typically http://localhost:2719)

## Using the Multilingual Transcription Notebook

The new multilingual transcription notebook (multilingual-transcription.py) offers these features:

### Model Selection
- Choose between OpenAI's GPT-4o and Google Gemini 1.5 Pro models
- Enter your API key for the selected provider

### Transcription Modes

#### Zero-Shot Transcription
- Upload an image of handwritten text
- The model will transcribe it without prior examples

#### One-Shot Transcription
- Upload an example image with its correct transcription
- Upload a target image to transcribe
- The model uses the example to improve transcription accuracy

#### HTR Improvement
- Upload an image and the output from a classical HTR model
- The multimodal model will correct errors in the HTR output

## Comparing Models

This notebook allows you to easily compare the transcription quality between OpenAI and Google Gemini models for medieval handwritten texts. You can:

1. Run the same transcription with both models
2. Compare accuracy and error patterns
3. Evaluate which model works better for your specific handwriting samples

## Troubleshooting

If you encounter any issues:

- Make sure your API keys are valid
- Check that your image is in a supported format (.png, .jpg, .jpeg)
- Try resizing large images before uploading
- For model-specific errors, refer to the appropriate API documentation 