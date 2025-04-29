# 🎓 Youniversity

Learn from YouTube videos with AI-powered conversations. Youniversity is a Streamlit application that allows you to:

1. Enter a YouTube URL and get the transcript
2. Chat with an AI about the video content
3. Receive answers with timestamped links to the relevant parts of the video

<img src="screenshots/youniversity.png" alt="This is a screenshot of the youtube video being loaded and transcribed." width="900">

*This is a screenshot of the youtube video being loaded and transcribed.*

<img src="screenshots/chat.png" alt="This is a screenshot of the chat interface." width="900">

*This is a screenshot of the chat interface.*

## Features

- **Transcript Extraction**: Get transcripts from YouTube videos automatically
- **Audio Transcription**: Generate transcripts from audio if no subtitle is available
- **AI Conversation**: Ask questions about the video content and get contextual answers
- **Timestamped References**: AI responses include links to the relevant parts of the video
- **Multiple LLM Providers**: Support for Ollama, OpenAI, and Anthropic LLMs
- **Custom Prompts**: Create, edit, and manage prompt templates in the browser

## Installation

### Prerequisites

- Python 3.8+
- [Ollama](https://ollama.ai/) (optional, for local LLM support)
- API keys for OpenAI or Anthropic (optional, for cloud LLM support)

### Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/youniversity.git
   cd youniversity
   ```

2. Create and activate a virtual environment:

   **Using venv:**

   **For Windows (PowerShell):**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   **For Windows (Command Prompt):**
   ```cmd
   python -m venv venv
   venv\Scripts\activate.bat
   ```

   **For macOS/Linux:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

   **Using uv:**
   ```bash
   # Install uv if you haven't already
   pip install uv
   
   # Create virtual environment with uv
   uv venv
   
   # For Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1
   
   # For macOS/Linux
   source .venv/bin/activate
   ```

3. Install dependencies:

   **Using pip:**
   ```bash
   pip install -r requirements.txt
   ```

   **Using uv:**
   ```bash
   uv pip install -r requirements.txt
   ```

4. Create configuration file:
   ```bash
   cp .env-template .env
   ```

5. Edit the `.env` file with your preferred settings and API keys.

## Running the Application

1. Make sure your virtual environment is activated.

2. If you plan to use Ollama, ensure it's running:
   ```bash
   ollama serve
   ```

3. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at http://localhost:8501

## Usage

### Getting a Transcript

1. In the "Video Transcript" tab, paste a YouTube URL in the input field.
2. Click "Fetch Transcript" to retrieve the video's transcript.
3. If no transcript is available, the application will generate one from the audio (this may take a while depending on the video length and your computer's capabilities).

### Chatting About the Video

1. Navigate to the "Chat" tab after fetching a transcript.
2. Type your question about the video content in the input field.
3. Click "Send" to submit your question.
4. The AI will respond with information from the video, including timestamped links to relevant sections.

### Managing Prompts

1. Use the sidebar to select, edit, or create prompt templates.
2. Prompt templates control how the AI interprets and responds to your questions.
3. Click "Edit Prompt Template" to modify the currently selected template.
4. Click "Create New Prompt" to create a new template.

### Selecting LLM Providers

1. Use the sidebar to select your preferred LLM provider.
2. Choose a model from the available options for the selected provider.
3. The default provider is Ollama with the llama3 model (if available).

## Configuration

The application can be configured by editing the `.env` file:

- `WHISPER_MODEL_SIZE`: Size of the Whisper model for audio transcription (tiny, base, small, medium, large).
- `TRANSCRIPTION_LANGUAGES`: Comma-separated list of language codes to try when fetching transcripts.
- `DEFAULT_PROVIDER`: Default LLM provider to use (ollama, openai, anthropic).
- `OLLAMA_API_URL`: URL for the Ollama API.
- `OPENAI_API_KEY`: Your OpenAI API key.
- `ANTHROPIC_API_KEY`: Your Anthropic API key.
- `PROMPTS_DIR`: Directory for prompt templates.
- `MODEL_TEMPERATURE`: Temperature parameter for LLM generation.
- `MODEL_MAX_TOKENS`: Maximum tokens to generate in responses.

## Troubleshooting

### Common Issues:

1. **No transcripts available**: Some YouTube videos don't have available transcripts. The application will attempt to generate one from the audio.

2. **Ollama not found**: Ensure Ollama is installed and running. The default API endpoint is http://localhost:11434.

3. **Slow transcription**: Generating transcripts from audio can be resource-intensive. Consider reducing the Whisper model size in the `.env` file.

4. **API key errors**: Check that your OpenAI or Anthropic API keys are correctly set in the `.env` file.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.