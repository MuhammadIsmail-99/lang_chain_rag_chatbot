# 🤖 LangChain + Gemini Chatbot

An interactive chatbot powered by Google's Gemini 2.5 Flash with optional Retrieval-Augmented Generation (RAG) for document-based conversations.

## ✨ Features

- **Two Chat Modes:**
  - 💬 **Simple Chat** - Direct conversation with Gemini AI
  - 📚 **RAG Chat** - Chat with your documents (answer questions based on uploaded files)

- **Smart Context Management:**
  - Maintains conversation history (last 10 messages)
  - Automatic document chunking and indexing
  - Fast semantic search using FAISS

- **Modern Stack:**
  - LangChain Expression Language (LCEL)
  - HuggingFace embeddings (free, local)
  - FAISS vector store (fast, reliable)

## 📋 Prerequisites

- Python 3.8+ (tested on Python 3.13)
- Google API Key ([Get one here](https://aistudio.google.com/app/apikey))

## 🚀 Quick Start

### 1. Clone & Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd langChainProject

# Create virtual environment
python -m venv langchain_env

# Activate virtual environment
# Windows PowerShell:
langchain_env\Scripts\Activate.ps1
# Windows CMD:
langchain_env\Scripts\activate.bat
# Linux/Mac:
source langchain_env/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Key

**Windows PowerShell:**
```powershell
$env:GOOGLE_API_KEY = "your-api-key-here"
```

**Windows CMD:**
```cmd
set GOOGLE_API_KEY=your-api-key-here
```

**Linux/Mac:**
```bash
export GOOGLE_API_KEY="your-api-key-here"
```

### 4. Run the Chatbot

```bash
python gemini_demo.py
```

## 📖 Usage

### Simple Chat Mode
```
Select mode:
1. Chat with document (RAG)
2. Simple chat

Choice (1/2): 2

👤 You: What is artificial intelligence?
🤖 AI: Artificial intelligence (AI) is...
```

### RAG Chat Mode
```
Choice (1/2): 1
Enter document path (default: notes.txt): notes.txt

📄 Loading notes.txt...
✓ Split into 5 chunks
🔍 Creating search index...
✓ Ready!

👤 You: What does the document say about LangChain?
🤖 AI: According to the document, LangChain is...
```

## 🎮 Commands

| Command | Description |
|---------|-------------|
| `quit` or `exit` | End the chat session |
| `clear` | Reset conversation history |
| `Ctrl+C` | Force quit |

## 📁 Project Structure

```
langChainProject/
│
├── gemini_demo.py      # Main chatbot script
├── notes.txt           # Sample document for RAG
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔧 Configuration

Edit `gemini_demo.py` to customize:

```python
# Change model
bot = GeminiChatbot(api_key, model="gemini-pro")

# Adjust temperature (0.0 = deterministic, 1.0 = creative)
self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)

# Change chunk size
chunks = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Retrieve more context
docs = self.vectordb.similarity_search(user_input, k=5)
```

## 📦 Dependencies

- `langchain` - LLM application framework
- `langchain-google-genai` - Gemini integration
- `langchain-community` - Community integrations
- `langchain-huggingface` - HuggingFace embeddings
- `faiss-cpu` - Vector similarity search
- `sentence-transformers` - Text embeddings

See `requirements.txt` for complete list.

## 🐛 Troubleshooting

### API Key Not Found
```
ValueError: GOOGLE_API_KEY not found
```
**Solution:** Set the environment variable before running

### Import Errors
```
ModuleNotFoundError: No module named 'faiss'
```
**Solution:** 
```bash
pip install faiss-cpu
```

### Deprecation Warnings
Warnings are automatically suppressed. To see them:
```python
# Comment out this line in gemini_demo.py
# warnings.filterwarnings('ignore')
```

### Python 3.13 Compatibility
This code is tested on Python 3.13. If you encounter issues:
- Try Python 3.10 or 3.11
- Update packages: `pip install --upgrade -r requirements.txt`

## 🎯 How RAG Works

```
User Question
    ↓
1. Convert to embeddings (vector)
    ↓
2. Search document for similar vectors
    ↓
3. Retrieve relevant text chunks
    ↓
4. Combine: Context + Question → Gemini
    ↓
5. AI answers based on YOUR document
```

## 📚 Learn More

- [LangChain Documentation](https://python.langchain.com/)
- [Gemini API Docs](https://ai.google.dev/docs)
- [FAISS Documentation](https://faiss.ai/)

## 🤝 Contributing

Feel free to:
- Report bugs
- Suggest features
- Submit pull requests

## 📄 License

MIT License - feel free to use this code for your projects!

## 🙏 Acknowledgments

- Google Gemini for the LLM
- LangChain for the framework
- HuggingFace for embeddings
- Facebook Research for FAISS

---

**Made with ❤️ using LangChain and Gemini**
