import os
import warnings
warnings.filterwarnings('ignore')

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter


class GeminiChatbot:
    """Simple chatbot with optional document context (RAG)."""
    
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.7)
        self.chat_history = []
        self.vectordb = None
    
    def load_document(self, file_path: str):
        """Load and index a document for RAG."""
        print(f"📄 Loading {file_path}...")
        
        # Load and split document
        docs = TextLoader(file_path, encoding='utf-8').load()
        chunks = CharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
        print(f"✓ Split into {len(chunks)} chunks")
        
        # Create embeddings and vector database
        print("🔍 Creating search index...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vectordb = FAISS.from_documents(chunks, embeddings)
        print("✓ Ready!\n")
    
    def chat(self, user_input: str) -> str:
        """Generate response with optional document context."""
        # Get document context if available
        context = ""
        if self.vectordb:
            docs = self.vectordb.similarity_search(user_input, k=2)
            context = "\n".join([doc.page_content for doc in docs])
        
        # Build prompt
        system_msg = "You are a helpful AI assistant."
        if context:
            system_msg += f"\n\nRelevant context:\n{context}"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        
        # Generate response
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"history": self.chat_history, "input": user_input})
        
        # Update history
        self.chat_history.append(HumanMessage(content=user_input))
        self.chat_history.append(AIMessage(content=response))
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
        
        return response
    
    def clear_history(self):
        """Clear conversation history."""
        self.chat_history = []
        print("✓ History cleared!")
    
    def run(self):
        """Start interactive chat loop."""
        mode = "RAG-Enhanced" if self.vectordb else "Simple"
        print("=" * 60)
        print(f"🤖 {mode} Chat")
        print("=" * 60)
        print("Commands: 'quit' to exit, 'clear' to reset history")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if not user_input:
                    continue
                if user_input.lower() in ['quit', 'exit']:
                    print("\n👋 Goodbye!")
                    break
                if user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                
                response = self.chat(user_input)
                print(f"\n🤖 AI: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")


def main():
    # Check API key
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Set it with: $env:GOOGLE_API_KEY='your-key'")
    
    # Initialize chatbot
    print("\n🚀 Initializing Gemini chatbot...\n")
    bot = GeminiChatbot(api_key)
    
    # Ask user for mode
    print("Select mode:")
    print("1. Chat with document (RAG)")
    print("2. Simple chat")
    choice = input("\nChoice (1/2): ").strip()
    
    # Load document if RAG mode
    if choice == "1":
        doc_path = input("Enter document path (default: notes.txt): ").strip() or "notes.txt"
        if os.path.exists(doc_path):
            bot.load_document(doc_path)
        else:
            print(f"❌ {doc_path} not found! Continuing without document...")
    
    # Start chat
    bot.run()


if __name__ == "__main__":
    main()