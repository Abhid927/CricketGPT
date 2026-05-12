#🏏 CricketGPT — AI Cricket Commentary + Match Insights Engine

CricketGPT is a locally deployed AI-powered chatbot that combines a custom-trained transformer model with a deterministic match statistics engine to generate realistic cricket commentary and answer factual match queries with high accuracy.

It is trained on ball-by-ball data from Cricsheet and supports both natural language generation and structured data retrieval.

🚀 Features
🧠 AI Commentary Generation
Generates realistic ball-by-ball cricket commentary
Supports prompts like:
"Continue this over: Over 16.3 Bumrah to Maxwell..."
"Summarize the last 3 overs in a hype style"
Built using a TinyGPT-style transformer (~11M parameters)
📊 Deterministic Match Stats Engine
Answers factual queries with high accuracy (no hallucination)
Examples:
"Score for the most recent Pakistan vs New Zealand match"
"Who top scored in the latest India vs Australia game?"
Uses structured data parsed from Cricsheet JSON files
🔍 Hybrid AI + Search System
Automatically routes queries:
Factual queries → Stats engine
Open-ended queries → AI model
Ensures:
Accurate statistics
Natural language responses
💻 Interactive Web Interface
Chat-based UI with:
Temperature and Top-K controls
Prompt suggestions
Chat history
Built with FastAPI backend and a lightweight frontend
🧱 Project Structure
Cricket-gpt/
│
├── app/
│   ├── server.py          # FastAPI backend
│   ├── stats_engine.py    # Match query system
│   └── static/
│       └── index.html     # Chat UI
│
├── model/
│   ├── train.py           # Transformer training loop
│   ├── dataset.py         # Dataset loader
│   └── generate.py        # Standalone generation
│
├── tokenizer/
│   ├── tokenizer.py
│   └── train_tokenizer.py
│
├── data/
│   ├── raw/               # Cricsheet JSON files
│   └── processed/         # train.txt, val.txt, match_index.json
│
├── checkpoints/           # Model checkpoints
│
└── preprocess_cricsheet.py
⚙️ Setup & Run
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/Cricket-gpt.git
cd Cricket-gpt
2. Install dependencies
pip install torch fastapi uvicorn sentencepiece
3. Run the backend server
$env:CKPT="checkpoints/ckpt_step9000.pt"
python -m uvicorn app.server:app --reload --port 8000

Open in browser:

http://127.0.0.1:8000
🧪 Example Prompts
📊 Factual Queries
Score for the most recent Pak vs NZ match
Who won the latest India vs Australia match?
Top scorer in the oldest England vs South Africa match
🎙️ Commentary Queries
Continue this over: Over 16.3 Bumrah to Maxwell,
Summarize the last 3 overs in a hype commentator style
Explain why this wicket was important: Over 14.2 Rashid Khan to Buttler
🧠 Model Details
Architecture: Decoder-only Transformer (TinyGPT)
Parameters: ~11.5M
Tokenizer: SentencePiece BPE (8,000 vocab)
Training Data: Ball-by-ball T20 data derived from Cricsheet
Training: Up to 10,000 steps using cross-entropy loss
📈 Key Design Decisions
Hybrid System (Core Idea)
Task Type	Approach
Match facts	Deterministic engine
Commentary	Transformer model

This avoids hallucination while keeping responses natural.

Context Handling
Input truncated to model block size (256 tokens)
Prevents overflow during inference
Team Name Normalization
Supports common aliases:
Pak → Pakistan
NZ → New Zealand
Uses regex-based extraction for robust query parsing
🚧 Future Improvements
Support ODI and Test formats
Add semantic search (embeddings)
Fine-tune model with chat-style data
Deploy with GPU inference
Integrate live match APIs
🙌 Acknowledgements
Cricsheet for open cricket datasets
PyTorch for model development
SentencePiece for tokenization
