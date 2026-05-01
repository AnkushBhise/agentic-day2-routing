# Agentic Day 2 - Routing Assignment

A **production-minded LangGraph workflow** for customer support routing that intelligently directs support tickets based on user tier and issue type.

## 🎯 Overview

This project demonstrates a **typed, auditable support routing system** using LangGraph that:

- **Tracks conversation state** in a typed `SupportState` 
- **Routes users to different support paths** based on customer tier (`vip` vs `standard`)
- **Determines issue type** (billing, shipping, technical, general)
- **Escalates tickets** when needed
- Makes routing logic **explicit, testable, and auditable**

## 📁 Project Structure

```
agentic-day2-routing/
├── app.py                    # Main entry point - run this file
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── LICENSE                   # Project license
├── .env                      # Environment variables (NOT in git)
├── .gitignore               # Git ignore rules
├── prompts/                 # Prompt configurations
│   ├── check_user_tier/
│   │   ├── current.yaml    # Active user tier check prompt
│   │   └── v1.0.0.yaml    # Version 1.0.0
│   ├── determine_issue_type/
│   │   ├── current.yaml    # Active issue type determination prompt
│   │   └── v1.0.0.yaml    # Version 1.0.0
│   ├── standard_agent/
│   │   ├── current.yaml    # Active standard tier agent prompt
│   │   └── v1.0.0.yaml    # Version 1.0.0
│   └── vip_agent/
│       ├── current.yaml    # Active VIP tier agent prompt
│       └── v1.0.0.yaml    # Version 1.0.0
└── utils/                   # Utility modules
    ├── prompt_manager.py    # Prompt loading and management
    └── state_printer.py     # State visualization utilities
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd agentic-day2-routing
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env  # Create from template (if available)
   ```
   
   Create a `.env` file in the project root with:
   ```env
   OPENAI_API_KEY=your_api_key_here
   ```

### Running the Application

```bash
python app.py
```

## 🔧 Configuration

### Environment Variables (.env)

The project uses a `.env` file for sensitive configuration. **This file must be added to `.gitignore`** and should never be committed to version control.

Required variables:
- `OPENAI_API_KEY` - Your OpenAI API key for accessing GPT models

Create `.env` file:
```bash
echo "OPENAI_API_KEY=your_key_here" > .env
```

### .gitignore

Ensure your `.gitignore` includes:
```gitignore
# Environment variables
.env
.env.local
.env.*.local

# Virtual environment
.venv/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/

# OS
.DS_Store
```

## 💡 How It Works

### Support Routing Flow

1. **Extract Customer ID** - Parse customer ID from user input
2. **Check User Tier** - Determine if customer is `vip` or `standard`
3. **Determine Issue Type** - Classify issue (billing, shipping, technical, general)
4. **Route to Agent** - Direct to VIP or standard agent based on tier
5. **Create Ticket** - Generate support ticket with appropriate priority
6. **Resolve or Escalate** - Handle issue or escalate as needed

### Built-in Tools

- `get_customer_tier(customer_id)` - Lookup customer tier
- `create_ticket(customer_id, issue, priority)` - Create support ticket

## 📦 Dependencies

Key dependencies:
- **langchain** - LLM orchestration
- **langchain-openai** - OpenAI integration
- **langgraph** - Graph-based workflow engine
- **python-dotenv** - Environment variable management
- **pydantic** - Data validation

See [requirements.txt](requirements.txt) for complete list.

## 📝 License

See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Pull requests welcome. Please ensure code passes linting and tests before submitting.
