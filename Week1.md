# Week 1 Journal
 Day 3: Dev environment and tooling installed. Ready to begin project coding tomorrow.
 **Day 3  Dev Environment Setup (Agentic AI Journey)**

##  Installed Core Tools
- Python 3.11+ 
- VSCode 
- Poetry (package manager) 

##  Created Project Folder
\\\ash
mkdir agentic-ai-journey
cd agentic-ai-journey
\\\

##  Initialized Poetry + Installed Packages
\\\ash
poetry init --name agentic-ai-journey --python \
^3.11\ --dependency openai --dependency langchain --dependency faiss-cpu -n
\\\

##  Created GitHub Repo + Linked Locally
- Repo: \gentic-ai-journey\ on GitHub
\\\ash
git init
git remote add origin https://github.com/YOUR_USERNAME/agentic-ai-journey.git
\\\

##  Added Journal File
\\\ash
echo \#
Week
1
Journal\ > Week1.md
\\\

##  Committed + Pushed to GitHub
\\\ash
git add .
git commit -m \Initial
setup:
Python
Poetry
dependencies
Week1
journal\
git branch -M main
git push -u origin main
\\\

Environment is now ready for Day 4  time to build the base agent. 

