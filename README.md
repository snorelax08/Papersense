<div align="center">
  
# 📄 PaperSense: AI-Powered Semantic PDF Search Engine

![License](https://img.shields.io/github/license/snorelax08/PaperSense?style=for-the-badge)
![GitHub top language](https://img.shields.io/github/languages/top/snorelax08/PaperSense?style=for-the-badge)
![Backend](https://img.shields.io/badge/Backend-FastAPI-009688?style=for-the-badge&logo=fastapi)
![Frontend](https://img.shields.io/badge/Frontend-React-61DAFB?style=for-the-badge&logo=react)
![Semantic Search](https://img.shields.io/badge/Search-SentenceTransformers-blue?style=for-the-badge&logo=pytorch)

<br>
<p align="center">
  PaperSense is an intelligent, local-first search engine built for students, researchers, and professionals who need Google-like search capabilities within their private document collection.
  <br>
  It allows users to upload PDFs and instantly search through them using a powerful **AI-driven hybrid search** model.
</p>

</div>

---

## 💡 About the Project

In today's knowledge-heavy world, finding specific information buried deep within local PDF libraries is a challenge. PaperSense solves this by combining the power of modern **SentenceTransformer embeddings** with traditional **TF-IDF keyword ranking** to deliver contextually accurate search results, even for vague or conceptual queries.

### 🎥 Demo

*Replace this markdown with an embedded GIF or a link to a video demonstration.*

****

---

## 🌟 Key Features

| Feature | Description | Tech/Benefit |
| :--- | :--- | :--- |
| **🔍 AI Hybrid Search** | Combines **SentenceTransformer embeddings** (`all-MiniLM-L6-v2`) and **TF-IDF keyword scoring** for maximum relevance. | Hybrid weighted ranking (**0.6 semantic + 0.4 keyword**) generates highly accurate results. |
| **📦 Instant PDF Uploads** | Simply drag & drop PDFs onto the UI. | Files are **auto-indexed on the Python backend** and available for semantic search immediately. |
| **🖼️ Live PDF Thumbnails** | Each search result includes a high-quality thumbnail preview of the relevant page. | Uses **PyMuPDF (fitz)** for fast rendering and a JPEG stream for instant UI preview. |
| **✨ Modern UI** | A clean, intuitive design supporting both **dark and light modes**. | Features **floating orbits**, **glass morphism**, smooth transitions, and real-time score badges. |
| **💾 Local-First & Private** | The entire system runs locally on your machine. | **PDFs never leave your computer**, ensuring privacy and full **offline capability** (after the initial model download). |
| **⚙️ Productivity Tools** | Includes search history, a clear indexed PDF list, and semantic **Score Badges** (High/Medium/Low). | Built for knowledge management and efficient document understanding. |

---

## 💻 Tech Stack

### Backend (Python) 🐍

PaperSense leverages a powerful Python stack for data processing, indexing, and serving:

* **Framework:** [FastAPI](https://fastapi.tiangolo.com/) (Async, high-performance)
* **Vector/Indexing:** [SentenceTransformers](https://www.sbert.net/) (for Semantic Embeddings)
* **Ranking:** [scikit-learn](https://scikit-learn.org/stable/) (for TF-IDF scoring and hybrid ranking)
* **PDF Parsing:** [PyPDF2](https://pypi.org/project/PyPDF2/)
* **Thumbnail Generation:** [PyMuPDF (fitz)](https://pypi.org/project/PyMuPDF/)
* **Server:** [Uvicorn](https://www.uvicorn.org/)

### Frontend (React) ⚛️

The user-facing interface is built for speed and aesthetics:

* **Framework:** [React](https://react.dev/) (via [Vite](https://vitejs.dev/))
* **Styling:** Custom CSS (Optimized for performance and includes glass morphism effects)
* **Communication:** Native Fetch API
* **Design:** Fully **Responsive Layout**

---

🚀 Getting Started (Local Setup)2. Backend Setup 🐍The backend handles all processing, indexing, and the API logic.Bashcd papersense-backend
pip install -r requirements.txt
python -m uvicorn api_main:app --reload --port 8000
The backend will run at: http://127.0.0.1:80003. Frontend Setup ⚛️The frontend is the modern UI that interacts with the backend API.Bashcd papersense-frontend
npm install
npm run dev
The frontend will run at: http://127.0.0.1:5173📁 Folder StructurePaperSense/
│ 
├── papersense-backend/
│   ├── api_main.py         # Main FastAPI application and logic
│   ├── pdfs/               # Directory where your PDFs are stored/uploaded
│   ├── requirements.txt
│   └── ...
│ 
└── papersense-frontend/
    ├── src/
    │   ├── index.css       # Custom, enhanced styling
    │   ├── App.jsx         # Main React component
    │   └── ...
🔗 API EndpointsRouteMethodDescription/searchPOSTRuns the hybrid semantic + keyword search./uploadPOSTUploads and automatically indexes a new PDF./filesGETReturns a list of all currently indexed PDFs./historyGETRetrieves the recent search history./thumbnail/{file}GETRenders and returns a PDF thumbnail image stream./reloadPOSTForces the backend to rebuild the search indexes./healthGETBasic API health check.🧑‍💻 Developer & LicenseDeveloper: Atharwa Vatsyayan (snorelax08)Description: Created for personal productivity, knowledge management, and document understanding.License: This project is licensed under the MIT License — free to use, modify, and distribute. See the LICENSE file for details.Action: Please replace the content of your README.md file with the code above. This should immediately render correctly on GitHub.
