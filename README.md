# 🛂 Biometric Check-In System

A modern, full-stack biometric face recognition and registration system for check-in, built with Python, SQL, Docker, gRPC, and a responsive web interface.

---

## 📑 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Database Setup (Docker)](#2-database-setup-docker)
  - [3. Backend Setup](#3-backend-setup)
  - [4. gRPC Protobuf Compilation](#4-grpc-protobuf-compilation)
  - [5. Edge/Web App Setup](#5-edgeweb-app-setup)
  - [6. Running the System](#6-running-the-system)
- [Usage Guide](#usage-guide)
  - [Face Recognition](#face-recognition)
  - [Register New Person](#register-new-person)
- [Environment Variables](#environment-variables)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [FAQ](#faq)
- [License](#license)
- [Contributing](#contributing)
- [Contact](#contact)

---

## ✨ Features

- **Real-time Face Recognition**: Instantly identify registered individuals from a live camera feed.
- **Biometric Registration**: Register new people with multi-pose support for robust recognition.
- **gRPC Microservice**: Fast, scalable communication between web app and backend.
- **Vector Database**: Uses PostgreSQL with pgvector for efficient similarity search of face embeddings.
- **Modern Web UI**: Responsive, user-friendly interface with live feedback, progress bars, and modals.
- **Dockerized Database**: Easy setup and portability for development and deployment.
- **Multi-Image Embedding**: Supports multiple face embeddings per person for improved accuracy.
- **Security**: Unique passport number enforcement(optional, not default) and safe database operations.
- **Extensible**: Easily add new features, models, or database fields.

---

## 🛠 Tech Stack

- **Python**: Flask, SQLAlchemy, gRPC, OpenCV, InsightFace, Facenet-PyTorch
- **PostgreSQL**: With [pgvector](https://github.com/pgvector/pgvector) extension for vector similarity search
- **Docker**: For database containerization
- **gRPC**: Protocol Buffers for service definitions and communication
- **HTML/CSS/JavaScript**: For the web interface
- **MTCNN**: For face detection on the edge/web app

---

## 🏗 Architecture

```
[Web App (Flask)] <-> [gRPC Client] <-> [gRPC Server (Face Recognition)] <-> [PostgreSQL + pgvector (Docker)]
```

- **Web App**: User interface for recognition and registration
- **gRPC Client**: Sends images and registration data to backend
- **gRPC Server**: Handles recognition, registration, and embedding extraction
- **Database**: Stores people and face embeddings, optimized for similarity search

### Data Flow

1. **Recognition**: Camera feed → Face detection (MTCNN) → gRPC call → Embedding extraction → Database similarity search → Result display
2. **Registration**: Form input + 5 photos → gRPC calls → Embedding extraction → Database insertions → Confirmation

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/ege-eker/BiometricCheckIn.git
cd BiometricCheckIn
```

### 2. Database Setup (Docker)

> **Requires Docker installed.**

- Edit `db/.env` with your desired credentials (default is fine for local use).
- Start the database:

```bash
cd db
docker-compose up -d
```

- This will:
  - Start PostgreSQL with pgvector extension
  - Run `init.sql` to create tables and indexes

### 3. Backend Setup

- Install Python dependencies:

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

- Edit `backend/.env` if needed (database credentials, model settings).

### 4. gRPC Protobuf Compilation

> **Required for both backend and edge.**

- Compile the proto file for Python:

```bash
python -m grpc_tools.protoc -I=proto --python_out=. --grpc_python_out=. proto/facerecognizer.proto
```

- Do this in both `backend` and `edge` folders if needed.

### 5. Edge/Web App Setup

- Install Python dependencies:

```bash
cd edge
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

- Edit `edge/.env` if needed.

### 6. Running the System

- **Start the backend server:**

```bash
cd backend
venv\Scripts\activate
python server.py
```

- **Start the web app:**

```bash
cd edge
venv\Scripts\activate
python web_app.py
```

- **Access the web interface:**
  - Open [http://localhost:5000](http://localhost:5000) in your browser.

---

## 📖 Usage Guide

### Face Recognition

1. Go to the **Recognize** tab.
2. Ensure your camera is active.
3. Click **Recognize Face**.
4. View results (name, age, nationality, passport, flight, similarity).

### Register New Person

1. Go to the **Register New Person** tab.
2. Fill in the form (name, surname, age, nationality, passport, flight).
3. Click **Start Registration**.
4. Take 5 photos with different poses.
5. Confirm registration in the modal.
6. View success message and person ID.

### Advanced Registration

- The system supports multi-image registration for robust recognition.
- Each photo is processed and stored as a separate embedding.
- You can review and confirm all images before final submission.

---

## ⚙️ Environment Variables

- `backend/.env` and `db/.env` control database and model settings.
- Example:

```
POSTGRES_DB=mydb
POSTGRES_USER=myuser
POSTGRES_PASSWORD=mypassword
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
MIN_SIMILARITY=0.75
DET_SIZE_W=640
DET_SIZE_H=640
model_name=buffalo_l
```

- Adjust `MIN_SIMILARITY` for recognition strictness.
- Change model parameters for different face recognition models.

---

## 📁 Project Structure

```
BiometricCheckIn/
├── backend/
│   ├── server.py
│   ├── db.py
│   ├── embedding_model.py
│   ├── proto/
│   │   ├── facerecognizer.proto
│   ├── requirements.txt
│   └── .env
├── edge/
│   ├── web_app.py
│   ├── client.py
│   ├── utils.py
│   ├── templates/
│   └── requirements.txt
├── db/
│   ├── docker-compose.yml
│   ├── init.sql
│   └── .env
```

- **backend/**: gRPC server, database logic, face embedding extraction
- **edge/**: Web app, gRPC client, camera and UI logic
- **db/**: Database setup scripts and Docker configuration

---

## 🛠 Troubleshooting

- **Database connection errors**: Check `.env` files and Docker status.
- **gRPC errors**: Ensure proto files are compiled and server is running.
- **Face not detected**: Ensure good lighting and camera quality.
- **Dependencies**: Use provided `requirements.txt` and Python 3.8+.
- **Camera issues**: Make sure no other application is using the camera.
- **Docker issues**: Restart containers and check logs for errors.

---

## ❓ FAQ

**Q: Can I use a different face recognition model?**  
A: Yes, update the model in `embedding_model.py` and adjust environment variables.

**Q: How do I add more fields to the registration form?**  
A: Update the HTML form, protobuf definitions, and database schema accordingly.

**Q: Is this system production-ready?**  
A: It is designed for prototyping and research. For production, add authentication, HTTPS, and further security.

**Q: Can I deploy this on the cloud?**  
A: Yes, Docker and Python make it portable. Adjust host and port settings as needed.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

---

## 📄 License

MIT License. See `LICENSE` for details.

---

## 📬 Contact

For questions or support, open an issue or contact [ege-eker](https://github.com/ege-eker).

---

> _Made with ❤️ by [ege-eker](https://github.com/ege-eker)_
