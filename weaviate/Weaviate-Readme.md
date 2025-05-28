# Weaviate Setup Guide for Windows 11

This guide walks through setting up Weaviate vector database on Windows 11 using Docker Desktop.

## Prerequisites

- Windows 11
- Administrator access
- Internet connection
- Python version 3.11.9

## Installation Steps

### 1. Install Docker Desktop

1. Download Docker Desktop from [Docker's official website](https://www.docker.com/products/docker-desktop/)
2. Run the installer
3. Restart your computer when prompted

### 2. Enable WSL 2 (Windows Subsystem for Linux)

Open PowerShell as Administrator and run:
```powershell
wsl --install
```
Restart your computer after installation.

### 3. Verify Docker Installation

Open Command Prompt or PowerShell and run:
```powershell
docker --version
docker-compose --version
```

### 4. Create Project Directory

```powershell
mkdir weaviate-project
cd weaviate-project
```

### 5. Create docker-compose.yml

Create a new file named `docker-compose.yml` and add the following content:

```yaml
version: '3.4'
services:
  weaviate:
    image: semitechnologies/weaviate:1.21.5
    ports:
      - "8080:8080"
    restart: on-failure:0
    environment:
      QUERY_DEFAULTS_LIMIT: 25
      AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED: 'true'
      PERSISTENCE_DATA_PATH: '/var/lib/weaviate'
      DEFAULT_VECTORIZER_MODULE: 'text2vec-transformers'
      ENABLE_MODULES: 'text2vec-transformers'
      TRANSFORMERS_INFERENCE_API: 'http://t2v-transformer:8080'
    volumes:
      - weaviate_data:/var/lib/weaviate
  
  t2v-transformer:
    image: semitechnologies/transformers-inference:sentence-transformers-multi-qa-MiniLM-L6-cos-v1
    environment:
      ENABLE_CUDA: '0'

volumes:
  weaviate_data:
```

### 6. Start Weaviate

```powershell
docker-compose up -d
```

### 7. Install Python Dependencies

```powershell
pip install weaviate-client
```

### 8. Create Test Script

Create a file named `test.py`:


### Frequently used commands
- for restarting docker - go to folder where docker compose is there
- make sure that wsl for windows is installed and docker is running
- ```docker-compose down```
- ```docker-compose up -d``` for restarting