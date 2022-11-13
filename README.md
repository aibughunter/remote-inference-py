# AIBugHunter Remote Inference Python Script

Python FastAPI/Uvicorn version of AIBugHunter Remote Inference Engine
This script is used in the [AIBugHunter](https://github.com/aibughunter/aibughunter) VSCode Extension to host the on-premise inference server.

## Installation and Deployment

### Install requirements

1. Clone the repository

```bash
git clone https://github.com/aibughunter/remote-inference-py
```

2. `cd` into the cloned repository

```bash
cd remote-inference-py
```

3. Install requirements.txt

```bash
pip install -r requirements.txt
# or for pip3
pip3 install -r requirements.txt
```

4. Install Uvicorn or Gunicorn to run the server script

Install one of them:
```bash
pip install uvicorn
```

```bash
pip install gunicorn
```

### Deploying the server

There are multiple ways you can deploy this server, mainly by using either `uvicorn` or `gunicorn`

The script is built to run the `uvicorn` or `gunicorn` command manually to allow easier configuration of the ports and the host on the fly.

```bash
# Uvicorn deployment
uvicorn deploy:app --reload --host 0.0.0.0 --port 8000
# You can add extra worker processes too
uvicorn deploy:app --reload --host 0.0.0.0 --port 8000 --workers 4
# But Gunicorn is superior for handling worker processes

# Gunicorn deployment
gunicorn deploy:app --workers 8 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```