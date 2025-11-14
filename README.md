# Examen BentoML

## Set up venv

### 1. Install pytest venv dependencis

Extract the zip file and open the extracted file in terminal or code editor. The 
run the following commands which setup the venv.
```bash
uv init
uv venv
uv pip install -r requirements.txt
```
--------

### 2. Activate venv

Windows 
```bash
source .venv/Scripts/activate
```

Linux
```bash
source .venv/bin/activate
```

### 3. Extract and run image -> Start API port 3000
```bash
docker load -i stud_adm_service.tar
docker image ls 
docker run --rm -p 3000:3000 stud_adm_service:5wg46n6bis6rvlv4 
```
--------

### 4. Extract and run image -> Start API port 3000

Open second terminal and activate venv.
```bash
pytest -v
```
--------
