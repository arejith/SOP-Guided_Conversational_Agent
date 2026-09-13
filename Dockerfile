FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent ./agent
COPY services ./services
COPY apps ./apps
COPY app.py .
RUN useradd --create-home demo
USER demo
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=5)" || exit 1
CMD ["python", "-m", "streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.headless=true", "--browser.gatherUsageStats=false"]
