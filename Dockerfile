FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN python -m compileall -q hybrid_rag app.py
ENTRYPOINT ["python", "app.py"]
CMD ["--help"]
