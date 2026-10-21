FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /workspace
COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install torch --index-url https://download.pytorch.org/whl/cpu && python -m pip install ".[dev,export]"
COPY configs ./configs
COPY tests ./tests
CMD ["python", "-m", "pytest", "-q"]
