FROM python:3.12-slim

# VTK precisa de OpenGL; sem tela, usa Mesa (software) sobre um X virtual (Xvfb).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    xvfb xauth x11-utils \
    libgl1 libglx-mesa0 libgl1-mesa-dri libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home app
USER app

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8080/', timeout=3)"

CMD ["xvfb-run", "-a", "-s", "-screen 0 1280x1024x24", \
    "temporal-graph", "--host", "0.0.0.0", "--port", "8080", "--server"]
