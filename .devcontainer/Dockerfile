# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>

FROM python:latest

ENV PYTHONIOENCODING=UTF-8
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update
RUN apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists


# Install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt

CMD ["/bin/bash"]

