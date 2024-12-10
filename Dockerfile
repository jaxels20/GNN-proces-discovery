# Use an official Python 3.12.7 slim image as the base
FROM python:3.12.7-slim

# Install necessary system dependencies for PyTorch, scientific libraries, and Graphviz
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libnss3-dev \
    libgdbm-dev \
    libsqlite3-dev \
    libreadline-dev \
    libffi-dev \
    libbz2-dev \
    liblzma-dev \
    graphviz \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install all Python packages explicitly
RUN pip install --no-cache-dir \
    aiohappyeyeballs==2.4.3 \
    aiohttp==3.11.8 \
    aiosignal==1.3.1 \
    annotated-types==0.7.0 \
    attrs==24.2.0 \
    certifi==2024.8.30 \
    charset-normalizer==3.4.0 \
    colorama==0.4.3 \
    colour==0.1.5 \
    contourpy==1.3.1 \
    cvxopt==1.3.2 \
    cycler==0.12.1 \
    deprecation==2.1.0 \
    filelock==3.16.1 \
    fonttools==4.55.0 \
    frozenlist==1.5.0 \
    fsspec==2024.10.0 \
    graphviz==0.20.3 \
    idna==3.10 \
    intervaltree==3.1.0 \
    Jinja2==3.1.4 \
    kiwisolver==1.4.7 \
    lxml==5.3.0 \
    MarkupSafe==3.0.2 \
    matplotlib==3.9.2 \
    mpmath==1.3.0 \
    multidict==6.1.0 \
    networkx==3.4.2 \
    numpy==2.1.2 \
    packaging==24.2 \
    pandas==2.2.3 \
    pillow==11.0.0 \
    pm4py==2.7.11.13 \
    propcache==0.2.0 \
    psutil==6.1.0 \
    PuLP==2.1 \
    pydantic==2.10.2 \
    pydantic_core==2.27.1 \
    pydotplus==2.0.2 \
    pyparsing==3.2.0 \
    python-dateutil==2.9.0.post0 \
    pytz==2024.2 \
    PyYAML==6.0.2 \
    requests==2.32.3 \
    scipy==1.14.1 \
    setuptools==75.1.0 \
    simpy==4.1.1 \
    six==1.16.0 \
    sortedcontainers==2.4.0 \
    sympy==1.13.3 \
    torch==2.4 \
    torch-geometric==2.6.1 \
    torchdata==0.9.0 \
    tqdm==4.67.1 \
    typing_extensions==4.12.2 \
    tzdata==2024.2 \
    urllib3==2.2.3 \
    wheel==0.44.0 \
    yarl==1.18.0

RUN pip install dgl -f https://data.dgl.ai/wheels/torch-2.4/repo.html

# Set the working directory
WORKDIR /workspace

# Set the default command
CMD ["bash"]