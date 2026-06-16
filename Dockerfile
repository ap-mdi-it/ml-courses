# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:python3.10-bookworm AS dev

# Create and activate a virtual environment [1].
# [1] https://docs.astral.sh/uv/concepts/projects/config/#project-environment-path
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=$VIRTUAL_ENV/bin:$PATH
ENV UV_PROJECT_ENVIRONMENT=$VIRTUAL_ENV

# Tell Git that the workspace is safe to avoid 'detected dubious ownership in repository' warnings.
RUN git config --system --add safe.directory '*'

# Create a non-root user and give it passwordless sudo access [1].
# [1] https://code.visualstudio.com/remote/advancedcontainers/add-nonroot-user
ARG TARGETARCH
RUN --mount=type=cache,id=apt-cache-$TARGETARCH,target=/var/cache/apt/ \
    --mount=type=cache,id=apt-lists-$TARGETARCH,target=/var/lib/apt/ \
    groupadd --gid 1000 user && \
    useradd --create-home --no-log-init --gid 1000 --uid 1000 --shell /usr/bin/bash user && \
    chown user:user /opt/ && \
    apt-get update && apt-get install --no-install-recommends --yes sudo && \
    echo 'user ALL=(root) NOPASSWD:ALL' > /etc/sudoers.d/user && chmod 0440 /etc/sudoers.d/user
USER user

# Install Node.js
RUN sudo apt-get update && sudo apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_lts.x | sudo -E bash - && \
    sudo apt-get install -y nodejs && \
    sudo npm install -g npm@11.5.2

# Install GraphViz
RUN sudo apt-get update && sudo apt-get install -y graphviz

# Install Playwright dependencies for Chromium
RUN npx --yes playwright@1.57.0 install-deps chromium

# Copy dependency files and install Python packages
COPY --chown=user:user pyproject.toml uv.lock README.md /opt/project/
WORKDIR /opt/project
RUN --mount=type=cache,id=uv-cache-$TARGETARCH,target=/home/user/.cache/uv,uid=1000,gid=1000 \
    uv sync --frozen --all-extras

# Install Playwright browsers (Chromium)
RUN /opt/venv/bin/playwright install chromium

# Configure the non-root user's shell.
RUN mkdir ~/.history/ && \
    echo 'HISTFILE=~/.history/.bash_history' >> ~/.bashrc && \
    echo 'bind "\"\e[A\": history-search-backward"' >> ~/.bashrc && \
    echo 'bind "\"\e[B\": history-search-forward"' >> ~/.bashrc

# Set working directory back to workspace
WORKDIR /workspaces
