# Dockerfile para Spec2AgentEval-v_esem
FROM python:3.11-slim

# Evita prompts interativos
ENV DEBIAN_FRONTEND=noninteractive

# Instala dependências do sistema + Node.js
RUN apt-get update && apt-get install -y \
    curl \
    unzip \
    git \
    bash \
    ca-certificates \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Instala Node.js 20 (necessário para Gemini CLI)
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && node -v \
    && npm -v

# Instala Gemini CLI via npm (global)
RUN npm install -g @google/gemini-cli

# Instala GitHub Copilot CLI via script oficial
RUN curl -fsSL https://gh.io/copilot-install | bash

# Garante que binários globais estejam no PATH
ENV PATH="/root/.npm-global/bin:/root/.local/bin:${PATH}"

# Copia arquivos do projeto
WORKDIR /app
COPY . .

# Instala dependências Python
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Garante que .env será lido corretamente
ENV PYTHONUNBUFFERED=1

# Comando padrão: executa o pipeline
CMD ["bash", "./pipe.sh"]