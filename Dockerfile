FROM node:20-bookworm-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip && rm -rf /var/lib/apt/lists/*

COPY package.json package-lock.json* ./
RUN npm install --omit=dev

COPY ml/requirements.txt ./ml/requirements.txt
RUN python3 -m pip install --no-cache-dir -r ml/requirements.txt

COPY server ./server
COPY ml ./ml

ENV NODE_ENV=production
ENV PORT=8787

EXPOSE 8787

CMD ["npm", "start"]
