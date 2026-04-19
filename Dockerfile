# ---- Build stage ----
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY src/ ./src/

# Bundle the Node.js server into a single file using esbuild
RUN npx esbuild src/node-server.ts \
  --bundle \
  --platform=node \
  --target=node20 \
  --format=esm \
  --outfile=dist-node/server.mjs

# ---- Runtime stage ----
FROM node:20-alpine

WORKDIR /app

# Copy only the bundled output — no node_modules needed at runtime
COPY --from=builder /app/dist-node/server.mjs ./server.mjs

EXPOSE 3000

CMD ["node", "server.mjs"]
