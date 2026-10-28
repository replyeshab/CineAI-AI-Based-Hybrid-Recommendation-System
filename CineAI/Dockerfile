# ---- web: build the SPA, serve with nginx, proxy /api to the backend (same origin => no CORS, SameSite=Strict cookies)
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json ./
COPY bun.lock* package-lock.json* ./
RUN npm install --no-audit --no-fund
COPY . .
ARG VITE_GOOGLE_CLIENT_ID=
ENV VITE_CINEAI_API_URL=/api VITE_GOOGLE_CLIENT_ID=$VITE_GOOGLE_CLIENT_ID
RUN npx vite build
FROM nginxinc/nginx-unprivileged:1.27-alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 8080
