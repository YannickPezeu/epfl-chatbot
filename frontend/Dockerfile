# Step 1: Build your Next.js application
FROM node:20.1.0-alpine as builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build



# Step 2: Serve your Next.js application
FROM node:20-alpine
WORKDIR /app

# Set environment variable
ENV NODE_ENV=production

# Install curl and wget
USER root
RUN apk --no-cache add curl wget

# Copy necessary files
COPY --from=builder /app/next.config.mjs ./
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/server.js ./

# Run as non-root user
RUN adduser -D appuser
USER appuser

# Expose the port the app runs on
EXPOSE 3000

# Start the application
CMD ["node", "server.js"]
# CMD ["npm", "start"]
