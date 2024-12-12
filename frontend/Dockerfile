# Step 1: Build your Next.js application
FROM node:20.1.0-alpine as builder
WORKDIR /app

# Install Python and build dependencies
RUN apk add --no-cache python3 make g++

# First copy package files and install dependencies
COPY package*.json ./
RUN npm ci  # Using ci instead of install for more reliable builds

# Copy the rest of the application code
COPY . .

# Clear any existing .next directory and build
RUN rm -rf .next && \
    npm run build && \
    ls -la .next  # Debug step to verify build output

# Step 2: Serve your Next.js application
FROM node:20-alpine
WORKDIR /app

# Set environment variable
ENV NODE_ENV=production

# Install necessary tools
USER root
RUN apk --no-cache add curl wget python3 make g++

# Copy ALL necessary files
COPY --from=builder /app/package*.json ./
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/next.config.mjs ./
COPY --from=builder /app/server.js ./

# Debug: Verify .next directory contents
RUN ls -la .next && \
    ls -la .next/static || true

# Ensure proper permissions
RUN mkdir -p /app/.next/cache && \
    chmod -R 777 /app/.next

# Create and switch to non-root user
RUN adduser -D appuser && chown -R appuser:appuser /app
USER appuser

# Expose the port the app runs on
EXPOSE 3000

# Start the application using npm scripts
CMD ["npm", "start"]