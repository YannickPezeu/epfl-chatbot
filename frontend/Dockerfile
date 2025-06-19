# Step 1: Build your Next.js application
FROM node:20.1.0-alpine AS builder
WORKDIR /app

# Install Python and build dependencies
RUN apk add --no-cache python3 make g++

# First copy package files and install dependencies
COPY package*.json ./
RUN npm ci

# Copy the rest of the application code
COPY . .

# Debug: Check what we have
RUN ls -la && echo "Checking for pages/app directories:" && find . -name "pages" -o -name "app" -type d

# Clear any existing .next directory and build
RUN rm -rf .next && \
    npm run build && \
    ls -la .next

# Step 2: Serve your Next.js application
FROM node:20-alpine
WORKDIR /app

# Set environment variable
ENV NODE_ENV=production

# Install necessary tools
USER root
RUN apk --no-cache add curl wget python3 make g++

# Copy package files first
COPY --from=builder /app/package*.json ./

# Copy everything from builder (simpler and more reliable)
COPY --from=builder /app .

# Install only production dependencies
RUN npm ci --only=production

# Debug: Verify structure
RUN echo "Final structure:" && ls -la && \
    echo "Pages/App directories:" && find . -name "pages" -o -name "app" -type d

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