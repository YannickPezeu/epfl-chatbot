const express = require('express');
const next = require('next');
const { createProxyMiddleware } = require('http-proxy-middleware');

const port = parseInt(process.env.PORT, 10) || 3000;
const dev = process.env.NODE_ENV !== 'production';
const app = next({ dev });
const handle = app.getRequestHandler();

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL|| "http://localhost:8000";

app.prepare().then(() => {
  const server = express();

  // Proxy api requests
  server.use(
    '/api',
    createProxyMiddleware({
      target: BACKEND_URL,
      changeOrigin: true,
      pathRewrite: { '^/api': '' },
      ws: true,
      secure: !dev,
      onError: (err, req, res) => {
        console.error('Proxy Error:', err);
        res.status(500).json({ error: 'Proxy Error' });
      }
    })
  );

  server.all('*', (req, res) => handle(req, res));

  server.listen(port, '0.0.0.0', (err) => {
    if (err) throw err;
    console.log(`> Ready on http://localhost:${port}`);
    console.log(`> Proxying API requests to ${BACKEND_URL}`);
  });
});