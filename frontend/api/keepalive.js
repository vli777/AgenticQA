// Vercel Serverless Function for backend keepalive
// This endpoint is called by Vercel Cron to prevent Render cold starts

export default async function handler(req, res) {
  const backendUrl = process.env.BACKEND_URL || process.env.VITE_API_BASE_URL;

  if (!backendUrl) {
    return res.status(500).json({ error: "BACKEND_URL not configured" });
  }

  try {
    const response = await fetch(`${backendUrl}/health`, {
      method: "GET",
      headers: {
        "User-Agent": "AgenticQA-Keepalive/1.0",
      },
    });

    const data = await response.json();

    return res.status(response.ok ? 200 : 502).json({
      status: response.ok ? "ok" : "backend_unhealthy",
      backend: data,
      timestamp: new Date().toISOString(),
    });
  } catch (error) {
    return res.status(502).json({
      status: "error",
      error: error.message,
      timestamp: new Date().toISOString(),
    });
  }
}
