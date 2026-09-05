import type { NextConfig } from "next";

/**
 * Minimal config for the Phase 0 scaffold. No image domains, no rewrites yet -
 * the dashboard talks to the FastAPI service directly via NEXT_PUBLIC_API_URL
 * (see lib/api.ts). Revisit when video/clip serving (R2 + hls.js) lands.
 */
const nextConfig: NextConfig = {
  reactStrictMode: true,
};

export default nextConfig;
