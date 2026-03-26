/** @type {import('next').NextConfig} */
const nextConfig = {
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "**" },
      { protocol: "https", hostname: "**" },
    ],
  },
  async rewrites() {
    const apiBase = process.env.API_BASE_SERVER ?? "http://localhost:8001";
    return [
      { source: "/api/v1/:path*", destination: `${apiBase}/api/v1/:path*` },
      { source: "/storage/:path*", destination: `${apiBase}/storage/:path*` },
    ];
  },
};

export default nextConfig;
