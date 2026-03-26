import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Elephant Game — Zoo Zurich",
  description: "Can you identify the elephants of Zoo Zurich?",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
