"use client";

import Link from "next/link";

export default function LandingPage() {
  return (
    <div className="fixed inset-0 overflow-auto bg-[#050510]">
      {/* Aurora gradient background */}
      <div className="fixed inset-0 -z-10 overflow-hidden">
        <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] rounded-full bg-[radial-gradient(circle,rgba(0,212,170,0.12)_0%,transparent_70%)] animate-aurora-1" />
        <div className="absolute top-[10%] right-[-15%] w-[55%] h-[55%] rounded-full bg-[radial-gradient(circle,rgba(123,97,255,0.10)_0%,transparent_70%)] animate-aurora-2" />
        <div className="absolute bottom-[-10%] left-[20%] w-[50%] h-[50%] rounded-full bg-[radial-gradient(circle,rgba(14,165,233,0.08)_0%,transparent_70%)] animate-aurora-3" />
        <div className="absolute bottom-[20%] right-[10%] w-[40%] h-[40%] rounded-full bg-[radial-gradient(circle,rgba(236,72,153,0.06)_0%,transparent_70%)] animate-aurora-1" />
      </div>

      <div className="relative z-10 flex min-h-screen flex-col items-center justify-center px-4 py-12">
        {/* Title */}
        <div className="mb-16 text-center">
          <h1 className="mb-3 text-5xl font-extralight tracking-wide text-white sm:text-6xl">
            Elephant Game
          </h1>
          <p className="text-lg font-light tracking-[0.2em] text-white/40 uppercase">
            Zoo Zurich
          </p>
        </div>

        {/* Cards */}
        <div className="grid w-full max-w-4xl grid-cols-1 gap-6 sm:grid-cols-2">
          {/* Elephant Match Card */}
          <Link href="/match" className="group">
            <div className="flex h-64 flex-col items-center justify-center rounded-3xl border border-white/[0.08] bg-white/[0.03] p-8 backdrop-blur-2xl transition-all duration-500 hover:border-cyan-400/20 hover:bg-white/[0.05] hover:shadow-[0_0_60px_rgba(0,212,170,0.1)] group-hover:scale-[1.02]">
              <div className="mb-6 flex items-center gap-3">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-cyan-400/10 text-2xl transition-all duration-500 group-hover:bg-cyan-400/20 group-hover:shadow-[0_0_20px_rgba(0,212,170,0.2)]">
                  🐘
                </div>
                <div className="text-2xl font-light text-white/20">vs</div>
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-purple-400/10 text-2xl transition-all duration-500 group-hover:bg-purple-400/20 group-hover:shadow-[0_0_20px_rgba(123,97,255,0.2)]">
                  🐘
                </div>
              </div>
              <h2 className="mb-2 text-xl font-light tracking-wide text-white">
                Elephant Match
              </h2>
              <p className="text-center text-sm font-light text-white/40">
                Can you tell them apart?
              </p>
            </div>
          </Link>

          {/* Name That Elephant Card */}
          <Link href="/name" className="group">
            <div className="flex h-64 flex-col items-center justify-center rounded-3xl border border-white/[0.08] bg-white/[0.03] p-8 backdrop-blur-2xl transition-all duration-500 hover:border-amber-400/20 hover:bg-white/[0.05] hover:shadow-[0_0_60px_rgba(245,158,11,0.1)] group-hover:scale-[1.02]">
              <div className="mb-6">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-amber-400/10 text-2xl transition-all duration-500 group-hover:bg-amber-400/20 group-hover:shadow-[0_0_20px_rgba(245,158,11,0.2)]">
                  🐘
                </div>
              </div>
              <h2 className="mb-2 text-xl font-light tracking-wide text-white">
                Name That Elephant
              </h2>
              <p className="text-center text-sm font-light text-white/40">
                Who is this elephant?
              </p>
            </div>
          </Link>

          {/* Behavior Quiz Card */}
          <Link href="/behavior" className="group">
            <div className="flex h-64 flex-col items-center justify-center rounded-3xl border border-white/[0.08] bg-white/[0.03] p-8 backdrop-blur-2xl transition-all duration-500 hover:border-emerald-400/20 hover:bg-white/[0.05] hover:shadow-[0_0_60px_rgba(16,185,129,0.1)] group-hover:scale-[1.02]">
              <div className="mb-6">
                <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-400/10 text-2xl transition-all duration-500 group-hover:bg-emerald-400/20 group-hover:shadow-[0_0_20px_rgba(16,185,129,0.2)]">
                  🔍
                </div>
              </div>
              <h2 className="mb-2 text-xl font-light tracking-wide text-white">
                Behavior Quiz
              </h2>
              <p className="text-center text-sm font-light text-white/40">
                What is the elephant doing?
              </p>
            </div>
          </Link>

          {/* Elephant Galaxy Card */}
          <Link href="/galaxy" className="group">
            <div className="flex h-64 flex-col items-center justify-center rounded-3xl border border-white/[0.08] bg-white/[0.03] p-8 backdrop-blur-2xl transition-all duration-500 hover:border-purple-400/20 hover:bg-white/[0.05] hover:shadow-[0_0_60px_rgba(123,97,255,0.1)] group-hover:scale-[1.02]">
              <div className="mb-6">
                <div className="relative flex h-16 w-16 items-center justify-center">
                  <div className="absolute inset-0 rounded-full bg-purple-500/10 transition-all duration-700 group-hover:bg-purple-500/20 group-hover:shadow-[0_0_30px_rgba(123,97,255,0.3)]" />
                  <div className="absolute inset-2 rounded-full border border-purple-400/20 transition-all duration-700 group-hover:border-purple-400/40" />
                  <div className="absolute inset-4 rounded-full border border-cyan-400/10 transition-all duration-700 group-hover:border-cyan-400/30" />
                  <span className="relative z-10 text-xl">✦</span>
                </div>
              </div>
              <h2 className="mb-2 text-xl font-light tracking-wide text-white">
                Elephant Galaxy
              </h2>
              <p className="text-center text-sm font-light text-white/40">
                Explore the feature space
              </p>
            </div>
          </Link>
        </div>

        <p className="mt-12 text-xs font-light tracking-widest text-white/20 uppercase">
          Zoo Zurich — Powered by Computer Vision & ReID
        </p>
      </div>
    </div>
  );
}
