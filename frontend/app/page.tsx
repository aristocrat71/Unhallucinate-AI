"use client";

import { useState, useEffect } from "react";
import { Navbar } from "@/components/navbar"
import { Hero } from "@/components/hero"

export default function LandingPage() {
  const [scrollProgress, setScrollProgress] = useState(0);

  useEffect(() => {
    const handleScroll = () => {
      const totalHeight = document.documentElement.scrollHeight - document.documentElement.clientHeight;
      const progress = (window.scrollY / totalHeight) * 100;
      setScrollProgress(progress);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <div className="relative min-h-screen font-sans selection:bg-primary/30 overflow-x-hidden">
      {/* Scroll Progress Indicator */}
      <div className="fixed top-0 right-0 w-1 h-full bg-slate-800/30 z-50">
        <div 
          className="w-full bg-gradient-to-b from-blue-500 via-purple-500 to-blue-600 transition-all duration-300 ease-out shadow-[0_0_10px_rgba(59,130,246,0.5)]"
          style={{ height: `${scrollProgress}%` }}
        />
      </div>

      <div className="fixed inset-0 grid-lines pointer-events-none opacity-20" />
      <Navbar />
      <main>
        <Hero />
      </main>
    </div>
  )
}
