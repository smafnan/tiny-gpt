import { useEffect, useRef, useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, Loader2, Cpu, Wand2, RotateCcw } from "lucide-react";

const SEEDS = ["ROMEO:", "To be, or not", "FIRST CITIZEN:\n", "O, "];

function Blobs() {
  return (
    <div className="pointer-events-none fixed inset-0 overflow-hidden">
      <div className="absolute -top-44 left-1/4 h-[34rem] w-[34rem] rounded-full bg-amber-600/15 blur-3xl animate-float" />
      <div className="absolute bottom-0 -right-40 h-[30rem] w-[30rem] rounded-full bg-yellow-500/10 blur-3xl animate-float [animation-delay:-6s]" />
      <div className="absolute -bottom-40 -left-32 h-[28rem] w-[28rem] rounded-full bg-orange-500/10 blur-3xl animate-float [animation-delay:-9s]" />
    </div>
  );
}

function Slider({
  label, value, min, max, step, onChange, fmt,
}: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; fmt?: (v: number) => string;
}) {
  return (
    <label className="text-sm">
      <span className="mb-1 flex justify-between text-slate-400">
        <span>{label}</span>
        <span className="font-mono text-amber-300">{fmt ? fmt(value) : value}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full accent-amber-400"
      />
    </label>
  );
}

export default function App() {
  const [prompt, setPrompt] = useState("ROMEO:");
  const [temp, setTemp] = useState(0.8);
  const [len, setLen] = useState(400);
  const [topK, setTopK] = useState(40);
  const [full, setFull] = useState("");
  const [shown, setShown] = useState("");
  const [loading, setLoading] = useState(false);
  const [info, setInfo] = useState<any>(null);
  const reveal = useRef<number | null>(null);

  useEffect(() => {
    fetch("/api/info").then((r) => r.json()).then(setInfo).catch(() => {});
  }, []);

  // Typewriter reveal of the generated text.
  useEffect(() => {
    if (!full) return;
    setShown("");
    let i = 0;
    if (reveal.current) clearInterval(reveal.current);
    reveal.current = window.setInterval(() => {
      i += 3;
      setShown(full.slice(0, i));
      if (i >= full.length && reveal.current) clearInterval(reveal.current);
    }, 16);
    return () => {
      if (reveal.current) clearInterval(reveal.current);
    };
  }, [full]);

  async function gen() {
    setLoading(true);
    setFull("");
    setShown("");
    try {
      const r = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, max_new_tokens: len, temperature: temp, top_k: topK }),
      });
      const d = await r.json();
      if (d.ok) setFull(d.text);
      else setFull("[" + d.error + "]");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="relative min-h-screen text-amber-50">
      <Blobs />
      <div className="relative mx-auto max-w-3xl px-5 py-14">
        <motion.header
          initial={{ opacity: 0, y: -16 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8 text-center"
        >
          <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-amber-400/20 bg-amber-400/5 px-4 py-1.5 text-xs text-amber-200/80 backdrop-blur">
            <Cpu size={14} className="text-amber-300" />
            {info?.ready
              ? `${(info.params / 1e6).toFixed(1)}M-param transformer · built from scratch`
              : "Char-level GPT"}
          </div>
          <h1 className="text-5xl font-extrabold tracking-tight text-white sm:text-6xl">
            Tiny{" "}
            <span className="bg-gradient-to-r from-amber-300 via-yellow-200 to-orange-300 bg-clip-text text-transparent">
              GPT
            </span>
          </h1>
          <p className="mx-auto mt-4 max-w-lg text-amber-200/60">
            A transformer with hand-written self-attention, trained on Shakespeare.
            Give it a seed and watch it dream up new lines, character by character.
          </p>
        </motion.header>

        {/* Prompt + controls */}
        <div className="rounded-2xl border border-amber-400/15 bg-white/[0.03] p-5 backdrop-blur">
          <textarea
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Seed text…"
            className="h-20 w-full resize-none bg-transparent font-mono text-base text-amber-100 placeholder:text-amber-200/30 outline-none"
          />
          <div className="mt-2 flex flex-wrap gap-2 border-t border-amber-400/10 pt-3">
            {SEEDS.map((s) => (
              <button
                key={s}
                onClick={() => setPrompt(s)}
                className="rounded-full border border-amber-400/15 bg-amber-400/5 px-3 py-1 font-mono text-xs text-amber-200/70 hover:bg-amber-400/10"
              >
                {JSON.stringify(s).slice(1, -1)}
              </button>
            ))}
          </div>
          <div className="mt-4 grid gap-4 sm:grid-cols-3">
            <Slider label="Temperature" value={temp} min={0.2} max={1.5} step={0.05} onChange={setTemp} fmt={(v) => v.toFixed(2)} />
            <Slider label="Length" value={len} min={100} max={800} step={50} onChange={setLen} />
            <Slider label="Top-k" value={topK} min={5} max={65} step={5} onChange={setTopK} />
          </div>
          <button
            onClick={gen}
            disabled={loading}
            className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-amber-500 to-orange-500 px-5 py-3 text-base font-semibold text-white shadow-lg shadow-amber-500/20 disabled:opacity-60"
          >
            {loading ? <Loader2 className="animate-spin" size={18} /> : <Wand2 size={18} />}
            {loading ? "Dreaming…" : "Generate"}
          </button>
        </div>

        {/* Output */}
        {(shown || loading) && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6 rounded-2xl border border-amber-400/15 bg-[#16100a]/60 p-7 backdrop-blur"
          >
            <div className="mb-3 flex items-center gap-2 text-xs uppercase tracking-wider text-amber-200/40">
              <Sparkles size={13} /> generated
              {shown && !loading && (
                <button onClick={gen} className="ml-auto inline-flex items-center gap-1 text-amber-200/50 hover:text-amber-200">
                  <RotateCcw size={12} /> again
                </button>
              )}
            </div>
            <pre className="whitespace-pre-wrap break-words font-serif text-lg leading-relaxed text-amber-50/90">
              {shown}
              {shown.length < full.length && <span className="animate-blink">▋</span>}
            </pre>
          </motion.div>
        )}

        <footer className="mt-16 text-center text-xs text-amber-200/30">
          Multi-head causal self-attention, written by hand in PyTorch · char-level ·
          trained on tiny-shakespeare
        </footer>
      </div>
    </div>
  );
}
