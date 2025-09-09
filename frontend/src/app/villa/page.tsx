"use client";

import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import { motion, AnimatePresence } from "framer-motion";

/** -------- Types (match FastAPI) -------- */
type RagRequest = {
  query?: string;
  top_k?: number;
  rebuild?: boolean;
  // we send these so backend logs/visualizations are produced
  return_contexts?: boolean;
  return_llm_prompt?: boolean;
  return_llm_answer_raw?: boolean;
};

type Hit = { score: number; text_preview: string; row_index?: number | null };
type RagResponse = {
  rebuilt: boolean;
  query?: string | null;
  top_k: number;
  reranked: boolean;
  hits: Hit[];               // not shown on UI
  answer?: string | null;    // only this is shown
  debug: Record<string, any>; // not shown on UI
};

const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL; // e.g. http://localhost:8000

// Animation variants with proper typing
const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.05
    }
  }
};

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      type: "spring" as const,
      stiffness: 300,
      damping: 24
    }
  }
};

const messageVariants = {
  hidden: { opacity: 0, y: 10, scale: 0.95 },
  visible: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      type: "spring" as const,
      stiffness: 500,
      damping: 30
    }
  }
};

const loadingVariants = {
  animate: {
    opacity: [0.5, 1, 0.5],
    transition: {
      duration: 1.5,
      repeat: Infinity,
      ease: "easeInOut"
    }
  }
};

const buttonVariants = {
  initial: { scale: 1 },
  tap: { scale: 0.97 },
  hover: { 
    scale: 1.02,
    boxShadow: "0 5px 15px rgba(59, 130, 246, 0.4)"
  }
};

export default function RagChatPage() {
  type Msg = { role: "user" | "assistant" | "system"; text: string; id: string };
  const [messages, setMessages] = useState<Msg[]>([]);

  const [input, setInput] = useState("");
  const [topK, setTopK] = useState(5);
  const [rebuildFirst, setRebuildFirst] = useState(false);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string>("");

  const endRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Auto-focus on input after sending a message
  useEffect(() => {
    if (!loading && inputRef.current) {
      inputRef.current.focus();
    }
  }, [loading]);

  async function send() {
    setErr("");
    if (!backendUrl) {
      setErr("NEXT_PUBLIC_BACKEND_URL is not set (e.g., http://localhost:8000).");
      return;
    }

    const q = input.trim();
    if (!q && !rebuildFirst) {
      setErr("Enter a query or enable 'Rebuild first'.");
      return;
    }

    setLoading(true);

    try {
      // Show the user's message (if any)
      if (q) {
        setMessages((m) => [...m, { 
          role: "user", 
          text: q, 
          id: `user-${Date.now()}-${Math.random().toString(36).substr(2, 9)}` 
        }]);
        setInput(""); // Clear input after sending
      }

      const payload: RagRequest = {
        rebuild: rebuildFirst,
        ...(q ? { query: q } : {}),
        top_k: topK,
        // keep backend verbose for logs/files, but we don't display them here
        return_contexts: true,
        return_llm_prompt: true,
        return_llm_answer_raw: true,
      };

      const res = await fetch(`${backendUrl}/api/rag_agent`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text().catch(() => "");
        throw new Error(`HTTP ${res.status}: ${text || res.statusText}`);
      }

      const data: RagResponse = await res.json();

      // If rebuild-only (no query), just post a concise system note
      if (!q) {
        const rebuiltMsg = data.rebuilt
          ? "Rebuild completed. You can ask a question now."
          : "No rebuild done.";
        setMessages((m) => [...m, { 
          role: "system", 
          text: rebuiltMsg, 
          id: `system-${Date.now()}-${Math.random().toString(36).substr(2, 9)}` 
        }]);
        return;
      }

      const answer =
        (data.answer && data.answer.trim()) ||
        "No relevant context found in CSV.";

      setMessages((m) => [...m, { 
        role: "assistant", 
        text: answer, 
        id: `assistant-${Date.now()}-${Math.random().toString(36).substr(2, 9)}` 
      }]);
    } catch (e: any) {
      setErr(e?.message || "Request failed.");
    } finally {
      setLoading(false);
      setRebuildFirst(false);
    }
  }

  function MarkdownMessage({ text }: { text: string }) {
    return (
      <div className="prose prose-sm max-w-none">
        <ReactMarkdown
          components={{
            strong: ({ node, ...props }) => <strong className="font-bold text-gray-900" {...props} />,
            ul: ({ node, ...props }) => <ul className="list-disc ml-6 space-y-1" {...props} />,
            ol: ({ node, ...props }) => <ol className="list-decimal ml-6 space-y-1" {...props} />,
            p: ({ node, ...props }) => <p className="my-1" {...props} />,
          }}
        >
          {text}
        </ReactMarkdown>
      </div>
    );
  }

  // Handle Enter key press (with Shift for new line)
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <motion.div 
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className="mx-auto max-w-3xl p-4 md:p-6 min-h-screen flex flex-col"
    >
      <motion.header 
        initial={{ y: -20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.4 }}
        className="mb-6"
      >
        <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
          Villa Finder
        </h1>
        <p className="text-sm text-gray-500 mt-1">
          Ask questions about your CSV data with AI assistance
        </p>
      </motion.header>

      {/* Chat messages container */}
      <div className="flex-1 overflow-y-auto mb-4 space-y-3 py-2">
        <AnimatePresence mode="popLayout">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="space-y-3"
          >
            {messages.map((m) => (
              <motion.div
                key={m.id}
                layout
                variants={itemVariants}
                initial="hidden"
                animate="visible"
                exit={{ opacity: 0, scale: 0.95 }}
                className={m.role === "user" ? "flex justify-end" : "flex justify-start"}
              >
                <motion.div
                  variants={messageVariants}
                  className={
                    "max-w-[85%] rounded-2xl px-4 py-2 text-sm shadow-sm " +
                    (m.role === "user" 
                      ? "bg-blue-600 text-white" 
                      : m.role === "system"
                        ? "bg-amber-100 text-amber-900 border border-amber-200"
                        : "bg-gray-100 text-gray-900 border border-gray-200")
                  }
                >
                  <MarkdownMessage text={m.text} />
                </motion.div>
              </motion.div>
            ))}
          </motion.div>
        </AnimatePresence>

        {loading && (
          <motion.div 
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-start"
          >
            <motion.div 
              animate={{ opacity: [0.5, 1, 0.5] }}
              transition={{ duration: 1.5, repeat: Infinity, ease: "easeInOut" }}
              className="max-w-[85%] rounded-2xl px-4 py-3 text-sm bg-gray-100 text-gray-900 border border-gray-200 flex items-center gap-2"
            >
              <motion.div 
                animate={{ rotate: 360 }}
                transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                className="w-4 h-4 border-2 border-gray-400 border-t-blue-600 rounded-full"
              />
              <span>Thinking…</span>
            </motion.div>
          </motion.div>
        )}
        <div ref={endRef} />
      </div>

      {/* Composer */}
      <motion.div 
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ delay: 0.2 }}
        className="rounded-xl border bg-white p-4 shadow-lg space-y-3 sticky bottom-0"
      >
        <AnimatePresence>
          {err && (
            <motion.div 
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: "auto", opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="overflow-hidden"
            >
              <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-sm text-red-800 flex items-start gap-2">
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 text-red-500 shrink-0 mt-0.5" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                </svg>
                <span>{err}</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <textarea
          ref={inputRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={2}
          className="w-full rounded-lg border border-gray-300 p-3 outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all resize-none"
          placeholder="Type your question…"
          disabled={loading}
        />
        
        <div className="flex flex-wrap items-center gap-4">
          <motion.label 
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
            className="flex items-center gap-2 text-sm text-gray-700 cursor-pointer select-none"
          >
            <input
              type="checkbox"
              checked={rebuildFirst}
              onChange={(e) => setRebuildFirst(e.target.checked)}
              className="rounded text-blue-600 focus:ring-blue-500"
              disabled={loading}
            />
            Rebuild first (re-ingest CSV)
          </motion.label>

          <motion.label 
            whileHover={{ scale: 1.05 }}
            className="flex items-center gap-2 text-sm text-gray-700"
          >
            Top-K
            <input
              type="number"
              min={1}
              max={50}
              value={topK}
              onChange={(e) => setTopK(parseInt(e.target.value || "5", 10))}
              className="w-20 rounded border border-gray-300 p-1 transition-colors focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              disabled={loading}
            />
          </motion.label>

          <motion.button
            variants={buttonVariants}
            initial="initial"
            whileHover="hover"
            whileTap="tap"
            onClick={send}
            disabled={loading || (!input.trim() && !rebuildFirst)}
            className="ml-auto rounded-lg bg-blue-600 px-4 py-2 text-white font-medium disabled:opacity-60 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <motion.svg 
                  animate={{ rotate: 360 }}
                  transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                  className="w-4 h-4 text-white" 
                  fill="none" 
                  viewBox="0 0 24 24"
                >
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </motion.svg>
                <span>Sending…</span>
              </>
            ) : (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 20 20" fill="currentColor">
                  <path fillRule="evenodd" d="M10.293 3.293a1 1 0 011.414 0l6 6a1 1 0 010 1.414l-6 6a1 1 0 01-1.414-1.414L14.586 11H3a1 1 0 110-2h11.586l-4.293-4.293a1 1 0 010-1.414z" clipRule="evenodd" />
                </svg>
                <span>Send</span>
              </>
            )}
          </motion.button>
        </div>
      </motion.div>
    </motion.div>
  );
}