"use client";

import Link from 'next/link';
import { motion } from 'framer-motion';

const Home = () => {
const agents = [
  {
    emoji: '🏡',
    title: 'Villa Bot',
    description: 'Find and locate beautiful hotels and villas with detailed information and recommendations',
    href: '/villa',
    buttonText: 'Find →'
  },
  {
    emoji: '📦',
    title: 'Tata Sampann Product Assistant',
    description: 'Get product details and recommendations',
    href: '/tata-sampann',
    buttonText: 'Explore Products →'
  },
  {
    emoji: '🍎',
    title: 'Nutrition Bot',
    description: 'Get personalized food advice and meal plans',
    href: '/nutrition-bot',
    buttonText: 'Get Nutrition Tips →'
  },
  {
    emoji: '🖼️',
    title: 'Image Analyzer',
    description: 'Upload images and get AI-powered insights',
    href: '/image-analyzer',
    buttonText: 'Analyze Images →'
  },
  {
    emoji: '📧',
    title: 'Document Mailer',
    description: 'Request documents and get them delivered to your email',
    href: '/document-mailer',
    buttonText: 'Request Documents →'
  },
  {
    emoji: "📄",
    title: "RAG Document Assistant",
    description: "Upload a PDF and ask questions about its content",
    href: "/rag",
    buttonText: "Ask about your PDF"
  },
  {
    emoji: '🔍',
    title: 'QR Code Detector',
    description: 'Upload an image to detect and decode QR codes',
    href: '/qr-detector',
    buttonText: 'Scan QR Code →'
  },
  {
    emoji: '🏷️',
    title: 'Brand Checker',
    description: 'Upload an image to identify cigarette brands',
    href: '/brand-check',
    buttonText: 'Check Brand →'
  },
  {
    emoji: '👁️',
    title: 'Vision Test',
    description: 'Test computer vision capabilities with various image analysis tasks',
    href: '/vision-test',
    buttonText: 'Test Vision →'
  },
  {
    emoji: '✨',
    title: 'More Coming Soon...',
    description: 'Additional AI agents in development',
    href: '#',
    buttonText: 'Coming soon...'
  }
];

  const container = {
    hidden: { opacity: 0 },
    show: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1
      }
    }
  };

  const item = {
    hidden: { opacity: 0, y: 20 },
    show: { opacity: 1, y: 0 }
  };

  return (
    <div className="min-h-screen bg-white flex flex-col items-center justify-center p-6">
      <motion.header
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6 }}
        className="text-center mb-12"
      >
        <h1 className="text-3xl md:text-4xl font-bold mb-4 text-black">
          GenAIssance
        </h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="text-lg text-gray-600"
        >
          Experiment with our specialized AI agents
        </motion.p>
      </motion.header>

      <motion.div
        variants={container}
        initial="hidden"
        animate="show"
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl w-full"
      >
        {agents.map((agent, index) => (
          <motion.div
            key={index}
            variants={item}
            className={`bg-white border border-gray-300 shadow-lg rounded-lg p-6 h-full flex flex-col ${index === agents.length - 1 ? 'opacity-70' : ''}`}
          >
            <div className="text-3xl mb-4">
              {agent.emoji}
            </div>
            <h2 className="text-xl font-semibold text-black mb-2">{agent.title}</h2>
            <p className="text-gray-600 mb-6 flex-grow">{agent.description}</p>
            <Link
              href={agent.href}
              className={`inline-block px-4 py-2 border border-black text-black rounded transition-colors duration-200 font-medium text-center hover:bg-black hover:text-white ${index === agents.length - 1 ? 'pointer-events-none opacity-70' : ''}`}
            >
              {agent.buttonText}
            </Link>
          </motion.div>
        ))}
      </motion.div>
    </div>
  );
};

export default Home;