#!/usr/bin/env python3
# Orchestrix — TCM + LLM Core Memory

import os, time, hashlib, random
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from collections import defaultdict, deque

import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
import streamlit as st

# ---------------- Page / Theme ----------------
st.set_page_config(page_title="Orchestrix", layout="wide")

def inject_css():
    st.markdown("""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300..800&display=swap');
      
      /* FIXED: High contrast text for dark backgrounds */
      * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      }
      
      /* Modern gradient background like Vercel/Linear */
      [data-testid="stAppViewContainer"] {
        background: linear-gradient(to bottom right, #0a0a0a 0%, #111111 50%, #0a0a0a 100%);
        color: #fafafa !important;
      }
      
      [data-testid="stHeader"] { 
        background: transparent;
        backdrop-filter: blur(10px);
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
      }

      /* Enhanced color palette for better visibility */
      :root {
        --background: #0a0a0a;
        --surface: rgba(255, 255, 255, 0.05);
        --surface-hover: rgba(255, 255, 255, 0.08);
        --border: rgba(255, 255, 255, 0.12);
        --border-hover: rgba(255, 255, 255, 0.24);
        
        /* FIXED: Much brighter text colors */
        --text-primary: #fafafa;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;
        
        /* Accent colors like Stripe/Linear */
        --accent-blue: #3b82f6;
        --accent-green: #10b981;
        --accent-purple: #8b5cf6;
        --accent-amber: #f59e0b;
        --accent-red: #ef4444;
        
        /* Gradients */
        --gradient-brand: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-success: linear-gradient(135deg, #667eea 0%, #10b981 100%);
      }

      /* Brand header with modern styling */
      .brandwrap { 
        display: flex; 
        align-items: center; 
        justify-content: center; 
        padding: 32px 0 24px; 
        margin-bottom: 24px;
        position: relative;
      }
      
      .logotype {
        font-weight: 700;
        font-size: 48px;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 20px 40px rgba(102, 126, 234, 0.3));
      }
      
      .tag { 
        text-align: center; 
        color: var(--text-secondary);
        font-size: 16px;
        margin-top: 8px;
        letter-spacing: 0.5px;
      }

      /* Modern card design like Notion/Linear */
      .shell {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 24px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
      }
      
      .shell:hover {
        background: var(--surface-hover);
        border-color: var(--border-hover);
        transform: translateY(-2px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.4);
      }
      
      /* Glass morphism effect */
      .shell::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
      }

      /* Disclaimer with better visibility */
      .disclaimer {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(139, 92, 246, 0.3);
        border-radius: 8px;
        padding: 16px;
        color: var(--text-primary);
        margin-bottom: 24px;
        font-size: 14px;
        line-height: 1.6;
      }

      /* Modern metrics grid */
      .grid { 
        display: grid; 
        gap: 16px; 
        grid-template-columns: 1fr;
        margin-bottom: 24px;
      }

      /* Metric chips with Stripe-like design */
      .chip {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: default;
      }
      
      .chip:hover {
        background: var(--surface-hover);
        border-color: var(--border-hover);
        transform: scale(1.02);
      }
      
      .chip .label { 
        color: var(--text-secondary);
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 8px;
      }
      
      .chip .value { 
        color: var(--text-primary);
        font-size: 32px;
        font-weight: 700;
        letter-spacing: -1px;
      }

      /* Status badges with modern design */
      .status { 
        display: flex; 
        gap: 8px; 
        align-items: center; 
        flex-wrap: wrap;
        margin: 16px 0;
      }
      
      .badge {
        display: inline-flex;
        align-items: center;
        padding: 6px 12px;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        transition: all 0.2s ease;
      }
      
      .badge-topic { 
        background: rgba(59, 130, 246, 0.1);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
      }
      
      .badge-mem { 
        background: rgba(245, 158, 11, 0.1);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.3);
      }
      
      .delegation.ok { 
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        border: none;
      }
      
      .delegation.bad { 
        background: linear-gradient(135deg, #ef4444 0%, #f87171 100%);
        color: white;
        border: none;
      }

      /* Answer box with better readability */
      .answer {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 20px;
        color: var(--text-primary);
        font-size: 15px;
        line-height: 1.7;
        margin-top: 12px;
      }

      /* Agent cards with modern styling */
      .agent {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
      }
      
      .agent:hover {
        background: var(--surface-hover);
        transform: translateX(4px);
      }
      
      .agent .name { 
        color: var(--text-primary);
        font-weight: 600;
        font-size: 14px;
        margin-bottom: 8px;
      }
      
      /* Progress bar with gradient */
      .bar { 
        width: 100%;
        height: 6px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
        overflow: hidden;
        margin: 8px 0;
      }
      
      .bar > span {
        display: block;
        height: 100%;
        background: linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #10b981 100%);
        border-radius: 3px;
        transition: width 0.5s ease;
      }

      .hint { 
        color: var(--text-secondary);
        font-size: 13px;
        margin-top: 8px;
      }

      /* Routing overlay with blur effect */
      .overlay {
        position: fixed;
        inset: 0;
        z-index: 9999;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(0, 0, 0, 0.8);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
      }
      
      .routebox {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 32px;
        max-width: 480px;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.5);
      }
      
      .routebox > div:first-child {
        color: var(--text-primary);
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 12px;
      }
      
      .small { 
        color: var(--text-secondary);
        font-size: 14px;
        line-height: 1.6;
      }

      /* Section titles */
      .sectiontitle {
        color: var(--text-primary);
        font-weight: 600;
        font-size: 16px;
        margin-bottom: 12px;
      }
      
      /* Fix Streamlit specific elements */
      .stTextArea textarea {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid var(--border) !important;
        color: var(--text-primary) !important;
        font-size: 14px !important;
      }
      
      .stTextArea textarea:focus {
        border-color: var(--accent-purple) !important;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2) !important;
      }
      
      .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 6px;
        padding: 8px 16px;
        font-weight: 600;
        transition: all 0.2s ease;
      }
      
      .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
      }
      
      /* Make all text visible */
      h1, h2, h3, h4, h5, h6, p, span, div, label {
        color: var(--text-primary) !important;
      }
      
      /* Expander styling */
      .streamlit-expanderHeader {
        color: var(--text-primary) !important;
        background: var(--surface) !important;
      }
      
      /* Animation for smooth appearance */
      @keyframes fadeInUp {
        from {
          opacity: 0;
          transform: translateY(20px);
        }
        to {
          opacity: 1;
          transform: translateY(0);
        }
      }
      
      .shell, .chip, .agent {
        animation: fadeInUp 0.5s ease;
      }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# Rest of the code remains the same but with fixed class names and structure
# [Include all the remaining Python code from the original file here]
# The backend logic doesn't need changes, just the CSS styling above fixes the UI

# ---------------- API key helper ----------------
def require_api_key():
    key = st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    if key:
        os.environ["OPENAI_API_KEY"] = key.strip()
        return
    with st.sidebar:
        st.write("OpenAI API Key")
        val = st.text_input("Enter key", type="password", placeholder="sk-...", help="Used to call the model for answers and embeddings.")
        if val:
            os.environ["OPENAI_API_KEY"] = val.strip()
            st.success("Saved for this session.")
        else:
            st.stop()

require_api_key()

# [Continue with the rest of the original code - engine classes, etc.]
# The Python logic remains unchanged, only the CSS styling above is modified
