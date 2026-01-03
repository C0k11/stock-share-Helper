# Changelog

## [2026-01-03] Phase 3.3: Live Paper Trading Engine + Mari Secretary

### Added
- **Live Paper Trading Engine** (`scripts/run_live_paper_trading.py`)
  - Real-time Multi-Agent strategy simulation with Mari's voice commentary
  - Event-driven architecture with TTS queue for lag-free voice output
  - Simulates: Planner → MoE Router → Expert → System 2 Debate → Execution

- **Multi-Agent Strategy** (`src/trading/strategy.py`)
  - Encapsulates trading logic from `run_trading_inference.py`
  - Mock implementation with hooks for real model integration (Phase 3.4)
  - Supports Planner, Gatekeeper, MoE Router, Expert inference, System 2 debate

- **Event System Enhancement** (`src/trading/event.py`)
  - Added `LOG` event type for Agent thought process streaming
  - Added `priority` field for TTS control (0=text, 1=bubble, 2=speak)

- **Secretary Integration**
  - `/api/v1/status` endpoint aggregates trading + voice training status
  - `/api/v1/chat` injects real-time status into LLM system prompt
  - Mari can now "see" what the system is doing and report on it

- **GPT-SoVITS Voice Training** (completed)
  - S1 (GPT semantic) trained: `epoch=14-step=15.ckpt`
  - S2 (SoVITS vocoder) trained: `G_64.pth`
  - Converted to inference format: `mari_s1_infer.ckpt`, `mari_s2G_infer.pth`
  - Mari's trained voice now used in all TTS output

### Changed
- **Secretary Persona** (`configs/secretary.yaml`)
  - User now addressed as "Sensei" (せんせい) in Japanese style
  - User's name is "Cokii" (コキー)
  - Added live commentary mode instructions
  - LLM backend switched to Ollama (`qwen2.5:7b-instruct`)

- **TradingEngine** (`src/trading/engine.py`)
  - Added LOG event handling for external listeners

### Services Configuration
- GPT-SoVITS TTS: `http://127.0.0.1:9880` (Mari weights auto-loaded)
- Stock FastAPI: `http://127.0.0.1:8000`
- Stock Streamlit: `http://127.0.0.1:8501`
- Ollama LLM: `http://localhost:11434/v1` (model: `qwen2.5:7b-instruct`)

### Next Steps (Phase 3.4)
- Replace mock strategy with real model inference
- Connect to live market data feed
- WebSocket for real-time UI updates
