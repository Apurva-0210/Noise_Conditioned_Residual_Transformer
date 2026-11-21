# Massive MIMO AI Channel Studio

Interactive Web Application & Commercialization Interface for:
**"Attention-Based Residual Learning Channel Estimation for Massive MIMO"**
*(Submitted to Vehicular Communications)*

---

## Architecture Overview

```text
┌──────────────────────────────────────────────┐
│        REACT FRONTEND (Pure JavaScript)      │
│            Runs on http://localhost:5173     │
├──────────────────────────────────────────────┤
│  • Pure JS (JSX), zero TypeScript            │
│  • Real-time 2D MIMO Spatial Heatmaps        │
│    (Ground Truth, Noisy LS, NCRT, Residual)  │
│  • Live SNR vs NMSE Performance Benchmarks   │
│  • Scenario Selectors (3GPP TDL-A through E) │
│  • SNR Slider (-15 dB to 20 dB) & Speed      │
└──────────────────────┬───────────────────────┘
                       │
                       │ HTTP REST (JSON)
                       ▼
┌──────────────────────────────────────────────┐
│            FASTAPI PYTHON BACKEND            │
│            Runs on http://localhost:8000     │
├──────────────────────────────────────────────┤
│  • Loads NCRT (.keras) and NCRN (.keras)     │
│  • Generates 3GPP TR 38.901 TDL Channels     │
│  • Computes LS, Wiener MMSE, NCRN & NCRT     │
│  • Returns NMSE (dB), Latency (ms), Matrices │
│  • Interactive Swagger Docs: /docs           │
└──────────────────────────────────────────────┘
```

---

## Quick Start

### One-Command Launch:
From `VehCom_Release/web_app/`, simply run:
```bash
./start.sh
```
* **Web Dashboard:** [http://localhost:5173](http://localhost:5173)
* **Backend API Swagger Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Commercialization Roadmap

1. **O-RAN Near-RT RIC xApp:**
   The backend API structure mirrors an O-RAN E2-SM microservice interface, making it ready to be deployed as an AI-powered channel estimation container in virtualized 5G/6G base stations.
2. **Digital Twin RF Simulator SaaS:**
   A cloud platform where telecom research teams can simulate and validate spatial MIMO channels under real-world Doppler and delay spread conditions without expensive hardware testbeds.
