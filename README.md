# 🏥 WARD ZERO: Intelligent Home Healthcare Monitoring System

> **Hackathon Submission:** Operation Ward Zero (Nursio Innovation)  
> **Mission:** Sense → Understand → Predict → Alert → Respond  
> *"Turning a bedroom into a smart hospital ward."*

## 📖 Introduction
Hospitals have continuous monitoring, trained nurses, and immediate alarm systems. However, at home, patients rely on family caregivers who may miss early warning signs of deterioration, especially at night. 

**Ward Zero** is an AI-powered, software-based healthcare monitoring solution that acts as a 24x7 "Digital Nurse." It bridges the gap between hospital-grade monitoring and home care by using IoT data simulation, a Full-Stack dashboard, and AI/ML predictive analytics.

## ✨ Key Features (End-to-End Workflow)

*   **SENSE (IoT):** Simulated medical sensors publish real-time patient vitals (Heart Rate, SpO2, BP, Temp) over MQTT (QoS 1).
*   **UNDERSTAND (AI/ML):** A Random Forest Machine Learning model processes the data to generate a real-time Risk Score (0.0 to 1.0).
*   **PREDICT (Explainability):** The system explains *why* the risk is high (e.g., "Hypoxia detected: SpO2 dropping to 88%").
*   **ALERT (Full-Stack):** If Risk > 0.6, a red alert is triggered on the futuristic dashboard and saved to Supabase.
*   **RESPOND (AI Assistant):** Integrated **Groq AI (LLM)** generates actionable, non-diagnostic advice for the caregiver. Caregivers can also chat with the AI, add context notes, and acknowledge alerts.
*   **RESILIENCE:** Built-in retry buffer ensures data isn't lost if the internet drops. The UI shows a "Connection Lost" overlay and auto-reconnects.
*   **MULTILINGUAL UI:** Holographic Sci-Fi UI supports English, Hindi, and Hinglish.

## 🏗️ System Architecture

```text
[Sensor Simulator] --(MQTT)--> [FastAPI Backend] --(REST/WS)--> [Holographic UI]
                                      |
                                      +---> [Scikit-Learn ML Model] (Risk Score)
                                      +---> [Groq AI API] (Caregiver Advice)
                                      +---> [Supabase PostgreSQL] (History & Logs)