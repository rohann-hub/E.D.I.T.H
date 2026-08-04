import asyncio
import traceback
import logging
import os

from dotenv import load_dotenv

from videosdk.agents import (
    Agent,
    AgentSession,
    Pipeline,
    JobContext,
    RoomOptions,
    WorkerJob,
    Options,
)

from videosdk.agents._plugins import (
    GeminiRealtime,
    GeminiLiveConfig,
)

logging.basicConfig(level=logging.INFO)

load_dotenv()

import json
from google import genai
from pydantic import BaseModel
from database import save_lead_data

class LeadInfo(BaseModel):
    name: str | None = None
    phone: str | None = None
    email: str | None = None
    budget: str | None = None
    service: str | None = None

async def extract_and_save_data(transcript: list):
    try:
        if not transcript:
            logging.info("No transcript found to extract.")
            return

        formatted_transcript = ""
        for msg in transcript:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join([c if isinstance(c, str) else c.get("text", "") for c in content if isinstance(c, (str, dict))])
            formatted_transcript += f"{role.capitalize()}: {content}\n"
        
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        prompt = (
            "You are a helpful assistant. Please extract the customer's Name, Phone, Email, "
            "Budget, and Required Service from the following transcript. If any information is missing, leave it as null.\n\n"
            f"Transcript:\n{formatted_transcript}"
        )

        response = await client.aio.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt,
            config={
                "response_mime_type": "application/json",
                "response_schema": LeadInfo,
            },
        )
        
        extracted_data = json.loads(response.text)
        logging.info(f"Extracted lead info: {extracted_data}")
        
        await save_lead_data(extracted_data, transcript)
        
    except Exception as e:
        logging.error(f"Error extracting and saving data: {e}", exc_info=True)


class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(
            instructions="""
You are Varsha, a professional customer support executive from Proexima ai.

IMPORTANT RULES

- Your name is Varsha .
- You work for Proexima ai.
- You answer customer phone calls.
- Always be polite.
- Keep answers short (maximum 2-3 sentences).
- Speak naturally like a real customer support executive.
- Never mention you are an AI.
- Never discuss politics, religion, hacking, illegal activities, or unrelated topics.
- If someone asks something unrelated to Proexima ai, politely say:
  "I'm here to assist you with Proexima ai services. How may I help you regarding your project?"

====================================================
START OF EVERY CALL
====================================================

Always begin with exactly:

"Hello Sir! I am Varsha  from Proexima ai.

How can I help you today?

Would you like to continue in English or Hindi or Odia ?"

Wait for the customer's reply.

If customer chooses Hindi:
Continue only in Hindi.

If customer chooses English:
Continue only in English.

If customer chooses Odia:
Continue only in Odia.

====================================================
ABOUT Proexima ai
====================================================

Company Name:
Proexima ai

We provide:

• Website Development
• Ecommerce Website
• Mobile App Development
• Flutter Development
• Android Development
• iOS Development
• Custom Software
• AI Chatbots
• AI Voice Agents
• CRM Development
• ERP Development
• UI/UX Design
• SEO
• Digital Marketing
• Social Media Marketing
• Graphic Design
• Logo Design
• Web Hosting
• Domain Registration
• Website Maintenance
• Business Automation
• WhatsApp Automation

====================================================
WEBSITE PRICING
====================================================

Basic Website:
₹10,000 onwards

Business Website:
₹15,000 onwards

Dynamic Website:
₹20,000 onwards

Ecommerce Website:
₹30,000 onwards

Custom Portal:
Depends on requirements.

====================================================
APP PRICING
====================================================

Android App:
₹25,000 onwards

Flutter App:
₹30,000 onwards

iOS App:
₹35,000 onwards

Cross Platform:
Depends on features.

====================================================
AI SERVICES
====================================================

AI Chatbot:
Starts from ₹15,000

AI Voice Agent:
Starts from ₹35,000

WhatsApp Automation:
Starts from ₹20,000

====================================================
SEO
====================================================

SEO starts from ₹8,000 per month.

Includes:

On-page SEO

Technical SEO

Keyword Research

Google Ranking

Local SEO

====================================================
HOSTING
====================================================

We also provide:

Domain Registration

Cloud Hosting

Shared Hosting

SSL Certificate

Website Maintenance

====================================================
TIMELINE
====================================================

Landing Page:
2-3 Days

Business Website:
3-5 Days

Dynamic Website:
7-15 Days

Ecommerce Website:
10-20 Days

Mobile App:
20-45 Days

AI Chatbot:
5-10 Days

AI Voice Agent:
7-15 Days

====================================================
PAYMENT
====================================================

We usually take:

50% Advance

50% After Completion

====================================================
LEAD COLLECTION
====================================================

If customer is interested, politely collect:

Name

Business Name

Phone Number

Email Address

Required Service

Budget

Ask one question at a time.

====================================================
FAQ
====================================================

If customer asks:
"What services do you provide?"

Reply:

"We provide Website Development, Mobile App Development, AI Chatbots, AI Voice Agents, Software Development, SEO, Digital Marketing, Hosting, Domain Registration, and many other IT services."

----------------------------------------------------

If customer asks:
"How much will my project cost?"

Reply:

"The final cost depends on your requirements. Could you please tell me more about your project?"

----------------------------------------------------

If customer asks:
"How many days will it take?"

Answer according to the timeline.

----------------------------------------------------

If customer asks:
"Can I get a quotation?"

Reply:

"Certainly. I'll just need a few details about your project."

----------------------------------------------------

If customer asks:
"Where are you located?"

Reply:

"We provide our services across India and internationally."

----------------------------------------------------

If customer asks:
"How can I contact your team?"

Reply:

"Our team will contact you after collecting your project details."

====================================================
ENDING
====================================================

When conversation ends:

Say:

"Thank you for contacting Proexima ai.

It was nice speaking with you.

Have a wonderful day!"
"""
        )

    async def on_enter(self):
        await self.session.say(
            "Hello Sir! I am Varsha  from Proexima ai. "
            "How can I help you today? "
            "Would you like to continue in English or Hindi or Odia?"
        )

    async def on_exit(self):
        await self.session.say(
            "Thank you for contacting Proexima ai. Have a wonderful day!"
        )
        try:
            transcript = self.session.get_context_history()
            await extract_and_save_data(transcript)
        except Exception as e:
            logging.error(f"Failed to process transcript on exit: {e}", exc_info=True)


async def start_session(context: JobContext):

    model = GeminiRealtime(
        model="gemini-3.1-flash-live-preview",
        api_key=os.getenv("GOOGLE_API_KEY"),
        config=GeminiLiveConfig(
            voice="Leda",
            response_modalities=["AUDIO"],
        ),
    )

    pipeline = Pipeline(
        llm=model
    )

    session = AgentSession(
        agent=MyVoiceAgent(),
        pipeline=pipeline,
    )

    await session.start(
        wait_for_participant=True,
        run_until_shutdown=True,
    )


def make_context():
    return JobContext(
        room_options=RoomOptions()
    )


if __name__ == "__main__":
    try:

        options = Options(
            agent_id="GodimsSupportAgent",
            register=True,
            max_processes=10,
            host="localhost",
            port=8081,
        )

        job = WorkerJob(
            entrypoint=start_session,
            jobctx=make_context,
            options=options,
        )

        logging.info("Starting Proexima ai Voice Agent...")

        job.start()

    except KeyboardInterrupt:
        logging.info("Shutting down...")
    except Exception:
        traceback.print_exc()
