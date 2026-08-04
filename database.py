import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient

logger = logging.getLogger(__name__)

_client = None
_db = None

def get_db():
    """Initializes and returns the MongoDB database instance."""
    global _client, _db
    if _client is None:
        mongo_uri = os.getenv("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MONGO_URI environment variable is not set. Please add it to your .env file.")
        
        # Create a new motor client
        _client = AsyncIOMotorClient(mongo_uri)
        
        # Use a default database name, can also be configured via env
        _db = _client.get_database(os.getenv("MONGO_DB_NAME", "voice_agent_db"))
        
    return _db

async def save_lead_data(structured_data: dict, transcript: list) -> bool:
    """
    Saves the extracted lead data and full transcript to the MongoDB collection.
    
    Args:
        structured_data (dict): The JSON object extracted by the LLM.
        transcript (list): The list of message dictionaries from the conversation context.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    try:
        db = get_db()
        collection = db["leads"]
        
        document = {
            "lead_info": structured_data,
            "transcript": transcript
        }
        
        result = await collection.insert_one(document)
        logger.info(f"Successfully saved lead data with ID: {result.inserted_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to save lead data to MongoDB: {e}", exc_info=True)
        return False
