"""
Script to clear all users from the database
Run this to remove all existing users and start fresh
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")


async def clear_all_users():
    """Clear all users and related data from the database"""

    # MongoDB connection
    mongo_url = os.environ.get("MONGO_URL")
    if not mongo_url:
        print("Error: MONGO_URL not found in environment variables")
        return

    db_name = os.environ.get("DB_NAME", "keystroke_auth")

    client = AsyncIOMotorClient(mongo_url)
    db = client[db_name]

    try:
        # Get collections
        users_collection = db["users"]
        patterns_collection = db["keystroke_patterns"]
        auth_attempts_collection = db["authentication_attempts"]

        # Count before deletion
        users_count = await users_collection.count_documents({})
        patterns_count = await patterns_collection.count_documents({})
        attempts_count = await auth_attempts_collection.count_documents({})

        print("\nFound:")
        print(f"  - {users_count} users")
        print(f"  - {patterns_count} keystroke patterns")
        print(f"  - {attempts_count} authentication attempts")

        # Delete all data
        users_result = await users_collection.delete_many({})
        patterns_result = await patterns_collection.delete_many({})
        attempts_result = await auth_attempts_collection.delete_many({})

        print("\nDeleted:")
        print(f"  - {users_result.deleted_count} users")
        print(f"  - {patterns_result.deleted_count} keystroke patterns")
        print(f"  - {attempts_result.deleted_count} authentication attempts")

        print("\n✓ Database cleared successfully!")

    except Exception as e:
        print(f"Error clearing database: {e}")
    finally:
        client.close()


if __name__ == "__main__":
    print("=" * 60)
    print("CLEAR ALL USERS FROM DATABASE")
    print("=" * 60)
    print("\nThis will delete ALL users and related data.")
    confirmation = input("Are you sure you want to continue? (yes/no): ")

    if confirmation.lower() == "yes":
        asyncio.run(clear_all_users())
    else:
        print("\nOperation cancelled.")
