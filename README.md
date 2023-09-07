# MemberBarry

![Just Barry](docs/img/barry.png)

## Dependencies
- `backoff`
- `chromadb`
- `openai`
- `openai[embeddings]`
- `python-dotenv`
- `tiktoken`
- `shortuuid`
---

## MemberBarry

### Overview

`MemberBarry` is the main class in this project. It is responsible for session management, conversation handling, and interaction with the OpenAI API. The class can summarize text, and manage conversation context. It can also do some cool tricks like transcribe audio using `Whisper`.


## AIDatabase

### Overview

`AIDatabase` is a Python class that handles SQLite database operations for storing and retrieving conversation sessions and summaries. It is designed to support the `MemberBarry` class by providing database functionalities such as creating tables, inserting data, and fetching records. There are two simultaneous data stores in this module. A standard SQLLite DB to store a plain text representation of ALL interactions which is used for immediate context and running summary, and a chroma vector db running on a persistant sqlite backend which provides similarity search and long-term memory.

---


## Links

- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Python SQLite Documentation](https://docs.python.org/3/library/sqlite3.html)
