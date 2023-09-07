# MemberBarry

![Just Barry](docs/img/barry.png)

## Overview

`MemberBarry` is a Python class that serves as a wrapper for the OpenAI API. It is designed to manage conversations with the API while retaining the context of each session. It provides methods for tasks such as audio transcription, text summarization, and conversation handling.

---

## Dependencies
- `openai`
- `openai[embeddings]`  # Soon
- `tiktoken`
- `backoff`
- `python-dotenv`

---

## MemberBarry

### Overview

`MemberBarry` is the main class in this project. It is responsible for session management, conversation handling, and interaction with the OpenAI API. The class can summarize text, and manage conversation context. It can also do some cool tricks like transcribe audio using `Whisper`.


## AIDatabase

### Overview

`AIDatabase` is a Python class that handles SQLite database operations for storing and retrieving conversation sessions and summaries. It is designed to support the `MemberBarry` class by providing database functionalities such as creating tables, inserting data, and fetching records.

---


## Links

- [OpenAI API Documentation](https://platform.openai.com/docs/)
- [Python SQLite Documentation](https://docs.python.org/3/library/sqlite3.html)
