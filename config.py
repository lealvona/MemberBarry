"""Global Configs"""
import os

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')

OPENAI_TEMPERATURE = os.environ.get('OPENAI_TEMPERATURE', 0.4)
OPENAI_MAX_TOKENS = os.environ.get('OPENAI_MAX_TOKENS', 8192)
OPENAI_STT_ENGINE = os.environ.get('OPENAI_STT_ENGINE', 'whisper-1')
OPENAI_MODEL = os.environ.get('OPENAI_MODEL', 'gpt-3.5-turbo')  # or "gpt-4" if ya got it  # noqa

# SUMMARY_TOKEN_LIMIT = os.environ.get('SUMMARY_TOKEN_LIMIT', 200)
CONTEXT_TOKEN_LIMIT = os.environ.get('CONTEXT_TOKEN_LIMIT', 500)
