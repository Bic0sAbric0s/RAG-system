from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, StateFilter
from aiogram.types import Message, FSInputFile
from simple_rag import ChromaRAG
from config import bot_token
import asyncio


bot = Bot(bot_token=bot_token)
dp = Dispatcher()

rag_system = ChromaRAG()

@dp.message(Command('start'))
async def strt_command(message: Message):
    welcome_text = f"""
                👋 Привет!

                Я RAG-бот для работы с документами. Вот что я умею:

                📚 Команды:
                /info - справка по использованию
                /help - основные команды
                /list - список загруженных документов
                /stats - статистика базы данных
                /add - добавление документов в базу

                📄 Загрузка документов:
                Просто отправьте мне PDF файл, и я его обработаю!

                ❓ Вопросы:
                Просто напишите вопрос, и я найду ответ в документах.

                🔍 Дополнительно:
                /clear - очистить историю диалога
                """
    await message.answer(welcome_text)

@dp.message(Command('help'))
async def strt_command(message: Message):
    help_text = f"""
                📖 Справка по использованию:

                1️⃣ Загрузите PDF документ (просто отправьте файл)
                2️⃣ Подождите обработку (обычно 10-30 секунд)
                3️⃣ Задавайте вопросы по документу

                💡 Советы:
                - Можно загружать несколько документов
                - Вопросы могут быть на русском или английском

                ⚠️ Ограничения:
                - Максимальный размер файла: 20MB
                - Поддерживаются только PDF
                - Время ответа: 10-15 секунд
                """
    await message.answer(help_text)

