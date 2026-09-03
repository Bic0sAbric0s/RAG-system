from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.enums import ChatAction
from aiogram.types import Message, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton, KeyboardButton
from simple_rag import ChromaRAG
from config import bot_token
import asyncio
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

bot = Bot(token=bot_token)
dp = Dispatcher()
router = Router()
dp.include_router(router)

rag_system = ChromaRAG(persist_directory='./my_documents_db')

user_sessions = {}

def get_user_session(user_id: int):
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            "history": [],
            "last_activity": datetime.now(),
            "questions_asked": 0,
            "documents_loaded": 0,
            "preferences": {
                "language": "ru",
                "detail_level": "normal",
                "show_sources": True
            }
        }
    return user_sessions[user_id]

@router.message(Command('start'))
async def start_command(message: Message):
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

@router.message(Command('help'))
async def help_command(message: Message):
    help_text = f"""
                📚 Команды:
                /info - справка по использованию
                /help - основные команды
                /list - список загруженных документов
                /stats - статистика базы данных
                /clearcache - очиста кэша
                /cleardb - очиста базы данных
                """
    await message.answer(help_text)

@router.message(Command('info'))
async def info_command(message: Message):
    info_text = f"""
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
    await message.answer(info_text)

@router.message(Command('list'))
async def list_command(message: Message):
    if hasattr(rag_system, 'list_document'):
        document = rag_system.list_document()

    text = f'В базе {len(document)} количество документов\n'
    for i, doc in enumerate(document, 1):
        if i <= 10:
            doc_name = doc.get('name', 'Unknown')
            doc_chunks = doc.get('chunks', 0)
            doc_date = doc.get('added_date', 'Unknown')
            
            text += f"{i}. {doc_name}\n"
            text += f"   📊 Чанков: {doc_chunks}\n"
            if doc_date != 'Unknown':
                text += f"   📅 Добавлен: {doc_date[:10]}\n"
            text += "\n"
        else:
            break
    
    await message.answer(text)

@router.message(Command('stats'))
async def stats_command(message: Message):
    try:
        user_id = message.from_user.id
        session = get_user_session(user_id)

        if hasattr(rag_system, 'get_stats'):
            rag_stats = rag_system.get_stats()
        else:
            rag_stats = {}
    
        stats_text = f"""
                    📚 База данных:
                    • Документов: {rag_stats.get('unique_documents', 0)}
                    • Всего чанков: {rag_stats.get('total_chunks', 0)}

                    👤 Ваша сессия:
                    • Вопросов задано: {session['questions_asked']}
                    • Документов загружено: {session['documents_loaded']}
                    • В диалоге сообщений: {len(session['history'])}
                    """
        await message.answer(stats_text) 

    except Exception:
        await message.answer("❌ Ошибка при получении статистики.")

@router.message(Command('clearcache'))
async def clearcache_command(message: Message):
    # try:
    if hasattr(rag_system, 'clear_cache'):
        result = rag_system.clear_cache()
    
        await message.answer(result)
    # except Exception:
    #     await message.answer("❌ Ошибка при очистке кэша")
    

@router.message(Command('cleardb'))
async def cleardb_command(message: Message):
    pass

@router.message(F.text)
async def handle_text(message: Message, state: FSMContext):
    text = message.text.strip()

    await message.bot.send_chat_action(
        chat_id=message.chat.id,
        action=ChatAction.TYPING
    )

    if hasattr(rag_system, 'ask'):
        result = rag_system.ask(text)
    
    if isinstance(result, dict):
        answer = result.get('answer', 'Нет ответа')
    else:
        answer = str(result)

    await message.answer(answer)


    # await message.answer("❌ RAG система не настроена для вопросов.")
    # return

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())

    