from aiogram import Bot, Dispatcher, Router, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardButton, CallbackQuery
from aiogram.enums import ChatAction
from aiogram.types import Message, FSInputFile, InlineKeyboardMarkup, InlineKeyboardButton, KeyboardButton
from simple_rag import ChromaRAG
from config import bot_token
import asyncio
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

bot = Bot(token=bot_token)
dp = Dispatcher()
router = Router()
dp.include_router(router)

rag_system = ChromaRAG(persist_directory='./my_documents_db')

docs_pee_page = 10
user_sessions = {}

def get_user_session(user_id: int):
    if user_id not in user_sessions:
        user_sessions[user_id] = {
            'history': [],
            'last_activity': datetime.now(),
            'questions_asked': 0,
            'documents_loaded': 0,
            'preferences': {
                'language': 'ru',
                'detail_level': 'normal',
                'show_sources': True
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
                /delete - удаление последнего документа в базе данных
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

def build_documents_text(documents: list, page: int = 0) -> str:
    start = page * docs_pee_page
    end = start + docs_pee_page
    page_docs = documents[start:end]

    total_pages = (len(documents) + docs_pee_page - 1) // docs_pee_page

    text = f'📚 В базе {len(documents)} документов\n'

    for i, doc in enumerate(page_docs, start=start + 1):
        doc_name = doc.get('name', 'Unknown')
        doc_chunks = doc.get('chunks', 0)
        doc_date = doc.get('added_date', 'Unknown')

        text += f'{i}. {doc_name}\n'
        text += f'   📊 Чанков: {doc_chunks}\n'
        if doc_date != 'Unknown':
            text += f'   📅 Добавлен: {doc_date[:10]}\n'
        text += '\n'

    return text

def build_pagination_keyboard(page: int, total_pages: int):
    builder = InlineKeyboardBuilder()

    buttons = []
    if page > 0:
        buttons.append(InlineKeyboardButton(
            text='⬅️ Предыдущие',
            callback_data=f'docs_page_{page - 1}'
        ))

    buttons.append(InlineKeyboardButton(
        text=f'{page + 1}/{total_pages}',
        callback_data='docs_page_ignore'
    ))

    if page < total_pages - 1:
        buttons.append(InlineKeyboardButton(
            text='Следующие ➡️',
            callback_data=f'docs_page_{page + 1}'
        ))

    builder.row(*buttons)
    return builder.as_markup()

@router.message(Command('list'))
async def list_command(message: Message):
    if hasattr(rag_system, 'list_document'):
        documents = rag_system.list_document()
    else:
        documents = []

    if not documents:
        await message.answer('В базе пока нет документов.')
        return

    total_pages = (len(documents) + docs_pee_page - 1) // docs_pee_page
    text = build_documents_text(documents, page=0)

    if total_pages <= 1:
        await message.answer(text)
        return

    await message.answer(
        text,
        reply_markup=build_pagination_keyboard(page=0, total_pages=total_pages)
    )

@router.callback_query(lambda c: c.data.startswith('docs_page_'))
async def paginate_documents(callback: CallbackQuery):
    page_str = callback.data.replace('docs_page_', '')

    if page_str == 'ignore':
        await callback.answer()
        return

    page = int(page_str)

    if hasattr(rag_system, 'list_document'):
        documents = rag_system.list_document()
    else:
        documents = []

    if not documents:
        await callback.answer('Документы не найдены')
        return

    total_pages = (len(documents) + docs_pee_page - 1) // docs_pee_page

    if page < 0:
        page = 0
    elif page >= total_pages:
        page = total_pages - 1

    text = build_documents_text(documents, page=page)

    try:
        await callback.message.edit_text(
            text,
            reply_markup=build_pagination_keyboard(page=page, total_pages=total_pages)
        )
    except Exception as e:
        await callback.answer()

    await callback.answer()

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
        await message.answer('❌ Ошибка при получении статистики.')

@router.message(Command('delete'))
async def list_command(message: Message):
    result = rag_system.delete_last_document()
    
    await message.answer(result[1])

@router.message(F.document)
async def handle_document(message: Message, state: FSMContext):
    document = message.document
    file_name = document.file_name
    file_extension = os.path.splitext(file_name)[1].lower()

    extensions = ['.pdf', '.docx', '.txt']
    if file_extension not in extensions:
        await message.answer('❌ Неподдерживаемый формат!')
        return
    
    processing_msg = await message.answer(
        '📥 Получаю документ...'
        f'📄 {document.file_name}'
    )

    # try:
    await processing_msg.edit_text('⌛ Документ получен! Это может занять 1-2 минуты')

    file = await bot.get_file(document.file_id)
    file_bytes = await bot.download_file(file.file_path)

    doc = rag_system.add_documet(file_name, 'bytes', file_bytes.read())
    await processing_msg.edit_text('✅ Документ успешно обработан!')

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


    # await message.answer('❌ RAG система не настроена для вопросов.')
    # return

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())

    