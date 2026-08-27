from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command, StateFilter
from aiogram.types import Message, FSInputFile
from config import bot_token
import asyncio


bot = Bot(bot_token=bot_token)
dp = Dispatcher()


@dp.message(Command('start'))
async def strt_command(message):
    await message.answer('Привет! Я бот для анализа доку')